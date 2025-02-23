require("dotenv").config();
const express = require("express");
const multer = require("multer");
const pdf = require("pdf-parse");
const csvParser = require("csv-parser");
const mammoth = require("mammoth");
const { Readable } = require("stream");
const { Pinecone } = require("@pinecone-database/pinecone");
const { Document } = require("langchain/document");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const axios = require("axios");
const { v4: uuidv4 } = require("uuid");
const cors = require("cors");
const { Mutex } = require("async-mutex"); // For thread-safe conversation history

const app = express();
app.use(cors());
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // Increased file size limit to 50MB
});
const port = process.env.PORT || 5000;

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const INDEX_CONFIG = {
  name: "doc-chat-index",
  dimension: 1024,
  metric: "cosine",
  spec: {
    serverless: {
      cloud: "aws",
      region: "us-east-1",
    },
  },
};

const EMBEDDING_MODEL = "multilingual-e5-large";

let pineconeIndex;

const conversationHistory = new Map();
const historyMutex = new Mutex(); // Mutex for thread-safe history updates

async function initializePinecone() {
  try {
    const indexList = await pinecone.listIndexes();

    if (!indexList.indexes?.some((idx) => idx.name === INDEX_CONFIG.name)) {
      await pinecone.createIndex(INDEX_CONFIG);
      await new Promise((resolve) => setTimeout(resolve, 60000));
    }

    pineconeIndex = pinecone.Index(INDEX_CONFIG.name);
  } catch (error) {
    console.error("Pinecone initialization failed:", error);
    process.exit(1);
  }
}

async function generateEmbeddings(texts, inputType = "passage") {
  try {
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new Error("Texts must be a non-empty array");
    }

    console.log("Generating embeddings for:", texts.slice(0, 3)); // Logs first 3 chunks

    const response = await pinecone.inference.embed(EMBEDDING_MODEL, texts, {
      inputType,
      truncate: "END",
      batchSize: 32,
    });

    console.log("Raw Embedding Response:", JSON.stringify(response, null, 2));

    let embeddingsArray;
    if (Array.isArray(response)) {
      embeddingsArray = response;
    } else if (response && Array.isArray(response.data)) {
      embeddingsArray = response.data;
    } else {
      throw new Error("Invalid embedding response format");
    }

    return embeddingsArray.map((e) => e.values);
  } catch (error) {
    console.error("Embedding generation failed:", error);
    throw new Error("Failed to generate embeddings: " + error.message);
  }
}

async function processDocument(fileBuffer, mimetype) {
  try {
    let text = "";

    if (mimetype === "application/pdf") {
      const data = await pdf(fileBuffer);
      text = data.text;
    } else if (mimetype === "text/plain") {
      text = fileBuffer.toString("utf-8");
    } else if (
      mimetype === "text/csv" ||
      mimetype === "application/vnd.ms-excel"
    ) {
      text = await processCSV(fileBuffer);
    } else if (
      mimetype ===
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ) {
      text = await processDOCX(fileBuffer);
    } else {
      throw new Error("Unsupported file type");
    }

    if (!text || text.trim().length === 0) {
      throw new Error("Document contains no readable text");
    }

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    return splitter.splitDocuments([new Document({ pageContent: text })]);
  } catch (error) {
    console.error("Document processing failed:", error);
    throw new Error("Failed to parse document: " + error.message);
  }
}

async function processCSV(fileBuffer) {
  return new Promise((resolve, reject) => {
    let text = "";
    const stream = Readable.from(fileBuffer.toString("utf-8"));

    stream
      .pipe(csvParser())
      .on("data", (row) => {
        text += Object.values(row).join(" ") + "\n"; // Combine row values
      })
      .on("end", () => resolve(text))
      .on("error", (err) => reject(err));
  });
}

async function processDOCX(fileBuffer) {
  const result = await mammoth.extractRawText({ buffer: fileBuffer });
  return result.value;
}

async function storeDocumentChunks(docs, sessionId) {
  try {
    const texts = docs.map((doc) => doc.pageContent);
    const embeddings = await generateEmbeddings(texts, "passage");

    const vectors = embeddings.map((values, i) => ({
      id: `doc-${sessionId}-${Date.now()}-${i}`,
      values,
      metadata: {
        text: texts[i],
        timestamp: Date.now(),
        sessionId, // Store session ID in metadata
      },
    }));

    await pineconeIndex.namespace(`user-${sessionId}`).upsert(vectors);
  } catch (error) {
    console.error("Vector storage failed:", error);
    throw new Error("Failed to store document");
  }
}

async function processQuery(question, sessionId) {
  try {
    const [embedding] = await generateEmbeddings([question], "query");

    const response = await pineconeIndex.namespace(`user-${sessionId}`).query({
      topK: 5,
      vector: embedding,
      includeMetadata: true,
      filter: {
        sessionId, // Ensure only this user's documents are queried
        timestamp: { $gte: Date.now() - 604800000 },
      },
    });

    return response.matches.map((m) => m.metadata.text);
  } catch (error) {
    console.error("Query processing failed:", error);
    throw new Error("Failed to process query");
  }
}

async function generateResponse(context, history, question) {
  const messages = [
    {
      role: "system",
      content: `You are a friendly and knowledgeable document expert. Use the following extracted context from the user's document to help answer their questions. Remember the conversation history and tailor your responses to be personalized and engaging.

Document context:
${context.join("\n")}

Conversation history:
${history.map(([q, a]) => `User: ${q}\nAssistant: ${a}`).join("\n")}

Now, answer the following question in a clear, conversational, and personalized tone:`,
    },
    { role: "user", content: question },
  ];

  try {
    const response = await axios.post(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        messages,
        model: "llama-3.3-70b-versatile",
        temperature: 0.7,
        max_tokens: 500,
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          "Content-Type": "application/json",
        },
      }
    );

    return response.data.choices[0].message.content;
  } catch (error) {
    console.error("LLM API error:", error.response?.data || error.message);
    throw new Error("Failed to generate response");
  }
}

app.post("/upload", upload.single("document"), async (req, res) => {
  try {
    if (!req.file) throw new Error("No file uploaded");
    const { sessionId } = req.body; // Require session ID

    if (!sessionId) throw new Error("Session ID is required");

    // Supported file types
    const allowedTypes = [
      "text/plain",
      "application/pdf",
      "text/csv",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ];

    if (!allowedTypes.includes(req.file.mimetype)) {
      throw new Error(
        "Invalid file type. Only TXT, PDF, CSV, and DOCX are allowed"
      );
    }

    const docs = await processDocument(req.file.buffer, req.file.mimetype);

    if (!docs || docs.length === 0) {
      throw new Error("No valid text extracted from document");
    }

    await storeDocumentChunks(docs, sessionId);
    res.json({ success: true, message: "Document processed successfully" });
  } catch (error) {
    res.status(400).json({ success: false, error: error.message });
  }
});

app.post("/chat", express.json(), async (req, res) => {
  try {
    const { question, sessionId } = req.body;

    if (!sessionId) throw new Error("Session ID is required");
    if (!question?.trim()) throw new Error("Question is required");
    if (question.length > 1000) throw new Error("Question too long");

    const release = await historyMutex.acquire(); // Acquire mutex lock
    try {
      let history = conversationHistory.get(sessionId) || [];
      const context = await processQuery(question, sessionId);
      const response = await generateResponse(context, history, question);

      const newHistory = [
        ...history,
        [question.slice(0, 500), response.slice(0, 2000)],
      ];
      if (newHistory.length > 10) newHistory.shift();
      conversationHistory.set(sessionId, newHistory);

      res.json({ response, sessionId });
    } finally {
      release();
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

initializePinecone().then(() => {
  app.listen(port, () => {
    console.log(`Server operational on port ${port}`);
    console.log(`Pinecone index: ${INDEX_CONFIG.name}`);
    console.log(`Embedding model: ${EMBEDDING_MODEL}`);
  });
});
