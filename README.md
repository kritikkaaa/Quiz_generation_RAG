# RAG-Based MCQ Quiz Generation API

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for generating high-quality MCQ quizzes from user-uploaded documents. Unlike generic QA systems, this pipeline is specifically designed to extract meaningful academic content, retrieve relevant context using embeddings + reranking, and generate structured MCQs using an LLM.

## Objective
To build an applied AI system demonstrating embedding-based retrieval, vector search, background processing, and API-based interaction.

## System Architecture
User → FastAPI → /upload → Background Job → Chunking → Embeddings → Vector DB → /generate-quiz → Retrieval → Reranking → LLM → MCQ Output

## Tech Stack
- Framework: FastAPI
- LLM: FLAN-T5 (HuggingFace Transformers)
- Embeddings: SentenceTransformers (all-mpnet-base-v2)
- Vector Database: ChromaDB
- Reranking: BM25 (rank_bm25)
- PDF Parsing: PyPDF
- Validation: Pydantic

## Repository Structure
.
├── main.py              # FastAPI application (API layer)
├── rag_banao_tech.py    # Core RAG pipeline
└── README.md

## Workflow
1. Document Upload: Accepts PDF and TXT files, files are temporarily stored
2. Background Processing: Extract text, split into chunks, generate embeddings, store in vector database
3. Quiz Generation: Retrieve relevant chunks using semantic search, apply BM25 reranking for precision, generate MCQs using LLM with strict formatting

## API Endpoints

POST /upload
Upload one or more PDF/TXT files. Processing happens asynchronously.

POST /generate-quiz
Request Body: {"topic": "Biology", "difficulty": "easy"}
Response: Returns structured MCQs from retrieved context

GET /retrieve
Query parameter: ?topic=Biology - Returns top retrieved chunks for debugging.

## Key Features
- RAG-based MCQ generation (not generic QA)
- Semantic search using embeddings
- Hybrid retrieval (Dense + BM25 reranking)
- Background ingestion pipeline
- Support for PDF & TXT documents
- Structured MCQ generation with constraints
- Basic rate limiting
- Latency tracking

## Setup Instructions

1. Clone Repository
git clone <https://github.com/kritikkaaa/Quiz_generation_RAG>
cd <Quiz_generation_RAG>

2. Install Dependencies
pip install fastapi uvicorn chromadb sentence-transformers transformers langchain-text-splitters rank_bm25 pypdf python-multipart

3. Run the API
uvicorn main:app --reload

4. Access API Docs
http://127.0.0.1:8000/docs

## Evaluation Criteria Mapping

| Requirement | Implementation |
|-------------|----------------|
| Chunking Strategy | Fixed-size chunking with overlap (500 chars, 90 overlap) |
| Retrieval Quality | Embeddings + BM25 reranking |
| API Design | FastAPI with modular endpoints |
| Metrics Awareness | Latency tracking |
| System Explanation | Detailed below |

## Design Decisions

Chunk Size (500 chars, 90 overlap):
500 characters fits ~3–5 sentences — enough context for a single concept without diluting the embedding vector with unrelated material. The 90-character overlap ensures a sentence split across a chunk boundary doesn't vanish from retrieval entirely.

Retrieval Failure Case Observed:
When a query used different vocabulary than the document (e.g. query: "glucose production", document: "sugar synthesis"), dense retrieval scored poorly. Adding BM25 reranking on top of the dense results significantly helped because BM25 does exact keyword matching regardless of semantic similarity.

Metric Tracked:
End-to-end query latency (latency_ms in every /generate-quiz response). Typical range: 800–2000ms depending on LLM generation. Retrieval alone is under 100ms; the bulk of latency is dominated by FLAN-T5 generation.

## Author
[Kritika Kamboj]
