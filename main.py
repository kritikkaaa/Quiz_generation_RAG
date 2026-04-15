# RAG API

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from pydantic import BaseModel
from typing import List
import os
import time

#RAG PIPELINE
from rag_pipeline import (
    load_pdfs,
    chunk_text,
    clear_collection,
    add_to_chroma,
    retrieve_with_rerank,
    generate_quiz_with_context
)

app = FastAPI(title="RAG MCQ Quiz Generation API")

# ================================
# RATE LIMITING
# ================================

last_called = {}

def rate_limit(ip):
    now = time.time()
    if ip in last_called and now - last_called[ip] < 2:
        return False
    last_called[ip] = now
    return True

# ================================
# REQUEST MODEL
# ================================

class QuizRequest(BaseModel):
    topic: str
    difficulty: str = "easy"

# ================================
# BACKGROUND INGESTION
# ================================

def process_documents(file_paths: List[str]):
    text = ""

    for path in file_paths:
        if path.endswith(".pdf"):
            text += load_pdfs([path])

        elif path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text += f.read()

    chunks = chunk_text(text)

    clear_collection()
    add_to_chroma(chunks)

    print("Documents processed and stored")

# ================================
# API ENDPOINTS
# ================================

#Upload Documents
@app.post("/upload")
def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    os.makedirs("temp", exist_ok=True)

    file_paths = []

    for file in files:
        path = f"temp/{file.filename}"
        with open(path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(path)

    background_tasks.add_task(process_documents, file_paths)

    return {
        "message": "Files uploaded. Processing started in background."
    }


#Generate MCQ Quiz (MAIN FEATURE)
@app.post("/generate-quiz")
def generate_quiz(request: QuizRequest, req: Request):

    ip = req.client.host
    if not rate_limit(ip):
        return {"error": "Too many requests. Try again later."}

    quiz = generate_quiz_with_context(
        topic=request.topic,
        difficulty=request.difficulty
    )

    return {
        "topic": request.topic,
        "difficulty": request.difficulty,
        "quiz": quiz
    }



# ================================
# RUN SERVER
# ================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
