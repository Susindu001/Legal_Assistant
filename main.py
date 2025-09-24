import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from modules.rag_pipeline import pdf_search, create_vector_db, list_uploaded_documents
from langchain_community.llms import Ollama

load_dotenv()
app = FastAPI(title="Legal Assistant API", description="AI-powered legal document assistant")

# Initialize local LLM (Ollama Mistral)
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
llm = Ollama(model=MODEL_NAME)

# === Request/Response Models ===
class UploadPDFResponse(BaseModel):
    message: str
    doc_name: str

class QuestionRequest(BaseModel):
    question: str
    doc_name: str = None  # Optional, filter by uploaded document

class QuestionResponse(BaseModel):
    answer: str

# === Endpoints ===
@app.get("/")
async def root():
    return {"message": "Legal Assistant API is running with local Mistral via Ollama!"}

@app.post("/upload_pdf", response_model=UploadPDFResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        upload_dir = "data"
        os.makedirs(upload_dir, exist_ok=True)
        file_content = await file.read()
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(file_content)

        doc_name = file.filename
        print(f"before entering create_vector_db with {file_path} and {doc_name}")
        create_vector_db(file_path, doc_name)
        print(f"after exiting create_vector_db with {file_path} and {doc_name}")

        return UploadPDFResponse(message="PDF uploaded and vector database updated.", doc_name=doc_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question_endpoint(request: QuestionRequest):
    try:
        answer = pdf_search(query=request.question, llm=llm, doc_name=request.doc_name)
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_documents")
async def list_documents():
    try:
        docs = list_uploaded_documents()
        return {"uploaded_documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv("MONGODB_URI"))
        collection = client["documents"]["documents_vectors"]
        doc_count = collection.count_documents({})
        return {"status": "healthy", "mongodb_documents": doc_count, "llm_initialized": llm is not None}
    except Exception as e:
        return {"status": "error", "error": str(e)}
