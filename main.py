import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from modules.rag_pipeline import pdf_search, create_vector_db
from langchain_community.llms import Ollama   # ✅ Ollama LLM

# Load environment variables
load_dotenv()

app = FastAPI(title="Legal Assistant API", description="AI-powered legal document assistant")

# Global PDF storage
PDF_PATH = None

# Initialize local LLM (Ollama Mistral)
print("[DEBUG] Initializing local Mistral (Ollama) at startup...")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
print(f"[DEBUG] Using local Ollama model: {MODEL_NAME}")

llm = Ollama(model=MODEL_NAME)
print("[DEBUG] Ollama Mistral model initialized successfully")

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Legal Assistant API is running with local Mistral via Ollama!"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global PDF_PATH
    try:
        upload_dir = "data"
        os.makedirs(upload_dir, exist_ok=True)
        file_content = await file.read()
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(file_content)

        PDF_PATH = file_path

        import modules.rag_pipeline as rag_pipeline
        rag_pipeline.PDF_PATH = PDF_PATH
        rag_pipeline.create_vector_db()

        return {"message": f"PDF uploaded and vector database created for {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question_endpoint(request: QuestionRequest):
    try:
        import modules.rag_pipeline as rag_pipeline
        answer = rag_pipeline.pdf_search(request.question, llm)
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    try:
        from pymongo import MongoClient
        client = MongoClient(os.getenv("MONGODB_URI"))
        collection = client["documents"]["documents_vectors"]
        doc_count = collection.count_documents({})
        
        return {
            "status": "healthy",
            "pdf_loaded": PDF_PATH is not None,
            "pdf_path": PDF_PATH,
            "mongodb_documents": doc_count,
            "llm_initialized": llm is not None
        }
    except Exception as e:
        return {
            "status": "error",
            "pdf_loaded": PDF_PATH is not None,
            "pdf_path": PDF_PATH,
            "mongodb_documents": "error",
            "llm_initialized": llm is not None,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
