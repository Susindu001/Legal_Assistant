from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from modules.vector_store import load_db
from config import MODEL_NAME

def ask_question(query: str):
    db = load_db()
    retriever = db.as_retriever()
    llm = HuggingFaceHub(repo_id=MODEL_NAME)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(query)
