from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from config import EMBED_MODEL, MONGODB_URI, MONGODB_DB, MONGODB_COLLECTION

# Initialize Mongo Client
client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
collection = db[MONGODB_COLLECTION]

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def create_or_update_db(chunks):
    vector_store = MongoDBAtlasVectorSearch.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection=collection,
        index_name="contract_vector_index"
    )
    return vector_store

def load_db():
    return MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=collection,
        index_name="contract_vector_index"
    )
