import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MODEL_NAME = "google/flan-t5-base"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Get MongoDB config from environment variables (fallback to defaults for development)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB = os.getenv("MONGODB_DB", "documents")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "documents_vectors")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
