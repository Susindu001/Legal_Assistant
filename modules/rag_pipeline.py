import os
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
#from main import llm  # Ensure llm is initialized in main.py before using here

# === Embedding model ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

# === Configuration ===
PDF_PATH = "./data/Lease-Agreement-Template-for-landlords-dd-15-Mar-22.pdf"
MONGODB_URI = "mongodb+srv://susindugajanayake:xFwZvwzWBIUtMuvL@cluster0.z40kp.mongodb.net/"
DB_NAME = "documents"
COLLECTION_NAME = "documents_vectors"
INDEX_NAME = "embedding_index"

# === Create Vector DB in MongoDB ===
def create_vector_db():
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found at: {PDF_PATH}")

    print(f"📄 Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    print("🌐 Connecting to MongoDB...")
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    print("📦 Storing vectors in MongoDBAtlasVectorSearch...")
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name=INDEX_NAME
    )

    vector_store.add_documents(chunks)
    print(f"✅ Stored {len(chunks)} vectors in MongoDB.")

# === Semantic Search on MongoDB ===
def pdf_search(query: str, llm) -> str:
    k = 10
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    db = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name=INDEX_NAME
    )

    print(f"🔍 Running similarity search for query: {query}")
    raw_results = db.similarity_search(query,k=k*2)  # Get more to allow dedup
    seen = set()
    results = []

    for doc in raw_results:
        if doc.page_content not in seen:
            results.append(doc)
            seen.add(doc.page_content)
        if len(results) == k:
            break

    if not results:
        print("⚠️ No relevant results found")
        return "No relevant information found in the manual."

    context_text = "\n\n---\n\n".join(
        [f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}" for doc in results]
    )

    print(context_text)

    PROMPT_TEMPLATE = """
    You are given a context that contains multiple clauses of a legal lease.
    Return ONLY the exact clause from the context that answers the question.
    Copy it word-for-word, no changes.
    Start with ###
    End with ###
    Nothing else.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    print(f"🧠 Sending prompt to LLM...")
    response = llm.invoke(prompt)
    #print("\nSearch result / LLM answer:\n", response)
    return response
