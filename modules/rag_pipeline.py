import os
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.text_splitter import SemanticChunker

# === Embedding model ===
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

# === Configuration ===
MONGODB_URI = "mongodb+srv://susindugajanayake:xFwZvwzWBIUtMuvL@cluster0.z40kp.mongodb.net/"
DB_NAME = "documents"
COLLECTION_NAME = "documents_vectors"
INDEX_NAME = "embedding_index"

# === Create Vector DB in MongoDB ===
def create_vector_db(doc_path, doc_name):
    if not doc_path or not os.path.exists(doc_path):
        raise FileNotFoundError(f"PDF file not found at: {doc_path}")

    print(f"📄 Loading PDF from {doc_path}...")
    loader = PyPDFLoader(doc_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    # Initialize SemanticChunker
    chunker = SemanticChunker(embedding_model)

    # Split documents semantically
    chunks = chunker.split_documents(documents)
    print(f"✅ Created {len(chunks)} semantic chunks")

    # Add document name to metadata
    for chunk in chunks:
        chunk.metadata["doc_name"] = doc_name

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
def pdf_search(query: str, llm, doc_name: str) -> str:
    k = 7 
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    print("in the function pdf_search")

    db = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name=INDEX_NAME
    )

    # # Filter by doc_name if provided
    # search_kwargs = {"k": k * 2}
    # if doc_name:
    #     search_kwargs["filter"] = {"metadata.doc_name": doc_name}

    raw_results = db.similarity_search(query,k=k*2,pre_filter={"doc_name": doc_name})

    # retriever = db.as_retriever(
    # search_kwargs={
    #     "k": 10,
    #     "search_filter": {"metadata.doc_name": doc_name}
    # })

    #raw_results = retriever.get_relevant_documents(query)
    # query_embedding = embedding_model.embed_query(query)
    # print("doc_name:", doc_name)
    # filter_query = {"doc_name": doc_name} if doc_name else {}

    # pipeline = [
    #     {
    #         "$vectorSearch": {
    #             "index": INDEX_NAME,
    #             "path": "embedding",
    #             "queryVector": query_embedding,
    #             "numCandidates": 200,
    #             "limit": 10,
    #             "filter": filter_query
    #         }
    #     },
    #     {"$project": {"_id": 0, "page_content": 1, "metadata": 1}}
    # ]

    #results = list(collection.aggregate(pipeline))
    #raw_results = [type('Document', (object,), res)() for res in results]

    # array_of_results = []
    # for doc in results:
    #     array_of_results.append(doc)
    #return array_of_results

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

    # context_docs = array_of_results
    # context_string = " ".join([doc["page_content"] for doc in context_docs])

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

    Answer (copy exact text only):
    """


    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    print(f"🧠 Sending prompt to LLM...")
    response = llm.invoke(prompt)
    return response

# === List all uploaded documents ===
def list_uploaded_documents():
    client = MongoClient(MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    doc_names = collection.distinct("doc_name")
    return doc_names