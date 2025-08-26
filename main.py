from modules.pdf_loader import extract_text_from_pdf
from modules.text_splitter import split_text
from modules.vector_store import create_or_update_db
from modules.rag_pipeline import ask_question

if __name__ == "__main__":
    # Step 1: Load contract
    file_path = "data/Lease-Agreement-Template-for-landlords-dd-15-Mar-22.pdf"
    text = extract_text_from_pdf(file_path)

    # Step 2: Chunk
    chunks = split_text(text)

    # Step 3: Store in MongoDB Vector DB
    db = create_or_update_db(chunks)

    # Step 4: Ask questions
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = ask_question(query)
        print(f"\nAnswer: {answer}\n")
