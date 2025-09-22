# from config import MODEL_NAME
# from langchain.llms import HuggingFaceHub

# llm = HuggingFaceHub(repo_id=MODEL_NAME)

# def summarize(text, mode="plain", lang="en"):
#     if mode == "plain":
#         prompt = f"Summarize this contract in {lang}:\n{text}"
#     else:
#         prompt = f"Summarize this contract focusing on risks in {lang}:\n{text}"
#     return llm(prompt)
