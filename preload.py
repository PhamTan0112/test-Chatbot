from embedder import get_embedding
from vectorstore import add_document, setup_collection
import os

setup_collection()

for idx, filename in enumerate(os.listdir("documents")):
    if filename.endswith(".txt") or filename.endswith(".md"):
        with open(f"documents/{filename}", encoding="utf-8") as f:
            text = f.read()
            emb = get_embedding(text)
            add_document(doc_id=idx, text=text, embedding=emb)
