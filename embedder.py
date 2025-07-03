import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyC8hGg01YBuaiyQ9FV73CUU_LFmLI7HdMU"))

def get_embedding(text: str) -> list[float]:
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        print(f"❌ Lỗi khi lấy embedding: {e}")
        return []