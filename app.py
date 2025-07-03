from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import generate_answer  
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class Query(BaseModel):
    question: str
    user_id: str

@app.post("/ask")
async def ask(query: Query):
    answer = generate_answer(query.question, query.user_id)
    return { "answer": answer }

# Bệnh nhân gặp phải tình trạng mất tập trung, thường xuyên chóng mặt, kèm theo các cơn đau đầu âm ỉ. Đôi lúc xuất hiện cảm giác đau ngực bất chợt và mất thăng bằng khi di chuyển hoặc đứng lên. Tôi bị bệnh gì?