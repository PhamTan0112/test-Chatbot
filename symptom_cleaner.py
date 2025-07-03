from symptom_normalizer import normalize_symptom
import re

def post_process_extracted(symptoms: list[str]) -> list[str]:
    cleaned = []
    for s in symptoms:
        s = s.lower().strip()
        # ✅ Loại tính từ mô tả không cần thiết
        s = re.sub(r'\b(persistent|frequent|sudden|dull|mild|severe|acute|chronic)\b', '', s)
        # ✅ Loại mô tả thời điểm hoặc hành động
        s = re.sub(r'\b(when.*|while.*|during.*|on standing|on moving)\b', '', s)
        s = re.sub(r'\s+', ' ', s).strip()

        norm = normalize_symptom(s)
        cleaned.append(norm)
    return list(set(cleaned))  # Loại trùng
