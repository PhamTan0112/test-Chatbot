from embedder import get_embedding
from vectorstore import search_similar
import google.generativeai as genai
import requests
from predict_specialty import predict_specialty
from db_session import append_session
from symptom_normalizer import normalize_symptom
import re
import json

genai.configure(api_key="AIzaSyC8hGg01YBuaiyQ9FV73CUU_LFmLI7HdMU")

def call_gemini_flash(prompt: str) -> str:
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

def classify_user_intent(question: str) -> str:
    prompt = f"""
    Bạn là trợ lý y tế thông minh. Hãy phân loại câu hỏi sau thành một trong 3 nhóm:
    - "health_query": nếu người dùng mô tả triệu chứng hoặc hỏi về bệnh
    - "personal_info": nếu người dùng hỏi về bản thân (tên, tuổi, lịch khám, bác sĩ từng gặp...)
    - "general_chat": nếu chỉ chào hỏi, hỏi vu vơ

    Trả về đúng một từ: health_query, personal_info hoặc general_chat

    Câu hỏi: "{question}"
    """
    try:
        result = call_gemini_flash(prompt)
        return result.strip().lower()
    except Exception:
        return "health_query"

def refine_question_if_needed(question: str) -> str:
    vague_keywords = [
        "mệt", "khó chịu", "không khỏe", "bị gì", "nên làm gì",
        "không ổn", "tôi cảm thấy", "hơi lạ", "thấy lạ", "khó tả"
    ]
    if any(kw in question.lower() for kw in vague_keywords):
        question += "\n\n👉 Mình cần thêm thông tin để giúp bạn tốt hơn:\n"
        question += "- Bạn cảm thấy không khỏe ở đâu? (ví dụ: đầu, bụng, ngực...)\n"
        question += "- Triệu chứng bắt đầu từ khi nào?\n"
        question += "- Mức độ nghiêm trọng: nhẹ, vừa hay dữ dội?"
    return question

def extract_symptoms_with_gemini(question: str) -> list:
    prompt = f'''
Bạn là bác sĩ. Trích xuất tất cả triệu chứng y tế có thể có từ câu hỏi sau. Trả lời dưới dạng JSON list tiếng Anh hợp lệ, không giải thích:
"{question}"
Ví dụ: ["headache", "chest pain"]
'''
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().replace("```", "").strip()
        if not text.startswith("[") or not text.endswith("]"):
            return []
        symptoms = json.loads(text)
        if isinstance(symptoms, list):
            return [s.strip().lower() for s in symptoms if isinstance(s, str)]
    except Exception:
        pass
    return []

def get_standard_symptoms(filepath="documents/benh_va_trieu_chung_chuan_hoa.txt"):
    symptoms_set = set()
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    blocks = content.split("---")
    for block in blocks:
        symp_match = re.search(r"🔍 Biểu hiện đi kèm:([\s\S]*?)(?:🛡️|$)", block)
        if symp_match:
            symptoms = [s.strip().lower() for s in symp_match.group(1).split(",") if s.strip()]
            symptoms_set.update(symptoms)
    return symptoms_set

def load_disease_symptoms(filepath="documents/benh_va_trieu_chung_chuan_hoa.txt"):
    diseases = []
    with open(filepath, encoding="utf-8") as f:
        content = f.read()
    blocks = content.split("---")
    for block in blocks:
        name_match = re.search(r"### [^:]+: (.+)", block)
        desc_match = re.search(r"📌 Mô tả:([\s\S]*?)(?:🔍|🛡️|$)", block)
        symp_match = re.search(r"🔍 Biểu hiện đi kèm:([\s\S]*?)(?:🛡️|$)", block)
        if name_match and desc_match and symp_match:
            name = name_match.group(1).strip()
            desc = desc_match.group(1).strip().replace("\n", " ")
            symptoms = [s.strip().lower() for s in symp_match.group(1).split(",") if s.strip()]
            diseases.append({"name": name, "desc": desc, "symptoms": symptoms})
    return diseases

def find_related_diseases(user_symptoms, diseases, min_match=3):
    user_symptoms = [s.strip().lower() for s in user_symptoms]
    related = []
    for d in diseases:
        disease_symptoms = [sym.strip().lower() for sym in d["symptoms"]]
        match_count = sum(1 for s in user_symptoms if s in disease_symptoms)
        if match_count >= min_match:
            related.append({"name": d["name"], "desc": d["desc"], "match": match_count})
    related.sort(key=lambda x: -x["match"])
    return related

def generate_answer(question: str, user_id: str) -> str:
    question = refine_question_if_needed(question)
    intent = classify_user_intent(question)

    # Xử lý hội thoại và thông tin cá nhân
    if intent in ["general_chat", "personal_info"]:
        try:
            res = requests.get(f"http://localhost:3000/api/external/analyze/{user_id}", timeout=5)
            res.raise_for_status()
            data = res.json()
        except Exception:
            return "Xin lỗi, tôi không thể truy xuất hồ sơ bệnh nhân lúc này."

        if intent == "general_chat":
            return f"Chào bạn {data.get('full_name', 'bạn')}! Tôi có thể hỗ trợ tư vấn sức khỏe nếu bạn cần."

        if intent == "personal_info":
            name = data.get("full_name", "Không rõ")
            dob = data.get("dob", "Không rõ")
            bp = data.get("blood_pressure", {}).get("text", "Chưa có dữ liệu")
            lab = data.get("last_lab_test", "Chưa có dữ liệu")
            last_doctor = data.get("last_doctor", {})
            doctor = f"{last_doctor.get('name', 'Không rõ')} ({last_doctor.get('specialization', 'Chưa rõ')})"
            return (
                f"Bạn là {name}, sinh ngày {dob}. Gần nhất, bạn khám với {doctor}. "
                f"Huyết áp: {bp}. Xét nghiệm gần nhất: {lab}."
            )

    # Xử lý health_query
    raw_symptoms_en = extract_symptoms_with_gemini(question)
    user_symptoms = [normalize_symptom(s) for s in raw_symptoms_en]
    standard_symptoms = get_standard_symptoms()
    user_symptoms = [s for s in user_symptoms if s in standard_symptoms]

    if user_symptoms:
        query_vec = get_embedding(f"Symptoms: {', '.join(user_symptoms)}")
    else:
        query_vec = get_embedding(question)

    if not query_vec:
        return "Không thể tạo embedding cho câu hỏi."

    docs = search_similar(query_vec)
    if not docs:
        return "Chưa có tài liệu phù hợp. Bạn có thể mô tả rõ hơn?"

    try:
        res = requests.get(f"http://localhost:3000/api/external/analyze/{user_id}", timeout=5)
        res.raise_for_status()
        data = res.json()
    except Exception:
        data = {}

    patient_context = data.get("summary_text", "Không có dữ liệu bệnh nhân.")
    active_doctors = data.get("active_doctors", [])

    predicted_specialty = predict_specialty(question)
    matching_doctors = [
        doc for doc in active_doctors
        if predicted_specialty.lower() in doc.get("specialization", "").lower()
    ] if predicted_specialty != "Không rõ" else []

    show_doctors = matching_doctors if matching_doctors else active_doctors
    doctor_list = "\n".join([
        f"{doc.get('name', 'Không rõ')} ({doc.get('specialization', 'Chuyên khoa chưa rõ')})"
        for doc in show_doctors[:3]
    ]) or "Không có dữ liệu bác sĩ"

    diseases = load_disease_symptoms()
    related_diseases = find_related_diseases(user_symptoms, diseases)
    likely_disease = next((d for d in related_diseases if d["match"] >= 3), None)
    disease_summary_details = "\n".join([
        f"- {d['name']}: {d['desc']}" for d in related_diseases[:2]
    ]) or "Chưa xác định rõ"

    # ❗ Nếu chưa có bệnh phù hợp → hỏi thêm triệu chứng
    if not likely_disease:
        followup_question = """
Mình chưa đủ thông tin để tư vấn chính xác. Bạn có thể giúp mình trả lời thêm nhé:
- Bạn cảm thấy không khỏe ở đâu (ví dụ: đầu, bụng, lưng...)?
- Triệu chứng xuất hiện từ khi nào?
- Mức độ: nhẹ, vừa hay dữ dội?

Trả lời thêm giúp mình nhé
"""
        append_session(user_id, question, followup_question.strip())
        return followup_question.strip()

    # 🔥 Tạo prompt đẹp mắt và rõ ràng
    prompt = f"""
🩺 Bạn là một **bác sĩ gia đình ảo**. Phân tích thông tin bên dưới và đưa ra phản hồi:

🎯 **Yêu cầu**: 
- Nhận định nguyên nhân có thể dựa trên triệu chứng
- Gợi ý chuyên khoa phù hợp
- Viết tự nhiên, nhẹ nhàng, không khẳng định chắc chắn
- Không liệt kê máy móc

---

📨 **Câu hỏi**: {question.strip()}

📋 **Triệu chứng**: {', '.join(user_symptoms) or 'Chưa rõ'}

📌 **Bệnh nghi ngờ**: {likely_disease['name'] if likely_disease else 'Chưa rõ'}

🧠 **Bệnh có thể liên quan**:
{disease_summary_details.strip()}

📈 **Chuyên khoa phù hợp**: {predicted_specialty or 'Chưa rõ'}

👨‍⚕️ **Bác sĩ gợi ý**:
{doctor_list.strip()}

---

📢 **Viết ngắn gọn (dưới 5 câu), không dùng dấu * trong câu trả lời.**
"""

    try:
        llm_answer = call_gemini_flash(prompt)
    except Exception:
        llm_answer = "Hiện tại hệ thống chưa thể tạo phản hồi chi tiết. Vui lòng thử lại sau."

    append_session(user_id, question, llm_answer)
    return llm_answer
