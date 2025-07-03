from embedder import get_embedding
from vectorstore import search_similar
import google.generativeai as genai
import requests
from generate_care_plan import generate_care_plan
from predict_specialty import predict_specialty
from db_session import append_session
from symptom_normalizer import normalize_symptom
import re
import json

# Cấu hình API Gemini
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

    Chỉ trả về đúng một từ: health_query, personal_info hoặc general_chat

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
Bạn là bác sĩ. Trích xuất tất cả triệu chứng y tế có thể có từ câu hỏi sau. Trả lời **chỉ dưới dạng danh sách JSON tiếng Anh hợp lệ**, không giải thích, không dùng markdown:
"{question}"

Ví dụ đầu ra: ["headache", "chest pain"]
'''
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = text.replace("```", "").strip()
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
    question_lower = question.lower()
    intent = classify_user_intent(question)

    # Trường hợp hội thoại hoặc yêu cầu cá nhân
    if intent in ["personal_info", "general_chat"]:
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

    # Nếu là health_query → tiếp tục quy trình chẩn đoán bệnh như cũ
    raw_symptoms_en = extract_symptoms_with_gemini(question)
    user_symptoms = [normalize_symptom(s) for s in raw_symptoms_en]
    standard_symptoms = get_standard_symptoms()
    user_symptoms = [s for s in user_symptoms if s in standard_symptoms]

    if user_symptoms:
        normalized_query = f"Symptoms: {', '.join(user_symptoms)}. What is the possible disease?"
        query_vec = get_embedding(normalized_query)
    else:
        query_vec = get_embedding(question)

    if not query_vec:
        return "Không thể tạo embedding cho câu hỏi."

    docs = search_similar(query_vec)
    if not docs:
        return "Hiện tại hệ thống không tìm thấy tài liệu liên quan để tư vấn. Vui lòng mô tả rõ hơn hoặc thử lại sau."

    try:
        res = requests.get(f"http://localhost:3000/api/external/analyze/{user_id}", timeout=5)
        res.raise_for_status()
        data = res.json()
    except Exception:
        data = {}

    patient_context = data.get("summary_text", "Không có dữ liệu bệnh nhân.")
    abnormal_flags = data.get("abnormal_flags", [])
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

    prompt = f"""
Bạn là một bác sĩ gia đình ảo, nhiệm vụ của bạn là:
1. Phân tích triệu chứng người bệnh cung cấp
2. Đưa ra một số bệnh lý có thể liên quan
3. Dự đoán chuyên khoa nên đến khám
4. Đưa ra khuyến nghị: nên đi khám sớm hay theo dõi thêm
5. Giữ ngữ điệu nhẹ nhàng, đồng cảm, dễ hiểu

Ngữ cảnh bệnh nhân:

📨 Câu hỏi:
"{question}"
📋 Triệu chứng trích xuất:
{', '.join(user_symptoms) if user_symptoms else 'Chưa rõ'}
📌 Bệnh nghi ngờ: {likely_disease['name'] if likely_disease else 'Chưa xác định'}
🧠 Các bệnh có thể liên quan:
{disease_summary_details}
📈 Chuyên khoa phù hợp: {predicted_specialty}
🔎 Dấu hiệu bất thường (nếu có):
{'; '.join(abnormal_flags) if abnormal_flags else 'Không có'}
👨‍⚕️ Bác sĩ sẵn có:
{doctor_list}
💼 Hồ sơ bệnh nhân:
{patient_context}
🎯 Hãy trả lời bệnh nhân bằng văn phong tự nhiên. Viết ngắn gọn, không quá 4–5 câu. Tránh liệt kê máy móc, không khẳng định chẩn đoán chắc chắn. Hãy hỗ trợ bệnh nhân ra quyết định.
"""

    try:
        llm_answer = call_gemini_flash(prompt)
    except Exception:
        llm_answer = "Hiện tại hệ thống chưa thể tạo phản hồi chi tiết. Dưới đây là tư vấn sơ bộ."

    abnormal_section = "\n🔹 Dấu hiệu bất thường:\n- " + "\n- ".join(abnormal_flags) if abnormal_flags else ""
    care_plan = generate_care_plan(user_id) if any(kw in question_lower for kw in ["tôi", "của tôi"]) and abnormal_flags else ""
    care_plan_section = f"\n\n📋 Kế hoạch chăm sóc cá nhân hóa:\n{care_plan.strip()}" if care_plan else ""

    final_answer = llm_answer + abnormal_section + care_plan_section
    append_session(user_id, question, final_answer)
    return final_answer
