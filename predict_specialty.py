import google.generativeai as genai

# Đảm bảo bạn đã cấu hình API key từ môi trường
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def predict_specialty(symptom_text: str) -> str:
    """
    Dự đoán chuyên khoa phù hợp với triệu chứng người dùng mô tả.
    Trả về chuỗi tên chuyên khoa: ví dụ "Tiêu hóa", "Thần kinh", "Hô hấp", v.v.
    """
    prompt = f"""
Bạn là bác sĩ phân loại triệu chứng. Dựa vào mô tả dưới đây, hãy trả lời chuyên khoa phù hợp nhất (chỉ trả lời đúng tên chuyên khoa, không thêm mô tả).

Triệu chứng: {symptom_text}

Chuyên khoa phù hợp nhất là:
"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt).text.strip()
        return response.split("\n")[0].strip()  # đảm bảo chỉ lấy dòng đầu
    except Exception as e:
        return "Không rõ"
