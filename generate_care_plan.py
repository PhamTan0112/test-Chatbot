# 📄 generate_care_plan.py

import requests
import google.generativeai as genai

def generate_care_plan(user_id: str) -> str:
    """
    Tạo kế hoạch chăm sóc sức khỏe cá nhân hóa dựa trên hồ sơ bệnh nhân và dấu hiệu bất thường.
    """
    try:
        response = requests.get(f"http://localhost:3000/api/external/analyze/{user_id}")
        response.raise_for_status()
        data = response.json()
    except Exception:
        return "Không thể lấy dữ liệu bệnh nhân."

    summary_text = data.get("summary_text", "")
    abnormal_flags = data.get("abnormal_flags", [])

    if not summary_text:
        return "Không có đủ dữ liệu để lập kế hoạch chăm sóc."

    flags_text = "- " + "\n- ".join(abnormal_flags) if abnormal_flags else "Không có dấu hiệu bất thường."

    prompt = f"""
Bạn là một bác sĩ AI. Dựa vào hồ sơ bệnh nhân và cảnh báo bên dưới, hãy đưa ra kế hoạch chăm sóc ngắn gọn, thực tế.

──────────────────────────────
Hồ sơ bệnh nhân:
{summary_text}

Dấu hiệu bất thường:
{flags_text}
──────────────────────────────

Yêu cầu:
- Viết 3–5 dòng gợi ý thiết thực(*tối đa 60 ký tự*).
- Không *bịa đặt*. Tránh từ chuyên môn phức tạp.
- Phù hợp với người bệnh phổ thông.
- **Không dùng dấu (*) (***)
- Dễ hiểu, ngắn gọn, gần gũi
Kế hoạch chăm sóc:
"""

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return "Không thể tạo kế hoạch chăm sóc lúc này."
