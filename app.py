"""
app.py
------------------------------------------------------------
Trợ lý AI Tin học 10 (RAG) - sinh câu hỏi theo FORMAT CHUẨN.

Chatbot trả lời theo đúng template có cấu trúc ([QUESTION_TYPE],
[SUBJECT], [BLOOM_LEVEL]...). Có thể sinh 1 hoặc nhiều câu, và
tải kết quả ra file .txt.

Dùng google-genai + faiss (index build sẵn trong faiss_index/).
------------------------------------------------------------
"""
import os
import pickle

import numpy as np
import faiss
import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# 0. CẤU HÌNH
# ============================================================
st.set_page_config(page_title="Trợ lý AI Tin học 10", page_icon="🤖")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.pkl")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash-lite"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Không tìm thấy GEMINI_API_KEY. Hãy thêm biến môi trường này.")
    st.stop()

client = genai.Client(api_key=api_key)

# ============================================================
# 1. FORMAT CHUẨN + HƯỚNG DẪN CHO AI
# ============================================================
# Template mẫu mà AI phải tuân theo cho MỖI câu hỏi.
TEMPLATE = """[QUESTION_TYPE]:
[SUBJECT]:
[GRADE]:
[TOPIC]:
[SUBTOPIC]:
[DIFFICULTY]:
[BLOOM_LEVEL]:
[SKILL]:

[QUESTION]:

[WORD_BANK]:

[CORRECT_ANSWER]:

[DISTRACTORS]:

[EXPLANATION]:

[KEYWORDS]:

[GENERATION_RULES]:"""

SYSTEM_INSTRUCTION = f"""Bạn là trợ lý ảo cho giáo viên Tin học lớp 10 tại Việt Nam, bám sát Chương trình GDPT 2018 (SGK Kết nối tri thức và Cánh diều).

NHIỆM VỤ DUY NHẤT: sinh dữ liệu câu hỏi theo ĐÚNG FORMAT CHUẨN dưới đây. Mỗi câu hỏi là MỘT khối theo template:

{TEMPLATE}

QUY TẮC BẮT BUỘC:
- LUÔN trả lời theo đúng format trên, điền đầy đủ mọi trường. KHÔNG thêm chữ giới thiệu, KHÔNG dùng markdown, KHÔNG bọc trong dấu ```.
- Nếu sinh NHIỀU câu, mỗi câu là một khối riêng, ngăn cách nhau bằng một dòng chứa đúng: ---
- Quy ước giá trị:
  + QUESTION_TYPE: DRAG_THE_WORDS | MULTIPLE_CHOICE | TRUE_FALSE | MATCHING
  + SUBJECT: Tin học
  + GRADE: 10
  + DIFFICULTY: Nhận biết | Thông hiểu | Vận dụng | Vận dụng cao
  + BLOOM_LEVEL: Remember | Understand | Apply | Analyze
- WORD_BANK chỉ dùng cho DRAG_THE_WORDS (các từ để kéo thả). DISTRACTORS là các phương án sai (cho MULTIPLE_CHOICE).
- KEYWORDS: vài từ khóa, ngăn cách bằng dấu phẩy.
- GENERATION_RULES: ghi ngắn gọn quy tắc/ràng buộc khi sinh câu này.
- Chỉ làm về Tin học lớp 10. Nếu yêu cầu không liên quan, trả lời đúng một dòng: "Tôi chỉ hỗ trợ nội dung Tin học lớp 10."
- Dựa vào tài liệu tham khảo (nếu có) để câu hỏi bám sát SGK.
"""


# ============================================================
# 2. NẠP FAISS INDEX
# ============================================================
@st.cache_resource(show_spinner=False)
def load_index():
    if not (os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE)):
        return None, None
    try:
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    except Exception as e:
        st.warning(f"⚠️ Không nạp được FAISS index: {e}")
        return None, None


faiss_index, doc_chunks = load_index()

if faiss_index is None:
    st.warning(
        "⚠️ Chưa có dữ liệu RAG (thiếu `faiss_index/`). Vẫn dùng được "
        "nhưng chỉ dựa vào kiến thức chung của AI."
    )


def search_context(query: str, k: int = 4) -> str:
    if faiss_index is None:
        return ""
    try:
        result = client.models.embed_content(
            model=EMBED_MODEL, contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        q_vec = np.array([result.embeddings[0].values], dtype="float32")
        _, indices = faiss_index.search(q_vec, k)
        parts = [doc_chunks[i] for i in indices[0] if 0 <= i < len(doc_chunks)]
        return "\n\n".join(parts)
    except Exception as e:
        st.warning(f"⚠️ Lỗi tìm kiếm RAG: {e}")
        return ""


# ============================================================
# 3. GỌI AI SINH CÂU HỎI
# ============================================================
def generate_questions(topic: str, qtype: str, difficulty: str,
                       n: int, user_request: str = "") -> str:
    """Sinh n câu hỏi theo format chuẩn về chủ đề topic.

    user_request: phần mô tả thêm của giáo viên (yêu cầu chi tiết)
    để AI bám theo - ví dụ: "câu hỏi gắn với ví dụ thực tế",
    "không dùng thư viện ngoài", "đáp án có giải thích chi tiết"...
    """
    context = search_context(topic + " " + user_request, k=4)
    user_prompt = f"""Hãy sinh {n} câu hỏi theo format chuẩn.
- Loại câu hỏi: {qtype}
- Mức độ: {difficulty}
- Chủ đề: {topic}

[YÊU CẦU CHI TIẾT TỪ GIÁO VIÊN]:
{user_request if user_request.strip() else '(không có yêu cầu thêm)'}

[TÀI LIỆU THAM KHẢO]:
{context if context else '(không có, dùng kiến thức của bạn)'}
"""
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
        ),
    )
    return response.text


# ============================================================
# 4. GIAO DIỆN
# ============================================================
st.title("🤖 Sinh câu hỏi Tin học 10 (Format chuẩn)")
st.caption("AI sinh câu hỏi theo format có cấu trúc, tải về file .txt.")

# Lưu kết quả gần nhất trong phiên
if "result" not in st.session_state:
    st.session_state.result = ""

col1, col2 = st.columns(2)
with col1:
    qtype = st.selectbox(
        "Loại câu hỏi",
        ["DRAG_THE_WORDS", "MULTIPLE_CHOICE", "TRUE_FALSE", "MATCHING"],
    )
    difficulty = st.selectbox(
        "Mức độ",
        ["Nhận biết", "Thông hiểu", "Vận dụng", "Vận dụng cao"],
    )
with col2:
    topic = st.text_input("Chủ đề", "Kiểu dữ liệu Python")
    n = st.number_input("Số câu cần sinh", min_value=1, max_value=20, value=1)

user_request = st.text_area(
    "✏️ Yêu cầu chi tiết (tùy chọn)",
    placeholder=(
        "Mô tả thêm để AI bám sát ý bạn. Ví dụ:\n"
        "- Câu hỏi gắn với ví dụ thực tế trong đời sống.\n"
        "- Tập trung vào lệnh print() và input().\n"
        "- Đáp án phải có lời giải thích chi tiết, dễ hiểu cho HS yếu.\n"
        "- Tránh dùng thư viện ngoài, chỉ Python thuần."
    ),
    height=120,
)

if st.button("✍️ Sinh câu hỏi", type="primary", use_container_width=True):
    with st.spinner(f"AI đang sinh {n} câu hỏi..."):
        try:
            st.session_state.result = generate_questions(
                topic, qtype, difficulty, int(n), user_request
            )
        except Exception as e:
            st.error(f"Lỗi khi sinh câu hỏi: {e}")
            st.session_state.result = ""

# Hiển thị + cho tải về
if st.session_state.result:
    st.divider()
    st.text_area("Kết quả (có thể chỉnh sửa)",
                 st.session_state.result, height=400, key="result_edit")
    final = st.session_state.get("result_edit", st.session_state.result)
    st.download_button(
        "⬇️ Tải file .txt",
        data=final.encode("utf-8"),
        file_name="cau_hoi_tin_hoc_10.txt",
        mime="text/plain",
        use_container_width=True,
    )
