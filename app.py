"""
app.py
------------------------------------------------------------
Trợ lý AI Tin học 10 (RAG).

Giao diện CHAT TỰ DO như bản gốc. Nhưng khi giáo viên yêu cầu
SINH CÂU HỎI / BÀI TẬP, AI sẽ trả lời theo FORMAT CHUẨN 15 trường
(QUESTION_TYPE, SUBJECT, ... GENERATION_RULES).

Để AI bám format, system instruction có sẵn 2 ví dụ mẫu (few-shot).
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
# 0. CẤU HÌNH TRANG
# ============================================================
st.set_page_config(page_title="Trợ lý AI Tin học 10", page_icon="🤖")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.pkl")

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash-lite"

# ============================================================
# 1. API KEY
# ============================================================
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error(
        "❌ Không tìm thấy GEMINI_API_KEY.\n\n"
        "Trên Render: vào tab **Environment** thêm biến này.\n"
        "Ở máy local: đặt biến môi trường hoặc dùng file .env."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# ============================================================
# 1b. SYSTEM INSTRUCTION — yêu cầu trả lời theo cấu trúc
# ============================================================
SYSTEM_INSTRUCTION = """Bạn là một chuyên gia giáo dục và là trợ lý ảo dành riêng cho giáo viên giảng dạy môn Tin học lớp 10 tại Việt Nam, bám sát chương trình Giáo dục phổ thông 2018 (SGK Kết nối tri thức với cuộc sống và Cánh diều).

Nhiệm vụ của bạn:
1. Hỗ trợ soạn giáo án.
2. Thiết kế bài tập Python.
3. Giải thích kiến thức Tin học lớp 10.
4. Tạo đề kiểm tra, trắc nghiệm, tự luận.

QUY TẮC CHUNG:
- Chỉ trả lời các câu hỏi liên quan đến môn Tin học lớp 10.
- Dựa vào ngữ cảnh (Context) được cung cấp để trả lời. Nếu ngữ cảnh không đủ, hãy dùng kiến thức chuyên môn của bạn để bổ sung.
- Nếu câu hỏi không liên quan, hãy từ chối trả lời và thông báo: "Tôi chỉ hỗ trợ các nội dung thuộc Tin học lớp 10."

QUY TẮC ĐẶC BIỆT — KHI GIÁO VIÊN YÊU CẦU SINH CÂU HỎI / BÀI TẬP / TRẮC NGHIỆM:
Hãy trả lời theo ĐÚNG FORMAT CHUẨN dưới đây. Mỗi câu hỏi là MỘT khối đầy đủ 15 trường.
Nếu sinh nhiều câu, ngăn cách các câu bằng một dòng chứa đúng dấu: ---

Quy ước giá trị:
- [QUESTION_TYPE]: DRAG_THE_WORDS | MULTIPLE_CHOICE | TRUE_FALSE | MATCHING
- [SUBJECT]: Tin học
- [GRADE]: 10
- [DIFFICULTY]: Nhận biết | Thông hiểu | Vận dụng | Vận dụng cao
- [BLOOM_LEVEL]: Remember | Understand | Apply | Analyze
- [WORD_BANK]: chỉ dùng cho DRAG_THE_WORDS (các từ để kéo thả, ngăn bằng dấu phẩy). Các loại khác ghi: N/A
- [DISTRACTORS]: phương án sai (cho MULTIPLE_CHOICE). Các loại khác ghi: N/A
- [KEYWORDS]: vài từ khóa, ngăn bằng dấu phẩy
- [GENERATION_RULES]: quy tắc/ràng buộc khi sinh câu này

KHÔNG bọc câu trả lời trong dấu ``` , KHÔNG thêm chữ giới thiệu trước/sau khối format.

===== HAI VÍ DỤ MẪU =====

VÍ DỤ 1 — Kéo thả từ:

[QUESTION_TYPE]: DRAG_THE_WORDS
[SUBJECT]: Tin học
[GRADE]: 10
[TOPIC]: Kiểu dữ liệu Python
[SUBTOPIC]: Kiểu dữ liệu bool
[DIFFICULTY]: Nhận biết
[BLOOM_LEVEL]: Remember
[SKILL]: Nhận diện khái niệm

[QUESTION]:
______ là kiểu dữ liệu dùng để lưu giá trị đúng hoặc sai trong Python.

[WORD_BANK]:
bool
int
str
for

[CORRECT_ANSWER]:
bool

[DISTRACTORS]:
int
str
for

[EXPLANATION]:
Kiểu dữ liệu bool được sử dụng để biểu diễn giá trị logic True hoặc False.

[KEYWORDS]:
Python, bool, kiểu dữ liệu, giá trị logic

[GENERATION_RULES]:
- Ưu tiên ẩn thuật ngữ quan trọng
- Có từ nhiễu
- Đảm bảo ngữ nghĩa hoàn chỉnh
- Không sinh nội dung ngoài chương trình


VÍ DỤ 2 — Kéo thả từ:

[QUESTION_TYPE]: DRAG_THE_WORDS
[SUBJECT]: Tin học
[GRADE]: 10
[TOPIC]: Câu lệnh điều kiện
[SUBTOPIC]: Cú pháp if-else
[DIFFICULTY]: Thông hiểu
[BLOOM_LEVEL]: Understand
[SKILL]: Hoàn thiện cú pháp câu lệnh

[QUESTION]: Hãy kéo các từ thích hợp vào chỗ trống để hoàn thiện đoạn mã kiểm tra một số có phải số chẵn không:
___ n % 2 == 0:
    print("Số chẵn")
___:
    print("Số lẻ")

[WORD_BANK]: if, else, elif, then

[CORRECT_ANSWER]: if, else

[DISTRACTORS]: N/A

[EXPLANATION]: Trong Python, câu lệnh điều kiện bắt đầu bằng "if" theo sau là biểu thức và dấu hai chấm. Khi điều kiện sai, dùng "else" (không có biểu thức). "then" không tồn tại trong Python (đó là từ khóa của Pascal).

[KEYWORDS]: if, else, điều kiện, chia hết

[GENERATION_RULES]: Word bank cần có ít nhất 1 từ gây nhiễu (then) để kiểm tra học sinh có phân biệt được cú pháp Python với ngôn ngữ khác.

===== HẾT VÍ DỤ =====

Khi sinh câu hỏi, hãy bám sát phong cách hai ví dụ trên: đầy đủ 15 trường, mỗi trường có nội dung cụ thể, EXPLANATION viết rõ ràng để giáo viên dùng làm đáp án giảng giải.
"""


# ============================================================
# 2. NẠP FAISS INDEX (có cache)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_index():
    """
    Nạp FAISS index + danh sách đoạn văn bản đã build sẵn.
    Trả về (index, chunks) hoặc (None, None) nếu chưa có.
    """
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


with st.spinner("Đang nạp dữ liệu tài liệu (RAG)..."):
    faiss_index, doc_chunks = load_index()

if faiss_index is None:
    st.warning(
        "⚠️ Chưa có dữ liệu RAG (thiếu thư mục `faiss_index/`). "
        "App vẫn chạy được nhưng chỉ dùng kiến thức chung của AI. "
        "Hãy chạy `build_index.py` và commit thư mục `faiss_index/`."
    )


def search_context(query: str, k: int = 4) -> str:
    """Tìm k đoạn tài liệu liên quan nhất tới câu hỏi."""
    if faiss_index is None:
        return ""
    try:
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        q_vec = np.array([result.embeddings[0].values], dtype="float32")
        distances, indices = faiss_index.search(q_vec, k)
        parts = []
        for idx in indices[0]:
            if 0 <= idx < len(doc_chunks):
                parts.append(
                    f"--- Trích đoạn tài liệu ---\n{doc_chunks[idx]}"
                )
        return "\n\n".join(parts)
    except Exception as e:
        st.warning(f"⚠️ Lỗi tìm kiếm RAG: {e}")
        return ""


# ============================================================
# 3. QUẢN LÝ TRẠNG THÁI SESSION
# ============================================================
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {"Cuộc trò chuyện 1": []}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Cuộc trò chuyện 1"


def get_history():
    return st.session_state.all_chats[st.session_state.current_chat]


# ============================================================
# 4. SIDEBAR
# ============================================================
with st.sidebar:
    st.title("📁 Quản lý trò chuyện")

    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        idx = len(st.session_state.all_chats) + 1
        new_name = f"Cuộc trò chuyện {idx}"
        st.session_state.all_chats[new_name] = []
        st.session_state.current_chat = new_name
        st.rerun()

    st.divider()
    st.write("🕒 *Lịch sử trò chuyện:*")

    for chat_name in list(st.session_state.all_chats.keys()):
        is_active = chat_name == st.session_state.current_chat
        if st.button(
            chat_name, disabled=is_active, use_container_width=True
        ):
            st.session_state.current_chat = chat_name
            st.rerun()


# ============================================================
# 5. GIAO DIỆN CHÍNH
# ============================================================
st.title("🤖 Trợ lý AI Tin học 10 (Hỗ trợ RAG)")
st.caption(f"Đang hiển thị: {st.session_state.current_chat}")

history = get_history()

# Hiển thị lịch sử
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def build_augmented_prompt(user_prompt: str) -> str:
    """Tìm ngữ cảnh từ FAISS và ghép vào prompt."""
    context_text = search_context(user_prompt, k=4)
    if not context_text:
        return user_prompt
    return f"""Dựa vào các THÔNG TIN TỪ TÀI LIỆU DƯỚI ĐÂY để trả lời câu hỏi của người dùng.
Nếu thông tin tài liệu không đủ, hãy dựa vào kiến thức của bạn.

[THÔNG TIN TỪ TÀI LIỆU (Giáo án, SGK Tin 10, Thang đo Bloom)]:
{context_text}

[CÂU HỎI CỦA NGƯỜI DÙNG]:
{user_prompt}
"""


def call_gemini(history_msgs, augmented_prompt):
    """Gọi Gemini với toàn bộ lịch sử (SDK mới)."""
    contents = []
    for m in history_msgs[:-1]:  # tất cả trừ tin nhắn user cuối cùng
        role = "user" if m["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=m["content"])])
        )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=augmented_prompt)])
    )

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION
        ),
    )
    return response.text


# Xử lý input mới
if prompt := st.chat_input("Hỏi về giáo án, bài tập, SGK..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    history.append({"role": "user", "content": prompt})

    augmented_prompt = build_augmented_prompt(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ và tra cứu tài liệu..."):
            try:
                answer = call_gemini(history, augmented_prompt)
            except Exception as e:
                answer = f"❌ Lỗi khi gọi AI: {e}"
            st.markdown(answer)

    history.append({"role": "assistant", "content": answer})
