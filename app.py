"""
app.py
------------------------------------------------------------
Trợ lý AI Tin học 10 (RAG) - phiên bản tối ưu để deploy lên Render.

Khác biệt chính so với bản cũ:
- KHÔNG đọc file .docx/.pdf lúc khởi động. Thay vào đó NẠP SẴN
  FAISS index đã build từ trước (thư mục faiss_index/).
- Đường dẫn tuyệt đối -> không lỗi "file not found" trên server.
- Bọc try/except quanh phần khởi tạo -> app không crash, chỉ
  hiện cảnh báo nếu thiếu gì đó.
- Không lưu object chat của SDK vào session_state (dễ lỗi khi
  Render restart). Thay vào đó tự quản lý history dạng list.
------------------------------------------------------------
"""
import os

import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# 0. CẤU HÌNH TRANG
# ============================================================
st.set_page_config(page_title="Trợ lý AI Tin học 10", page_icon="🤖")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

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

genai.configure(api_key=api_key)

SYSTEM_INSTRUCTION = """Bạn là một chuyên gia giáo dục và là trợ lý ảo dành riêng cho giáo viên giảng dạy môn Tin học lớp 10 tại Việt Nam, bám sát chương trình Giáo dục phổ thông 2018 (SGK Kết nối tri thức với cuộc sống và Cánh diều).

Nhiệm vụ của bạn:
1. Hỗ trợ soạn giáo án.
2. Thiết kế bài tập Python.
3. Giải thích kiến thức Tin học lớp 10.
4. Tạo đề kiểm tra, trắc nghiệm, tự luận.

QUY TẮC BẮT BUỘC:
- Chỉ trả lời các câu hỏi liên quan đến môn Tin học lớp 10.
- Dựa vào ngữ cảnh (Context) được cung cấp để trả lời. Nếu ngữ cảnh không đủ, hãy dùng kiến thức chuyên môn của bạn để bổ sung.
- Nếu câu hỏi không liên quan, hãy từ chối trả lời và thông báo: "Tôi chỉ hỗ trợ các nội dung thuộc Tin học lớp 10."
"""


# ============================================================
# 2. NẠP MODEL VÀ FAISS INDEX (có cache)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    return genai.GenerativeModel(
        "gemini-2.5-flash-lite",
        system_instruction=SYSTEM_INSTRUCTION,
    )


@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Nạp FAISS index đã build sẵn. Trả về None nếu chưa có index."""
    if not os.path.isdir(INDEX_DIR):
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key
        )
        # allow_dangerous_deserialization=True: bắt buộc với FAISS bản mới
        # khi nạp index do chính mình tạo ra.
        return FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        st.warning(f"⚠️ Không nạp được FAISS index: {e}")
        return None


model = load_model()

with st.spinner("Đang nạp dữ liệu tài liệu (RAG)..."):
    vector_store = load_vector_store()

if vector_store is None:
    st.warning(
        "⚠️ Chưa có dữ liệu RAG (thư mục `faiss_index/` không tồn tại). "
        "App vẫn chạy được nhưng chỉ dùng kiến thức chung của AI. "
        "Hãy chạy `build_index.py` và commit thư mục `faiss_index/`."
    )


# ============================================================
# 3. QUẢN LÝ TRẠNG THÁI SESSION
# ============================================================
# Mỗi cuộc trò chuyện chỉ lưu 1 list các message dạng:
#   {"role": "user"|"assistant", "content": "..."}
# KHÔNG lưu object chat của SDK -> ổn định hơn.
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
    if vector_store is None:
        return user_prompt
    try:
        docs = vector_store.similarity_search(user_prompt, k=4)
    except Exception as e:
        st.warning(f"⚠️ Lỗi tìm kiếm RAG: {e}")
        return user_prompt

    if not docs:
        return user_prompt

    context_text = "\n\n".join(
        f"--- Trích đoạn tài liệu ---\n{d.page_content}" for d in docs
    )
    return f"""Dựa vào các THÔNG TIN TỪ TÀI LIỆU DƯỚI ĐÂY để trả lời câu hỏi của người dùng.
Nếu thông tin tài liệu không đủ, hãy dựa vào kiến thức của bạn.

[THÔNG TIN TỪ TÀI LIỆU (Giáo án, SGK Tin 10, Thang đo Bloom)]:
{context_text}

[CÂU HỎI CỦA NGƯỜI DÙNG]:
{user_prompt}
"""


def call_gemini(history_msgs, augmented_prompt):
    """
    Gọi Gemini với toàn bộ lịch sử. Lịch sử được dựng lại mỗi lần
    từ list message -> không phụ thuộc object chat đã lưu.
    Tin nhắn cuối (user) được thay bằng augmented_prompt.
    """
    contents = []
    for m in history_msgs[:-1]:  # tất cả trừ tin nhắn user cuối cùng
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [m["content"]]})
    # tin nhắn user cuối -> dùng prompt đã tăng cường
    contents.append({"role": "user", "parts": [augmented_prompt]})

    response = model.generate_content(contents)
    return response.text


# Xử lý input mới
if prompt := st.chat_input("Hỏi về giáo án, bài tập, SGK..."):
    # 1. Hiển thị + lưu tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    history.append({"role": "user", "content": prompt})

    # 2. Tạo prompt tăng cường (RAG)
    augmented_prompt = build_augmented_prompt(prompt)

    # 3. Gọi AI
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ và tra cứu tài liệu..."):
            try:
                answer = call_gemini(history, augmented_prompt)
            except Exception as e:
                answer = f"❌ Lỗi khi gọi AI: {e}"
            st.markdown(answer)

    # 4. Lưu phản hồi
    history.append({"role": "assistant", "content": answer})
