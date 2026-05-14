import os
import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CẤU HÌNH API VÀ MODEL ---
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ Không tìm thấy GEMINI_API_KEY. Vui lòng kiểm tra lại file .env hoặc biến môi trường!")
    st.stop()

genai.configure(api_key=api_key) 

instruction = """Bạn là một chuyên gia giáo dục và là trợ lý ảo dành riêng cho giáo viên giảng dạy môn Tin học lớp 10 tại Việt Nam, bám sát chương trình Giáo dục phổ thông 2018 (SGK Kết nối tri thức với cuộc sống và Cánh diều).

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

model = genai.GenerativeModel('gemini-2.5-flash-lite', system_instruction=instruction)

# --- 2. HÀM XỬ LÝ DỮ LIỆU RAG (ĐƯỢC CACHE ĐỂ CHẠY NHANH) ---
@st.cache_resource
def setup_rag_system():
    text_data = ""
    
    # Hàm phụ để đọc file docx
    def read_docx(file_path):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs]) + "\n"
        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")
            return ""

    # Hàm phụ để đọc file pdf
    def read_pdf(file_path):
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")
        return text

    # Đọc các file dữ liệu (Cần đảm bảo file nằm cùng thư mục với script)
    # Lưu ý: File .doc đã được giả định chuyển thành .docx
    text_data += read_docx("Miền nhận thức.docx")
    text_data += read_docx("Kế hoạch bài dạy_Công văn 5512.docx") 
    text_data += read_pdf("TIN HỌC 10 KNTT.pdf")

    if not text_data.strip():
        return None # Không có dữ liệu

    # Chia nhỏ văn bản (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_text(text_data)
    
    # Tạo Vector Store (Sử dụng Embedding của Google)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

# Khởi tạo Vector Store
with st.spinner("Đang tải dữ liệu tài liệu (RAG)... Vui lòng đợi trong giây lát."):
    vector_store = setup_rag_system()

# --- 3. QUẢN LÝ TRẠNG THÁI SESSION ---
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {"Cuộc trò chuyện 1": model.start_chat(history=[])}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Cuộc trò chuyện 1"

# Hiển thị lịch sử chat cho UI (Tách biệt lịch sử hiển thị và lịch sử API nếu cần)
if "chat_display_history" not in st.session_state:
    st.session_state.chat_display_history = {"Cuộc trò chuyện 1": []}

# --- 4. GIAO DIỆN SIDEBAR ---
with st.sidebar:
    st.title("📁 Quản lý trò chuyện")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        new_chat_name = f"Cuộc trò chuyện {len(st.session_state.all_chats) + 1}"
        st.session_state.all_chats[new_chat_name] = model.start_chat(history=[])
        st.session_state.chat_display_history[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.rerun()

    st.divider()
    st.write("🕒 *Lịch sử trò chuyện:*")
    
    for chat_name in st.session_state.all_chats.keys():
        is_active = (chat_name == st.session_state.current_chat)
        if st.button(chat_name, disabled=is_active, use_container_width=True):
            st.session_state.current_chat = chat_name
            st.rerun()

# --- 5. KHU VỰC GIAO DIỆN CHÍNH ---
st.title("🤖 Trợ lý AI Tin học 10 (Hỗ trợ RAG)")
st.caption(f"Đang hiển thị: {st.session_state.current_chat}")

active_chat = st.session_state.all_chats[st.session_state.current_chat]
display_history = st.session_state.chat_display_history[st.session_state.current_chat]

# Hiển thị lịch sử tin nhắn trên UI
for msg in display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Xử lý khi nhập câu hỏi mới
if prompt := st.chat_input("Hỏi về giáo án, bài tập, SGK..."):
    # 1. Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    display_history.append({"role": "user", "content": prompt})

    # 2. Tìm kiếm ngữ cảnh (RAG Retrieval)
    context_text = ""
    if vector_store:
        # Lấy 3 đoạn văn bản liên quan nhất từ các file
        docs = vector_store.similarity_search(prompt, k=4)
        context_text = "\n\n".join([f"--- Trích đoạn tài liệu ---\n{doc.page_content}" for doc in docs])

    # 3. Tạo Prompt tăng cường (Augmented Prompt)
    if context_text:
        augmented_prompt = f"""Dựa vào các THÔNG TIN TỪ TÀI LIỆU DƯỚI ĐÂY để trả lời câu hỏi của người dùng. 
        Nếu thông tin tài liệu không đủ, hãy dựa vào kiến thức của bạn.
        
        [THÔNG TIN TỪ TÀI LIỆU (Giáo án, SGK Tin 10 CD, Thang đo Bloom)]:
        {context_text}
        
        [CÂU HỎI CỦA NGƯỜI DÙNG]: 
        {prompt}
        """
    else:
        augmented_prompt = prompt

    # 4. Gửi đến AI
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ và tra cứu tài liệu..."):
            # Gửi tin nhắn chứa ngữ cảnh cho API
            response = active_chat.send_message(augmented_prompt)
            st.markdown(response.text)
            
    # Lưu tin nhắn phản hồi vào lịch sử hiển thị
    display_history.append({"role": "assistant", "content": response.text})
