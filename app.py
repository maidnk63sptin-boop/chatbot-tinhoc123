import os
import streamlit as st
import google.generativeai as genai

# 2. Lấy API Key từ hệ thống
api_key = os.getenv("GEMINI_API_KEY")

# 3. Kiểm tra xem đã lấy được Key chưa để báo lỗi rõ ràng trên giao diện
if not api_key:
    st.error("🚨 Không tìm thấy GEMINI_API_KEY. Vui lòng kiểm tra lại file .env hoặc biến môi trường trên Render!")
    st.stop() # Dừng chạy app nếu không có key

instruction = """Bạn là một chuyên gia giáo dục và là trợ lý ảo dành riêng cho giáo viên giảng dạy môn Tin học lớp 10 tại Việt Nam, bám sát chương trình Giáo dục phổ thông 2018 (SGK Kết nối tri thức với cuộc sống và Cánh diều).

Nhiệm vụ của bạn:
1. Hỗ trợ soạn giáo án.
2. Thiết kế bài tập Python.
3. Giải thích kiến thức Tin học lớp 10 (mạng máy tính, Internet, đạo đức số, lập trình).
4. Tạo đề kiểm tra, trắc nghiệm, tự luận.

QUY TẮC BẮT BUỘC:
- Chỉ trả lời các câu hỏi liên quan đến môn Tin học lớp 10.
- Nếu câu hỏi không liên quan, hãy từ chối trả lời và thông báo: "Tôi chỉ hỗ trợ các nội dung thuộc Tin học lớp 10."
"""

# 4. Cấu hình API Key
genai.configure(api_key=api_key) 

model = genai.GenerativeModel('gemini-2.5-flash-lite', 
                              system_instruction=instruction)

if "all_chats" not in st.session_state:
    st.session_state.all_chats = {"Cuộc trò chuyện 1": model.start_chat(history=[])}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Cuộc trò chuyện 1"

with st.sidebar:
    st.title("🗂️ Quản lý trò chuyện")
    
    # Nút tạo cuộc trò chuyện mới
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        # Đặt tên tự động theo số thứ tự
        new_chat_name = f"Cuộc trò chuyện {len(st.session_state.all_chats) + 1}"
        # Tạo phiên chat mới và lưu vào danh sách
        st.session_state.all_chats[new_chat_name] = model.start_chat(history=[])
        # Chuyển màn hình sang đoạn chat vừa tạo
        st.session_state.current_chat = new_chat_name
        st.rerun()

    st.divider()
    st.write("🕒 **Lịch sử trò chuyện:**")
    
    # Hiển thị danh sách các cuộc trò chuyện cũ dưới dạng nút bấm
    for chat_name in st.session_state.all_chats.keys():
        # Nếu là đoạn chat đang mở thì làm mờ nút đi (disabled)
        is_active = (chat_name == st.session_state.current_chat)
        if st.button(chat_name, disabled=is_active, use_container_width=True):
            # Khi bấm vào nút, chuyển trạng thái sang đoạn chat đó
            st.session_state.current_chat = chat_name
            st.rerun()

# --- KHU VỰC GIAO DIỆN CHÍNH ---
st.title("Trợ lý AI Tin học 10 🤖")
st.caption(f"Đang hiển thị: {st.session_state.current_chat}")

# Lấy phiên chat đang được chọn ra để xử lý
active_chat = st.session_state.all_chats[st.session_state.current_chat]

# Hiển thị lịch sử của đoạn chat đang được chọn
for message in active_chat.history:
    role = "assistant" if message.role == "model" else "user"
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

# Xử lý khi nhập câu hỏi mới vào đoạn chat hiện tại
if prompt := st.chat_input("Hỏi về giáo án, bài tập..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    response = active_chat.send_message(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response.text)
