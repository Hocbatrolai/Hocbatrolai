# ---------------------------------------------------------------------------
# ROUTE CHATBOT AI ĐÃ NÂNG CẤP (GỘP CHUNG VISION, RAG VÀ TRÍ NHỚ)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ROUTE CHATBOT AI ĐÃ NÂNG CẤP (VISION + RAG + INTERNET SEARCH + TRÍ NHỚ)
# ---------------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
if 'user' not in session:
@@ -236,58 +239,60 @@ def chat():

data = request.json
user_message = data.get('message', '')
    image_data = data.get('image', None) # Lấy dữ liệu ảnh nếu có
    chat_history = data.get('history', []) # Nhận lịch sử chat từ giao diện
    image_data = data.get('image', None) 
    chat_history = data.get('history', []) 

try:
        # 1. Rút trích dữ liệu RAG (nếu người dùng có nhắn text)
        retrieved_context = retrieve_kma_info(user_message) if user_message else ""
        # 1. Rút trích dữ liệu RAG (Nội bộ KMA)
        kma_context = retrieve_kma_info(user_message) if user_message else ""
        
        # 2. Tìm kiếm thêm trên Internet (Google/DuckDuckGo)
        web_context = search_internet(user_message) if user_message else ""
        
current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        # 2. Cấu hình System Prompt (Kết hợp cả nhiệm vụ đọc ảnh và kiến thức KMA)
        system_prompt = f"""Bạn là Lavie, trợ lý ảo cực kỳ thân thiện, thông minh của Học viện Kỹ thuật Mật mã (KMA).
        # 3. Cấu hình System Prompt (Bơm cả 2 luồng dữ liệu vào)
        system_prompt = f"""Bạn là Lavie, trợ lý ảo thông minh của Học viện Kỹ thuật Mật mã (KMA).
Hôm nay là: {current_time}

Dữ liệu nội bộ MỚI NHẤT của trường (ưu tiên sử dụng):
---
{retrieved_context if retrieved_context else "Không có thông tin nội bộ đặc biệt nào được tìm thấy."}
---
[THÔNG TIN NỘI BỘ KMA TỪ CƠ SỞ DỮ LIỆU]:
{kma_context if kma_context else "Không có thông tin nội bộ liên quan."}

[THÔNG TIN TÌM KIẾM TỪ INTERNET]:
{web_context if web_context else "Không có thông tin từ internet."}

Nhiệm vụ:
1. Nếu người dùng gửi ảnh đồ thị, hãy phân tích kỹ các đỉnh, cạnh, suy luận logic tìm sắc số.
2. Nếu được hỏi thông tin, dùng dữ liệu nội bộ ở trên để trả lời chính xác.
3. Xưng hô 'Lavie' và gọi người dùng là 'bạn'. Trả lời tự nhiên, hiện đại, có emoji."""
1. Trả lời câu hỏi của người dùng. Ưu tiên dùng [THÔNG TIN NỘI BỘ] nếu câu hỏi liên quan đến KMA.
2. Nếu hỏi kiến thức ngoài hoặc tin tức mới, hãy tổng hợp từ [THÔNG TIN TÌM KIẾM TỪ INTERNET] để trả lời.
3. Nếu gửi ảnh đồ thị, hãy tập trung phân tích tìm sắc số (chromatic number).
4. Xưng hô 'Lavie' và 'bạn', nói chuyện tự nhiên, lưu loát, có emoji."""

messages = [{"role": "system", "content": system_prompt}]

        # 3. Thêm lịch sử trò chuyện (Giữ 8 tin nhắn gần nhất để AI hiểu ngữ cảnh)
if chat_history:
messages.extend(chat_history[-8:])

        # 4. Phân luồng: Nếu có ảnh thì dùng Model Vision, nếu không thì dùng Model Text siêu tốc
if image_data:
            prompt_text = user_message if user_message else "Hãy phân tích đồ thị trong ảnh này và tìm sắc số (chromatic number) của nó."
            prompt_text = user_message if user_message else "Hãy phân tích đồ thị trong ảnh này."
messages.append({
"role": "user",
"content": [
{"type": "text", "text": prompt_text},
{"type": "image_url", "image_url": {"url": image_data}}
]
})
            # Model hỗ trợ Vision trên Groq
active_model = "llama-3.2-11b-vision-preview" 
else:
messages.append({
"role": "user",
"content": user_message
})
            # Model xử lý Text siêu tốc
active_model = "llama-3.1-8b-instant"

        # 5. Gọi API Groq
chat_completion = client.chat.completions.create(
messages=messages,
model=active_model,
            temperature=0.7, # Độ sáng tạo vừa phải
            temperature=0.7, 
)
return jsonify({"answer": chat_completion.choices[0].message.content})
