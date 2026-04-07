import os
import psycopg2
import datetime
import numpy as np
from groq import Groq
from psycopg2 import IntegrityError
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS  

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'kma_secret_key_sieu_bao_mat')

# ---------------------------------------------------------------------------
# CẤU HÌNH AI & RAG
# ---------------------------------------------------------------------------
GROQ_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

kma_knowledge_base = [
    # (Giữ nguyên mảng kma_knowledge_base khổng lồ của bạn ở đây - Đảm bảo có dấu phẩy cuối mỗi câu)
    "Tên gọi: Học viện Kỹ thuật Mật mã có tên tiếng Anh là Vietnam Academy of Cryptography Techniques (hoặc Academy of Cryptography Techniques), mã trường là KMA.",
    "Cơ quan chủ quản: Học viện Kỹ thuật Mật mã (KMA) trực thuộc Ban Cơ yếu Chính phủ của Bộ Quốc phòng.",
    # ... các câu khác của bạn ...
    "Kênh YouTube chính thức của Học viện Kỹ thuật Mật mã (KMA) cung cấp các video sự kiện, giới thiệu và bài giảng có đường link là: https://www.youtube.com/channel/UCXy1Pqmu4v_5DTfL8pIVfkw/featured"
]

# Khởi tạo TF-IDF
vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(kma_knowledge_base)

def retrieve_kma_info(query, top_k=2):
    try:
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, kb_vectors)[0]
        top_indices = sims.argsort()[-top_k:][::-1]
        results = [kma_knowledge_base[idx] for idx in top_indices if sims[idx] > 0.1]
        return "\n".join(results)
    except:
        return ""

def search_internet(query, max_results=3):
    try:
        with DDGS() as ddgs:
            # Ép kiểu list và thêm giới hạn để tránh treo server
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        return ""
    except Exception as e:
        print(f"Lỗi tìm kiếm Web: {e}")
        return ""

# ---------------------------------------------------------------------------
# DATABASE - TỐI ƯU HÓA ĐỂ TRÁNH SẬP SERVER
# ---------------------------------------------------------------------------
def get_db_connection():
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        return None # Trả về None nếu không có DB để app không bị sập
    return psycopg2.connect(db_url)

def init_db():
    conn = get_db_connection()
    if not conn: 
        print("⚠️ Cảnh báo: Chưa có DATABASE_URL, tính năng Đăng nhập sẽ không hoạt động.")
        return
    try:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS Users (
                id SERIAL PRIMARY KEY, 
                username VARCHAR(50) UNIQUE NOT NULL, 
                password VARCHAR(255) NOT NULL
            )
        ''')
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Lỗi khởi tạo DB: {e}")

# Gọi khởi tạo DB trong khối try-except để đảm bảo App vẫn chạy
try:
    init_db()
except:
    pass

# (Các Route login/register/logout giữ nguyên như code của bạn)

# ---------------------------------------------------------------------------
# ROUTE CHATBOT AI (FIXED)
# ---------------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    # Kiểm tra login (Bạn có thể tạm comment 2 dòng dưới để test chat trước)
    if 'user' not in session:
        return jsonify({"answer": "Vui lòng đăng nhập để trò chuyện cùng Lavie!"}), 401

    if not client:
        return jsonify({"answer": "Lavie chưa được cấp API Key để hoạt động. Vui lòng kiểm tra lại cấu hình!"}), 500

    data = request.json
    user_message = data.get('message', '')
    image_data = data.get('image', None) 
    chat_history = data.get('history', []) 

    try:
        kma_context = retrieve_kma_info(user_message) if user_message else ""
        web_context = search_internet(user_message) if user_message else ""
        current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        system_prompt = f"""Bạn là Lavie, trợ lý ảo của Học viện Kỹ thuật Mật mã (KMA).
Thời gian: {current_time}
[DỮ LIỆU KMA]: {kma_context if kma_context else "Không có."}
[DỮ LIỆU WEB]: {web_context if web_context else "Không có."}

QUY TẮC: 
- Trả lời thân thiện, sử dụng emoji.
- Chỉ dựa trên dữ liệu được cung cấp. 
- Nếu không biết, hãy bảo bạn liên hệ Fanpage KMA.
"""
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history[-6:]) # Lấy 6 tin nhắn gần nhất

        if image_data:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message or "Phân tích ảnh này"},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            })
            model = "llama-3.2-11b-vision-preview"
        else:
            messages.append({"role": "user", "content": user_message})
            model = "llama-3.3-70b-versatile" # Dùng model 70B cho chất lượng tốt nhất

        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.4
        )
        return jsonify({"answer": response.choices[0].message.content})
        
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        return jsonify({"answer": "Lavie đang bận một chút, bạn thử lại sau giây lát nhé!"}), 500

if __name__ == '__main__':
    app.run(debug=True)
