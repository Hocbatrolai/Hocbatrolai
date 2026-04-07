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
# 1. CẤU HÌNH AI & RAG
# ---------------------------------------------------------------------------
GROQ_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

kma_knowledge_base = [
    "Tên gọi: Học viện Kỹ thuật Mật mã có tên tiếng Anh là Vietnam Academy of Cryptography Techniques (hoặc Academy of Cryptography Techniques), mã trường là KMA.",
    "Cơ quan chủ quản: Học viện Kỹ thuật Mật mã (KMA) trực thuộc Ban Cơ yếu Chính phủ của Bộ Quốc phòng.",
    "Ngày truyền thống: Ngày truyền thống của Học viện Kỹ thuật Mật mã (KMA) là ngày 15/4/1976.",
    "Lịch sử thành lập: Tên gọi Trường Đại học Kỹ thuật Mật mã bắt đầu được sử dụng từ ngày 5/6/1985.",
    "Lịch sử thành lập: Tên gọi Học viện Kỹ thuật Mật mã (KMA) chính thức bắt đầu từ tháng 2 năm 1995 trên cơ sở sáp nhập Trường Đại học Kỹ thuật Mật mã và Viện Nghiên cứu Khoa học Kỹ thuật Mật mã.",
    "Tiền thân: Tiền thân của KMA bao gồm Trường Cán bộ Cơ yếu Trung ương, Trường Đại học Kỹ thuật Mật mã và Viện Nghiên cứu Khoa học Kỹ thuật Mật mã.",
    "Vị thế: KMA được chính phủ Việt Nam lựa chọn là một trong tám cơ sở trọng điểm đào tạo nhân lực an toàn thông tin quốc gia.",
    "Cơ sở đào tạo: Học viện Kỹ thuật Mật mã (KMA) hiện có 02 cơ sở. Trụ sở chính tại 141 Chiến Thắng, Hà Nội. Cơ sở phía Nam tại 17A Cộng Hòa, TP.HCM.",
    "Nhiệm vụ: Nhiệm vụ quan trọng của Học viện Kỹ thuật Mật mã là tổ chức đào tạo chuyên ngành mật mã, an toàn thông tin.",
    "Ký túc xá: KMA có ký túc xá dành riêng cho sinh viên hệ quân sự tại cơ sở Hà Nội.",
    "Hệ đào tạo: Học viện Kỹ thuật Mật mã (KMA) có chương trình đào tạo hệ quân sự và hệ dân sự.",
    "Tuyển sinh đại học: Hiện tại, KMA tuyển sinh 3 ngành: An toàn thông tin, Công nghệ thông tin, Kỹ thuật điện tử viễn thông.",
    "Điểm chuẩn 2025: An toàn thông tin: 24,42; Công nghệ thông tin: 24,17; Điện tử viễn thông: 23,48.",
    "Học phí: Học phí hệ dân sự KMA năm 2024-2025 khoảng 525.000 VNĐ/tín chỉ.",
    "Cách tính điểm tổng kết: Điểm tổng kết = Điểm quá trình * 0.3 + Điểm thi cuối kỳ * 0.7.",
    "Trang web chính thức KMA: https://actvn.edu.vn/",
    "Cổng thông tin đào tạo KMA: https://ktdbcl.actvn.edu.vn/dang-nhap.html"
]

vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(kma_knowledge_base)

def retrieve_kma_info(query, top_k=2):
    try:
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, kb_vectors)[0]
        top_indices = sims.argsort()[-top_k:][::-1]
        results = [kma_knowledge_base[idx] for idx in top_indices if sims[idx] > 0.1]
        return "\n".join(results)
    except: return ""

def search_internet(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if results:
                return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        return ""
    except: return ""

# ---------------------------------------------------------------------------
# 2. DATABASE NỘI BỘ
# ---------------------------------------------------------------------------
def get_db_connection():
    db_url = os.environ.get('DATABASE_URL')
    if not db_url: return None
    return psycopg2.connect(db_url)

def init_db():
    conn = get_db_connection()
    if not conn: return
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

init_db()

# ---------------------------------------------------------------------------
# 3. CÁC ROUTE GIAO DIỆN (ĐÃ FIX LỖI 404)
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session['user'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db_connection()
        if not conn: return "Lỗi kết nối Database!"
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT * FROM Users WHERE username = %s', (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            return redirect(url_for('home'))
        flash("Sai tài khoản hoặc mật khẩu!", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash("Vui lòng điền đủ thông tin!", "warning")
            return redirect(url_for('register'))
        hashed_pw = generate_password_hash(password)
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute('INSERT INTO Users (username, password) VALUES (%s, %s)', (username, hashed_pw))
            conn.commit()
            flash("Đăng ký thành công!", "success")
            return redirect(url_for('login'))
        except IntegrityError:
            flash("Tên đăng nhập đã tồn tại!", "warning")
        finally:
            cur.close()
            conn.close()
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------------------------------------------------------------------------
# 4. ROUTE CHATBOT AI
# ---------------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({"answer": "Vui lòng đăng nhập!"}), 401
    
    data = request.json
    user_message = data.get('message', '')
    image_data = data.get('image', None)
    chat_history = data.get('history', [])

    try:
        kma_context = retrieve_kma_info(user_message)
        web_context = search_internet(user_message)
        current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        system_prompt = f"Bạn là Lavie, trợ lý KMA. Thời gian: {current_time}. Thông tin: {kma_context} {web_context}"
        
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history: messages.extend(chat_history[-6:])

        if image_data:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_message or "Phân tích ảnh"}, {"type": "image_url", "image_url": {"url": image_data}}]})
            model = "llama-3.2-11b-vision-preview"
        else:
            messages.append({"role": "user", "content": user_message})
            model = "llama-3.3-70b-versatile"

        response = client.chat.completions.create(messages=messages, model=model)
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"answer": f"Lỗi: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
