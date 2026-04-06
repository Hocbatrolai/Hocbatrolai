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

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'kma_secret_key_sieu_bao_mat')

# ---------------------------------------------------------------------------
# CẤU HÌNH AI & RAG (TÍNH NĂNG 3: RETRIEVAL-AUGMENTED GENERATION)
# ---------------------------------------------------------------------------
GROQ_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# Đây là Cơ sở dữ liệu tri thức giả lập (Bạn có thể thêm hàng trăm câu vào đây)
# CƠ SỞ DỮ LIỆU TRI THỨC NỘI BỘ KMA (ĐÃ GỘP TẤT CẢ THÔNG TIN MỚI NHẤT)
kma_knowledge_base = [
    # --- THÔNG TIN CHUNG & LỊCH SỬ ---
    "Học viện Kỹ thuật Mật mã (KMA) có tên tiếng Anh là Vietnam Academy of Cryptography Techniques (hoặc Academy of Cryptography Techniques).",
    "Học viện Kỹ thuật Mật mã trực thuộc Ban Cơ yếu Chính phủ của Bộ Quốc phòng.",
    "Học viện Kỹ thuật Mật mã được thành lập ngày 17 tháng 2 năm 1995 trên cơ sở sáp nhập Trường Đại học Kỹ thuật Mật mã và Viện Nghiên cứu Khoa học Kỹ thuật Mật mã.",
    "Tiền thân của KMA bao gồm Trường Cán bộ Cơ yếu Trung ương (thành lập 15/4/1976), Trường Đại học Kỹ thuật Mật mã (5/6/1985) và Viện Nghiên cứu Khoa học Kỹ thuật Mật mã (17/2/1980).",
    "KMA được chính phủ Việt Nam lựa chọn là một trong tám cơ sở trọng điểm đào tạo nhân lực an toàn thông tin Việt Nam theo Đề án đến năm 2025.",
    "KMA là một trong những trường hàng đầu tại Việt Nam về đào tạo An toàn thông tin và mật mã học.",
    "Mã trường của Học viện Kỹ thuật Mật mã là KMA.",
    
    # --- ĐÀO TẠO, TUYỂN SINH & ĐIỂM CHUẨN ---
    "Trường có chương trình đào tạo hệ quân sự và hệ dân sự.",
    "KMA đào tạo 4 ngành: Kỹ thuật mật mã, An toàn thông tin, Công nghệ thông tin, Điện tử viễn thông.",
    "KMA tuyển sinh hệ dân sự 3 ngành: An toàn thông tin, Công nghệ thông tin (chuyên ngành Kỹ thuật phần mềm nhúng và di động), và Kỹ thuật điện tử viễn thông.",
    "Ngành Kỹ thuật mật mã thuộc ngành cử tuyển, các trường quân đội, công an cử sinh viên mình sang học.",
    "Về đào tạo sau đại học, KMA đào tạo Tiến sĩ ngành Mật mã và Thạc sĩ chuyên ngành An toàn thông tin.",
    "Điểm chuẩn ngành An toàn thông tin năm 2025 của KMA là 24,42 điểm.",
    "Điểm chuẩn ngành Công nghệ thông tin năm 2025 của KMA là 24,17 điểm.",
    "Điểm chuẩn ngành Kỹ thuật điện tử viễn thông năm 2025 của KMA là 23,48 điểm.",
    "Điểm chuẩn các ngành của KMA thường ở mức cao so với các trường kỹ thuật khác.",
    "Học phí hệ đóng học phí của KMA dự kiến năm học 2024-2025 là khoảng 525.000 VNĐ/tín chỉ.",
    "Sinh viên hệ quân sự được miễn học phí và có chế độ sinh hoạt theo quy định của nhà nước.",
    "Ngành Công nghệ thông tin tại KMA tập trung vào phát triển phần mềm và hệ thống nhúng.",
    "Ngành Kỹ thuật điện tử viễn thông đào tạo về hệ thống truyền thông và thiết bị điện tử.",
    "Ngành An toàn thông tin của KMA được đánh giá rất cao trong các bảng xếp hạng ngành IT tại Việt Nam.",
    "KMA chú trọng đào tạo cả lý thuyết và thực hành cho sinh viên.",
    
    # --- CƠ SỞ VẬT CHẤT & ĐỜI SỐNG SINH VIÊN ---
    "Học viện Kỹ thuật Mật mã (KMA) có trụ sở chính tại 141 Chiến Thắng, Tân Triều, Thanh Trì, Hà Nội.",
    "Trường có cơ sở phía Nam (Phân hiệu KMA) tại 17A Cộng Hòa, Phường 4, Quận Tân Bình, TP.HCM.",
    "KMA có ký túc xá dành cho sinh viên hệ quân sự tại cơ sở Hà Nội.",
    "Thư viện của KMA cung cấp nhiều tài liệu chuyên sâu về mật mã, an ninh mạng và công nghệ thông tin.",
    "Môi trường học tập tại KMA mang tính kỷ luật cao, đặc biệt với hệ quân sự.",
    "Sinh viên KMA thường tham gia các cuộc thi an toàn thông tin như CTF (Capture The Flag).",
    "KMA có các câu lạc bộ học thuật và phong trào như: CLB An toàn thông tin, CLB Xung kích, CLB Máu mật mã.",
    
    # --- HỢP TÁC, NGHIÊN CỨU & CƠ HỘI VIỆC LÀM ---
    "Trường thường xuyên tổ chức hội thảo, seminar về công nghệ và bảo mật như VCIS 2024, VCIS 2025 và sắp tới là VCIS 2026.",
    "Trường có hợp tác với nhiều tổ chức và doanh nghiệp trong lĩnh vực công nghệ và an ninh mạng như SamSung, FPT, Viettel.",
    "Sinh viên KMA có thể tham gia các chương trình trao đổi và hợp tác quốc tế.",
    "Sinh viên KMA được khuyến khích nghiên cứu khoa học và tham gia các đề tài thực tế.",
    "Sinh viên KMA có cơ hội thực tập và làm việc tại các cơ quan nhà nước, doanh nghiệp an ninh mạng và công nghệ.",
    "Sinh viên tốt nghiệp KMA có thể làm việc trong các lĩnh vực như an ninh mạng, kiểm thử xâm nhập (pentest), phát triển phần mềm và quản trị hệ thống.",
    "Sinh viên KMA có nhiều cơ hội việc làm sau khi tốt nghiệp với mức lương hấp dẫn.",
    "KMA là lựa chọn hàng đầu cho những ai muốn theo đuổi lĩnh vực an ninh mạng tại Việt Nam.",
    
    # --- TỔ CHỨC & LÃNH ĐẠO ---
    "Giám đốc hiện nay của Học viện Kỹ thuật Mật mã là TS Hoàng Văn Thức (từ tháng 12/2022).",
    "Các Phó giám đốc của KMA hiện nay bao gồm: PGS.TS Lương Thế Dũng, TS Nguyễn Tân Đăng và GS.TS Nguyễn Hiếu Minh.",
    "KMA có đội ngũ giảng viên giàu kinh nghiệm, nhiều người là chuyên gia trong lĩnh vực mật mã và an toàn thông tin.",
    "KMA có các khoa đào tạo gồm: Khoa Mật mã, Khoa An toàn thông tin, Khoa Công nghệ thông tin, Khoa Điện tử Viễn thông, Khoa Lý luận chính trị, Khoa Cơ bản, Khoa Quân sự và Giáo dục thể chất.",
    "Giám đốc KMA qua các thời kỳ bao gồm: Đại tá PGS.TS Lê Mỹ Tú, Thiếu tướng TS Đặng Vũ Sơn, Thiếu tướng TS Nguyễn Nam Hải, PGS.TS Nguyễn Hồng Quang, Đại tá TS Hoàng Văn Quân, Đại tá TS Nguyễn Hữu Hùng.",
    
    # --- KHEN THƯỞNG ---
    "Học viện Kỹ thuật Mật mã đã được tặng thưởng Huân chương Độc lập hạng Nhất (2011), hạng Nhì (2005) và hạng Ba (2000).",
    "Học viện Kỹ thuật Mật mã vinh dự nhận Huân chương Lao động hạng Nhất (1996), hạng Nhì (1990, 2009) và hạng Ba (1991, 2005).",
    "KMA được tặng thưởng Huân chương Bảo vệ Tổ quốc hạng Ba (2016).",
    "Học viện Kỹ thuật Mật mã đã được nhận các phần thưởng quốc tế như: Huân chương Tự do hạng Nhì và Huân chương Lao động hạng Nhì của Nhà nước Lào; Huân chương Bảo vệ Tổ quốc hạng Nhì của Nhà nước Campuchia."


    # --- TOÁN RỜI RẠC & LÝ THUYẾT ĐỒ THỊ ---
    "Toán rời rạc là nền tảng của công nghệ thông tin, nghiên cứu các cấu trúc dữ liệu rời rạc như tập hợp, logic, quan hệ và đồ thị.",
    "Đồ thị (Graph) là một cấu trúc gồm tập các đỉnh (V) và tập các cạnh (E) kết nối các cặp đỉnh đó. Ký hiệu G = (V, E).",
    "Đồ thị vô hướng là đồ thị mà các cạnh không có hướng. Đồ thị có hướng (mạng) là đồ thị mà mỗi cạnh là một cặp đỉnh có thứ tự.",
    "Bậc của một đỉnh trong đồ thị vô hướng là số cạnh đi vào đỉnh đó. Trong đồ thị có hướng, ta có bán bậc vào và bán bậc ra.",
    "Đường đi (Path) là một dãy các đỉnh sao cho giữa hai đỉnh liên tiếp có một cạnh nối. Chu trình (Cycle) là đường đi có đỉnh đầu và đỉnh cuối trùng nhau.",
    "Đồ thị liên thông là đồ thị mà giữa bất kỳ hai đỉnh nào cũng có ít nhất một đường đi nối chúng.",
    
    # --- THUẬT TOÁN ĐỒ THỊ ---
    "Thuật toán Duyệt theo chiều rộng (BFS) sử dụng hàng đợi (Queue) để thăm tất cả các đỉnh gần đỉnh xuất phát trước, thường dùng tìm đường đi ngắn nhất trên đồ thị không trọng số.",
    "Thuật toán Duyệt theo chiều sâu (DFS) sử dụng ngăn xếp (Stack) hoặc đệ quy để đi sâu nhất có thể vào một nhánh trước khi quay lui.",
    "Thuật toán Dijkstra dùng để tìm đường đi ngắn nhất từ một đỉnh nguồn đến tất cả các đỉnh khác trong đồ thị có trọng số không âm.",
    "Thuật toán Bellman-Ford có thể tìm đường đi ngắn nhất trong đồ thị có trọng số âm, đồng thời phát hiện chu trình âm.",
    "Thuật toán Prim và Kruskal đều được dùng để tìm cây khung nhỏ nhất (Minimum Spanning Tree) của đồ thị vô hướng liên thông có trọng số.",
    "Thuật toán Floyd-Warshall dùng để tìm đường đi ngắn nhất giữa mọi cặp đỉnh trong đồ thị.",

    # --- SẮC SỐ & TÔ MÀU ĐỒ THỊ ---
    "Tô màu đồ thị (Graph Coloring) là việc gán màu cho các đỉnh sao cho không có hai đỉnh kề nhau nào có cùng màu.",
    "Sắc số của đồ thị (Chromatic Number), ký hiệu là chi(G), là số màu tối thiểu cần thiết để tô màu đồ thị G.",
    "Bài toán tìm sắc số đồ thị là một bài toán NP-khó, không có thuật toán thời gian đa thức để giải chính xác cho mọi đồ thị.",
    "Thuật toán Tham lam (Greedy Coloring) là phương pháp phổ biến để tô màu đồ thị: duyệt qua các đỉnh và gán màu nhỏ nhất chưa được dùng bởi các đỉnh kề nó.",
    "Định lý Bốn màu khẳng định rằng mọi bản đồ phẳng (đồ thị phẳng) có thể được tô bằng tối đa 4 màu sao cho các vùng kề nhau khác màu.",
    "Đồ thị đầy đủ K_n có sắc số bằng n vì mọi đỉnh đều kề nhau.",
    "Đồ thị lưỡng phân (Bipartite Graph) là đồ thị có sắc số bằng 2.",

    # --- KHÁI NIỆM TOÁN RỜI RẠC KHÁC ---
    "Nguyên lý Dirichlet (Nguyên lý chuồng bồ bưu): Nếu nhốt n+1 con thỏ vào n cái chuồng thì ít nhất có một chuồng chứa từ 2 con thỏ trở lên.",
    "Chỉnh hợp và Tổ hợp là các khái niệm cơ bản trong phép đếm. Tổ hợp chập k của n phần tử là cách chọn k phần tử không tính thứ tự.",
    "Hệ thức truy hồi là công thức biểu diễn một số hạng của dãy số thông qua các số hạng đứng trước nó, ví dụ dãy Fibonacci.",
    "Logic mệnh đề nghiên cứu các khẳng định có giá trị Chân (True) hoặc Giả (False). Các phép toán logic cơ bản gồm: Hội (AND), Tuyển (OR), Phủ định (NOT), Kéo theo (->).",
    "Cây (Tree) là một đồ thị liên thông và không có chu trình. Một cây có n đỉnh thì luôn có đúng n-1 cạnh."
]

# Khởi tạo công cụ tìm kiếm siêu nhẹ (TF-IDF Vectorizer)
vectorizer = TfidfVectorizer()
kb_vectors = vectorizer.fit_transform(kma_knowledge_base)

def retrieve_kma_info(query, top_k=2):
    """Hàm tìm kiếm thông tin liên quan nhất từ thư viện kiến thức nội bộ"""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, kb_vectors)[0]
    
    top_indices = sims.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if sims[idx] > 0.05: # Ngưỡng tương đồng tối thiểu
            results.append(kma_knowledge_base[idx])
    
    return "\n".join(results)

# ---------------------------------------------------------------------------
# DATABASE & ROUTE GIAO DIỆN (Giữ nguyên như cũ)
# ---------------------------------------------------------------------------
def get_db_connection():
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("Chưa tìm thấy DATABASE_URL!")
    return psycopg2.connect(db_url)

def init_db():
    try:
        conn = get_db_connection()
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
        print(f"Lỗi DB: {e}")

init_db()

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
            flash("Vui lòng nhập đầy đủ thông tin!", "warning")
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute('INSERT INTO Users (username, password) VALUES (%s, %s)', (username, hashed_pw))
            conn.commit()
            flash("Đăng ký thành công! Mời bạn đăng nhập.", "success")
            return redirect(url_for('login'))
        except IntegrityError:
            conn.rollback()
            flash("Tên đăng nhập đã tồn tại!", "warning")
        except Exception as e:
            conn.rollback()
            flash(f"Lỗi hệ thống: {str(e)}", "danger")
        finally:
            cur.close()
            conn.close()
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---------------------------------------------------------------------------
# ROUTE CHATBOT AI ĐÃ NÂNG CẤP
# ---------------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({"answer": "Vui lòng đăng nhập!"}), 401

    if not client:
        return jsonify({"answer": "Lỗi: Chưa cấu hình GROQ_API_KEY!"}), 500

    data = request.json
    user_message = data.get('message', '')
    # TÍNH NĂNG 1: TRÍ NHỚ - Nhận lịch sử chat từ giao diện
    chat_history = data.get('history', []) 

    try:
        # TÍNH NĂNG 3 (Tiếp): Rút trích dữ liệu liên quan từ câu hỏi
        retrieved_context = retrieve_kma_info(user_message)
        
        # TÍNH NĂNG 4: BƠM DỮ LIỆU ĐỘNG (Thời gian thực)
        current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        # Khởi tạo System Prompt với RAG
        system_prompt = f"""Bạn là Lavie, trợ lý ảo cực kỳ thân thiện, thông minh và năng động của Học viện Kỹ thuật Mật mã (KMA).
Hôm nay là: {current_time}

Dưới đây là cơ sở dữ liệu nội bộ MỚI NHẤT của trường. Hãy LUÔN ƯU TIÊN dùng thông tin này để trả lời câu hỏi:
---
{retrieved_context if retrieved_context else "Không có thông tin nội bộ đặc biệt nào được tìm thấy cho câu hỏi này."}
---

Quy tắc:
1. Xưng hô là 'Lavie' và gọi người dùng là 'bạn'.
2. Nếu câu hỏi nằm trong cơ sở dữ liệu nội bộ, hãy trả lời chính xác theo đó.
3. Nếu không có dữ liệu nội bộ, hãy dùng kiến thức chung nhưng thể hiện sự khiêm tốn.
4. Trả lời ngắn gọn, tự nhiên, hiện đại, có emoji."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Thêm lịch sử trò chuyện (giữ 8 tin nhắn gần nhất để AI hiểu ngữ cảnh)
        messages.extend(chat_history[-8:])
        
        # Thêm câu hỏi hiện tại
        messages.append({"role": "user", "content": user_message})

        # TÍNH NĂNG 2: MODEL XỊN HƠN (Llama 3.1 70B)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7, # Độ sáng tạo vừa phải
        )
        return jsonify({"answer": chat_completion.choices[0].message.content})
        
    except Exception as e:
        print(f"Lỗi AI: {e}")
        return jsonify({"answer": f"Lỗi: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
