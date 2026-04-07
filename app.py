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
# CẤU HÌNH AI & RAG (TÍNH NĂNG 3: RETRIEVAL-AUGMENTED GENERATION)
# ---------------------------------------------------------------------------
GROQ_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# Đây là Cơ sở dữ liệu tri thức giả lập (Bạn có thể thêm hàng trăm câu vào đây)
# CƠ SỞ DỮ LIỆU TRI THỨC NỘI BỘ KMA (ĐÃ GỘP TẤT CẢ THÔNG TIN MỚI NHẤT)
kma_knowledge_base = [
    # --- THÔNG TIN CHUNG & LỊCH SỬ KMA ---
    "Tên gọi: Học viện Kỹ thuật Mật mã có tên tiếng Anh là Vietnam Academy of Cryptography Techniques (hoặc Academy of Cryptography Techniques), mã trường là KMA.",
    "Cơ quan chủ quản: Học viện Kỹ thuật Mật mã (KMA) trực thuộc Ban Cơ yếu Chính phủ của Bộ Quốc phòng.",
    "Ngày truyền thống: Ngày truyền thống của Học viện Kỹ thuật Mật mã (KMA) là ngày 15/4/1976.",
    "Lịch sử thành lập: Tên gọi Trường Đại học Kỹ thuật Mật mã bắt đầu được sử dụng từ ngày 5/6/1985.",
    "Lịch sử thành lập: Tên gọi Học viện Kỹ thuật Mật mã (KMA) chính thức bắt đầu từ tháng 2 năm 1995 trên cơ sở sáp nhập Trường Đại học Kỹ thuật Mật mã và Viện Nghiên cứu Khoa học Kỹ thuật Mật mã.",
    "Tiền thân: Tiền thân của KMA bao gồm Trường Cán bộ Cơ yếu Trung ương, Trường Đại học Kỹ thuật Mật mã và Viện Nghiên cứu Khoa học Kỹ thuật Mật mã.",
    "Vị thế: KMA được chính phủ Việt Nam lựa chọn là một trong tám cơ sở trọng điểm đào tạo nhân lực an toàn thông tin quốc gia.",
    
    # --- CƠ SỞ ĐÀO TẠO & NHIỆM VỤ ---
    "Cơ sở đào tạo: Học viện Kỹ thuật Mật mã (KMA) hiện có 02 cơ sở. Trụ sở chính tại 141 Chiến Thắng, Tân Triều, Thanh Trì, Hà Nội. Cơ sở phía Nam tại 17A Cộng Hòa, Phường 4, Quận Tân Bình, TP.HCM.",
    "Nhiệm vụ: Nhiệm vụ quan trọng của Học viện Kỹ thuật Mật mã là tổ chức đào tạo chuyên ngành mật mã, an toàn thông tin và một số ngành công nghệ khác.",
    "Ký túc xá: KMA có ký túc xá dành riêng cho sinh viên hệ quân sự tại cơ sở Hà Nội.",
    
    # --- ĐÀO TẠO, TUYỂN SINH & ĐIỂM CHUẨN ---
    "Hệ đào tạo: Học viện Kỹ thuật Mật mã (KMA) có chương trình đào tạo hệ quân sự (miễn học phí, có chế độ) và hệ dân sự (đóng học phí).",
    "Tuyển sinh đại học: Hiện tại, Học viện Kỹ thuật Mật mã tuyển sinh hệ dân sự 3 ngành: An toàn thông tin, Công nghệ thông tin (phần mềm nhúng và di động), Kỹ thuật điện tử viễn thông.",
    "Ngành Mật mã: Ngành Kỹ thuật mật mã tại KMA thuộc hệ cử tuyển, dành cho sinh viên do các trường quân đội, công an cử sang học.",
    "Đào tạo sau đại học: KMA đào tạo Tiến sĩ ngành Mật mã và Thạc sĩ chuyên ngành An toàn thông tin.",
    "Điểm chuẩn 2025: Điểm chuẩn ngành An toàn thông tin năm 2025 của KMA là 24,42 điểm.",
    "Điểm chuẩn 2025: Điểm chuẩn ngành Công nghệ thông tin năm 2025 của KMA là 24,17 điểm.",
    "Điểm chuẩn 2025: Điểm chuẩn ngành Kỹ thuật điện tử viễn thông năm 2025 của KMA là 23,48 điểm.",
    "Học phí: Học phí hệ dân sự của KMA dự kiến năm học 2024-2025 là khoảng 525.000 VNĐ/tín chỉ.",
    "Mở ngành mới: Năm 2017, Học viện Kỹ thuật Mật mã mở thêm chuyên ngành Quản trị mạng an toàn.",
    "Khóa sinh viên 2025: Năm 2025, KMA thành lập khóa AT22 (An toàn thông tin), khóa DT10 (Điện tử Viễn thông) và khóa H36 (Kỹ thuật mật mã).",
    
    # --- CHƯƠNG TRÌNH ĐÀO TẠO & MÔN HỌC ---
    "Môn học khối kiến thức chung KMA: Toán học và Khoa học cơ bản gồm Giải tích 1, Giải tích 2, Đại số tuyến tính, Xác suất thống kê, Toán rời rạc, Vật lý đại cương 1, 2 và Thực hành vật lý đại cương.",
    "Môn học Lý luận chính trị KMA: Triết học Mác – Lênin, Chủ nghĩa xã hội Khoa học, Lịch sử Đảng Cộng sản Việt Nam, Kinh tế chính trị Mác – Lênin, Tư tưởng Hồ Chí Minh.",
    "Môn tự chọn KMA: Sinh viên chọn một trong các môn Khoa học quản lý, Pháp luật VN đại cương, Tâm lý học đại cương, hoặc Logic học.",
    "Môn học Ngoại ngữ & Thể chất KMA: Tiếng Anh 1, Tiếng Anh 2, Tiếng Anh 3, Tin học đại cương, Kỹ năng mềm, Giáo dục thể chất và Giáo dục quốc phòng an ninh.",
    "Môn học Lập trình & Phần mềm KMA (cơ sở ngành): Lập trình căn bản, Cấu trúc dữ liệu và giải thuật (CTDL&GT), Lập trình hướng đối tượng (OOP), Quản trị dự án phần mềm, Phát triển ứng dụng web, Phân tích thiết kế hệ thống, Phát triển phần mềm ứng dụng, Công nghệ phần mềm.",
    "Môn học Mạng & Bảo mật KMA (cơ sở ngành): Mạng máy tính, Quản trị mạng máy tính, Cơ sở an toàn và bảo mật thông tin (ATTT), Kiến trúc máy tính, Nguyên lý hệ điều hành, Linux và phần mềm nguồn mở.",
    "Môn học Điện tử & Viễn thông KMA (cơ sở ngành): Điện tử tương tự và điện tử số, Cơ sở lý thuyết truyền tin, Kỹ thuật vi xử lý, Xử lý tín hiệu số, Kỹ thuật truyền số liệu, Hệ thống viễn thông, Hệ thống thông tin di động.",
    "Môn học Khoa học máy tính & Dữ liệu KMA (cơ sở ngành): Phương pháp tính, Ôtômát và ngôn ngữ hình thức, Chương trình dịch, Lý thuyết độ phức tạp tính toán, Lý thuyết cơ sở dữ liệu, Hệ quản trị cơ sở dữ liệu (HQTCSDL).",
    "Thực tập KMA: Sinh viên KMA phải học Tiếng Anh chuyên ngành và tham gia Thực tập cơ sở chuyên ngành.",
    
    # --- TỔ CHỨC & LÃNH ĐẠO KMA ---
    "Giám đốc: Giám đốc hiện nay của Học viện Kỹ thuật Mật mã là Đại tá, TS Hoàng Văn Thức.",
    "Phó giám đốc: Các Phó giám đốc của KMA hiện nay bao gồm PGS.TS Lương Thế Dũng, TS Nguyễn Tân Đăng và GS.TS Nguyễn Hiếu Minh.",
    "Giám đốc qua các thời kỳ: Đại tá PGS.TS Lê Mỹ Tú, Thiếu tướng TS Đặng Vũ Sơn, Thiếu tướng TS Nguyễn Nam Hải, PGS.TS Nguyễn Hồng Quang, Đại tá TS Hoàng Văn Quân, Đại tá TS Nguyễn Hữu Hùng.",
    "Hiệu trưởng đầu tiên: Hiệu trưởng đầu tiên của Trường Cán bộ cơ yếu Trung ương (tiền thân KMA) là đồng chí Võ Doãn Tiếu.",
    "Giáo sư KMA: Năm 2023, Học viện KMA bổ nhiệm chức danh Giáo sư ngành Công nghệ thông tin lần đầu tiên cho đồng chí Nguyễn Hiếu Minh.",
    "Đoàn thanh niên KMA: Tại Đại hội Đoàn Thanh niên KMA 2025–2030, đồng chí Trịnh Duy Hải được bầu làm Bí thư Đoàn Thanh niên.",
    
    # --- KHEN THƯỞNG, HỢP TÁC & HOẠT ĐỘNG ---
    "Hợp tác quốc tế: KMA đào tạo giúp nước bạn Lào từ năm 1980 và Campuchia từ năm 1981.",
    "Hội thảo: KMA tổ chức Hội thảo Khoa học quốc tế về 'Mật mã và An toàn thông tin' lần thứ nhất vào năm 2024.",
    "Huân chương: Năm 1986, KMA nhận Huân chương Chiến công hạng Nhì. Năm 2021 (kỷ niệm 45 năm), KMA nhận Huân chương Bảo vệ Tổ quốc hạng Nhì.",
    "Phong trào sinh viên: Sinh viên KMA thường tham gia thi Capture The Flag (CTF) và các CLB như An toàn thông tin, Xung kích, Máu mật mã.",

    # --- KIẾN THỨC TOÁN RỜI RẠC & LÝ THUYẾT ĐỒ THỊ ---
    "Khái niệm Toán rời rạc: Toán rời rạc là nền tảng của công nghệ thông tin, nghiên cứu các cấu trúc dữ liệu rời rạc như tập hợp, logic, quan hệ và đồ thị.",
    "Định nghĩa Đồ thị (Graph): Đồ thị là một cấu trúc gồm tập các đỉnh (V) và tập các cạnh (E) kết nối các cặp đỉnh đó. Ký hiệu G = (V, E).",
    "Phân loại đồ thị: Đồ thị vô hướng là đồ thị các cạnh không có hướng. Đồ thị có hướng (mạng) là đồ thị mỗi cạnh là một cặp đỉnh có thứ tự.",
    "Bậc của đỉnh: Bậc của một đỉnh trong đồ thị vô hướng là số cạnh đi vào đỉnh đó. Đồ thị có hướng có bán bậc vào và bán bậc ra.",
    "Đường đi và Chu trình: Đường đi (Path) là dãy các đỉnh nối tiếp nhau bằng cạnh. Chu trình (Cycle) là đường đi có đỉnh đầu và đỉnh cuối trùng nhau.",
    "Đồ thị liên thông: Đồ thị liên thông là đồ thị mà giữa bất kỳ hai đỉnh nào cũng có ít nhất một đường đi nối chúng.",
    
    # --- THUẬT TOÁN ĐỒ THỊ ---
    "Thuật toán duyệt đồ thị: Duyệt theo chiều rộng (BFS) dùng hàng đợi (Queue). Duyệt theo chiều sâu (DFS) dùng ngăn xếp (Stack) hoặc đệ quy.",
    "Thuật toán đường đi ngắn nhất: Thuật toán Dijkstra tìm đường đi ngắn nhất từ một đỉnh trên đồ thị trọng số không âm. Thuật toán Bellman-Ford xử lý được trọng số âm.",
    "Thuật toán cây khung nhỏ nhất: Thuật toán Prim và Kruskal được dùng để tìm cây khung nhỏ nhất (Minimum Spanning Tree) của đồ thị.",
    "Thuật toán mọi cặp đỉnh: Thuật toán Floyd-Warshall dùng để tìm đường đi ngắn nhất giữa mọi cặp đỉnh trong đồ thị.",

    # --- SẮC SỐ & TÔ MÀU ĐỒ THỊ ---
    "Khái niệm Tô màu đồ thị (Graph Coloring): Tô màu đồ thị là việc gán màu cho các đỉnh sao cho không có hai đỉnh kề nhau nào có cùng màu.",
    "Khái niệm Sắc số: Sắc số của đồ thị (Chromatic Number), ký hiệu là chi(G), là số lượng màu tối thiểu cần thiết để tô màu đồ thị hợp lệ.",
    "Độ phức tạp bài toán sắc số: Bài toán tìm sắc số đồ thị là một bài toán NP-khó, chưa có thuật toán thời gian đa thức giải chính xác.",
    "Thuật toán tô màu Tham lam: Greedy Coloring duyệt qua các đỉnh và gán màu nhỏ nhất chưa được dùng bởi các đỉnh kề nó.",
    "Sắc số đồ thị đặc biệt: Đồ thị đầy đủ K_n có sắc số bằng n. Đồ thị lưỡng phân (Bipartite Graph) có sắc số bằng 2.",
    "Định lý tô màu bản đồ: Định lý Bốn màu khẳng định mọi đồ thị phẳng đều có thể được tô bằng tối đa 4 màu.",

    # --- KHÁI NIỆM TOÁN RỜI RẠC KHÁC ---
    "Nguyên lý Dirichlet: Nguyên lý chuồng bồ câu phát biểu nếu nhốt n+1 con thỏ vào n cái chuồng thì ít nhất một chuồng có từ 2 con thỏ trở lên.",
    "Tổ hợp: Tổ hợp chập k của n phần tử là cách chọn k phần tử từ tập n phần tử mà không tính đến thứ tự.",
    "Hệ thức truy hồi: Là công thức biểu diễn một số hạng của dãy số thông qua các số hạng đứng trước nó (ví dụ dãy Fibonacci).",
    "Logic mệnh đề: Nghiên cứu các khẳng định có giá trị Chân (True) hoặc Giả (False) với các phép toán Hội (AND), Tuyển (OR), Phủ định (NOT), Kéo theo.",
    "Khái niệm Cây (Tree): Cây là một đồ thị liên thông và không có chu trình. Cây có n đỉnh thì luôn có đúng n-1 cạnh.",

    # --- QUY ĐỊNH VĂN HÓA HỌC ĐƯỜNG & KIỂM SOÁT TẠI CỔNG KMA ---
    
    "Quy định thẻ sinh viên KMA: Sinh viên, học viên Học viện Kỹ thuật Mật mã (KMA) bắt buộc phải đeo Thẻ học viên - sinh viên do Học viện cấp khi đến trường.",
    "Quy định trang phục áo KMA: Khi đến trường, sinh viên KMA phải mặc áo có cổ, tay áo lịch sự. Học viện khuyến khích sinh viên mặc áo có logo KMA hoặc đồng phục lớp.",
    "Quy định quần áo và giày dép KMA: Sinh viên Học viện Kỹ thuật Mật mã bắt buộc phải mặc quần dài. Tuyệt đối không được đi dép lê, dép xỏ ngón khi vào trường.",
    "Quy định trang phục nữ sinh viên KMA: Sinh viên nữ tại KMA tuyệt đối không được mặc váy trên đầu gối, váy xẻ cao hoặc các trang phục quá mỏng, gây phản cảm.",
    "Hình thức xử lý vi phạm cổng KMA: Bắt đầu từ thứ Hai, ngày 30/3/2026, Học viện Kỹ thuật Mật mã sẽ tăng cường kiểm tra tại cổng. Những trường hợp vi phạm sẽ không được vào trường.",
    "Lỗi không được vào cổng KMA: Sinh viên KMA sẽ không được phép vào Học viện nếu vi phạm quy định về trang phục (như đi dép lê, dép xỏ ngón) hoặc đi học muộn quá thời gian quy định.",
    # --- QUY ĐỊNH & ĐIỀU KIỆN XÉT HỌC BỔNG KHUYẾN KHÍCH HỌC TẬP KMA ---
    
    "Điều kiện chung xét học bổng KMA: Sinh viên Học viện Kỹ thuật Mật mã muốn xét học bổng khuyến khích học tập phải đăng ký đủ số môn, đủ số tín chỉ theo chuẩn khung đào tạo từng học kỳ và phải nộp học phí đúng hạn quy định.",
    "Điều kiện điểm học tập xét học bổng KMA: Điểm trung bình chung tích lũy (TBCTL) học tập phải đạt loại Khá (từ 3.0 trở lên). Trừ các môn không tính điểm cần phải đạt, Học viện chỉ lấy điểm thi lần 1, sinh viên tuyệt đối không có môn thi lại, học lại.",
    "Điều kiện điểm rèn luyện xét học bổng KMA: Để dự xét học bổng, sinh viên KMA phải có điểm rèn luyện từ loại Khá (70 điểm trở lên) và nộp phiếu đánh giá điểm rèn luyện (mẫu 01 và 02) đúng thời gian quy định.",
    "Mức học bổng loại Khá tại KMA: Học bổng loại khá yêu cầu điểm TBCTL đạt loại khá và điểm rèn luyện từ loại khá trở lên. Số tiền học bổng bằng 15 tín chỉ nhân với mức học phí cơ sở của 1 tín chỉ.",
    "Mức học bổng loại Giỏi tại KMA: Học bổng loại giỏi yêu cầu điểm TBCTL đạt loại giỏi và điểm rèn luyện đạt loại tốt. Số tiền học bổng loại giỏi bằng 1,1 lần mức học bổng loại khá.",
    "Mức học bổng loại Xuất sắc tại KMA: Học bổng loại xuất sắc yêu cầu điểm TBCTL đạt loại xuất sắc và điểm rèn luyện đạt loại xuất sắc. Số tiền học bổng loại xuất sắc bằng 1,2 lần mức học bổng loại khá.",
    "Nguyên tắc xét duyệt học bổng KMA: Căn cứ vào nguồn quỹ học bổng khuyến khích học tập, Văn phòng KMA sẽ xác định số suất học bổng và lấy danh sách theo thứ tự điểm từ cao xuống thấp cho từng khóa học, ngành học.",
    # --- QUY CHẾ TÍNH ĐIỂM & GPA KMA ---
    "Cách tính điểm tổng kết 1 môn học (GPA môn) tại KMA: Điểm học phần được tính theo tỷ lệ 30% và 70%. Cụ thể: Điểm tổng kết = Điểm quá trình (chuyên cần, kiểm tra giữa kỳ) * 0.3 + Điểm thi kết thúc học phần (cuối kỳ) * 0.7.",
    
    "Cách tính điểm trung bình chung tích lũy (GPA toàn khóa) tại KMA: Điểm tích lũy được tính bằng tổng của (điểm môn học thứ i nhân với số tín chỉ của môn đó) rồi chia cho tổng số tín chỉ đã học.",
    # --- CÁC ĐƯỜNG LINK & NỀN TẢNG CHÍNH THỨC CỦA KMA ---
    
    "Trang web (Website) trang chủ chính thức của Học viện Kỹ thuật Mật mã (KMA) có địa chỉ đường link là: https://actvn.edu.vn/",
    "Fanpage Facebook chính thức của Học viện Kỹ thuật Mật mã (KMA) dùng để cập nhật tin tức, sự kiện và thông báo có đường link là: https://www.facebook.com/hocvienkythuatmatma",
    "Cổng thông tin đào tạo (Trang quản lý sinh viên KMA) dùng để xem lịch học, thời khóa biểu, đăng ký tín chỉ và xem điểm có địa chỉ đường link là: https://ktdbcl.actvn.edu.vn/dang-nhap.html).",
    "Kênh YouTube chính thức của Học viện Kỹ thuật Mật mã (KMA) cung cấp các video sự kiện, giới thiệu và bài giảng có đường link là: https://www.youtube.com/channel/UCXy1Pqmu4v_5DTfL8pIVfkw/featured"
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
def search_internet(query, max_results=3):
    """Hàm tự động lướt web tìm kiếm thông tin mới nhất"""
    try:
        results = DDGS().text(query, max_results=max_results)
        if results:
            snippets = [f"- {r['title']}: {r['body']}" for r in results]
            return "\n".join(snippets)
        return ""
    except Exception as e:
        print(f"Lỗi tìm kiếm Web: {e}")
        return ""
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
# ---------------------------------------------------------------------------
# ROUTE CHATBOT AI ĐÃ NÂNG CẤP (GỘP CHUNG VISION, RAG VÀ TRÍ NHỚ)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ROUTE CHATBOT AI ĐÃ NÂNG CẤP (VISION + RAG + INTERNET SEARCH + TRÍ NHỚ)
# ---------------------------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    if 'user' not in session:
        return jsonify({"answer": "Vui lòng đăng nhập!"}), 401

    if not client:
        return jsonify({"answer": "Lỗi: Chưa cấu hình GROQ_API_KEY!"}), 500

    data = request.json
    user_message = data.get('message', '')
    image_data = data.get('image', None) 
    chat_history = data.get('history', []) 

    try:
        # 1. Rút trích dữ liệu RAG (Nội bộ KMA)
        kma_context = retrieve_kma_info(user_message) if user_message else ""
        
        # 2. Tìm kiếm thêm trên Internet (Google/DuckDuckGo)
        web_context = search_internet(user_message) if user_message else ""
        
        current_time = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

        # 3. Cấu hình System Prompt (Bơm cả 2 luồng dữ liệu vào)
        system_prompt = f"""Bạn là Lavie, trợ lý ảo thông minh của Học viện Kỹ thuật Mật mã (KMA).
Hôm nay là: {current_time}

[THÔNG TIN NỘI BỘ KMA]:
{kma_context if kma_context else "KHÔNG CÓ DỮ LIỆU."}

[THÔNG TIN TỪ INTERNET]:
{web_context if web_context else "KHÔNG CÓ DỮ LIỆU."}

QUY TẮC BẮT BUỘC (TUYỆT ĐỐI TUÂN THỦ):
1. CHỈ TRẢ LỜI dựa trên [THÔNG TIN NỘI BỘ KMA] và [THÔNG TIN TỪ INTERNET] được cung cấp ở trên.
2. Nếu cả 2 nguồn thông tin trên đều "KHÔNG CÓ DỮ LIỆU" hoặc không chứa câu trả lời, bạn BẮT BUỘC phải nói: "Xin lỗi, hiện tại Lavie chưa có thông tin chính xác về vấn đề này. Bạn có thể liên hệ fanpage KMA để biết thêm chi tiết nhé!". TUYỆT ĐỐI KHÔNG ĐƯỢC TỰ BỊA RA HAY ĐOÁN CÂU TRẢ LỜI.
3. Khi trả lời, hãy xưng 'Lavie' và gọi người dùng là 'bạn'. Giữ thái độ thân thiện, năng động, dùng emoji.
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            messages.extend(chat_history[-8:])

        if image_data:
            prompt_text = user_message if user_message else "Hãy phân tích đồ thị trong ảnh này."
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            })
            active_model = "llama-3.2-11b-vision-preview" 
        else:
            messages.append({
                "role": "user",
                "content": user_message
            })
            active_model = "llama-3.1-8b-instant"

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=active_model,
            temperature=0.7, 
        )
        return jsonify({"answer": chat_completion.choices[0].message.content})
        
    except Exception as e:
        print(f"Lỗi AI: {e}")
        return jsonify({"answer": f"Xin lỗi, Lavie gặp lỗi khi xử lý: {str(e)}"}), 500
