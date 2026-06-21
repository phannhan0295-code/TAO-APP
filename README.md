# 📈 Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư Cổ Phiếu HOSE (Streamlit App)

Ứng dụng web được xây dựng bằng **Streamlit** giúp trực quan hóa, kiểm định lịch sử (backtest) và tối ưu hóa danh mục đầu tư cổ phiếu trên sàn HOSE giai đoạn 2020 - 2023. Chiến lược này kết hợp bộ chọn cổ phiếu bằng **chỉ báo MACD** kết hợp với **lọc chế độ thị trường (Regime Filter)** qua đường trung bình động SMA của chỉ số VN-Index để giảm thiểu rủi ro trong các pha sập của thị trường.

---

## 🌟 Tính Năng Chính

1. **Chọn lọc Danh mục tối ưu theo chu kỳ (Quý):**
   - Đánh giá tất cả cổ phiếu đủ điều kiện thanh khoản và xu hướng tăng.
   - Chạy thử thuật toán MACD trên dữ liệu 1 năm gần nhất để chấm điểm (Sharpe Ratio).
   - Chọn ra 5 mã cổ phiếu tối ưu nhất để nắm giữ trong quý tiếp theo.

2. **Lớp Phòng thủ Thị trường (Market-Regime Filter):**
   - Kiểm tra vị thế của VN-Index so với đường trung bình động SMA 200 ngày.
   - Khi VN-Index ở dưới đường SMA 200 (thị trường gấu) $\rightarrow$ Tự động bán toàn bộ danh mục và chuyển 100% tài sản về tiền mặt để né tránh các đợt sụt giảm mạnh (như năm 2022).
   - Khi VN-Index vượt lên trên SMA 200 $\rightarrow$ Giải ngân lại theo trọng số mục tiêu.

3. **Tương tác trực quan & Báo cáo Hiệu quả chuyên nghiệp:**
   - Tải lên tệp CSV dữ liệu giao dịch động.
   - Tùy chỉnh trực tiếp vốn ban đầu, phí giao dịch, số lượng cổ phiếu, chu kỳ MACD/SMA ngay trên giao diện sidebar.
   - Thống kê các chỉ số hiệu quả KPI: Tổng lợi nhuận, CAGR, biến động năm, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio.
   - Biểu đồ động Plotly (Equity Curve, Drawdown, Asset allocation, Quarterly returns).
   - Nhật ký lịch sử giao dịch mua/bán và cho phép xuất dữ liệu ra file CSV.
   - Kiểm định ý nghĩa thống kê (t-test và Wilcoxon signed-rank test) chứng minh tính hiệu quả của chiến lược so với VN-Index.

---

## 📁 Cấu Trúc File CSV Dữ Liệu Đầu Vào

Tệp dữ liệu CSV đầu vào cần chứa dữ liệu giá của các mã cổ phiếu sàn HOSE và chỉ số VN-Index (ticker viết thường: `vnindex`).
Các cột bắt buộc bao gồm:

- `date`: Định dạng ngày tháng (ví dụ: `MM/DD/YYYY` hoặc `YYYY-MM-DD`).
- `ticker`: Tên mã cổ phiếu (ví dụ: `aaa`, `fpt`, `vnindex` - viết thường).
- `open`, `high`, `low`, `close`: Giá mở cửa, cao nhất, thấp nhất, đóng cửa của phiên (tính bằng nghìn đồng).
- `volume`: Khối lượng giao dịch.
- `adj_open`: Giá mở cửa đã điều chỉnh chia tách/cổ tức (tính bằng nghìn đồng).
- `adj_close`: Giá đóng cửa đã điều chỉnh (tính bằng nghìn đồng).

*Lưu ý: Ứng dụng tự động nhân giá cổ phiếu với 1,000 để quy đổi về Đồng thực tế, riêng VN-Index giữ nguyên thang điểm.*

---

## 💻 Hướng Dẫn Cài Đặt và Chạy Cục Bộ (Local)

Để chạy ứng dụng trên máy tính của bạn, vui lòng làm theo các bước sau:

### 1. Chuẩn bị thư mục và môi trường ảo
Mở Terminal (macOS/Linux) hoặc Command Prompt/PowerShell (Windows) và di chuyển vào thư mục dự án:
```bash
cd "/Users/phanthinhan/Desktop/TAO APP"
```

Khởi tạo và kích hoạt môi trường ảo (khuyến nghị):
- **macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Windows:**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### 2. Cài đặt các thư viện cần thiết
Cài đặt toàn bộ các thư viện được định nghĩa trong file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng Streamlit
Khởi chạy máy chủ phát triển cục bộ:
```bash
streamlit run app.py
```
Sau khi khởi chạy thành công, trình duyệt của bạn sẽ tự động mở trang web ứng dụng tại địa chỉ mặc định: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Deploy Lên Streamlit Community Cloud

Để deploy ứng dụng này lên mạng và chia sẻ với mọi người, bạn có thể sử dụng Streamlit Community Cloud miễn phí theo các bước:

### Bước 1: Đẩy dự án lên GitHub
1. Tạo một Repository mới trên GitHub của bạn (ví dụ tên: `hose-portfolio-optimization`).
2. Khởi tạo Git trong thư mục dự án cục bộ và đẩy mã nguồn lên:
   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "Initial commit: Streamlit Portfolio Optimization app"
   git branch -M main
   git remote add origin https://github.com/tai-khoan-github-cua-ban/hose-portfolio-optimization.git
   git push -u origin main
   ```

### Bước 2: Deploy ứng dụng trên Streamlit Cloud
1. Truy cập vào trang web [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấp vào nút **"New app"** ở góc trên cùng bên phải.
3. Cấu hình các thông tin deploy:
   - **Repository:** Chọn kho lưu trữ `hose-portfolio-optimization` vừa tạo.
   - **Branch:** Chọn nhánh `main`.
   - **Main file path:** Nhập `app.py`.
4. Nhấp vào **"Deploy!"**. Chỉ sau 1-2 phút, ứng dụng của bạn sẽ được kích hoạt trực tuyến và cung cấp một đường dẫn công khai dạng `https://<ten-app>.streamlit.app/`.

---

## 🛠️ Công Nghệ Sử Dụng

- **Streamlit:** Bộ khung xây dựng giao diện ứng dụng Web.
- **Pandas & NumPy:** Tiền xử lý dữ liệu và tính toán chỉ báo số lớn.
- **Plotly:** Thiết kế biểu đồ động, hỗ trợ zoom và xem thông số khi hover chuột.
- **SciPy:** Thực hiện các phép kiểm định thống kê toán học (t-test, Wilcoxon).
