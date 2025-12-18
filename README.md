# AutoDriver Algo - Traffic Routing Simulation

Một hệ thống mô phỏng và tối ưu hóa định tuyến giao thông tự động sử dụng các thuật toán đồ thị và phân bổ băng thông thông minh.

## 📋 Mô tả Dự án

Dự án này cung cấp các giải pháp định tuyến giao thông (traffic routing) dựa trên:
- **Đường đi ngắn nhất (Shortest Path)**: Sử dụng thuật toán Dijkstra
- **Yen's K-Shortest Paths**: Tìm k đường đi ngắn nhất thay thế
- **Greedy Regret Assignment**: Phân bổ xe dựa trên độ "hối tiếc" (regret)
- **Time-based Bandwidth Tracking**: Quản lý băng thông dựa trên khung thời gian

## 🏗️ Cấu trúc Dự án

```
autodriver_Algo/
├── README.md
├── results.html              # Kết quả hiển thị HTML
├── src/
│   ├── main.py              # Điểm vào chính
│   ├── fake_data.py         # Sinh dữ liệu giao thông giả lập
│   ├── visualize.py         # Trực quan hóa bằng Plotly
│   ├── visualize_html.py    # Xuất kết quả ra HTML
│   ├── algorithms/
│   │   ├── common/
│   │   │   ├── dijkstra.py           # Thuật toán Dijkstra
│   │   │   ├── path_cost.py          # Tính toán chi phí đường đi
│   │   │   └── time_slots.py         # Quản lý khung thời gian
│   │   └── time_regret/
│   │       ├── greedy_regret_time.py # Greedy + Regret Assignment
│   │       ├── evaluation_time.py    # Đánh giá kết quả
│   │       └── time_slots.py         # Quản lý slot thời gian
```

## 🚀 Cách Sử dụng

### Cài đặt

Đảm bảo bạn có Python 3.8+ và các thư viện cần thiết:

```bash
pip install numpy networkx plotly numba
```

### Chạy Mô phỏng

```bash
python -m src.main
```

Chương trình sẽ:
1. **Sinh dữ liệu**: Tạo đồ thị giao thông phẳng với các xe (vehicles)
2. **Chạy Baseline**: Giải pháp đường đi ngắn nhất không xét báng thông
3. **Chạy Giải pháp Tối ưu**: Sử dụng Greedy Regret với theo dõi băng thông
4. **So sánh**: Đánh giá hiệu suất giữa các phương pháp
5. **Xuất kết quả**: Tạo file `results.html` với biểu đồ tương tác

## 🔧 Các Thành phần Chính

### Sinh Dữ liệu (`fake_data.py`)

```python
data = generate_planar_traffic_data(
    n_nodes=90,           # Số nút trong đồ thị
    n_vehicles=90,        # Số xe cần định tuyến
    n_communities=3,      # Số cộng đồng (clusters)
    p_in=0.7,            # Xác suất kết nối trong cộng đồng
    p_out=0.5,           # Xác suất kết nối giữa cộng đồng
    bandwidth_low=5,      # Băng thông tối thiểu
    bandwidth_high=6,     # Băng thông tối đa
    seed=42              # Seed để tái tạo kết quả
)
```

### Thuật toán Dijkstra (`algorithms/common/dijkstra.py`)

Triển khai Dijkstra với độ phức tạp O(n²) sử dụng Numba để tối ưu hóa tốc độ:
- Tìm đường đi ngắn nhất từ một nguồn đến một đích
- Hỗ trợ đồ thị với ma trận kề

### Greedy Regret Assignment (`algorithms/time_regret/greedy_regret_time.py`)

Phân bổ xe sử dụng:
- **Yen's K-Shortest Paths**: Tìm k đường đi tốt nhất cho mỗi xe
- **Regret-based Selection**: Chọn đường đi dựa trên độ "hối tiếc" 
- **Time-based Bandwidth**: Quản lý băng thông theo khung thời gian để giảm tắc nghẽn

### Trực quan Hóa (`visualize.py`, `visualize_html.py`)

Tạo các biểu đồ tương tác:
- Sơ đồ tuyến đường xe lên đồ thị
- Biểu đồ tải trên các cạnh theo thời gian
- So sánh tải giữa các giải pháp
- Thanh trượt thời gian để xem mô phỏng theo từng bước

## 📊 Đầu ra

Chương trình tạo ra:
- **results.html**: Bộ trang tương tác với các biểu đồ
- **Console output**: Báo cáo chi tiết về hiệu suất

### Metrics Đánh giá

- **Max Edge Load**: Tải trên cạnh cao nhất
- **Average Edge Load**: Tải bình quân trên các cạnh
- **Total Cost**: Chi phí định tuyến toàn bộ
- **Overload Summary**: Thống kê tràn băng thông

## 🔍 Nguyên lý Hoạt động

### 1. Sinh Dữ liệu
- Tạo đồ thị phẳng dạng lưới với các cộng đồng
- Gán xác suất kết nối khác nhau trong/ngoài cộng đồng
- Gán ngẫu nhiên các cặp origin-destination cho mỗi xe
- Gán băng thông và thời gian di chuyển cho mỗi cạnh

### 2. Giải pháp Baseline
- Sử dụng Dijkstra tìm đường đi ngắn nhất
- Không xét đến báng thông
- Dùng để so sánh với các giải pháp tối ưu

### 3. Greedy Regret Assignment
- Tính k đường đi tốt nhất cho mỗi xe sử dụng Yen's
- Tính "regret" cho mỗi đường đi: hiệu số giữa đường đi thứ 2 và đường đi hiện tại
- Lặp:
  - Chọn xe có regret cao nhất
  - Phân bổ xe đó vào đường đi tốt nhất
  - Cập nhật băng thông (cập nhật chi phí cạnh)
  - Tính lại regret cho các xe còn lại

### 4. Quản lý Băng thông Theo Thời gian
- Mỗi cạnh có `max_slots_per_edge` (số lượng xe tối đa)
- Các xe di chuyển vào cạnh ở các thời điểm khác nhau
- Tính lại khả năng truy cập (availability) tại các slot thời gian khác nhau

## ⚙️ Tham số Cấu hình

Chỉnh sửa `src/main.py` để thay đổi:

```python
k_paths = 10                    # Số đường đi kỹ lặp tìm kiếm
beta_penalty = 10              # Hệ số phạt cho cạnh quá tải
max_slots_per_edge = 200       # Lượng xe tối đa trên mỗi cạnh
n_nodes = 90                   # Số nút
n_vehicles = 90                # Số xe
```

## 📈 Hiệu suất

- Sử dụng **Numba JIT Compilation** để tăng tốc độ tính toán
- Các thuật toán chính được tối ưu hóa cho xử lý nhanh
- Đủ khả năng xử lý đồ thị trung bình (100+ nút, 100+ xe)

## 🤝 Đóng góp

Để cải thiện dự án, vui lòng:
1. Fork dự án
2. Tạo nhánh tính năng (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add AmazingFeature'`)
4. Push lên nhánh (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 Ghi chú

- Dữ liệu được sinh giả lập cho mục đích thử nghiệm
- Các kết quả có thể thay đổi tùy thuộc vào seed ngẫu nhiên
- HTML output tương tác, xem tốt nhất trên trình duyệt hiện đại

## 📧 Liên hệ

- **Repository**: [GitHub](https://github.com/katan-chan/autodriver_Algo)
- **Ngôn ngữ**: Python 3.8+
- **License**: Xem file LICENSE (nếu có)

---

**Cập nhật**: December 2025
