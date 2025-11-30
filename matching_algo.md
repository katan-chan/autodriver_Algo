# Thuật Toán Ghép Cặp Rank-Maximal Cho Bài Toán Autodriver Có Tắc Đường

## 1. Mô tả bài toán

Bài toán mô phỏng quá trình phân bổ **các chuyến xe tự hành (trip)** vào **các tuyến đường (path)** trên một **đồ thị giao thông có tắc đường**.  
Mỗi cạnh của đồ thị có **sức chứa giới hạn (bandwidth)** và **hàm penalty** phản ánh mức độ tắc nghẽn khi vượt quá sức chứa.

Mục tiêu là gán lần lượt từng chuyến vào mạng sao cho:
- Giảm thiểu tổng thời gian di chuyển của hệ thống (system cost),
- Tránh hiện tượng tắc nghẽn cục bộ,
- Ưu tiên những chuyến có quãng đường cơ bản dài hơn,
- Và chỉ thực hiện các thay đổi cục bộ tối thiểu khi cần thiết.

Thuật toán kết hợp **greedy priority** (ưu tiên theo thời gian di chuyển cơ bản) và **rank-maximal incremental assignment**, trong đó mỗi bước có thể **điều chỉnh K** (thứ hạng của path) của một hoặc hai chuyến để đạt trade-off tốt nhất.

---

## 2. Đối tượng trong bài toán

### 2.1. Trip (chuyến xe)
- Ký hiệu: `i = 1..T`
- Thuộc tính:
  - `origin_i`, `dest_i`: nút đầu và nút cuối.
  - `paths_i = [p_{i,1}, p_{i,2}, ..., p_{i,K_i}]`: danh sách K đường đi ứng viên, được sắp theo chi phí tăng dần khi **không có penalty**.
  - `B_i = cost(p_{i,1})`: chi phí lý tưởng (free-flow).
  - `assigned_path_index`: thứ hạng hiện tại (ban đầu = `None`).

### 2.2. Path (tuyến đường)
- Tập các cạnh `E(p)` trên đồ thị.
- `cost(p) = Σ_e t_e(N_e)`: tổng chi phí các cạnh theo trạng thái hiện tại.
- `K_i`: số lượng path ứng viên mà trip i có thể chọn.

### 2.3. Edge (cạnh của đồ thị)
- `t_e^0`: thời gian di chuyển khi không tắc.
- `B_e`: năng lực (bandwidth / capacity).
- `N_e`: số xe hiện tại trên cạnh e.
- `t_e(N_e) = t_e^0 * exp(max(0, N_e - B_e))`: hàm penalty (có thể thay bằng dạng khác).

---

## 3. Ràng buộc bài toán

1. Mỗi trip phải được gán đúng một path trong danh sách `paths_i`.
2. Trạng thái tắc đường trên cạnh `e` được xác định bởi tổng số xe qua đó:  
   `N_e = Σ_i 1[e ∈ path_i]`.
3. Khi thêm hoặc đổi path, cần cập nhật lại toàn bộ `N_e` của các cạnh liên quan.
4. Không có bước nào làm giảm số lượng trip đã được gán hợp lệ.

---

## 4. Quy trình thuật toán

### 4.1. Bước khởi tạo
1. Với mỗi trip `i`, tính chi phí **free-flow** của các path ứng viên:  
   `C_{i,k} = Σ_e t_e^0`.
2. Sắp xếp các path của `i` theo `C_{i,k}` tăng dần.
3. Xác định `B_i = C_{i,1}` là chi phí tốt nhất lý tưởng.
4. **Sắp xếp toàn bộ các trip theo `B_i` giảm dần** — tức là *thằng đi lâu nhất xét trước*.
5. Khởi tạo `N_e = 0` cho mọi cạnh.

---

### 4.2. Vòng lặp chính

Duyệt lần lượt từng trip theo thứ tự đã sắp.

#### Bước 1: Thử thêm trực tiếp
- Giả sử gán trip `i` vào path tốt nhất `p_{i,1}`.
- Cập nhật tạm `N'_e = N_e + 1` với mọi cạnh `e ∈ p_{i,1}`.
- Tính **ΔSystemCost** = `SysCost' - SysCost`.

Nếu `ΔSystemCost` nhỏ hơn một ngưỡng cho trước (penalty chưa đáng kể),  
→ **chấp nhận trực tiếp**:  
  - `assigned_path_index[i] = 1`
  - Cập nhật `N_e = N'_e`

Ngược lại → sang Bước 2.

---

#### Bước 2: Tối ưu cục bộ (giảm K của một trong hai chuyến)

Tìm cặp thay đổi `(i, j)` có thể giảm tổng chi phí ít nhất:

1. **Phương án A (đẩy trip mới xuống):**
   - i dùng path thứ hai `p_{i,2}`.
   - Tính `SysCost_A` → `ΔA = SysCost_A - SysCost`.

2. **Phương án B (đẩy một trip cũ xuống):**
   - Xét từng trip `j` đang dùng các cạnh trùng với `p_{i,1}`.
   - Nếu `j` có path kế tiếp `p_{j,k_j+1}`, thử:
     - `i`: dùng `p_{i,1}`
     - `j`: dùng `p_{j,k_j+1}`
   - Tính `SysCost_B(j)` → `ΔB(j) = SysCost_B(j) - SysCost`.

3. **Chọn phương án có Δ nhỏ nhất:**
   - Nếu `ΔA` nhỏ nhất → i bị đẩy xuống path 2.
   - Nếu có `ΔB(j)` nhỏ hơn → đẩy j xuống path tệ hơn, giữ i ở path tốt nhất.
   - Nếu tất cả Δ > 0, chọn Δ nhỏ nhất để giảm thiệt hại tổng thể.

4. Cập nhật `N_e` và `assigned_path_index` tương ứng.

---

### 4.3. Kết thúc
- Thuật toán dừng khi tất cả trip đã được gán path hợp lệ.
- Đầu ra:
  - `assigned_path_index[i]` cho mỗi trip.
  - Tổng system cost cuối cùng: `Σ_e N_e * t_e(N_e)`.

---

## 5. Ý nghĩa và đặc tính

- Greedy với **thứ tự theo thời gian lý tưởng** giúp tránh việc những chuyến đi dài bị kẹt sau.
- Cơ chế “**giảm K một bên**” mô phỏng hành vi nhường đường hợp lý,  
  đảm bảo mỗi bước thêm xe gây tăng chi phí ít nhất.
- Thuật toán **hội tụ sau T bước** vì mỗi trip chỉ thêm một lần và chỉ có số hữu hạn thao tác swap.
- Kết quả là một nghiệm **cục bộ ổn định** (local optimum) theo lớp move “swap hai chuyến”.

---

## 6. Gợi ý mở rộng
- Cho phép hoán đổi đồng thời >2 trip để tránh local minima.
- Dùng hàm penalty trơn hơn (logistic, polynomial) để ổn định gradient.
- Chạy iterative refinement:
  - Sau khi gán hết, re-evaluate cost, re-sort trip, và chạy lại vòng 2 cho fine-tuning.

---
