# Tài liệu: Mốc Thời gian trong Hệ thống Routing

## Định nghĩa Mốc Thời gian

### 1. `start_time` (Thời điểm xuất phát)
- **Ý nghĩa**: Thời điểm xe **SẴN SÀNG** vào cạnh đầu tiên
- **Không phải**: Thời điểm xe "xuất hiện" tại node
- **Ví dụ**: 
  - `start_time = 0`: Xe đã ở node origin, sẵn sàng đi ngay tại t=0
  - Xe VÀO cạnh đầu tiên ngay tại t=0

### 2. `entry_time` và `exit_time` (Thời gian trên cạnh)
- **`entry_time`**: Thời điểm xe **BẮT ĐẦU** đi vào cạnh
- **`exit_time`**: Thời điểm xe **RỜI KHỎI** cạnh (đến node tiếp theo)
- **Công thức không có penalty**:
  ```python
  entry_time = current_time
  exit_time = entry_time + travel_time
  ```

- **Công thức CÓ penalty (tắc đường)**:
  ```python
  entry_time = current_time
  overflow_ratio = (current_load + 1) / bandwidth - 1.0
  if overflow_ratio > 0:
      penalty_time = travel_time × (e^(β×overflow_ratio) - 1)
      exit_time = entry_time + travel_time + penalty_time
  else:
      exit_time = entry_time + travel_time
  ```

## Timeline Ví dụ

### Xe A: Origin=0, Destination=3
**Path**: 0 → 1 → 2 → 3  
**Travel times**: 
- Edge(0,1) = 10s
- Edge(1,2) = 15s (có tắc đường, penalty=5s)
- Edge(2,3) = 8s

**start_time = 0**

```
t=0:     Xe tại node 0, SẴN SÀNG đi
         ↓ (VÀO cạnh 0→1 ngay lập tức)
         
t=0-10:  Xe ĐANG Ở trên cạnh 0→1
         [entry_time=0, exit_time=10]
         
t=10:    Xe ĐẾN node 1
         ↓ (VÀO cạnh 1→2 ngay lập tức)
         
t=10-30: Xe ĐANG Ở trên cạnh 1→2 (bị tắc!)
         [entry_time=10, exit_time=10+15+5=30]
         (Penalty: +5s do overflow)
         
t=30:    Xe ĐẾN node 2
         ↓ (VÀO cạnh 2→3 ngay lập tức)
         
t=30-38: Xe ĐANG Ở trên cạnh 2→3
         [entry_time=30, exit_time=38]
         
t=38:    Xe ĐẾN node 3 (DESTINATION)
```

**Tổng thời gian**: 38 giây (10 + 20 + 8)

## Visualization Timeline

### Time Range trong main.py
```python
time_min = min(vehicle_start_time)  # Thường = 0
time_max = time_min + 600.0          # 10 phút
n_samples = 200                      # Lấy mẫu 200 điểm
```

### Sampling
- **Khoảng thời gian**: [time_min, time_max] = [0, 600]
- **Số điểm**: 200
- **Khoảng cách giữa 2 điểm**: 600 / 200 = 3 giây

### Animation Steps
```python
n_time_steps = 20  # Số frame animation
```
- **Mỗi frame**: 600 / 20 = 30 giây

## Kiểm tra Load tại một thời điểm

### Hàm `count_vehicles_at_time(edge_idx, query_time)`
```python
Điều kiện đếm: entry_time <= query_time < exit_time
```

**Ví dụ với edge (0,1) có 3 xe**:
```
Xe A: [entry=0,  exit=10]
Xe B: [entry=2,  exit=12]
Xe C: [entry=10, exit=20]

Tại t=0:  Count = 1 (chỉ A, vì 0 <= 0 < 10)
Tại t=5:  Count = 2 (A và B, vì 0<=5<10 và 2<=5<12)
Tại t=10: Count = 1 (chỉ C, vì 10<=10<20. A và B đã ra khỏi)
Tại t=12: Count = 1 (chỉ C)
Tại t=20: Count = 0 (C đã ra khỏi)
```

**Lưu ý**: 
- `exit_time` KHÔNG được tính trong interval
- Tại `exit_time`, xe đã RỜI KHỎI cạnh

## Penalty Impact

### Trước khi có penalty (chỉ cost)
- Penalty chỉ ảnh hưởng **cost function**
- Thời gian thực tế = `travel_time` (không đổi)

### Sau khi có penalty (ảnh hưởng time)
- Penalty làm xe đi **CHẬM HƠN**
- Thời gian thực tế = `travel_time + penalty_time`
- **Cascading effect**: Xe chậm → vào cạnh tiếp theo muộn hơn → ảnh hưởng load của cạnh sau

### Công thức Penalty
```python
overflow_ratio = (current_load + 1) / bandwidth - 1.0

if overflow_ratio > 0:
    penalty_time = travel_time × (e^(β×overflow_ratio) - 1)
else:
    penalty_time = 0
```

**Ví dụ**:
- bandwidth = 5 xe
- current_load = 6 xe → overflow_ratio = 7/5 - 1 = 0.4
- β = 1
- travel_time = 10s
- penalty_time = 10 × (e^0.4 - 1) ≈ 10 × 0.49 ≈ 4.9s
- **Actual time**: 10 + 4.9 = **14.9s**
