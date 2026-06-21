# Kế hoạch lập trình — Phát hiện bất thường phổ điểm THPT (PU Learning)

> Tài liệu này chia công việc code thành **các phần độc lập**. Mỗi phần là một module, có đầu vào – đầu ra rõ ràng và một checklist. Làm tuần tự từ Phần 0 → Phần 9; phần sau dùng kết quả phần trước.

**Gắn với đề cương:** mỗi phần ghi rõ ứng với Mục nào trong `De_cuong_hoan_chinh.docx`.

**Ba phần chịu lực nhất (làm kỹ):** Phần 2 (đặc trưng), Phần 4 (tiêm nhiễu), Phần 8 (đánh giá). Phần 4 là cái khiến mọi con số đáng tin — đừng làm qua loa.

---

## Cấu trúc thư mục dự án

```
exam-anomaly/
├── data/
│   ├── raw/            # CSV cấp thí sinh tải về (du_lieu_diem_thi_2018.csv ...)
│   └── processed/      # phổ điểm + bảng đặc trưng đã gộp
├── src/
│   ├── data_prep.py    # Phần 1
│   ├── features.py     # Phần 2
│   ├── detectors.py    # Phần 3 (baseline không giám sát)
│   ├── injection.py    # Phần 4 (tiêm nhiễu hiệu chỉnh theo 2018)
│   ├── pu_models.py    # Phần 5 (PU Learning)
│   ├── controls.py     # Phần 6 (đối chứng âm + null)
│   ├── explain.py      # Phần 7 (SHAP + phiếu sức khỏe)
│   ├── evaluate.py     # Phần 8 (chỉ số, seed, kiểm định)
│   └── pipeline.py     # Phần 9 (đóng gói chạy 1 lệnh)
├── notebooks/          # thử nghiệm, vẽ biểu đồ
├── outputs/            # kết quả, hình, bảng
├── requirements.txt
└── README.md
```

---

## Phần 0 — Chuẩn bị dự án

**Mục tiêu:** dựng khung dự án chạy được.

- [ ] Tạo thư mục như cây trên, khởi tạo Git (`git init`).
- [ ] Tạo môi trường ảo: `python -m venv .venv` rồi kích hoạt.
- [ ] `requirements.txt`: `pandas numpy scipy scikit-learn matplotlib seaborn shap pulearn`
- [ ] `pip install -r requirements.txt`
- [ ] Cố định seed mặc định (ví dụ 0) ở một chỗ dùng chung.

---

## Phần 1 — Dữ liệu  ·  `data_prep.py`  ·  *(Mục 6)*

**Mục tiêu:** từ dữ liệu cấp thí sinh → bảng phổ điểm theo (tỉnh × môn × năm).

**Input:** `data/raw/du_lieu_diem_thi_YYYY.csv` (2018–2025).
**Output:** `data/processed/pho_diem_{nam}.csv`, `data/processed/diem_tho_gop.parquet`.

- [ ] Tải dữ liệu 2018–2025 (xem README dataset). Lõi phân tích: **2018–2024** (chung 9 môn); **2025 để riêng**.
- [ ] Đọc CSV, map mã tỉnh → tên tỉnh (bảng 1–64).
- [ ] Làm sạch: bỏ điểm thiếu/NaN (= không thi); quyết định cách xử lý điểm 0 và ghi chú rõ.
- [ ] Gộp về đơn vị (tỉnh, môn, năm); lưu phổ điểm theo bin 0.2.
- [ ] Tách riêng đợt 2 của 2020, 2021.
- [ ] Đánh dấu 3 tỉnh mỏ neo 2018: **05 Hà Giang, 14 Sơn La, 23 Hòa Bình**.

---

## Phần 2 — Thiết kế đặc trưng  ·  `features.py`  ·  *(Mục 7.1)*

**Mục tiêu:** mỗi đơn vị (tỉnh, môn, năm) → một vector đặc trưng. **Tất cả đều là so sánh tương đối** để khử độ khó của đề.

**Input:** phổ điểm/điểm thô từ Phần 1.
**Output:** `data/processed/dac_trung.csv` (mỗi dòng 1 đơn vị).

- [ ] Đuôi điểm cao: tỷ lệ điểm ≥ 9, ≥ 8, tỷ lệ điểm 10.
- [ ] Hình dạng phân phối: mean, median, std, skewness, kurtosis.
- [ ] **Lệch so với toàn quốc cùng năm:** Jensen–Shannon, Wasserstein.
- [ ] **Bất nhất chéo-môn:** đuôi điểm cao của môn này so với các môn khác cùng tỉnh.
- [ ] **Cú nhảy theo thời gian:** thứ hạng tương đối của tỉnh so với cả nước, năm nay vs năm trước.
- [ ] Tín hiệu vi mô: độ răng cưa/“dồn điểm” quanh các mốc làm tròn.
- [ ] Chuẩn hóa đặc trưng; lưu kèm metadata (tỉnh, môn, năm).

---

## Phần 3 — Baseline không giám sát  ·  `detectors.py`  ·  *(Mục 7.2)*

**Mục tiêu:** đường cơ sở vững, không cần nhãn — đảm bảo luôn có kết quả.

**Input:** `dac_trung.csv`.
**Output:** điểm bất thường cho mỗi đơn vị (`outputs/scores_baseline.csv`).

- [ ] Isolation Forest.
- [ ] Local Outlier Factor (LOF).
- [ ] DBSCAN (đánh dấu điểm ngoài cụm).
- [ ] Xếp hạng theo độ phân kỳ phân phối (từ Phần 2).
- [ ] Hàm chung trả về “điểm bất thường” chuẩn hóa 0–1 cho mọi mô hình.
- [ ] Kiểm tra nhanh: 3 tỉnh mỏ neo 2018 có lọt top bất thường không.

---

## Phần 4 — Máy tiêm nhiễu hiệu chỉnh theo 2018  ·  `injection.py`  ·  *(Mục 7.3)* ⭐

**Mục tiêu:** tạo hàng trăm ca “gian lận giả” *giống ca thật* để có nhãn mà đánh giá. **Đây là trụ cột — chống đánh giá vòng tròn.**

**Input:** phổ điểm sạch + chữ ký 2018.
**Output:** tập dữ liệu có nhãn tổng hợp + báo cáo kiểm chứng.

- [ ] **Bước 1 – Trích chữ ký:** đo phổ điểm 3 tỉnh 2018 lệch khỏi mặt bằng thế nào (đuôi cao phình bao nhiêu, méo ra sao).
- [ ] **Bước 2 – Dựng máy mô phỏng:** lấy phổ điểm sạch, bóp méo theo đúng chữ ký đó ở nhiều **cường độ** (nhẹ → nặng).
- [ ] **Bước 3 – Kiểm chứng (GT3):** đo khoảng cách phân phối giữa ca mô phỏng và ca thật cùng cường độ; chỉ dùng khi đủ gần.
- [ ] Sinh tập ca giả ở nhiều cường độ, gắn nhãn để dùng cho Phần 8.

---

## Phần 5 — PU Learning  ·  `pu_models.py`  ·  *(Mục 7.2)*

**Mục tiêu:** khai thác số ít nhãn dương + dữ liệu không nhãn; **khảo sát** hành vi khi nhãn cực hiếm (không cá cược phải thắng baseline).

**Input:** đặc trưng + nhãn (ca thật 2018 + ca giả từ Phần 4).
**Output:** điểm bất thường theo PU (`outputs/scores_pu.csv`).

- [ ] Elkan–Noto.
- [ ] Biased SVM hoặc PU-bagging.
- [ ] Phân loại hai bước (two-step).
- [ ] Tổ hợp (ensemble) kết hợp điểm từ nhiều mô hình.
- [ ] Ghi lại trung thực cả trường hợp PU **không** vượt baseline (đó vẫn là kết quả khoa học).

---

## Phần 6 — Đối chứng âm + mô hình null  ·  `controls.py`  ·  *(Mục 7.4)*

**Mục tiêu:** đảm bảo mô hình **không** báo nhầm tỉnh giỏi hợp pháp; đo tỷ lệ báo nhầm nền.

**Input:** đặc trưng + điểm bất thường.
**Output:** chỉ số báo nhầm trên đối chứng âm và null.

- [ ] Bộ “giỏi hợp pháp”: tỉnh mạnh ổn định nhiều năm, nhiều môn (ví dụ nhiều trường chuyên).
- [ ] Kiểm tra mô hình KHÔNG đánh dấu nhóm này.
- [ ] Mô hình null/hoán vị: trộn ngẫu nhiên để đo tỷ lệ báo nhầm nền.
- [ ] Báo cáo FPR trên cả hai.

---

## Phần 7 — Giải thích (XAI) + phiếu sức khỏe  ·  `explain.py`  ·  *(Mục 7.5)*

**Mục tiêu:** mỗi cảnh báo kèm lý do; đóng gói thành sản phẩm dùng được.

**Input:** mô hình + đặc trưng của đơn vị bị đánh dấu.
**Output:** `outputs/phieu_suc_khoe_{tinh}_{mon}_{nam}.json|html`.

- [ ] Tính SHAP cho điểm bất thường.
- [ ] Mỗi đơn vị: điểm bất thường + xếp hạng + 3–5 đặc trưng đóng góp lớn nhất kèm diễn giải bằng lời.
- [ ] Xuất “phiếu sức khỏe phổ điểm” gọn cho người rà soát.

---

## Phần 8 — Đánh giá & độ tin cậy  ·  `evaluate.py`  ·  *(Mục 9)* ⭐

**Mục tiêu:** đo hiệu quả một cách chắc chắn, không tự lừa mình.

**Input:** điểm bất thường (Phần 3, 5) + nhãn (Phần 4) + đối chứng (Phần 6).
**Output:** bảng kết quả + hình trong `outputs/`.

- [ ] **Chốt quy trình đánh giá TRƯỚC khi xem kết quả** (chỉ số, ngưỡng, cách so sánh).
- [ ] ROC-AUC, PR-AUC trên nhãn tổng hợp đã kiểm chứng.
- [ ] Precision@k, Recall với các ca đã biết (2018).
- [ ] Đường cong độ nhạy theo cường độ can thiệp.
- [ ] Chạy **nhiều seed (0,1,2)** → trung bình + khoảng tin cậy; kiểm định thống kê.
- [ ] **Ablation**: bật/tắt từng nhóm đặc trưng.
- [ ] **Phân tích độ nhạy**: đổi giả định máy tiêm nhiễu, xem kết luận có ổn định.
- [ ] Phân tích lỗi: khi nào sai, vì sao.

---

## Phần 9 — Đóng gói pipeline  ·  `pipeline.py`  ·  *(Mục 8)*

**Mục tiêu:** một lệnh chạy hết: phổ điểm mới → danh sách theo dõi + giải thích.

- [ ] Nối Phần 1 → 8 thành một luồng.
- [ ] Tham số dòng lệnh: chọn năm, môn, mô hình.
- [ ] Xuất danh sách tỉnh xếp hạng theo điểm bất thường + phiếu sức khỏe.
- [ ] Viết README hướng dẫn chạy lại.

---

## Định nghĩa “tối thiểu coi như xong” *(Mục 3)*

1. Pipeline dữ liệu + đặc trưng tái lập được (Phần 1–2).
2. ≥ 3 baseline chạy ổn định và bắt được ca 2018 (Phần 3).
3. Máy tiêm nhiễu đã kiểm chứng + đường cong độ nhạy (Phần 4, 8).
4. Phân tích báo nhầm có đối chứng âm (Phần 6).

> PU vượt baseline là **kỳ vọng**, không phải điều kiện sống còn.

---

## Thứ tự khuyến nghị

`Phần 0 → 1 → 2 → 3 → 4 → 8 (đánh giá baseline) → 5 → 6 → 7 → 9`

Lý do: làm xong baseline + tiêm nhiễu + đánh giá (0–4, 8) là đã có một kết quả hoàn chỉnh để báo cáo. PU, đối chứng, XAI, đóng gói (5–7, 9) bồi thêm sau.
