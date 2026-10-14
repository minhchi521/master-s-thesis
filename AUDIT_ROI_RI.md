# Kiểm tra rò rỉ dữ liệu (data leakage) — pipeline phát hiện bất thường phổ điểm

## Kết luận ngắn
Có rò rỉ và nhiều lỗi liên quan. Rò rỉ **không phải** lý do khiến kết quả đẹp: sau khi sửa, ROC-AUC vẫn cao
(tổ hợp PU cổ điển 0,952 → 0,975). Nhưng PR-AUC bị thổi phồng (tổ hợp cổ điển 0,817 → 0,744; Two-Step 0,788 → 0,583),
và **thứ hạng mô hình bị đảo** (ScalePU-style 0,944 → 0,975; PUICL-style 0,923 → 0,832). Kết luận về CH4/GT4 vì thế đã thay đổi.

Môi trường chạy lại tái hiện đúng số liệu Colab của bản đầu (ví dụ ScalePU 0,929 / 0,546; PUICL 0,911 / 0,365; PU-bagging 0,921 / 0,721),
nên các so sánh trước/sau dưới đây là đáng tin.

## Các lỗi rò rỉ (xếp theo mức nghiêm trọng)
| # | Lỗi | Ở đâu | Bằng chứng | Cách sửa |
|---|---|---|---|---|
| 1 | Huấn luyện trên tất cả đơn vị rồi chấm lại chính các đơn vị đó (kể cả tập kiểm tra, tỉnh bị giấu bị gán nhãn 0 khi huấn luyện) | evaluate.py (LOPO cũ), pu_models.run_* | Chỉ sửa quy trình đã làm PR-AUC Two-Step giảm 0,788 → 0,674 | `fit_xxx(X_train, y) → predict(X_eval)`; tập kiểm tra bị loại hẳn khỏi huấn luyện |
| 2 | Tách kiểm tra theo đơn vị thay vì theo tỉnh: 9 môn của một tỉnh có đặc trưng cấp tỉnh (chéo-môn) giống hệt nhau | evaluate.py | Đặc trưng `cross_subject_*` tính theo (tỉnh, năm) | Chọn tập kiểm tra âm theo tỉnh (18 tỉnh × 9 môn) |
| 3 | Đặc trưng thời gian là hằng số giả cho cả năm 2018 (0 / 0,5 / 0,5), mà mọi nhãn dương đều ở 2018 | features.py | 3 đặc trưng này đoán "năm 2018" với AUC 0,9995; chiếm 5–9% độ quan trọng trong PU-bagging | Để NaN cho 2018 và loại nhóm thời gian khỏi huấn luyện PU |
| 4 | Đặc trưng tuyệt đối mang dấu ấn năm thi (vi phạm nguyên tắc "chỉ so sánh tương đối") | features.py | Trên tỉnh chưa từng thấy, 17 đặc trưng PU đoán "năm 2018" với AUC 0,986; sau chuẩn hóa tương đối còn 0,846 | Chuẩn hóa vững (trung vị, MAD) trong từng (môn, năm) |
| 5 | Phân phối tham chiếu = trung bình 63 tỉnh, có chứa chính đơn vị đang chấm và 3 tỉnh gian lận | features.py | Đọc mã | Trung vị của các tỉnh khác (loại chính tỉnh đó) |
| 6 | Chuẩn hóa học trên toàn bộ dữ liệu, gồm cả tập kiểm tra | features.py, pu_models.py | Đọc mã | Chuẩn hóa chỉ học trên tập huấn luyện mỗi fold |
| 7 | Tiêm nhiễu: chia ngẫu nhiên tập tổng hợp (cùng đơn vị sạch lặp lại ở nhiều cường độ) rồi huấn luyện bộ phân loại trên chính ca giả — vừa rò rỉ gần-trùng-lặp vừa "tự ra đề tự chấm" | evaluate.py, injection.py | Đường cong độ nhạy phẳng ≈ 1,0 ngay từ cường độ 0,2 | Ca giả chỉ dùng để chấm; chữ ký lấy từ tỉnh bị giấu; mô hình chưa thấy ca giả |

## Các lỗi khác (không phải rò rỉ nhưng làm sai kết luận)
- **Đối chứng âm dùng sai mã tỉnh.** Mã 31/40/48 thực ra là Quảng Bình/Đắk Lắk/Đồng Nai, mã 79 không tồn tại trong dữ liệu. Nhận xét trước đây về "Hải Phòng, Nghệ An, Đà Nẵng" không còn đúng; đã dùng mã thi đúng (01, 02, 03, 04, 29) và có hàm kiểm tra tên khớp.
- **Mô hình null vô nghĩa:** hoán vị điểm rồi đếm tỷ lệ vượt ngưỡng cố định luôn cho cùng một số (độ lệch chuẩn 0). Thay bằng hoán vị nhãn (ROC-AUC) và hoán vị tỉnh.
- **Đặc trưng "tỷ lệ điểm tối đa" luôn bằng 0** vì bin cuối là [9,8–10] nên ngưỡng ≥ 10,0 không bao giờ khớp. Đổi thành tỷ lệ ở bin cuối.
- **Kiểm định Wilcoxon với n = 3** không bao giờ cho p < 0,25; thay bằng bootstrap phân tầng theo fold. So sánh "baseline vs PU" cũ còn lấy nhầm hàng đầu và hàng cuối của bảng.
- **pulearn:** `WeightedUnlabelledPuClassifier` không còn trong bản 0.2 nên import lỗi và Elkan–Noto bị bỏ qua (điểm toàn 0); pulearn giữ lại 20% toàn bộ dòng nên đôi khi không có mẫu dương nào để ước lượng c — đã thêm thử lại.

## Mức ảnh hưởng của từng nhóm sửa (trung bình 24 lượt: 3 fold × 8 lần chia)
Cột đầu là cách làm cũ chạy lại; cột cuối là bản đã sửa hoàn toàn. Mỗi ô là ROC-AUC / PR-AUC.
Bảng đầy đủ: `outputs/audit_attribution.csv` (chạy `python audit_leakage.py`, khoảng 10 phút).

| Mô hình | Cũ (đặc trưng cũ + quy trình cũ) | Chỉ sửa đặc trưng | Chỉ sửa quy trình | Đã sửa hoàn toàn |
|---|---|---|---|---|
| Elkan–Noto | 0,962 / 0,774 | 0,970 / 0,746 | 0,966 / 0,763 | 0,972 / 0,744 |
| PU-bagging | 0,931 / 0,681 | 0,974 / 0,703 | 0,930 / 0,666 | 0,974 / 0,705 |
| Two-Step | 0,914 / 0,788 | 0,831 / 0,586 | 0,905 / 0,674 | 0,870 / 0,583 |
| ScalePU-style | 0,944 / 0,600 | 0,975 / 0,768 | 0,941 / 0,592 | 0,975 / 0,766 |
| PUICL-style | 0,923 / 0,553 | 0,826 / 0,491 | 0,933 / 0,589 | 0,832 / 0,541 |
| Tổ hợp cổ điển | 0,952 / 0,817 | 0,975 / 0,752 | 0,950 / 0,782 | 0,975 / 0,744 |
| Tổ hợp thế hệ mới | 0,941 / 0,651 | 0,910 / 0,586 | 0,946 / 0,678 | 0,911 / 0,611 |
| Tổ hợp tất cả | 0,954 / 0,800 | 0,963 / 0,683 | 0,953 / 0,792 | 0,964 / 0,688 |

## Những điều cần nói rõ khi báo cáo
- Quyết định chuẩn hóa tương đối được đưa ra vì chẩn đoán "đặc trưng đoán được năm 2018", nhưng trước đó tôi đã xem một so sánh thăm dò nhanh (Elkan–Noto 0,962 → 0,971; Two-Step 0,906 → 0,886), nên đây không phải lựa chọn hoàn toàn mù về kết quả. Nên báo cáo trung thực điều này.
- Ablation và so sánh bộ đặc trưng (GT1) là thăm dò. Không được chọn bộ đặc trưng cuối dựa trên chính các fold dùng để báo cáo.
- Các thống kê chuẩn hóa theo (môn, năm) và các baseline không giám sát được tính trên toàn bộ tỉnh của năm đó (không dùng nhãn) — đúng bối cảnh triển khai, nhưng vẫn là tính "truyền dẫn".
- Chỉ có 3 tỉnh dương thật, cùng một năm 2018: kết quả có độ biến thiên lớn giữa các fold và không kiểm định được khả năng tổng quát hóa theo thời gian. Nhãn dương là các ca "bị bắt", không chắc đại diện cho kiểu can thiệp tinh vi chưa bị phát hiện (giả định nhãn dương chọn ngẫu nhiên không thỏa).
- GT1 (đặc trưng kết hợp tốt hơn mean/median): sau khi so sánh tương đối đúng cách, mean/median một mình đạt ROC-AUC 0,981 và nhạy nhất trên ca tiêm nhiễu; giả thuyết chưa được ủng hộ.
