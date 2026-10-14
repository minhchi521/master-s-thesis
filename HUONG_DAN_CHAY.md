# Hướng dẫn chạy bản đã sửa

1. Thay `exam_anomaly_code.zip` trên Google Drive bằng file zip này cùng với file dữ liệu test đã đính kèm trong bài báo cáo `du_lieu_diem_thi_2016-2025.rar` (bỏ thư mục `.git` nặng ~450 MB; zip chỉ vài trăm KB).
2. Cell 3 (giải nén) giữ nguyên bản đã sửa trước đó. Cell 4 chỉ cần `pip install shap pulearn` — KHÔNG cần lệnh `sed` vá `pu_models.py` nữa.
3. Cell chạy pipeline: `python pipeline.py --model pu --force` (bước đánh giá mất ~5 phút trên máy 1 CPU).
4. Muốn tái hiện bảng "trước/sau khi sửa lỗi": `python audit_leakage.py` (~10 phút) → `outputs/audit_attribution.csv`.

Thư mục `ket_qua_sau_sua/` là kết quả đã chạy lại (hai lần chạy cho kết quả trùng khớp tuyệt đối). `AUDIT_ROI_RI.md` là báo cáo chi tiết các lỗi.
Lưu ý: `scores_pu.csv` là điểm trong-mẫu chỉ để lập danh sách theo dõi — không dùng để báo cáo hiệu năng.
