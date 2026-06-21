"""
config.py — Hằng số dùng chung toàn dự án.

Mọi tham số "ma thuật" (seed, đường dẫn, danh sách năm/môn/tỉnh đặc biệt)
đều đặt ở đây để các module khác import, tránh hard-code rải rác.
"""

import pathlib

# ──────────────────────────────────────────────
# Đường dẫn gốc
# ──────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]   # thư mục exam-anomaly/
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS        = ROOT / "outputs"

# ──────────────────────────────────────────────
# Seed cố định — dùng ở MỌI nơi cần random
# ──────────────────────────────────────────────
RANDOM_SEED = 0

# ──────────────────────────────────────────────
# Khoảng năm phân tích
# ──────────────────────────────────────────────
YEARS_MAIN = list(range(2018, 2025))   # 2018–2024: chung 9 môn, dùng để huấn luyện + đánh giá
YEAR_HOLDOUT = 2025                    # 2025: tách riêng, chỉ dùng để demo pipeline

# ──────────────────────────────────────────────
# Danh sách 9 môn chuẩn (mã cột trong CSV gốc)
# ──────────────────────────────────────────────
SUBJECTS = ["toan", "ngu_van", "ngoai_ngu", "vat_li", "hoa_hoc",
            "sinh_hoc", "lich_su", "dia_li", "gdcd"]

# ──────────────────────────────────────────────
# 3 tỉnh "mỏ neo" — gian lận đã xác nhận năm 2018
# (dùng làm nhãn dương thật sự trong PU Learning)
# ──────────────────────────────────────────────
ANCHOR_PROVINCES_2018 = {
    "05": "Hà Giang",
    "14": "Sơn La",
    "23": "Hòa Bình",
}

# ──────────────────────────────────────────────
# Độ rộng bin phổ điểm (thang 10 chia đều)
# ──────────────────────────────────────────────
BIN_WIDTH = 0.2
SCORE_MIN, SCORE_MAX = 0.0, 10.0

# ──────────────────────────────────────────────
# Năm có 2 đợt thi (cần tách đợt)
# ──────────────────────────────────────────────
DUAL_SESSION_YEARS = [2020, 2021]

# ──────────────────────────────────────────────
# Ngưỡng "đuôi điểm cao" dùng trong feature engineering
# ──────────────────────────────────────────────
HIGH_SCORE_THRESHOLDS = [8.0, 9.0, 10.0]

# ──────────────────────────────────────────────
# Top-k tỉnh nghi vấn cần xem khi đánh giá nhanh
# ──────────────────────────────────────────────
TOPK_ALERT = 10
