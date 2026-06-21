"""
data_prep.py — Phần 1: Chuẩn bị dữ liệu
=========================================
Mục tiêu: CSV cấp thí sinh → bảng phổ điểm (tỉnh × môn × năm).

Input:  data/raw/du_lieu_diem_thi_YYYY.csv   (2018–2025)
Output:
  - data/processed/pho_diem_{nam}.csv        — phổ điểm theo bin 0.2 cho mỗi (tỉnh, môn)
  - data/processed/diem_tho_gop.parquet      — bảng điểm thô đã gộp tất cả năm

Ứng với Mục 6 trong đề cương.
"""

import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    DATA_RAW, DATA_PROCESSED,
    YEARS_MAIN, YEAR_HOLDOUT,
    SUBJECTS, ANCHOR_PROVINCES_2018,
    BIN_WIDTH, SCORE_MIN, SCORE_MAX,
    DUAL_SESSION_YEARS,
)

# ──────────────────────────────────────────────────────────────
# 1. Bảng mã tỉnh (01–64 + một số mã đặc biệt)
# ──────────────────────────────────────────────────────────────
# Nguồn: danh mục đơn vị hành chính Việt Nam
PROVINCE_MAP = {
    "01": "Hà Nội",        "02": "Hà Giang",      "04": "Cao Bằng",
    "06": "Bắc Kạn",       "08": "Tuyên Quang",   "10": "Lào Cai",
    "11": "Điện Biên",     "12": "Lai Châu",       "14": "Sơn La",
    "15": "Yên Bái",       "17": "Hòa Bình",       "19": "Thái Nguyên",
    "20": "Lạng Sơn",      "22": "Quảng Ninh",     "24": "Bắc Giang",
    "25": "Phú Thọ",       "26": "Vĩnh Phúc",      "27": "Bắc Ninh",
    "30": "Hải Dương",     "31": "Hải Phòng",      "33": "Hưng Yên",
    "34": "Thái Bình",     "35": "Hà Nam",         "36": "Nam Định",
    "37": "Ninh Bình",     "38": "Thanh Hóa",      "40": "Nghệ An",
    "42": "Hà Tĩnh",       "44": "Quảng Bình",     "45": "Quảng Trị",
    "46": "Thừa Thiên Huế","48": "Đà Nẵng",        "49": "Quảng Nam",
    "51": "Quảng Ngãi",    "52": "Bình Định",      "54": "Phú Yên",
    "56": "Khánh Hòa",     "58": "Ninh Thuận",     "60": "Bình Thuận",
    "62": "Kon Tum",       "64": "Gia Lai",         "66": "Đắk Lắk",
    "67": "Đắk Nông",      "68": "Lâm Đồng",       "70": "Bình Phước",
    "72": "Tây Ninh",      "74": "Bình Dương",      "75": "Đồng Nai",
    "77": "Bà Rịa - Vũng Tàu", "79": "TP. Hồ Chí Minh",
    "80": "Long An",       "82": "Tiền Giang",      "83": "Bến Tre",
    "84": "Trà Vinh",      "86": "Vĩnh Long",       "87": "Đồng Tháp",
    "89": "An Giang",      "91": "Kiên Giang",       "92": "Cần Thơ",
    "93": "Hậu Giang",     "94": "Sóc Trăng",       "95": "Bạc Liêu",
    "96": "Cà Mau",
    # Một số mã cũ hoặc dự phòng
    "05": "Hà Giang",      "23": "Hòa Bình",
}


# ──────────────────────────────────────────────────────────────
# 2. Hàm đọc 1 file CSV thô
# ──────────────────────────────────────────────────────────────
def load_raw_csv(year: int, session: int = 1) -> pd.DataFrame:
    """
    Đọc file CSV điểm thi cho một năm (và đợt nếu có).

    Giả định cột trong CSV:
      - sbd       : số báo danh (có mã tỉnh ở 2 ký tự đầu)
      - toan, ngu_van, ... : điểm từng môn (float, NaN nếu không thi)
      - dot       : đợt (1 hoặc 2) — chỉ có với 2020, 2021

    Trả về DataFrame gồm: [sbd, tinh_ma, dot, toan, ngu_van, ...]
    """
    suffix = f"_dot{session}" if year in DUAL_SESSION_YEARS and session == 2 else ""
    fname = DATA_RAW / f"du_lieu_diem_thi_{year}{suffix}.csv"

    if not fname.exists():
        print(f"  [WARN] Không tìm thấy: {fname}")
        return pd.DataFrame()

    df = pd.read_csv(fname, dtype={"sbd": str}, low_memory=False)

    # Chuẩn hóa tên cột về chữ thường, bỏ khoảng trắng
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Trích mã tỉnh từ 2 ký tự đầu của SBD
    df["tinh_ma"] = df["sbd"].str[:2]

    # Map sang tên tỉnh (giữ mã gốc nếu không có trong bảng)
    df["tinh_ten"] = df["tinh_ma"].map(PROVINCE_MAP).fillna(df["tinh_ma"])

    # Ghi nhận năm và đợt
    df["nam"] = year
    df["dot"] = session if year in DUAL_SESSION_YEARS else 1

    return df


# ──────────────────────────────────────────────────────────────
# 3. Làm sạch điểm
# ──────────────────────────────────────────────────────────────
def clean_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch cột điểm:
      - NaN (= không đăng ký thi môn này) → bỏ dòng khi tính phổ điểm môn đó,
        KHÔNG điền 0 vì điều đó sẽ làm méo phân phối.
      - Điểm 0 hợp lệ (thi nhưng bị 0) → GIỮ NGUYÊN; ghi chú lại tỷ lệ.
      - Điểm ngoài [0, 10] → coi là lỗi nhập liệu, đặt thành NaN.

    Quyết định thiết kế: không thi ≠ điểm 0.
    """
    for subj in SUBJECTS:
        if subj not in df.columns:
            continue
        col = pd.to_numeric(df[subj], errors="coerce")

        # Đánh dấu điểm bất hợp lệ (ngoài thang 0–10)
        invalid_mask = col.notna() & ((col < SCORE_MIN) | (col > SCORE_MAX))
        if invalid_mask.sum() > 0:
            print(f"  [WARN] {subj}: {invalid_mask.sum()} điểm ngoài [0,10] → NaN")
        col[invalid_mask] = np.nan

        df[subj] = col

    return df


# ──────────────────────────────────────────────────────────────
# 4. Xây dựng phổ điểm theo bin
# ──────────────────────────────────────────────────────────────
def build_score_distribution(scores: pd.Series, bin_width: float = BIN_WIDTH) -> pd.Series:
    """
    Từ mảng điểm (1 tỉnh, 1 môn, 1 năm) → vector phân phối chuẩn hóa.

    Bins: [0.0–0.2), [0.2–0.4), ..., [9.8–10.0], tổng cộng 50 bin.
    Trả về Series có index là cạnh trái mỗi bin, giá trị là tỷ lệ (sum=1).
    """
    bins = np.arange(SCORE_MIN, SCORE_MAX + bin_width, bin_width)
    counts, edges = np.histogram(scores.dropna(), bins=bins)
    dist = pd.Series(counts / max(counts.sum(), 1), index=np.round(edges[:-1], 2))
    return dist


# ──────────────────────────────────────────────────────────────
# 5. Tính phổ điểm cho cả năm
# ──────────────────────────────────────────────────────────────
def compute_pho_diem(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Từ DataFrame điểm thô (1 năm) → bảng phổ điểm phẳng.

    Mỗi dòng: (tinh_ma, tinh_ten, mon, nam, dot, n_thi, bin_0.0, bin_0.2, ..., bin_9.8)
    """
    records = []

    for (tinh_ma, dot), grp_tinh in df.groupby(["tinh_ma", "dot"]):
        tinh_ten = grp_tinh["tinh_ten"].iloc[0]

        for subj in SUBJECTS:
            if subj not in grp_tinh.columns:
                continue

            scores = grp_tinh[subj].dropna()
            n_thi = len(scores)

            if n_thi < 30:
                # Quá ít thí sinh → phổ điểm không đại diện, bỏ qua
                continue

            dist = build_score_distribution(scores)
            record = {
                "tinh_ma": tinh_ma,
                "tinh_ten": tinh_ten,
                "mon": subj,
                "nam": year,
                "dot": int(dot),
                "n_thi": n_thi,
                # Có phải tỉnh mỏ neo 2018 không
                "la_mo_neo_2018": (tinh_ma in ANCHOR_PROVINCES_2018) and (year == 2018),
            }
            record.update({f"bin_{b:.1f}": v for b, v in dist.items()})
            records.append(record)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# 6. Hàm chính — chạy toàn bộ pipeline Phần 1
# ──────────────────────────────────────────────────────────────
def run(years: list[int] | None = None):
    """
    Đọc, làm sạch và lưu phổ điểm + điểm thô gộp.

    Args:
        years: danh sách năm cần xử lý; mặc định là YEARS_MAIN + YEAR_HOLDOUT.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if years is None:
        years = YEARS_MAIN + [YEAR_HOLDOUT]

    all_raw_frames = []   # tích lũy điểm thô để lưu parquet
    all_pho_frames = []   # tích lũy phổ điểm

    for year in tqdm(years, desc="Xử lý từng năm"):
        print(f"\n=== Năm {year} ===")

        # Xác định số đợt cần đọc
        sessions = [1, 2] if year in DUAL_SESSION_YEARS else [1]

        year_frames = []
        for session in sessions:
            print(f"  Đợt {session}...")
            df = load_raw_csv(year, session)
            if df.empty:
                continue
            df = clean_scores(df)
            year_frames.append(df)

        if not year_frames:
            print(f"  [SKIP] Không có dữ liệu năm {year}")
            continue

        df_year = pd.concat(year_frames, ignore_index=True)
        all_raw_frames.append(df_year)

        # Tính phổ điểm và lưu file riêng theo năm
        pho = compute_pho_diem(df_year, year)
        out_path = DATA_PROCESSED / f"pho_diem_{year}.csv"
        pho.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Đã lưu phổ điểm: {out_path}  ({len(pho)} dòng)")

        all_pho_frames.append(pho)

    # Lưu điểm thô gộp (dùng parquet để tiết kiệm dung lượng)
    if all_raw_frames:
        df_gop = pd.concat(all_raw_frames, ignore_index=True)
        parquet_path = DATA_PROCESSED / "diem_tho_gop.parquet"
        df_gop.to_parquet(parquet_path, index=False)
        print(f"\nĐã lưu điểm thô gộp: {parquet_path}  ({len(df_gop):,} dòng)")

    # Lưu phổ điểm tổng hợp tất cả năm
    if all_pho_frames:
        df_pho_all = pd.concat(all_pho_frames, ignore_index=True)
        all_path = DATA_PROCESSED / "pho_diem_tat_ca.csv"
        df_pho_all.to_csv(all_path, index=False, encoding="utf-8-sig")
        print(f"Đã lưu phổ điểm tổng hợp: {all_path}  ({len(df_pho_all)} dòng)")

    print("\nPhần 1 hoàn thành.")


# ──────────────────────────────────────────────────────────────
# 7. Hàm tiện ích: đọc lại phổ điểm đã xử lý
# ──────────────────────────────────────────────────────────────
def load_pho_diem(years: list[int] | None = None) -> pd.DataFrame:
    """
    Đọc file phổ điểm đã lưu từ data/processed/.
    Tiện dùng cho các module sau (features.py, detectors.py, ...).
    """
    if years is None:
        path = DATA_PROCESSED / "pho_diem_tat_ca.csv"
        if path.exists():
            return pd.read_csv(path)
        years = YEARS_MAIN

    frames = []
    for y in years:
        p = DATA_PROCESSED / f"pho_diem_{y}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            print(f"[WARN] Chưa có phổ điểm năm {y}. Chạy data_prep.run() trước.")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    run()
