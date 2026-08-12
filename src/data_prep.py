# -*- coding: utf-8 -*-
"""
data_prep.py — Phần 1: Chuẩn bị dữ liệu  (BẢN ĐỌC FILE GỐC GỘP)
================================================================
Đọc thẳng file điểm thi cấp thí sinh đã gộp (≈1 triệu dòng/năm) và nén
về bảng phổ điểm theo (tỉnh × môn × năm).

Input:  data/raw/du_lieu_diem_thi_2016-2025.csv   (1 file gộp, cột gốc tiếng Việt)
Output: data/processed/pho_diem_tat_ca.csv         (phổ điểm bin 0.2 cho mỗi tỉnh×môn×năm)
        data/processed/pho_diem_{nam}.csv          (tách theo năm)

Ứng với Mục 6 trong đề cương.
"""

import pathlib
import numpy as np
import pandas as pd
from collections import defaultdict

from config import (
    DATA_RAW, DATA_PROCESSED, YEARS_MAIN, SUBJECTS,
    ANCHOR_PROVINCES_2018, BIN_WIDTH, SCORE_MIN, SCORE_MAX,
)

# ──────────────────────────────────────────────────────────────
# 1. Ánh xạ tên cột GỐC trong file (tiếng Việt viết liền) -> mã môn dùng nội bộ
#    Đây là chỗ trước đây bị lệch khiến không tìm thấy môn.
# ──────────────────────────────────────────────────────────────
COLUMN_MAP = {
    "Toan": "toan", "NguVan": "ngu_van", "NgoaiNgu": "ngoai_ngu",
    "VatLy": "vat_li", "HoaHoc": "hoa_hoc", "SinhHoc": "sinh_hoc",
    "LichSu": "lich_su", "DiaLy": "dia_li", "GDCD": "gdcd",
}

# ──────────────────────────────────────────────────────────────
# 2. Bảng mã tỉnh THI (1–64) -> tên. Đây là hệ mã đúng của dữ liệu điểm thi
#    (KHÁC hệ mã hành chính 01–96). Mã nằm ở cột "Tinh".
# ──────────────────────────────────────────────────────────────
_TINH = {
    1:"Hà Nội",2:"TP HCM",3:"Hải Phòng",4:"Đà Nẵng",5:"Hà Giang",6:"Cao Bằng",
    7:"Lai Châu",8:"Lào Cai",9:"Tuyên Quang",10:"Lạng Sơn",11:"Bắc Kạn",12:"Thái Nguyên",
    13:"Yên Bái",14:"Sơn La",15:"Phú Thọ",16:"Vĩnh Phúc",17:"Quảng Ninh",18:"Bắc Giang",
    19:"Bắc Ninh",21:"Hải Dương",22:"Hưng Yên",23:"Hòa Bình",24:"Hà Nam",25:"Nam Định",
    26:"Thái Bình",27:"Ninh Bình",28:"Thanh Hóa",29:"Nghệ An",30:"Hà Tĩnh",31:"Quảng Bình",
    32:"Quảng Trị",33:"TT Huế",34:"Quảng Nam",35:"Quảng Ngãi",36:"Kon Tum",37:"Bình Định",
    38:"Gia Lai",39:"Phú Yên",40:"Đắk Lắk",41:"Khánh Hòa",42:"Lâm Đồng",43:"Bình Phước",
    44:"Bình Dương",45:"Ninh Thuận",46:"Tây Ninh",47:"Bình Thuận",48:"Đồng Nai",49:"Long An",
    50:"Đồng Tháp",51:"An Giang",52:"BR-Vũng Tàu",53:"Tiền Giang",54:"Kiên Giang",55:"Cần Thơ",
    56:"Bến Tre",57:"Vĩnh Long",58:"Trà Vinh",59:"Sóc Trăng",60:"Bạc Liêu",61:"Cà Mau",
    62:"Điện Biên",63:"Đắk Nông",64:"Hậu Giang",
}
PROVINCE_MAP = {f"{k:02d}": v for k, v in _TINH.items()}

# Các mốc bin: [0.0,0.2), ... ,[9.8,10.0]  -> 50 bin
_EDGES = np.arange(SCORE_MIN, SCORE_MAX + 1e-9, BIN_WIDTH)
_NB = len(_EDGES) - 1
_BIN_LABELS = [f"bin_{e:.1f}" for e in _EDGES[:-1]]


def _find_combined_file() -> pathlib.Path | None:
    """Tìm file gộp trong data/raw (chấp nhận vài tên thường gặp)."""
    for name in ["du_lieu_diem_thi_2016-2025.csv", "du_lieu_diem_thi.csv"]:
        p = DATA_RAW / name
        if p.exists():
            return p
    hits = list(DATA_RAW.glob("du_lieu_diem_thi*2025*.csv")) + list(DATA_RAW.glob("*.csv"))
    return hits[0] if hits else None


def build_pho_diem(years=None, chunksize=1_000_000) -> pd.DataFrame:
    """
    Đọc file gộp theo từng khối (tiết kiệm RAM), nén về phổ điểm
    theo (tỉnh × môn × năm). Trả về DataFrame đã sẵn cho Phần 2.
    """
    if years is None:
        years = YEARS_MAIN
    year_codes = {f"{y % 100:02d}" for y in years}   # 2018 -> "18"

    path = _find_combined_file()
    if path is None:
        raise FileNotFoundError(f"Không thấy file gốc trong {DATA_RAW}. "
                                f"Hãy đặt du_lieu_diem_thi_2016-2025.csv vào đó.")
    print(f"Đọc file gốc: {path}")

    usecols = ["Nam", "Tinh"] + list(COLUMN_MAP)
    hist = defaultdict(lambda: np.zeros(_NB))   # (nam, tinh, mon) -> đếm theo bin

    for ch in pd.read_csv(path, usecols=usecols, dtype={"Nam": str, "Tinh": str},
                          chunksize=chunksize, low_memory=False):
        ch["Nam"] = ch["Nam"].str.strip()
        ch = ch[ch["Nam"].isin(year_codes)]          # chỉ giữ năm chính (đợt 1)
        if ch.empty:
            continue
        t = pd.to_numeric(ch["Tinh"], errors="coerce")
        keep = t.notna()
        ch, t = ch[keep], t[keep].astype(int)
        for raw_col, mon in COLUMN_MAP.items():
            s = pd.to_numeric(ch[raw_col], errors="coerce")
            m = s.notna() & (s >= SCORE_MIN) & (s <= SCORE_MAX)   # bỏ thiếu & điểm lỗi
            if not m.any():
                continue
            idx = np.clip((s[m].values / BIN_WIDTH).astype(int), 0, _NB - 1)
            g = pd.DataFrame({"nam": ch["Nam"].values[m], "t": t.values[m], "i": idx})
            for (nam, tinh, i), cnt in g.groupby(["nam", "t", "i"]).size().items():
                hist[(nam, tinh, mon)][i] += cnt

    # Dựng DataFrame phổ điểm
    rows = []
    for (nam, tinh, mon), h in hist.items():
        n = int(h.sum())
        if n < 30:                                   # bỏ nhóm quá ít thí sinh
            continue
        p = h / n
        ma = f"{tinh:02d}"
        nam_full = 2000 + int(nam)
        rec = {
            "tinh_ma": ma,
            "tinh_ten": PROVINCE_MAP.get(ma, f"T{ma}"),
            "mon": mon,
            "nam": nam_full,
            "dot": 1,
            "n_thi": n,
            "la_mo_neo_2018": (ma in ANCHOR_PROVINCES_2018) and (nam_full == 2018),
        }
        rec.update({lbl: round(float(v), 6) for lbl, v in zip(_BIN_LABELS, p)})
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values(["nam", "tinh_ma", "mon"]).reset_index(drop=True)
    return df


def run(years=None):
    """Chạy Phần 1: từ file gốc -> phổ điểm, lưu ra data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_pho_diem(years)

    # Lưu tổng hợp + tách theo năm
    all_path = DATA_PROCESSED / "pho_diem_tat_ca.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    for nam, g in df.groupby("nam"):
        g.to_csv(DATA_PROCESSED / f"pho_diem_{nam}.csv", index=False, encoding="utf-8-sig")

    print(f"Đã lưu phổ điểm: {all_path}  ({len(df)} dòng, "
          f"{df['nam'].nunique()} năm, {df['tinh_ma'].nunique()} tỉnh)")
    print("Phần 1 hoàn thành.")
    return df


def load_pho_diem(years=None) -> pd.DataFrame:
    """Đọc lại phổ điểm đã lưu. LUÔN giữ tinh_ma là chuỗi (tránh mất số 0 đầu)."""
    path = DATA_PROCESSED / "pho_diem_tat_ca.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"tinh_ma": str})
    frames = []
    for y in (years or YEARS_MAIN):
        p = DATA_PROCESSED / f"pho_diem_{y}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, dtype={"tinh_ma": str}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    run()