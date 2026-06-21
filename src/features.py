"""
features.py — Phần 2: Thiết kế đặc trưng
==========================================
Mục tiêu: mỗi đơn vị (tỉnh × môn × năm) → một vector đặc trưng số.
Tất cả đặc trưng đều là SO SÁNH TƯƠNG ĐỐI để khử ảnh hưởng độ khó đề.

Input:  data/processed/pho_diem_tat_ca.csv  (từ Phần 1)
Output: data/processed/dac_trung.csv

Ứng với Mục 7.1 trong đề cương.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PROCESSED,
    YEARS_MAIN, SUBJECTS,
    HIGH_SCORE_THRESHOLDS, BIN_WIDTH, SCORE_MIN, SCORE_MAX,
)
from data_prep import load_pho_diem


# ──────────────────────────────────────────────────────────────
# Hàm trợ giúp: lấy vector bin từ một dòng DataFrame
# ──────────────────────────────────────────────────────────────
def _get_bin_columns(df: pd.DataFrame) -> list[str]:
    """Trả về danh sách cột bin theo đúng thứ tự."""
    cols = sorted(
        [c for c in df.columns if c.startswith("bin_")],
        key=lambda x: float(x.replace("bin_", ""))
    )
    return cols


def _row_to_dist(row: pd.Series, bin_cols: list[str]) -> np.ndarray:
    """Lấy vector phân phối (đã chuẩn hóa) từ một dòng."""
    v = row[bin_cols].values.astype(float)
    s = v.sum()
    return v / s if s > 0 else v


# ──────────────────────────────────────────────────────────────
# Nhóm A: Đặc trưng đuôi điểm cao
# ──────────────────────────────────────────────────────────────
def feat_high_tail(row: pd.Series, bin_cols: list[str]) -> dict:
    """
    Tỷ lệ thí sinh đạt điểm cao ở các ngưỡng khác nhau.
    Đuôi cao phình bất thường là dấu hiệu cổ điển của can thiệp điểm.
    """
    dist = _row_to_dist(row, bin_cols)
    bins_left = [float(c.replace("bin_", "")) for c in bin_cols]
    features = {}

    for thr in HIGH_SCORE_THRESHOLDS:
        # Tổng xác suất ở những bin có cạnh trái >= thr
        tail = sum(p for b, p in zip(bins_left, dist) if b >= thr)
        features[f"tail_gte_{int(thr*10):02d}"] = tail   # vd: tail_gte_80, tail_gte_90, tail_gte_100

    return features


# ──────────────────────────────────────────────────────────────
# Nhóm B: Hình dạng phân phối (moments)
# ──────────────────────────────────────────────────────────────
def feat_distribution_shape(row: pd.Series, bin_cols: list[str]) -> dict:
    """
    Mean, median, std, skewness, kurtosis — tính từ phổ điểm.
    Phân phối bất thường thường có skewness âm mạnh (nghiêng về phải).
    """
    dist = _row_to_dist(row, bin_cols)
    bins_mid = np.array([float(c.replace("bin_", "")) + BIN_WIDTH / 2 for c in bin_cols])

    mean = np.sum(bins_mid * dist)
    variance = np.sum(((bins_mid - mean) ** 2) * dist)
    std = np.sqrt(variance) if variance > 0 else 0.0

    # Tránh chia 0 khi phân phối hoàn toàn phẳng
    if std > 0:
        skew = np.sum(((bins_mid - mean) ** 3) * dist) / (std ** 3)
        kurt = np.sum(((bins_mid - mean) ** 4) * dist) / (std ** 4) - 3
    else:
        skew, kurt = 0.0, 0.0

    # Median xấp xỉ từ CDF
    cdf = np.cumsum(dist)
    median_idx = np.searchsorted(cdf, 0.5)
    median = bins_mid[median_idx] if median_idx < len(bins_mid) else bins_mid[-1]

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "skewness": skew,
        "kurtosis": kurt,
    }


# ──────────────────────────────────────────────────────────────
# Nhóm C: Lệch so với mặt bằng toàn quốc
# ──────────────────────────────────────────────────────────────
def feat_divergence_from_national(
    row: pd.Series, national_dist: np.ndarray, bin_cols: list[str]
) -> dict:
    """
    Jensen–Shannon divergence và Wasserstein distance so với phân phối quốc gia
    (cùng môn, cùng năm).

    Dùng so sánh tương đối để loại bỏ ảnh hưởng độ khó đề.
    """
    local_dist = _row_to_dist(row, bin_cols)

    # Jensen–Shannon: 0 = giống hệt, 1 = khác tối đa
    js = jensenshannon(local_dist + 1e-10, national_dist + 1e-10)

    # Wasserstein: "khoảng cách vận chuyển" giữa hai phân phối
    bins_left = np.array([float(c.replace("bin_", "")) for c in bin_cols])
    ws = wasserstein_distance(bins_left, bins_left, local_dist, national_dist)

    return {
        "js_vs_national": float(js),
        "wasserstein_vs_national": float(ws),
    }


# ──────────────────────────────────────────────────────────────
# Nhóm D: Bất nhất chéo-môn
# ──────────────────────────────────────────────────────────────
def feat_cross_subject_inconsistency(
    tinh_ma: str, nam: int, dot: int,
    df_year: pd.DataFrame, bin_cols: list[str]
) -> dict:
    """
    Đuôi điểm cao của một môn so với trung bình các môn khác trong cùng tỉnh.

    Nếu tỉnh A đột nhiên có đuôi cao MÔN C vượt trội nhưng các môn khác bình thường,
    đó là tín hiệu bất nhất đáng ngờ.
    """
    rows_tinh = df_year[
        (df_year["tinh_ma"] == tinh_ma) &
        (df_year["nam"] == nam) &
        (df_year["dot"] == dot)
    ]

    tails_all = {}
    for _, r in rows_tinh.iterrows():
        d = _row_to_dist(r, bin_cols)
        bins_left = [float(c.replace("bin_", "")) for c in bin_cols]
        tail_90 = sum(p for b, p in zip(bins_left, d) if b >= 9.0)
        tails_all[r["mon"]] = tail_90

    if len(tails_all) < 2:
        return {"cross_subject_tail_std": 0.0, "cross_subject_max_z": 0.0}

    vals = list(tails_all.values())
    mean_tail = np.mean(vals)
    std_tail = np.std(vals)

    return {
        "cross_subject_tail_std": std_tail,
        # Z-score cao nhất: môn nào nổi trội nhất so với phần còn lại
        "cross_subject_max_z": (max(vals) - mean_tail) / std_tail if std_tail > 0 else 0.0,
    }


# ──────────────────────────────────────────────────────────────
# Nhóm E: Cú nhảy theo thời gian
# ──────────────────────────────────────────────────────────────
def feat_temporal_jump(
    tinh_ma: str, mon: str, nam: int,
    df_all: pd.DataFrame, bin_cols: list[str]
) -> dict:
    """
    Thứ hạng tương đối của tỉnh (về đuôi điểm ≥ 9) năm nay so với năm trước.

    Bước nhảy thứ hạng lớn (đặc biệt lên cao) trong 1 năm là cờ đỏ.
    """
    # Lọc tất cả tỉnh, cùng môn, hai năm liên tiếp
    prev_year = nam - 1
    for y in [nam, prev_year]:
        if y not in YEARS_MAIN:
            return {"rank_jump": 0.0, "rank_pct_now": 0.5, "rank_pct_prev": 0.5}

    def get_tail90_rank(year):
        sub = df_all[(df_all["mon"] == mon) & (df_all["nam"] == year) & (df_all["dot"] == 1)]
        if sub.empty:
            return None, None
        tails = {}
        for _, r in sub.iterrows():
            d = _row_to_dist(r, bin_cols)
            bins_left = [float(c.replace("bin_", "")) for c in bin_cols]
            tails[r["tinh_ma"]] = sum(p for b, p in zip(bins_left, d) if b >= 9.0)
        sorted_tails = sorted(tails.items(), key=lambda x: x[1], reverse=True)
        ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_tails)}
        n = len(ranks)
        return ranks.get(tinh_ma, n), n

    rank_now, n_now = get_tail90_rank(nam)
    rank_prev, n_prev = get_tail90_rank(prev_year)

    if rank_now is None or rank_prev is None:
        return {"rank_jump": 0.0, "rank_pct_now": 0.5, "rank_pct_prev": 0.5}

    # Phần trăm xếp hạng (0=kém nhất, 1=tốt nhất)
    pct_now  = 1 - (rank_now  - 1) / max(n_now  - 1, 1)
    pct_prev = 1 - (rank_prev - 1) / max(n_prev - 1, 1)

    return {
        "rank_pct_now": pct_now,
        "rank_pct_prev": pct_prev,
        "rank_jump": pct_now - pct_prev,   # dương = leo hạng, âm = tụt hạng
    }


# ──────────────────────────────────────────────────────────────
# Nhóm F: Tín hiệu vi mô — độ "dồn điểm"
# ──────────────────────────────────────────────────────────────
def feat_score_bunching(row: pd.Series, bin_cols: list[str]) -> dict:
    """
    Đo mức độ "răng cưa" bất thường quanh các mốc làm tròn (5.0, 6.0, 7.0, 8.0, 9.0).

    Can thiệp điểm thường đẩy điểm vượt ngưỡng làm tròn → bin ngay dưới ngưỡng
    mỏng hơn bình thường, bin ngay trên dày hơn.

    Chỉ số: tỷ lệ bin_ngay_trên / bin_ngay_dưới cho mỗi mốc.
    """
    dist = _row_to_dist(row, bin_cols)
    bins_left = [float(c.replace("bin_", "")) for c in bin_cols]

    milestones = [5.0, 6.0, 7.0, 8.0, 9.0]
    feats = {}

    for ms in milestones:
        # Bin ngay dưới ngưỡng: [ms-0.2, ms)
        idx_below = next((i for i, b in enumerate(bins_left) if abs(b - (ms - BIN_WIDTH)) < 1e-9), None)
        # Bin ngay trên ngưỡng: [ms, ms+0.2)
        idx_above = next((i for i, b in enumerate(bins_left) if abs(b - ms) < 1e-9), None)

        if idx_below is not None and idx_above is not None:
            below = dist[idx_below]
            above = dist[idx_above]
            # Tỷ lệ: >1 có nghĩa bin trên đông hơn bin dưới (bất thường)
            ratio = above / below if below > 1e-6 else 0.0
        else:
            ratio = 1.0

        key_ms = int(ms * 10)  # vd: 50, 60, 70, 80, 90
        feats[f"bunching_ratio_{key_ms}"] = ratio

    return feats


# ──────────────────────────────────────────────────────────────
# Hàm tổng hợp: tính toàn bộ đặc trưng cho một dataset
# ──────────────────────────────────────────────────────────────
def compute_all_features(df_pho: pd.DataFrame) -> pd.DataFrame:
    """
    Tính vector đặc trưng cho mọi đơn vị (tỉnh, môn, năm, đợt) trong df_pho.

    Trả về DataFrame mỗi dòng là một đơn vị với đầy đủ đặc trưng.
    """
    bin_cols = _get_bin_columns(df_pho)

    # Tính phân phối toàn quốc (trung bình trọng số) theo (môn, năm, đợt)
    # Dùng làm tham chiếu cho Nhóm C
    national_dists = {}
    for (mon, nam, dot), grp in df_pho.groupby(["mon", "nam", "dot"]):
        # Trung bình phân phối (không trọng số — mỗi tỉnh tính bằng nhau)
        nd = grp[bin_cols].values.astype(float)
        row_sums = nd.sum(axis=1, keepdims=True)
        nd_norm = np.where(row_sums > 0, nd / row_sums, 0)
        national_dists[(mon, nam, dot)] = nd_norm.mean(axis=0)

    all_records = []

    for idx, row in df_pho.iterrows():
        tinh_ma = row["tinh_ma"]
        mon     = row["mon"]
        nam     = row["nam"]
        dot     = int(row["dot"])

        # Metadata
        record = {
            "tinh_ma":  tinh_ma,
            "tinh_ten": row["tinh_ten"],
            "mon":      mon,
            "nam":      nam,
            "dot":      dot,
            "n_thi":    row["n_thi"],
            "la_mo_neo_2018": row.get("la_mo_neo_2018", False),
        }

        # Nhóm A: đuôi điểm cao
        record.update(feat_high_tail(row, bin_cols))

        # Nhóm B: hình dạng
        record.update(feat_distribution_shape(row, bin_cols))

        # Nhóm C: lệch so với quốc gia
        nd = national_dists.get((mon, nam, dot), np.ones(len(bin_cols)) / len(bin_cols))
        record.update(feat_divergence_from_national(row, nd, bin_cols))

        # Nhóm D: bất nhất chéo-môn
        df_year = df_pho[(df_pho["nam"] == nam) & (df_pho["dot"] == dot)]
        record.update(feat_cross_subject_inconsistency(tinh_ma, nam, dot, df_year, bin_cols))

        # Nhóm E: nhảy thứ hạng theo thời gian
        record.update(feat_temporal_jump(tinh_ma, mon, nam, df_pho, bin_cols))

        # Nhóm F: dồn điểm quanh ngưỡng
        record.update(feat_score_bunching(row, bin_cols))

        all_records.append(record)

    return pd.DataFrame(all_records)


# ──────────────────────────────────────────────────────────────
# Chuẩn hóa đặc trưng
# ──────────────────────────────────────────────────────────────
def normalize_features(df_feat: pd.DataFrame) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    """
    Chuẩn hóa các cột đặc trưng số về mean=0, std=1.
    Trả về (DataFrame đã chuẩn hóa, danh sách cột đặc trưng, scaler).

    Scaler lưu lại để dùng khi nhúng dữ liệu mới (pipeline.py).
    """
    meta_cols = ["tinh_ma", "tinh_ten", "mon", "nam", "dot", "n_thi", "la_mo_neo_2018"]
    feat_cols = [c for c in df_feat.columns if c not in meta_cols]

    scaler = StandardScaler()
    df_out = df_feat.copy()
    df_out[feat_cols] = scaler.fit_transform(df_feat[feat_cols].fillna(0))

    return df_out, feat_cols, scaler


# ──────────────────────────────────────────────────────────────
# Hàm chính
# ──────────────────────────────────────────────────────────────
def run():
    print("=== Phần 2: Thiết kế đặc trưng ===")

    df_pho = load_pho_diem()
    if df_pho.empty:
        print("[LỖI] Chưa có dữ liệu phổ điểm. Chạy data_prep.run() trước.")
        return

    print(f"Dữ liệu phổ điểm: {len(df_pho)} đơn vị")

    df_feat = compute_all_features(df_pho)
    print(f"Đặc trưng thô: {df_feat.shape[1]} cột")

    df_norm, feat_cols, _ = normalize_features(df_feat)
    print(f"Số đặc trưng sau chuẩn hóa: {len(feat_cols)}")

    out_path = DATA_PROCESSED / "dac_trung.csv"
    df_norm.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {out_path}")

    print("\nPhần 2 hoàn thành.")
    return df_norm


if __name__ == "__main__":
    run()
