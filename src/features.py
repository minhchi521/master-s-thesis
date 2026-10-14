"""
features.py — Phần 2: Thiết kế đặc trưng (BẢN ĐÃ SỬA RÒ RỈ, vector hóa)
=========================================================================
Mỗi đơn vị (tỉnh × môn × năm) → một vector đặc trưng số. Mọi đặc trưng đều là
SO SÁNH TƯƠNG ĐỐI để khử ảnh hưởng độ khó đề. Ứng với Mục 7.1 trong đề cương.

Các thay đổi so với bản đầu (xem AUDIT_ROI_RI.md):
  1. Phân phối tham chiếu quốc gia = TRUNG VỊ bỏ-chính-tỉnh-đó (leave-one-out
     median) thay vì trung bình toàn bộ 63 tỉnh. Trước đây tham chiếu có chứa
     chính đơn vị đang chấm điểm và cả 3 tỉnh gian lận → làm "mềm" tín hiệu.
     Cách mới không dùng nhãn nào (label-free).
  2. Nhóm E (nhảy thứ hạng theo thời gian) KHÔNG xác định được cho 2018 (không
     có 2017). Bản cũ điền hằng số giả (0.0 / 0.5 / 0.5) cho TOÀN BỘ 2018 → mọi
     ca dương đều nằm ở năm 2018 nên mô hình học được "năm 2018" như một lối tắt.
     Bản mới để NaN cho 2018, và nhóm E bị loại khỏi bộ đặc trưng huấn luyện PU
     (PU_FEATURE_COLS).
  3. File đầu ra vẫn được chuẩn hóa toàn cục (label-free, chỉ dùng cho baseline
     không giám sát). Mọi mô hình có dùng nhãn tự chuẩn hóa lại BÊN TRONG từng
     fold, chỉ trên dữ liệu huấn luyện (xem pu_models.py).
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PROCESSED, YEARS_MAIN, HIGH_SCORE_THRESHOLDS, BIN_WIDTH,
)
from data_prep import load_pho_diem

META_COLS = ["tinh_ma", "tinh_ten", "mon", "nam", "dot", "n_thi", "la_mo_neo_2018"]

FEATURE_GROUPS = {
    # "tail_gte_98" = tỷ lệ rơi vào bin cuối [9.8, 10] (điểm gần/đạt tối đa). Bản đầu dùng ngưỡng 10.0 nên
    # cột "tail_gte_100" LUÔN bằng 0 (không bin nào có cạnh trái ≥ 10.0) — đặc trưng "tỷ lệ điểm tối đa" bị hỏng.
    "tail_high":     ["tail_gte_80", "tail_gte_90", "tail_gte_98"],
    "shape":         ["mean", "median", "std", "skewness", "kurtosis"],
    "divergence":    ["js_vs_national", "wasserstein_vs_national"],
    "cross_subject": ["cross_subject_tail_std", "cross_subject_max_z"],
    "temporal":      ["rank_pct_now", "rank_pct_prev", "rank_jump"],
    "bunching":      ["bunching_ratio_50", "bunching_ratio_60", "bunching_ratio_70",
                      "bunching_ratio_80", "bunching_ratio_90"],
}
# Bộ đặc trưng dùng để HUẤN LUYỆN mô hình có dùng nhãn 2018 (bỏ nhóm temporal)
PU_FEATURE_COLS = [c for g, cols in FEATURE_GROUPS.items() if g != "temporal" for c in cols]
ALL_FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]
# Danh sách cột của bản gốc (chỉ dùng để tái hiện lỗi trong audit_leakage.py)
LEGACY_FEATURE_COLS = [("tail_gte_100" if c == "tail_gte_98" else c) for c in ALL_FEATURE_COLS]
RELATIVE_SKIP = set(FEATURE_GROUPS["divergence"] + FEATURE_GROUPS["temporal"])   # đã là so sánh tương đối


def _bin_columns(df):
    return sorted([c for c in df.columns if c.startswith("bin_")],
                  key=lambda x: float(x.replace("bin_", "")))


def _js(p, q):
    """Jensen–Shannon distance (log tự nhiên) theo từng dòng, giống scipy.jensenshannon."""
    p = p + 1e-10; q = q + 1e-10
    p = p / p.sum(axis=1, keepdims=True); q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b), axis=1)
    return np.sqrt(np.maximum(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def _wasserstein(p, q, centers):
    """Wasserstein-1 giữa hai phân phối rời rạc trên cùng lưới điểm."""
    dx = np.diff(centers)
    cp = np.cumsum(p, axis=1)[:, :-1]; cq = np.cumsum(q, axis=1)[:, :-1]
    return np.sum(np.abs(cp - cq) * dx, axis=1)


def compute_reference(df_pho, reference="loo_median"):
    """Trả về mảng tham chiếu quốc gia (n × số bin), cùng thứ tự dòng với df_pho."""
    df = df_pho.reset_index(drop=True)
    bins = _bin_columns(df)
    raw = df[bins].values.astype(float); s = raw.sum(axis=1, keepdims=True)
    D = np.where(s > 0, raw / np.where(s > 0, s, 1), 0.0)
    ref = np.zeros_like(D)
    for _, idx in df.groupby(["mon", "nam", "dot"]).indices.items():
        sub = D[idx]
        for k, i in enumerate(idx):
            if reference == "mean_all":
                ref[i] = sub.mean(axis=0)
            else:
                m = np.median(np.delete(sub, k, axis=0), axis=0)
                ref[i] = m / m.sum() if m.sum() > 0 else m
    return ref


def compute_all_features(df_pho, reference="loo_median", temporal="nan", ref_override=None):
    """
    reference: "loo_median" (bản sửa) | "mean_all" (bản gốc, chỉ để tái hiện lỗi)
    ref_override: mảng (n × số bin) tham chiếu tính sẵn; nếu có thì bỏ qua `reference`
    temporal : "nan" (bản sửa)       | "const2018" (bản gốc, chỉ để tái hiện lỗi)
    """
    df = df_pho.reset_index(drop=True).copy()
    bins = _bin_columns(df)
    left = np.array([float(c.replace("bin_", "")) for c in bins])
    mid = left + BIN_WIDTH / 2
    raw = df[bins].values.astype(float)
    s = raw.sum(axis=1, keepdims=True)
    D = np.where(s > 0, raw / np.where(s > 0, s, 1), 0.0)
    n = len(df)
    out = {c: df[c].values for c in META_COLS if c in df.columns}
    if "la_mo_neo_2018" not in out:
        out["la_mo_neo_2018"] = np.zeros(n, dtype=bool)

    # Nhóm A — đuôi điểm cao
    thr_list = [8.0, 9.0, 10.0] if reference == "mean_all" else [8.0, 9.0, 9.8]
    for thr in thr_list:
        out[f"tail_gte_{int(round(thr * 10)):02d}"] = D[:, left >= thr - 1e-9].sum(axis=1)

    # Nhóm B — hình dạng
    mean = D @ mid
    var = D @ (mid ** 2) - mean ** 2
    var = np.maximum(((D * (mid[None, :] - mean[:, None]) ** 2).sum(axis=1)), 0)
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(std > 0, (D * (mid[None, :] - mean[:, None]) ** 3).sum(axis=1) / std ** 3, 0.0)
        kurt = np.where(std > 0, (D * (mid[None, :] - mean[:, None]) ** 4).sum(axis=1) / std ** 4 - 3, 0.0)
    cdf = np.cumsum(D, axis=1)
    med_idx = np.array([np.searchsorted(cdf[i], 0.5) for i in range(n)])
    med = np.where(med_idx < len(mid), mid[np.minimum(med_idx, len(mid) - 1)], mid[-1])
    out.update(mean=mean, median=med, std=std, skewness=skew, kurtosis=kurt)

    # Nhóm C — lệch so với tham chiếu quốc gia (cùng môn, năm, đợt)
    ref = np.zeros_like(D)
    groups = {} if ref_override is not None else df.groupby(["mon", "nam", "dot"]).indices
    if ref_override is not None:        # dùng tham chiếu đã tính sẵn (để chấm ca tiêm nhiễu mà không làm lệch tham chiếu)
        ref = np.asarray(ref_override, dtype=float)
    for _, idx in groups.items():
        sub = D[idx]
        if reference == "mean_all":
            ref[idx] = sub.mean(axis=0)
        elif reference == "loo_median":
            for k, i in enumerate(idx):
                others = np.delete(sub, k, axis=0)
                m = np.median(others, axis=0)
                ref[i] = m / m.sum() if m.sum() > 0 else m
        else:
            raise ValueError(reference)
    out["js_vs_national"] = _js(D, ref)
    out["wasserstein_vs_national"] = _wasserstein(D, ref, left)

    # Nhóm D — bất nhất chéo-môn (đuôi ≥ 9.0 của môn so với các môn khác cùng tỉnh)
    tail90 = D[:, left >= 9.0 - 1e-9].sum(axis=1)
    df["_t90"] = tail90
    g = df.groupby(["tinh_ma", "nam", "dot"])["_t90"]
    gstd = g.transform(lambda x: np.std(x.values)) if True else None
    gmean = g.transform("mean"); gmax = g.transform("max"); gcnt = g.transform("count")
    out["cross_subject_tail_std"] = np.where(gcnt >= 2, gstd.values, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        zmax = np.where((gcnt >= 2) & (gstd > 0), (gmax - gmean) / gstd, 0.0)
    out["cross_subject_max_z"] = np.asarray(zmax)

    # Nhóm E — thứ hạng đuôi ≥ 9 của tỉnh trong năm nay so với năm trước (cùng môn)
    pct = np.full(n, np.nan)
    key = df.assign(_t=tail90)
    for (mon, nam, dot), idx in key.groupby(["mon", "nam", "dot"]).indices.items():
        t = tail90[idx]
        r = rankdata(-t, method="ordinal")
        pct[idx] = 1 - (r - 1) / max(len(idx) - 1, 1)
    lookup = {(df.at[i, "mon"], df.at[i, "tinh_ma"], df.at[i, "nam"]): pct[i] for i in range(n)}
    now = np.array([lookup.get((df.at[i, "mon"], df.at[i, "tinh_ma"], df.at[i, "nam"]), np.nan) for i in range(n)])
    prev = np.array([lookup.get((df.at[i, "mon"], df.at[i, "tinh_ma"], df.at[i, "nam"] - 1), np.nan) for i in range(n)])
    valid = np.array([(y in YEARS_MAIN) and ((y - 1) in YEARS_MAIN) for y in df["nam"]])
    now = np.where(valid, now, np.nan); prev = np.where(valid, prev, np.nan)
    jump = now - prev
    if temporal == "const2018":     # tái hiện lỗi gốc: điền hằng số giả
        now = np.where(valid, now, 0.5); prev = np.where(valid, prev, 0.5); jump = np.where(valid, jump, 0.0)
    out.update(rank_pct_now=now, rank_pct_prev=prev, rank_jump=jump)

    # Nhóm F — dồn điểm quanh các mốc 5,6,7,8,9
    for ms in [5.0, 6.0, 7.0, 8.0, 9.0]:
        ib = np.where(np.isclose(left, ms - BIN_WIDTH))[0]; ia = np.where(np.isclose(left, ms))[0]
        if len(ib) and len(ia):
            below = D[:, ib[0]]; above = D[:, ia[0]]
            ratio = np.where(below > 1e-6, above / np.where(below > 1e-6, below, 1), 0.0)
        else:
            ratio = np.ones(n)
        out[f"bunching_ratio_{int(ms * 10)}"] = ratio

    cols = META_COLS + (LEGACY_FEATURE_COLS if reference == "mean_all" else ALL_FEATURE_COLS)
    return pd.DataFrame({c: out[c] for c in cols})


# ──────────────────────────────────────────────────────────────
# So sánh tương đối trong từng (môn, năm, đợt)  — khử "độ khó của đề"
# ──────────────────────────────────────────────────────────────
def fit_relative_stats(df_raw, cols=None):
    """Trung vị và MAD (đã nhân 1.4826) của từng đặc trưng trong từng nhóm (môn, năm, đợt).
    Đặc trưng tuyệt đối (đuôi điểm cao, mean/median/std/skew/kurt, chéo-môn, dồn điểm) đều mang dấu ấn của
    NĂM THI: chỉ riêng nhóm chéo-môn đã đoán được "năm 2018" với AUC ≈ 1.0. Vì mọi ca dương đều ở 2018, mô hình
    sẽ học "năm 2018" thay vì học "bất thường". Chuẩn hóa robust trong từng (môn, năm) biến mọi đặc trưng thành
    "tỉnh này lệch bao nhiêu độ lệch chuẩn so với các tỉnh khác CÙNG MÔN CÙNG NĂM" — đúng nguyên tắc "chỉ so sánh
    tương đối" của đề cương (Mục 7.4). Không dùng nhãn."""
    cols = cols or [c for c in ALL_FEATURE_COLS if c not in RELATIVE_SKIP]
    stats = {}
    for key, idx in df_raw.groupby(["mon", "nam", "dot"]).indices.items():
        med, mad, sd = {}, {}, {}
        for c in cols:
            x = df_raw[c].values[idx].astype(float)
            if c.startswith("bunching"):
                x = np.log1p(x)                       # tỷ lệ bin trên/bin dưới đuôi nặng → log
            med[c] = np.median(x); mad[c] = 1.4826 * np.median(np.abs(x - med[c])); sd[c] = x.std()
        stats[key] = (med, mad, sd)
    return stats


def apply_relative_stats(df_raw, stats, cols=None, clip=10.0):
    cols = cols or [c for c in ALL_FEATURE_COLS if c not in RELATIVE_SKIP]
    out = df_raw.copy()
    for key, idx in df_raw.groupby(["mon", "nam", "dot"]).indices.items():
        med, mad, sd = stats[key]
        for c in cols:
            x = df_raw[c].values[idx].astype(float)
            if c.startswith("bunching"):
                x = np.log1p(x)
            scale = mad[c] if mad[c] > 1e-9 else (sd[c] + 1e-9)
            out.loc[out.index[idx], c] = np.clip((x - med[c]) / scale, -clip, clip)
    return out


def normalize_features(df_feat):
    """Chuẩn hóa toàn cục (label-free) — CHỈ dùng cho baseline không giám sát."""
    feat_cols = [c for c in df_feat.columns if c not in META_COLS]
    scaler = StandardScaler()
    df_out = df_feat.copy()
    df_out[feat_cols] = scaler.fit_transform(df_feat[feat_cols]).astype(float)
    df_out[feat_cols] = df_out[feat_cols].fillna(0.0)     # NaN (2018 chưa có năm trước) → 0 = mức trung bình
    return df_out, feat_cols, scaler


def run():
    print("=== Phần 2: Thiết kế đặc trưng (bản đã sửa rò rỉ) ===")
    df_pho = load_pho_diem()
    if df_pho.empty:
        print("[LỖI] Chưa có dữ liệu phổ điểm. Chạy data_prep.run() trước.")
        return
    print(f"Dữ liệu phổ điểm: {len(df_pho)} đơn vị")
    df_feat = compute_all_features(df_pho)
    df_feat = apply_relative_stats(df_feat, fit_relative_stats(df_feat))
    df_norm, feat_cols, _ = normalize_features(df_feat)
    print(f"Số đặc trưng: {len(feat_cols)}  (huấn luyện PU dùng {len(PU_FEATURE_COLS)}, bỏ nhóm temporal)")
    out_path = DATA_PROCESSED / "dac_trung.csv"
    df_norm.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {out_path}")
    return df_norm


if __name__ == "__main__":
    run()
