"""
detectors.py — Phần 3: Baseline không giám sát
================================================
Mục tiêu: phát hiện bất thường KHÔNG cần nhãn — đảm bảo luôn có kết quả
          ngay cả khi PU Learning thất bại.

Input:  data/processed/dac_trung.csv  (từ Phần 2)
Output: outputs/scores_baseline.csv   (điểm bất thường 0–1 cho mọi đơn vị)

Ứng với Mục 7.2 (phần baseline) trong đề cương.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_PROCESSED, OUTPUTS, RANDOM_SEED,
    ANCHOR_PROVINCES_2018, TOPK_ALERT,
)


# ──────────────────────────────────────────────────────────────
# Hàm trợ giúp: chuẩn hóa điểm bất thường về [0, 1]
# 1 = bất thường nhất, 0 = bình thường nhất
# ──────────────────────────────────────────────────────────────
def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Đưa mảng bất kỳ về [0,1] dùng min-max."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(raw_scores.reshape(-1, 1)).ravel()


# ──────────────────────────────────────────────────────────────
# 1. Isolation Forest
# ──────────────────────────────────────────────────────────────
def run_isolation_forest(X: np.ndarray) -> np.ndarray:
    """
    Isolation Forest: cô lập điểm dữ liệu bằng cách phân vùng ngẫu nhiên.
    Điểm bất thường bị cô lập nhanh hơn (cần ít cây hơn).

    score_samples() trả về giá trị âm — càng âm càng bất thường.
    Đảo dấu để 1 = bất thường nhất.
    """
    model = IsolationForest(
        n_estimators=300,        # số cây; nhiều hơn = ổn định hơn
        contamination="auto",    # không giả định tỷ lệ bất thường biết trước
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X)
    # score_samples trả về giá trị âm, flip để "cao = bất thường"
    raw = -model.score_samples(X)
    return _normalize_scores(raw)


# ──────────────────────────────────────────────────────────────
# 2. Local Outlier Factor (LOF)
# ──────────────────────────────────────────────────────────────
def run_lof(X: np.ndarray) -> np.ndarray:
    """
    LOF: so sánh mật độ cục bộ của mỗi điểm với hàng xóm của nó.
    Điểm có mật độ thấp hơn nhiều so với hàng xóm → bất thường.

    LOF không có predict() riêng cho outlier score nên dùng
    negative_outlier_factor_ trực tiếp sau fit.
    """
    model = LocalOutlierFactor(
        n_neighbors=20,       # k = 20 là mặc định khuyến nghị
        contamination="auto",
        n_jobs=-1,
        novelty=False,        # transductive: dùng toàn bộ X để tính
    )
    model.fit(X)
    # negative_outlier_factor_: càng âm càng bất thường
    raw = -model.negative_outlier_factor_
    return _normalize_scores(raw)


# ──────────────────────────────────────────────────────────────
# 3. DBSCAN
# ──────────────────────────────────────────────────────────────
def run_dbscan(X: np.ndarray) -> np.ndarray:
    """
    DBSCAN: gom cụm dựa theo mật độ.
    Điểm bị gán nhãn -1 (noise) → bất thường.
    Điểm thuộc cụm nhỏ (ít thành viên) → nghi ngờ hơn điểm cụm lớn.

    Trả về điểm bất thường:
      - noise (-1): 1.0
      - cụm nhỏ:   tỷ lệ nghịch kích thước cụm
      - cụm lớn:   0.0
    """
    # eps và min_samples cần tinh chỉnh theo số chiều + số điểm
    model = DBSCAN(eps=1.5, min_samples=5, n_jobs=-1)
    labels = model.fit_predict(X)

    n = len(labels)
    scores = np.zeros(n)

    # Đếm kích thước từng cụm (bỏ noise = -1)
    cluster_sizes = {}
    for lb in set(labels):
        if lb == -1:
            continue
        cluster_sizes[lb] = (labels == lb).sum()

    max_size = max(cluster_sizes.values()) if cluster_sizes else 1

    for i, lb in enumerate(labels):
        if lb == -1:
            scores[i] = 1.0                              # noise = bất thường tối đa
        else:
            # Cụm nhỏ: điểm bất thường cao hơn cụm lớn
            scores[i] = 1.0 - cluster_sizes[lb] / max_size

    return scores   # đã trong [0,1] theo định nghĩa


# ──────────────────────────────────────────────────────────────
# 4. Xếp hạng theo phân kỳ phân phối (JS distance từ features.py)
# ──────────────────────────────────────────────────────────────
def run_divergence_rank(df_feat: pd.DataFrame) -> np.ndarray:
    """
    Xếp hạng đơn giản dựa trên Jensen–Shannon divergence và Wasserstein
    (đã tính ở Phần 2). Trung bình cộng hai giá trị → điểm tổng hợp.

    Đây là phương pháp không cần mô hình, thuần thống kê.
    """
    if "js_vs_national" not in df_feat.columns:
        return np.zeros(len(df_feat))

    js  = df_feat["js_vs_national"].fillna(0).values
    ws  = df_feat["wasserstein_vs_national"].fillna(0).values

    combined = (_normalize_scores(js) + _normalize_scores(ws)) / 2
    return combined


# ──────────────────────────────────────────────────────────────
# 5. Ensemble: tổng hợp điểm từ nhiều mô hình
# ──────────────────────────────────────────────────────────────
def ensemble_scores(scores_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Kết hợp điểm bất thường từ nhiều mô hình bằng trung bình đơn giản.
    Tất cả điểm đã được chuẩn hóa [0,1] trước khi truyền vào.
    """
    matrix = np.column_stack(list(scores_dict.values()))
    return matrix.mean(axis=1)


# ──────────────────────────────────────────────────────────────
# 6. Kiểm tra nhanh: tỉnh mỏ neo 2018 có lọt top không?
# ──────────────────────────────────────────────────────────────
def quick_sanity_check(df_feat: pd.DataFrame, scores: np.ndarray, model_name: str):
    """
    Kiểm tra xem 3 tỉnh gian lận 2018 (Hà Giang, Sơn La, Hòa Bình)
    có nằm trong top bất thường không.

    Nếu KHÔNG có trong top → đặt trưng / mô hình có vấn đề.
    """
    df_check = df_feat.copy()
    df_check["score"] = scores
    df_check_2018 = df_check[df_check["nam"] == 2018].copy()

    if df_check_2018.empty:
        print(f"  [{model_name}] Không có dữ liệu 2018 để kiểm tra.")
        return

    df_check_2018 = df_check_2018.sort_values("score", ascending=False)
    total = len(df_check_2018)

    print(f"\n  [{model_name}] Kiểm tra tỉnh mỏ neo 2018 (top {TOPK_ALERT}/{total}):")
    for code, name in ANCHOR_PROVINCES_2018.items():
        subset = df_check_2018[df_check_2018["tinh_ma"] == code]
        if subset.empty:
            print(f"    {name} ({code}): không có dữ liệu")
            continue
        avg_score = subset["score"].mean()
        avg_rank  = (df_check_2018["score"] >= avg_score).sum()
        in_top    = avg_rank <= TOPK_ALERT
        flag = "✓ TOP" if in_top else "✗ ngoài top"
        print(f"    {name} ({code}): hạng trung bình ~{avg_rank}/{total}  [{flag}]")


# ──────────────────────────────────────────────────────────────
# 7. Hàm chính
# ──────────────────────────────────────────────────────────────
def run():
    print("=== Phần 3: Baseline không giám sát ===")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Đọc đặc trưng
    feat_path = DATA_PROCESSED / "dac_trung.csv"
    if not feat_path.exists():
        print("[LỖI] Chưa có dac_trung.csv. Chạy features.run() trước.")
        return

    df_feat = pd.read_csv(feat_path)
    meta_cols = ["tinh_ma", "tinh_ten", "mon", "nam", "dot", "n_thi", "la_mo_neo_2018"]
    feat_cols = [c for c in df_feat.columns if c not in meta_cols]
    X = df_feat[feat_cols].fillna(0).values

    print(f"Ma trận đặc trưng: {X.shape}")

    # Chạy từng mô hình
    print("\nChạy Isolation Forest...")
    sc_if  = run_isolation_forest(X)

    print("Chạy LOF...")
    sc_lof = run_lof(X)

    print("Chạy DBSCAN...")
    sc_db  = run_dbscan(X)

    print("Tính điểm phân kỳ phân phối...")
    sc_div = run_divergence_rank(df_feat)

    # Ensemble
    scores_dict = {
        "isolation_forest": sc_if,
        "lof":              sc_lof,
        "dbscan":           sc_db,
        "divergence":       sc_div,
    }
    sc_ensemble = ensemble_scores(scores_dict)

    # Kiểm tra nhanh
    for name, sc in scores_dict.items():
        quick_sanity_check(df_feat, sc, name)
    quick_sanity_check(df_feat, sc_ensemble, "ENSEMBLE")

    # Lưu kết quả
    df_out = df_feat[meta_cols].copy()
    for name, sc in scores_dict.items():
        df_out[f"score_{name}"] = sc
    df_out["score_ensemble"] = sc_ensemble

    out_path = OUTPUTS / "scores_baseline.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu điểm baseline: {out_path}")
    print("Phần 3 hoàn thành.")

    return df_out


if __name__ == "__main__":
    run()
