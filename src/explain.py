"""
explain.py — Phần 7: Giải thích (XAI) + Phiếu sức khỏe phổ điểm
==================================================================
Mục tiêu: mỗi đơn vị bị cảnh báo → phiếu giải thích có thể đọc được,
          kèm SHAP values cho biết ĐẶC TRƯNG NÀO đóng góp vào bất thường.

Input:
  - data/processed/dac_trung.csv  (đặc trưng)
  - outputs/scores_baseline.csv hoặc scores_pu.csv
Output:
  - outputs/phieu_suc_khoe_{tinh}_{mon}_{nam}.json
  - outputs/shap_summary.png      (biểu đồ tổng quan)

Ứng với Mục 7.5 trong đề cương.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # chạy không cần GUI

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap chưa cài. Bỏ qua SHAP, dùng feature importance thay thế.")

from sklearn.ensemble import RandomForestClassifier

from config import DATA_PROCESSED, OUTPUTS, RANDOM_SEED, TOPK_ALERT, ANCHOR_PROVINCES_2018


# ──────────────────────────────────────────────────────────────
# Mô tả tiếng Việt cho từng tên đặc trưng
# ──────────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "tail_gte_80":             "Tỷ lệ điểm ≥ 8",
    "tail_gte_90":             "Tỷ lệ điểm ≥ 9",
    "tail_gte_100":            "Tỷ lệ điểm 10 tuyệt đối",
    "mean":                    "Điểm trung bình",
    "median":                  "Điểm trung vị",
    "std":                     "Độ lệch chuẩn",
    "skewness":                "Độ lệch (skewness) — âm = nghiêng phải",
    "kurtosis":                "Độ nhọn (kurtosis)",
    "js_vs_national":          "Phân kỳ Jensen–Shannon so toàn quốc",
    "wasserstein_vs_national": "Khoảng cách Wasserstein so toàn quốc",
    "cross_subject_tail_std":  "Độ lệch đuôi cao giữa các môn",
    "cross_subject_max_z":     "Z-score môn nổi trội nhất so phần còn lại",
    "rank_pct_now":            "Thứ hạng tương đối năm hiện tại (1=tốt nhất)",
    "rank_pct_prev":           "Thứ hạng tương đối năm trước",
    "rank_jump":               "Bước nhảy thứ hạng so năm trước (dương=lên hạng)",
    "bunching_ratio_50":       "Dồn điểm quanh ngưỡng 5.0",
    "bunching_ratio_60":       "Dồn điểm quanh ngưỡng 6.0",
    "bunching_ratio_70":       "Dồn điểm quanh ngưỡng 7.0",
    "bunching_ratio_80":       "Dồn điểm quanh ngưỡng 8.0",
    "bunching_ratio_90":       "Dồn điểm quanh ngưỡng 9.0",
}


# ──────────────────────────────────────────────────────────────
# Huấn luyện mô hình proxy để tính SHAP
# ──────────────────────────────────────────────────────────────
def _train_proxy_model(
    X: np.ndarray,
    scores: np.ndarray,
    n_top: int = 100,
) -> RandomForestClassifier:
    """
    Huấn luyện RandomForest proxy:
      Dương = top n_top bất thường nhất
      Âm   = n_top bình thường nhất (phần dưới phân phối)

    Dùng để tính SHAP — không phải mô hình chính thức.
    """
    sorted_idx = np.argsort(scores)
    neg_idx = sorted_idx[:n_top]          # bình thường nhất
    pos_idx = sorted_idx[-n_top:]         # bất thường nhất

    idx_all = np.concatenate([neg_idx, pos_idx])
    y_proxy = np.concatenate([np.zeros(n_top), np.ones(n_top)])

    clf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
    )
    clf.fit(X[idx_all], y_proxy)
    return clf


# ──────────────────────────────────────────────────────────────
# Tính SHAP values (hoặc feature importance nếu không có shap)
# ──────────────────────────────────────────────────────────────
def compute_shap_values(
    clf: RandomForestClassifier,
    X: np.ndarray,
    feat_cols: list[str],
    max_display: int = 15,
    save_path=None,
) -> np.ndarray:
    """
    Tính SHAP values và vẽ summary plot.
    Trả về mảng shap_values shape (n_samples, n_features).
    """
    if not HAS_SHAP:
        # Fallback: dùng feature importance từ Random Forest
        fi = clf.feature_importances_
        # Trả về mảng giả (mỗi mẫu dùng chung feature importance)
        return np.tile(fi, (len(X), 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(clf)
        # Lấy SHAP cho lớp dương (index 1)
        sv = explainer.shap_values(X)
        shap_pos = sv[1] if isinstance(sv, list) else sv

    # Vẽ summary plot
    if save_path is not None:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_pos, X,
            feature_names=feat_cols,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Summary — Đặc trưng quan trọng nhất")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Đã lưu SHAP summary plot: {save_path}")

    return shap_pos


# ──────────────────────────────────────────────────────────────
# Tạo phiếu sức khỏe cho 1 đơn vị
# ──────────────────────────────────────────────────────────────
def make_health_card(
    row_feat: pd.Series,
    shap_vals: np.ndarray,
    feat_cols: list[str],
    score: float,
    rank: int,
    total: int,
    n_top_features: int = 5,
) -> dict:
    """
    Tạo phiếu sức khỏe dạng dict cho một đơn vị bị cảnh báo.

    Nội dung phiếu:
      - Thông tin định danh
      - Điểm bất thường + xếp hạng
      - 3–5 đặc trưng đóng góp nhiều nhất kèm diễn giải tiếng Việt
    """
    # Sắp xếp đặc trưng theo SHAP trị tuyệt đối
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:n_top_features]

    top_features = []
    for i in top_idx:
        fname = feat_cols[i]
        fval  = float(row_feat.get(fname, 0))
        fshap = float(shap_vals[i])
        fdesc = FEATURE_DESCRIPTIONS.get(fname, fname)

        direction = "tăng nghi ngờ" if fshap > 0 else "giảm nghi ngờ"
        top_features.append({
            "feature":     fname,
            "description": fdesc,
            "value":       round(fval, 4),
            "shap":        round(fshap, 4),
            "direction":   direction,
        })

    card = {
        "tinh_ma":   row_feat.get("tinh_ma", ""),
        "tinh_ten":  row_feat.get("tinh_ten", ""),
        "mon":       row_feat.get("mon", ""),
        "nam":       int(row_feat.get("nam", 0)),
        "dot":       int(row_feat.get("dot", 1)),
        "n_thi":     int(row_feat.get("n_thi", 0)),
        "la_mo_neo": bool(row_feat.get("la_mo_neo_2018", False)),
        "score_anomaly": round(score, 4),
        "rank":      rank,
        "total":     total,
        "rank_pct":  round(rank / total * 100, 1),
        "top_features": top_features,
        "nhan_xet_tom_tat": _generate_summary(row_feat, top_features, score),
    }
    return card


def _generate_summary(row_feat: pd.Series, top_features: list[dict], score: float) -> str:
    """Tạo câu nhận xét tóm tắt ngắn gọn bằng tiếng Việt."""
    parts = []
    tieu_chi = [f["description"] for f in top_features[:3]]
    if tieu_chi:
        parts.append(f"Bất thường chủ yếu do: {'; '.join(tieu_chi)}.")
    if score >= 0.8:
        parts.append("Mức độ nghi ngờ CAO — cần xem xét ưu tiên.")
    elif score >= 0.6:
        parts.append("Mức độ nghi ngờ TRUNG BÌNH.")
    else:
        parts.append("Mức độ nghi ngờ THẤP.")
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# Hàm chính
# ──────────────────────────────────────────────────────────────
def run(score_col: str = "score_ensemble", top_k: int = TOPK_ALERT):
    print("=== Phần 7: Giải thích (XAI) + Phiếu sức khỏe ===")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Đọc đặc trưng
    feat_path = DATA_PROCESSED / "dac_trung.csv"
    if not feat_path.exists():
        print("[LỖI] Chưa có dac_trung.csv.")
        return

    df_feat = pd.read_csv(feat_path)
    meta_cols = ["tinh_ma", "tinh_ten", "mon", "nam", "dot", "n_thi", "la_mo_neo_2018"]
    feat_cols = [c for c in df_feat.columns if c not in meta_cols]
    X = df_feat[feat_cols].fillna(0).values

    # Đọc điểm bất thường (ưu tiên baseline, fallback PU)
    score_file = OUTPUTS / "scores_baseline.csv"
    if not score_file.exists():
        score_file = OUTPUTS / "scores_pu.csv"
        score_col  = "score_pu_ensemble"

    if not score_file.exists():
        print("[LỖI] Chưa có file điểm. Chạy detectors.run() trước.")
        return

    df_scores = pd.read_csv(score_file)
    scores = df_scores[score_col].values

    # Huấn luyện mô hình proxy để tính SHAP
    print("Huấn luyện mô hình proxy...")
    clf_proxy = _train_proxy_model(X, scores)

    # Tính SHAP
    print("Tính SHAP values...")
    shap_vals = compute_shap_values(
        clf_proxy, X, feat_cols,
        save_path=OUTPUTS / "shap_summary.png"
    )

    # Xếp hạng
    sorted_idx = np.argsort(scores)[::-1]
    ranks = np.empty(len(scores), dtype=int)
    ranks[sorted_idx] = np.arange(1, len(scores) + 1)

    # Tạo phiếu cho top_k đơn vị bất thường nhất
    print(f"Tạo phiếu sức khỏe cho top {top_k} đơn vị...")
    for i in sorted_idx[:top_k]:
        row_feat = df_feat.iloc[i]
        card = make_health_card(
            row_feat=row_feat,
            shap_vals=shap_vals[i],
            feat_cols=feat_cols,
            score=float(scores[i]),
            rank=int(ranks[i]),
            total=len(scores),
        )

        fname = (
            f"phieu_suc_khoe"
            f"_{card['tinh_ma']}"
            f"_{card['mon']}"
            f"_{card['nam']}.json"
        )
        out_path = OUTPUTS / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(card, f, ensure_ascii=False, indent=2)

    print(f"  Đã tạo {top_k} phiếu sức khỏe trong {OUTPUTS}/")
    print("Phần 7 hoàn thành.")

    return shap_vals


if __name__ == "__main__":
    run()
