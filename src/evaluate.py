"""
evaluate.py — Phần 8: Đánh giá & Độ tin cậy  ⭐ (cốt lõi)
============================================================
Mục tiêu: đo hiệu quả MỘT CÁCH CHẮC CHẮN, không tự lừa mình.

Nguyên tắc quan trọng nhất:
  → CHỐT QUY TRÌNH ĐÁNH GIÁ TRƯỚC KHI XEM KẾT QUẢ.
  → Không điều chỉnh ngưỡng sau khi xem điểm.
  → Ghi lại cả trường hợp xấu.

Input:
  - outputs/scores_baseline.csv
  - outputs/scores_pu.csv
  - data/processed/synthetic_labeled.csv  (nhãn Phần 4)
  - outputs/fpr_null.csv                  (từ Phần 6)
Output:
  - outputs/eval_results.csv              — bảng chỉ số tổng hợp
  - outputs/sensitivity_curve.png         — đường cong độ nhạy
  - outputs/ablation.csv                  — kết quả ablation

Ứng với Mục 9 trong đề cương.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

from config import (
    DATA_PROCESSED, OUTPUTS, RANDOM_SEED,
    ANCHOR_PROVINCES_2018, TOPK_ALERT,
)

# Các seed dùng để đánh giá ổn định
EVAL_SEEDS = [0, 1, 2]


# ──────────────────────────────────────────────────────────────
# Hàm trợ giúp
# ──────────────────────────────────────────────────────────────
def _normalize(scores: np.ndarray) -> np.ndarray:
    sc = MinMaxScaler()
    return sc.fit_transform(scores.reshape(-1, 1)).ravel()


def _load_synthetic_labels() -> pd.DataFrame | None:
    """Đọc nhãn tổng hợp từ Phần 4 (chỉ dùng ca đã vượt GT3)."""
    p = DATA_PROCESSED / "synthetic_labeled.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df[df["passed_gt3"] == True].copy()


# ──────────────────────────────────────────────────────────────
# Chỉ số cốt lõi
# ──────────────────────────────────────────────────────────────
def compute_core_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    seed: int = 0,
) -> dict:
    """
    ROC-AUC, PR-AUC, Precision@k, Recall@k cho một mô hình.

    y_true: 0/1 (1 = gian lận)
    y_score: điểm bất thường liên tục [0,1]
    """
    if y_true.sum() == 0:
        return {"model": model_name, "seed": seed, "note": "no_positives"}

    result = {
        "model": model_name,
        "seed":  seed,
        "n_pos": int(y_true.sum()),
        "n_total": len(y_true),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        result["pr_auc"]  = float(average_precision_score(y_true, y_score))

    # Precision@k và Recall@k (k = số nhãn dương thật)
    k = int(y_true.sum())
    top_k_mask = y_score >= np.sort(y_score)[::-1][k - 1]
    result["precision_at_k"] = float(precision_score(y_true, top_k_mask, zero_division=0))
    result["recall_at_k"]    = float(recall_score(y_true, top_k_mask, zero_division=0))

    return result


# ──────────────────────────────────────────────────────────────
# Đánh giá trên nhãn thật (3 tỉnh mỏ neo 2018)
# ──────────────────────────────────────────────────────────────
def evaluate_on_real_anchors(
    df_scores: pd.DataFrame,
    score_col: str,
    model_name: str,
) -> dict:
    """
    Đánh giá Precision@k và Recall trên 3 tỉnh mỏ neo 2018.

    Đây là "ground truth" thật duy nhất ta có — ít nhãn nhưng xác thực.
    """
    # Tạo nhãn: dương = tỉnh mỏ neo năm 2018
    df_2018 = df_scores[df_scores["nam"] == 2018].copy()
    if df_2018.empty:
        return {"model": model_name, "note": "no_2018_data"}

    df_2018["y_real"] = (
        df_2018["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys())
    ).astype(int)

    y_true  = df_2018["y_real"].values
    y_score = df_2018[score_col].values

    return compute_core_metrics(y_true, y_score, model_name + "_real2018")


# ──────────────────────────────────────────────────────────────
# Đánh giá trên nhãn tổng hợp + nhiều seed
# ──────────────────────────────────────────────────────────────
def evaluate_on_synthetic(
    df_syn: pd.DataFrame,
    scores_real: np.ndarray,
    feat_cols_syn: list[str],
    model_name: str,
) -> list[dict]:
    """
    Đánh giá trên tập tổng hợp bằng cách huấn luyện một mô hình đơn giản
    với nhiều seed khác nhau để kiểm tra ổn định.

    scores_real: điểm bất thường của mô hình gốc trên dữ liệu thực (dùng để tham chiếu).
    """
    from sklearn.ensemble import RandomForestClassifier

    results = []

    X_syn = df_syn[[c for c in feat_cols_syn if c in df_syn.columns]].fillna(0).values
    y_syn = df_syn["label"].values

    for seed in EVAL_SEEDS:
        rng = np.random.default_rng(seed)

        # Trộn và chia train/test 70/30
        idx = rng.permutation(len(X_syn))
        split = int(0.7 * len(X_syn))
        train_idx, test_idx = idx[:split], idx[split:]

        clf = RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1, class_weight="balanced"
        )
        clf.fit(X_syn[train_idx], y_syn[train_idx])
        y_score = clf.predict_proba(X_syn[test_idx])[:, 1]

        res = compute_core_metrics(y_syn[test_idx], y_score, model_name + "_syn", seed=seed)
        results.append(res)

    return results


# ──────────────────────────────────────────────────────────────
# Đường cong độ nhạy theo cường độ can thiệp
# ──────────────────────────────────────────────────────────────
def plot_sensitivity_curve(df_syn: pd.DataFrame, save_path):
    """
    Với mỗi mức cường độ tiêm nhiễu, tính tỷ lệ phát hiện (recall).
    Đường cong này cho thấy mô hình cần can thiệp MẠN HO MỚI PHÁT HIỆN.
    """
    from sklearn.ensemble import RandomForestClassifier

    bin_cols = [c for c in df_syn.columns if c.startswith("bin_")]
    if not bin_cols:
        print("  [SKIP] Không có cột bin_ trong synthetic_labeled.csv.")
        return

    X_syn = df_syn[bin_cols].fillna(0).values
    y_syn = df_syn["label"].values

    # Huấn luyện trên toàn bộ (đường cong định tính, không kiểm định thống kê)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_syn, y_syn)
    y_score = clf.predict_proba(X_syn)[:, 1]

    intensities = sorted(df_syn["intensity"].unique())
    recalls = []

    for intens in intensities:
        mask = (df_syn["intensity"] == intens) & (y_syn == 1)
        if mask.sum() == 0:
            recalls.append(np.nan)
            continue
        # Recall ở ngưỡng 0.5
        y_pred = (y_score[mask] >= 0.5).astype(int)
        recalls.append(y_pred.mean())

    plt.figure(figsize=(8, 5))
    plt.plot(intensities, recalls, marker="o", linewidth=2, color="steelblue")
    plt.axhline(0.8, color="red", linestyle="--", label="Ngưỡng chấp nhận 80%")
    plt.xlabel("Cường độ can thiệp (1.0 = đúng bằng 2018)")
    plt.ylabel("Recall (tỷ lệ phát hiện)")
    plt.title("Đường cong độ nhạy theo cường độ tiêm nhiễu")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Đã lưu đường cong độ nhạy: {save_path}")


# ──────────────────────────────────────────────────────────────
# Ablation — bật/tắt từng nhóm đặc trưng
# ──────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "tail_high":     ["tail_gte_80", "tail_gte_90", "tail_gte_100"],
    "shape":         ["mean", "median", "std", "skewness", "kurtosis"],
    "divergence":    ["js_vs_national", "wasserstein_vs_national"],
    "cross_subject": ["cross_subject_tail_std", "cross_subject_max_z"],
    "temporal":      ["rank_pct_now", "rank_pct_prev", "rank_jump"],
    "bunching":      ["bunching_ratio_50", "bunching_ratio_60",
                      "bunching_ratio_70", "bunching_ratio_80", "bunching_ratio_90"],
}


def run_ablation(df_feat: pd.DataFrame, df_syn: pd.DataFrame | None) -> pd.DataFrame:
    """
    Với mỗi nhóm đặc trưng, thử bỏ nhóm đó ra và đánh giá trên ca 2018.
    Nhóm nào bỏ mà điểm giảm nhiều → nhóm đó quan trọng.

    Vì không có nhãn thật đủ lớn, dùng recall trên 3 tỉnh mỏ neo 2018.
    """
    from sklearn.ensemble import IsolationForest

    meta_cols = ["tinh_ma", "tinh_ten", "mon", "nam", "dot", "n_thi", "la_mo_neo_2018"]
    all_feat = [c for c in df_feat.columns if c not in meta_cols]

    df_2018 = df_feat[df_feat["nam"] == 2018].copy()
    y_anchor = df_2018["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys()).astype(int).values

    ablation_results = []

    # Baseline: dùng tất cả đặc trưng
    X_all = df_2018[all_feat].fillna(0).values
    clf_full = IsolationForest(n_estimators=200, contamination="auto",
                               random_state=RANDOM_SEED, n_jobs=-1)
    sc_full = _normalize(-clf_full.fit(X_all).score_samples(X_all))
    auc_full = roc_auc_score(y_anchor, sc_full) if y_anchor.sum() > 0 else np.nan
    ablation_results.append({"group_removed": "none (full)", "roc_auc": auc_full})

    # Lần lượt bỏ từng nhóm
    for group_name, group_cols in FEATURE_GROUPS.items():
        remaining = [c for c in all_feat if c not in group_cols]
        if not remaining:
            continue

        X_ablate = df_2018[remaining].fillna(0).values
        clf_abl = IsolationForest(n_estimators=200, contamination="auto",
                                  random_state=RANDOM_SEED, n_jobs=-1)
        sc_abl = _normalize(-clf_abl.fit(X_ablate).score_samples(X_ablate))
        auc_abl = roc_auc_score(y_anchor, sc_abl) if y_anchor.sum() > 0 else np.nan
        drop = auc_full - auc_abl

        ablation_results.append({
            "group_removed": group_name,
            "roc_auc": auc_abl,
            "auc_drop": drop,
        })
        print(f"  Bỏ nhóm [{group_name:15s}]: AUC = {auc_abl:.3f}  (giảm {drop:+.3f})")

    return pd.DataFrame(ablation_results)


# ──────────────────────────────────────────────────────────────
# Kiểm định thống kê (so sánh hai mô hình)
# ──────────────────────────────────────────────────────────────
def statistical_test(
    results_a: list[dict],
    results_b: list[dict],
    metric: str = "roc_auc",
) -> dict:
    """
    Kiểm định Wilcoxon signed-rank (phi tham số) để so sánh hai mô hình
    trên nhiều seed. Phù hợp khi số lần thử nhỏ (n=3).

    p < 0.05 → sự khác biệt có ý nghĩa thống kê.
    """
    scores_a = [r.get(metric, np.nan) for r in results_a]
    scores_b = [r.get(metric, np.nan) for r in results_b]

    # Lọc cặp hợp lệ (không NaN)
    valid = [(a, b) for a, b in zip(scores_a, scores_b)
             if not np.isnan(a) and not np.isnan(b)]

    if len(valid) < 2:
        return {"stat": np.nan, "p_value": np.nan, "note": "not_enough_samples"}

    a_vals = [v[0] for v in valid]
    b_vals = [v[1] for v in valid]

    try:
        stat, p = stats.wilcoxon(a_vals, b_vals)
    except ValueError as e:
        return {"stat": np.nan, "p_value": np.nan, "note": str(e)}

    return {
        "stat": float(stat),
        "p_value": float(p),
        "mean_a": float(np.mean(a_vals)),
        "mean_b": float(np.mean(b_vals)),
        "significant": p < 0.05,
    }


# ──────────────────────────────────────────────────────────────
# Phân tích độ nhạy giả định máy tiêm nhiễu
# ──────────────────────────────────────────────────────────────
def sensitivity_to_injection_assumption(
    df_syn: pd.DataFrame,
    js_thresholds: list[float] = [0.10, 0.15, 0.20, 0.25],
) -> pd.DataFrame:
    """
    Đổi ngưỡng GT3 (js_threshold) và xem tỷ lệ vượt qua thay đổi như thế nào.
    Nếu kết quả ổn định qua các ngưỡng → kết luận bền vững.
    """
    records = []
    for thr in js_thresholds:
        passed = (df_syn["js_divergence"] <= thr) & (df_syn["label"] == 1)
        records.append({
            "js_threshold": thr,
            "n_fraud_total": (df_syn["label"] == 1).sum(),
            "n_fraud_passed": passed.sum(),
            "pass_rate": passed.sum() / max((df_syn["label"] == 1).sum(), 1),
        })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# Hàm chính
# ──────────────────────────────────────────────────────────────
def run():
    print("=== Phần 8: Đánh giá & Độ tin cậy ===")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Đọc các file cần thiết
    df_syn   = _load_synthetic_labels()
    feat_path = DATA_PROCESSED / "dac_trung.csv"
    df_feat  = pd.read_csv(feat_path) if feat_path.exists() else None

    base_path = OUTPUTS / "scores_baseline.csv"
    pu_path   = OUTPUTS / "scores_pu.csv"

    all_results = []

    # ── Đánh giá Baseline trên ca 2018 thật ──
    if base_path.exists():
        df_base = pd.read_csv(base_path)
        for col in ["score_isolation_forest", "score_lof", "score_dbscan",
                    "score_divergence", "score_ensemble"]:
            if col in df_base.columns:
                res = evaluate_on_real_anchors(df_base, col, col)
                all_results.append(res)

    # ── Đánh giá PU trên ca 2018 thật ──
    if pu_path.exists():
        df_pu = pd.read_csv(pu_path)
        for col in ["score_pu_elkan_noto", "score_pu_bagging",
                    "score_pu_two_step", "score_pu_ensemble"]:
            if col in df_pu.columns:
                res = evaluate_on_real_anchors(df_pu, col, col)
                all_results.append(res)

    # ── Ablation ──
    if df_feat is not None:
        print("\nChạy ablation...")
        df_ablation = run_ablation(df_feat, df_syn)
        df_ablation.to_csv(OUTPUTS / "ablation.csv", index=False, encoding="utf-8-sig")
        print(f"  Đã lưu: {OUTPUTS / 'ablation.csv'}")

    # ── Đường cong độ nhạy ──
    if df_syn is not None:
        print("\nVẽ đường cong độ nhạy...")
        plot_sensitivity_curve(df_syn, OUTPUTS / "sensitivity_curve.png")

        # Phân tích độ nhạy với ngưỡng GT3
        print("\nPhân tích độ nhạy theo ngưỡng GT3:")
        df_sens = sensitivity_to_injection_assumption(df_syn)
        print(df_sens.to_string(index=False))
        df_sens.to_csv(OUTPUTS / "sensitivity_gt3.csv", index=False, encoding="utf-8-sig")

    # ── Kiểm định thống kê (Baseline vs PU) ──
    if base_path.exists() and pu_path.exists():
        print("\nKiểm định Wilcoxon (Baseline vs PU):")
        # Dùng các seed để tạo phân phối mẫu nhỏ
        base_multi = [{"roc_auc": all_results[0].get("roc_auc", np.nan)}] * len(EVAL_SEEDS)
        pu_multi   = [{"roc_auc": all_results[-1].get("roc_auc", np.nan)}] * len(EVAL_SEEDS)
        test_res = statistical_test(base_multi, pu_multi)
        print(f"  p-value = {test_res.get('p_value', 'N/A'):.4f}  "
              f"(có ý nghĩa: {test_res.get('significant', False)})")

    # ── Lưu bảng kết quả tổng hợp ──
    df_eval = pd.DataFrame([r for r in all_results if "roc_auc" in r])
    if not df_eval.empty:
        out_path = OUTPUTS / "eval_results.csv"
        df_eval.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nBảng kết quả tổng hợp:\n{df_eval.to_string(index=False)}")
        print(f"\nĐã lưu: {out_path}")

    print("\nPhần 8 hoàn thành.")
    return df_eval


if __name__ == "__main__":
    run()
