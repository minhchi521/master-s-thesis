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
    df = pd.read_csv(p, dtype={"src_tinh_ma": str, "base_tinh_ma": str})
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

    Đây cũng chính là "đối chứng dương" (Mục 7.4b) — ca gian lận giả được
    tiêm chủ động, dùng để kiểm tra pipeline CÓ khả năng bắt được bất
    thường đã biết chắc là bất thường hay không (bù cho đối chứng âm ở
    controls.py, vốn chỉ kiểm tra chiều ngược lại: không báo nhầm tỉnh giỏi).
    Trả về DataFrame (intensity, recall) để lưu thành doi_chung_duong.csv.
    """
    from sklearn.ensemble import RandomForestClassifier

    bin_cols = [c for c in df_syn.columns if c.startswith("bin_")]
    if not bin_cols:
        print("  [SKIP] Không có cột bin_ trong synthetic_labeled.csv.")
        return None

    X_syn = df_syn[bin_cols].fillna(0).values
    y_syn = df_syn["label"].values
    intensity_syn = df_syn["intensity"].values

    # QUAN TRỌNG: phải tách train/test — nếu huấn luyện và đánh giá trên CÙNG
    # dữ liệu, recall sẽ luôn ~100% (mô hình học thuộc lòng), không phản ánh
    # khả năng phát hiện thật. Đây là lỗi từng có ở bản trước (đối chứng dương
    # báo 100% recall ở mọi cường độ, kể cả 0.2 — dấu hiệu điển hình của học
    # thuộc lòng), nay tách 70/30 có phân tầng theo nhãn.
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(len(X_syn))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.3, stratify=y_syn, random_state=RANDOM_SEED
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_syn[idx_train], y_syn[idx_train])
    y_score_test = clf.predict_proba(X_syn[idx_test])[:, 1]
    y_true_test = y_syn[idx_test]
    intensity_test = intensity_syn[idx_test]

    intensities = sorted(df_syn["intensity"].unique())
    recalls = []
    n_test_pos = []

    for intens in intensities:
        mask = (intensity_test == intens) & (y_true_test == 1)
        n_test_pos.append(int(mask.sum()))
        if mask.sum() == 0:
            recalls.append(np.nan)
            continue
        # Recall ở ngưỡng 0.5, CHỈ trên tập test chưa từng thấy lúc huấn luyện
        y_pred = (y_score_test[mask] >= 0.5).astype(int)
        recalls.append(y_pred.mean())

    if any(n < 10 for n in n_test_pos if n > 0):
        print("  [CẢNH BÁO] Một vài mức cường độ có rất ít mẫu dương trong tập "
              "test (n<10) sau khi tách 70/30 — recall ở mức đó có thể không ổn định, "
              "nên báo cáo kèm cỡ mẫu trong luận văn.")

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

    return pd.DataFrame({"intensity": intensities, "recall": recalls, "n_test_pos": n_test_pos})


# ──────────────────────────────────────────────────────────────
# Phần 9 — Đánh giá KHÔNG VÒNG TRÒN: Leave-One-Province-Out (LOPO)
# ──────────────────────────────────────────────────────────────
# VẤN ĐỀ với eval_results.csv (evaluate_on_real_anchors ở trên): các mô hình
# PU được HUẤN LUYỆN với 3 tỉnh mỏ neo đã gán nhãn dương, rồi được ĐÁNH GIÁ
# lại trên chính 3 tỉnh đó → gần như chắc chắn AUC ~1.0, nhưng đó là hiệu
# năng "học thuộc lòng" (vòng tròn), không phải khả năng phát hiện case
# CHƯA TỪNG THẤY. Khối dưới đây khắc phục: với mỗi tỉnh mỏ neo, giữ lại tỉnh
# đó (xoá nhãn dương, coi như "chưa biết"), huấn luyện lại trên phần còn
# lại, rồi xem mô hình có tự tìm lại được tỉnh đã giấu hay không.
def run_lopo_evaluation() -> pd.DataFrame | None:
    """
    Leave-one-province-out cho cả 5 mô hình PU + 3 ensemble
    (elkan_noto, pu_bagging, two_step, scale_pu, puicl_style,
     ensemble, ensemble_classic, ensemble_modern).

    Trả về DataFrame với một dòng "GỘP CHUNG" mỗi mô hình — đây là con số
    PU thật, không vòng tròn, dùng để báo cáo trong luận văn (không dùng
    dòng PU trong eval_results.csv cho việc này).
    """
    import pu_models  # cùng thư mục src/ — import trễ để tránh vòng tròn import

    if not (DATA_PROCESSED / "dac_trung.csv").exists():
        print("  [SKIP] LOPO: chưa có dac_trung.csv.")
        return None

    df_feat, X, feat_cols = pu_models._load_features()
    df_2018 = df_feat[df_feat["nam"] == 2018].copy()
    if df_2018.empty:
        print("  [SKIP] LOPO: không có dữ liệu năm 2018.")
        return None

    anchors = list(ANCHOR_PROVINCES_2018.keys())
    is_2018 = (df_feat["nam"] == 2018).values
    is_anchor_2018 = is_2018 & df_feat["tinh_ma"].isin(anchors).values

    model_fns = {
        "elkan_noto":  pu_models.run_elkan_noto,
        "pu_bagging":  pu_models.run_pu_bagging,
        "two_step":    pu_models.run_two_step,
        "scale_pu":    pu_models.run_scale_pu,
        "puicl_style": pu_models.run_puicl_style,
    }
    classic_names = ["elkan_noto", "pu_bagging", "two_step"]
    modern_names  = ["scale_pu", "puicl_style"]

    # score_sum_neg / count_neg: tích lũy điểm trung bình ngoài mẫu cho các
    # dòng KHÔNG phải mỏ neo trên toàn bộ 3 fold (dùng làm "nền âm" khi gộp)
    fold_scores = {name: {} for name in list(model_fns) + ["ensemble", "ensemble_classic", "ensemble_modern"]}
    per_fold_rows = []

    for held_out in anchors:
        held_out_mask = (
            df_feat["tinh_ma"].values == held_out
        ) & is_2018

        # Nhãn PU với tỉnh held_out bị xoá nhãn dương (coi như chưa biết)
        y_fold = np.zeros(len(df_feat), dtype=int)
        other_anchor_mask = is_anchor_2018 & ~held_out_mask
        y_fold[other_anchor_mask] = 1

        # Bộ đánh giá của fold này: tỉnh held_out (dương thật, CHƯA từng
        # được gán nhãn dương lúc huấn luyện) so với các dòng KHÔNG phải
        # mỏ neo (âm) — bỏ 2 tỉnh mỏ neo còn lại ra khỏi tập đánh giá vì
        # bản thân chúng vẫn là dương thật (không phải nền âm hợp lệ).
        eval_mask = is_2018 & (~is_anchor_2018 | held_out_mask)
        y_true_fold = held_out_mask[eval_mask].astype(int)

        scores_this_fold = {}
        for name, fn in model_fns.items():
            try:
                sc = fn(X, y_fold)
            except Exception as e:
                print(f"  [WARN] LOPO fold={held_out} model={name} lỗi: {e}")
                sc = np.zeros(len(df_feat))
            scores_this_fold[name] = sc
            fold_scores[name][held_out] = sc[eval_mask]

            res = compute_core_metrics(y_true_fold, sc[eval_mask], name, seed=0)
            res["fold_tinh_giu_lai"] = held_out
            per_fold_rows.append(res)

        classic_ens = pu_models.ensemble_pu({k: scores_this_fold[k] for k in classic_names})
        modern_ens  = pu_models.ensemble_pu({k: scores_this_fold[k] for k in modern_names})
        full_ens    = pu_models.ensemble_pu(scores_this_fold)
        fold_scores["ensemble_classic"][held_out] = classic_ens[eval_mask]
        fold_scores["ensemble_modern"][held_out]  = modern_ens[eval_mask]
        fold_scores["ensemble"][held_out]         = full_ens[eval_mask]

        for ens_name, ens_sc in [("ensemble_classic", classic_ens),
                                  ("ensemble_modern", modern_ens),
                                  ("ensemble", full_ens)]:
            res = compute_core_metrics(y_true_fold, ens_sc[eval_mask], ens_name, seed=0)
            res["fold_tinh_giu_lai"] = held_out
            per_fold_rows.append(res)

    df_folds = pd.DataFrame(per_fold_rows)

    # ── Gộp chung (pooled): dương = điểm out-of-fold của đúng tỉnh bị giữ
    # lại ở fold tương ứng; âm = điểm trung bình qua 3 fold (mỗi dòng âm
    # được chấm bởi 3 mô hình hơi khác nhau — lấy trung bình cho công bằng).
    combined_rows = []
    non_anchor_mask = is_2018 & ~is_anchor_2018
    y_true_pooled_neg = np.zeros(non_anchor_mask.sum(), dtype=int)

    for name in fold_scores:
        pos_scores, pos_labels = [], []
        neg_score_stack = []
        for held_out in anchors:
            fold_full = np.zeros(len(df_feat))
            eval_mask = is_2018 & (~is_anchor_2018 | (df_feat["tinh_ma"].values == held_out))
            fold_full[eval_mask] = fold_scores[name][held_out]
            # điểm của chính tỉnh bị giữ lại (1 dòng/môn tương ứng)
            held_out_mask = (df_feat["tinh_ma"].values == held_out) & is_2018
            pos_scores.append(fold_full[held_out_mask])
            pos_labels.append(np.ones(held_out_mask.sum()))
            neg_score_stack.append(fold_full[non_anchor_mask])

        y_score_pos = np.concatenate(pos_scores)
        y_true_pos  = np.concatenate(pos_labels)
        y_score_neg = np.mean(np.vstack(neg_score_stack), axis=0)

        y_true_all  = np.concatenate([y_true_pos, y_true_pooled_neg])
        y_score_all = np.concatenate([y_score_pos, y_score_neg])

        res = compute_core_metrics(y_true_all, y_score_all, name, seed=0)
        res["fold_tinh_giu_lai"] = "GỘP CHUNG"
        combined_rows.append(res)

        # lưu lại để dùng cho đường cong phát hiện + Wilcoxon
        fold_scores[name]["__pooled_y_true__"] = y_true_all
        fold_scores[name]["__pooled_y_score__"] = y_score_all

    df_combined = pd.DataFrame(combined_rows)
    df_lopo = pd.concat([df_folds, df_combined], ignore_index=True)

    out_path = OUTPUTS / "pu_lopo_results.csv"
    df_lopo.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Đã lưu (LOPO — không vòng tròn): {out_path}")
    print(df_combined[["model", "roc_auc", "pr_auc", "recall_at_k"]].to_string(index=False))

    run_lopo_evaluation._pooled = fold_scores  # cache cho các hàm dùng lại bên dưới
    return df_lopo


def build_detection_curve(fold_scores: dict, model_name: str = "ensemble") -> tuple[pd.DataFrame, dict]:
    """
    Dùng điểm gộp (pooled) không vòng tròn của một mô hình để dựng bảng
    ROC + PR + recall theo ngưỡng — xuất duong_cong_phat_hien.csv/.png.
    """
    if model_name not in fold_scores or "__pooled_y_true__" not in fold_scores[model_name]:
        return pd.DataFrame(), {}

    y_true  = fold_scores[model_name]["__pooled_y_true__"]
    y_score = fold_scores[model_name]["__pooled_y_score__"]

    fpr, tpr, thr_roc = roc_curve(y_true, y_score)
    prec, rec, thr_pr = precision_recall_curve(y_true, y_score)

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc  = float(average_precision_score(y_true, y_score))

    df_roc = pd.DataFrame({"loai": "roc", "fpr": fpr, "tpr": tpr, "threshold": thr_roc})
    df_pr = pd.DataFrame({"loai": "pr", "precision": prec, "recall": rec,
                           "threshold": np.append(thr_pr, np.nan)})
    df_curve = pd.concat([df_roc, df_pr], ignore_index=True, sort=False)

    summary = {"model": model_name, "roc_auc": roc_auc, "pr_auc": pr_auc}
    return df_curve, summary


def plot_detection_curve(fold_scores: dict, model_names: list[str], save_path):
    """Vẽ ROC + PR side-by-side cho các mô hình chỉ định, dùng điểm gộp không vòng tròn."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in model_names:
        if name not in fold_scores or "__pooled_y_true__" not in fold_scores[name]:
            continue
        y_true  = fold_scores[name]["__pooled_y_true__"]
        y_score = fold_scores[name]["__pooled_y_score__"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        ap_val = average_precision_score(y_true, y_score)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")
        axes[1].plot(rec, prec, label=f"{name} (AP={ap_val:.2f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC — LOPO (không vòng tròn)"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall — LOPO (không vòng tròn)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Đã lưu: {save_path}")



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
    df_feat  = pd.read_csv(feat_path, dtype={"tinh_ma": str}) if feat_path.exists() else None

    base_path = OUTPUTS / "scores_baseline.csv"
    pu_path   = OUTPUTS / "scores_pu.csv"

    all_results = []

    # ── Đánh giá Baseline trên ca 2018 thật ──
    if base_path.exists():
        df_base = pd.read_csv(base_path, dtype={"tinh_ma": str})
        for col in ["score_isolation_forest", "score_lof", "score_dbscan",
                    "score_divergence", "score_ensemble"]:
            if col in df_base.columns:
                res = evaluate_on_real_anchors(df_base, col, col)
                all_results.append(res)

    # ── Đánh giá PU trên ca 2018 thật ──
    if pu_path.exists():
        df_pu = pd.read_csv(pu_path, dtype={"tinh_ma": str})
        for col in ["score_pu_elkan_noto", "score_pu_pu_bagging",
                    "score_pu_two_step", "score_pu_ensemble",
                    "score_pu_scale_pu", "score_pu_puicl_style",
                    "score_pu_ensemble_classic", "score_pu_ensemble_modern"]:
            if col in df_pu.columns:
                res = evaluate_on_real_anchors(df_pu, col, col)
                all_results.append(res)

    # ── Ablation ──
    if df_feat is not None:
        print("\nChạy ablation...")
        df_ablation = run_ablation(df_feat, df_syn)
        df_ablation.to_csv(OUTPUTS / "ablation.csv", index=False, encoding="utf-8-sig")
        print(f"  Đã lưu: {OUTPUTS / 'ablation.csv'}")

    # ── Đường cong độ nhạy + đối chứng dương (Mục 7.4b) ──
    if df_syn is not None:
        print("\nVẽ đường cong độ nhạy / đối chứng dương...")
        df_doi_chung = plot_sensitivity_curve(df_syn, OUTPUTS / "sensitivity_curve.png")
        if df_doi_chung is not None:
            df_doi_chung.to_csv(OUTPUTS / "doi_chung_duong.csv", index=False, encoding="utf-8-sig")
            print(f"  Đã lưu: {OUTPUTS / 'doi_chung_duong.csv'}")

        # Phân tích độ nhạy với ngưỡng GT3
        print("\nPhân tích độ nhạy theo ngưỡng GT3:")
        df_sens = sensitivity_to_injection_assumption(df_syn)
        print(df_sens.to_string(index=False))
        df_sens.to_csv(OUTPUTS / "sensitivity_gt3.csv", index=False, encoding="utf-8-sig")

    # ── Phần 9: LOPO — đánh giá PU KHÔNG VÒNG TRÒN (quan trọng nhất) ──
    print("\n" + "=" * 60)
    print("  PHẦN 9 — Leave-one-province-out (không vòng tròn)")
    print("=" * 60)
    df_lopo = run_lopo_evaluation()
    wilcoxon_lines = []
    if df_lopo is not None:
        fold_scores = run_lopo_evaluation._pooled

        # Đường cong phát hiện (dùng điểm gộp, không vòng tròn)
        curve_frames = []
        for name in ["ensemble", "ensemble_classic", "ensemble_modern"]:
            df_curve, summary = build_detection_curve(fold_scores, name)
            if not df_curve.empty:
                df_curve.insert(0, "model", name)
                curve_frames.append(df_curve)
        if curve_frames:
            df_all_curves = pd.concat(curve_frames, ignore_index=True)
            df_all_curves.to_csv(OUTPUTS / "duong_cong_phat_hien.csv", index=False, encoding="utf-8-sig")
            print(f"  Đã lưu: {OUTPUTS / 'duong_cong_phat_hien.csv'}")
            plot_detection_curve(
                fold_scores,
                ["ensemble", "ensemble_classic", "ensemble_modern"],
                OUTPUTS / "duong_cong_phat_hien.png",
            )

        # Wilcoxon: PU thế hệ mới (CH4/GT4) so với PU cổ điển, trên 3 fold LOPO
        df_by_fold = df_lopo[df_lopo["fold_tinh_giu_lai"] != "GỘP CHUNG"]
        classic_by_fold = (
            df_by_fold[df_by_fold["model"] == "ensemble_classic"]
            .sort_values("fold_tinh_giu_lai")[["roc_auc"]].to_dict("records")
        )
        modern_by_fold = (
            df_by_fold[df_by_fold["model"] == "ensemble_modern"]
            .sort_values("fold_tinh_giu_lai")[["roc_auc"]].to_dict("records")
        )
        test_lopo = statistical_test(classic_by_fold, modern_by_fold)
        wilcoxon_lines.append("=== Wilcoxon: PU thế hệ mới (ScalePU/PUICL-style) vs PU cổ điển, trên LOPO (n=3 tỉnh) ===")
        wilcoxon_lines.append(f"stat = {test_lopo.get('stat', 'N/A')}")
        wilcoxon_lines.append(f"p-value = {test_lopo.get('p_value', 'N/A')}")
        wilcoxon_lines.append(f"note = {test_lopo.get('note', '')}")
        wilcoxon_lines.append(
            "Lưu ý: n=3 (một cặp mỗi tỉnh mỏ neo) — kiểm định chỉ mang tính tham khảo, "
            "không đủ mạnh để kết luận chắc chắn; nên báo cáo cùng khoảng tin cậy bootstrap "
            "nếu cần lập luận chặt hơn trong luận văn."
        )

    # ── Kiểm định thống kê Baseline vs PU (giữ như bản gốc, dùng eval_results.csv) ──
    if base_path.exists() and pu_path.exists():
        print("\nKiểm định Wilcoxon (Baseline vs PU, trên eval_results.csv — vòng tròn, chỉ tham khảo):")
        base_multi = [{"roc_auc": all_results[0].get("roc_auc", np.nan)}] * len(EVAL_SEEDS)
        pu_multi   = [{"roc_auc": all_results[-1].get("roc_auc", np.nan)}] * len(EVAL_SEEDS)
        test_res = statistical_test(base_multi, pu_multi)
        print(f"  p-value = {test_res.get('p_value', 'N/A'):.4f}  "
              f"(có ý nghĩa: {test_res.get('significant', False)})")
        wilcoxon_lines.append("")
        wilcoxon_lines.append("=== Wilcoxon: Baseline vs PU (eval_results.csv — VÒNG TRÒN, chỉ tham khảo, đừng trích luận văn) ===")
        wilcoxon_lines.append(f"stat = {test_res.get('stat', 'N/A')}")
        wilcoxon_lines.append(f"p-value = {test_res.get('p_value', 'N/A')}")

    if wilcoxon_lines:
        (OUTPUTS / "wilcoxon_result.txt").write_text("\n".join(wilcoxon_lines), encoding="utf-8")
        print(f"\nĐã lưu: {OUTPUTS / 'wilcoxon_result.txt'}")

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
