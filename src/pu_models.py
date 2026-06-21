"""
pu_models.py — Phần 5: PU Learning
=====================================
Mục tiêu: khai thác số ít nhãn dương (ca gian lận đã biết) + dữ liệu không nhãn
          để phân loại bất thường tốt hơn baseline không giám sát.

Lưu ý trung thực: PU Learning với nhãn cực hiếm không đảm bảo vượt baseline.
Ghi lại cả trường hợp KHÔNG vượt — đó vẫn là kết quả khoa học.

Input:
  - data/processed/dac_trung.csv        — đặc trưng từ Phần 2
  - data/processed/synthetic_labeled.csv — nhãn tổng hợp từ Phần 4
Output:
  - outputs/scores_pu.csv               — điểm bất thường theo PU

Ứng với Mục 7.2 trong đề cương.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from pulearn import ElkanotoPuClassifier, WeightedUnlabelledPuClassifier
    HAS_PULEARN = True
except ImportError:
    HAS_PULEARN = False
    print("[WARN] pulearn chưa cài. Chỉ dùng PU-Bagging và Two-Step.")

from config import DATA_PROCESSED, OUTPUTS, RANDOM_SEED, ANCHOR_PROVINCES_2018


# ──────────────────────────────────────────────────────────────
# Hàm trợ giúp dùng chung
# ──────────────────────────────────────────────────────────────
def _normalize(scores: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).ravel()


def _load_features() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Đọc dac_trung.csv, trả về (df, X, feat_cols)."""
    feat_path = DATA_PROCESSED / "dac_trung.csv"
    if not feat_path.exists():
        raise FileNotFoundError("Chưa có dac_trung.csv. Chạy features.run() trước.")
    df = pd.read_csv(feat_path)
    meta_cols = ["tinh_ma", "tinh_ten", "mon", "nam", "dot", "n_thi", "la_mo_neo_2018"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].fillna(0).values
    return df, X, feat_cols


def _make_pu_labels(df_feat: pd.DataFrame) -> np.ndarray:
    """
    Gán nhãn PU từ dữ liệu thực:
      +1  — 3 tỉnh mỏ neo năm 2018 (nhãn dương đã biết)
       0  — tất cả còn lại (không nhãn = unknown)

    Trong PU Learning, 0 KHÔNG có nghĩa là âm — chỉ là chưa biết.
    """
    labels = np.zeros(len(df_feat), dtype=int)
    anchor_mask = (
        df_feat["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys()) &
        (df_feat["nam"] == 2018)
    )
    labels[anchor_mask] = 1
    return labels


def _load_synthetic_features(feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Đọc tập tổng hợp từ Phần 4 và chuyển về không gian đặc trưng (bin_*).
    Chỉ dùng các ca đã vượt kiểm chứng GT3.

    Trả về (X_syn, y_syn).
    """
    syn_path = DATA_PROCESSED / "synthetic_labeled.csv"
    if not syn_path.exists():
        return np.empty((0, len(feat_cols))), np.empty(0)

    df_syn = pd.read_csv(syn_path)
    df_syn = df_syn[df_syn["passed_gt3"] == True]

    # Tập tổng hợp dùng cột bin_* làm đặc trưng (chưa qua feature engineering)
    # → để khớp với X_real, cần chỉ dùng các cột chung
    bin_cols = [c for c in feat_cols if c.startswith("bin_") or
                c in df_syn.columns]
    common = [c for c in feat_cols if c in df_syn.columns]

    if not common:
        # Không khớp được cột → bỏ qua dữ liệu tổng hợp
        return np.empty((0, len(feat_cols))), np.empty(0)

    X_syn = np.zeros((len(df_syn), len(feat_cols)))
    for i, col in enumerate(feat_cols):
        if col in df_syn.columns:
            X_syn[:, i] = df_syn[col].fillna(0).values

    y_syn = df_syn["label"].values
    return X_syn, y_syn


# ──────────────────────────────────────────────────────────────
# Mô hình 1 — Elkan–Noto (dùng thư viện pulearn)
# ──────────────────────────────────────────────────────────────
def run_elkan_noto(X: np.ndarray, y_pu: np.ndarray) -> np.ndarray:
    """
    Elkan–Noto: ước lượng c = P(s=1|y=1) (xác suất gán nhãn cho mẫu dương)
    rồi hiệu chỉnh đầu ra của bộ phân loại thông thường.

    Khi nhãn rất ít (< 20), ổn định kém — ghi nhận và không ép phải thắng baseline.
    """
    if not HAS_PULEARN:
        print("  [SKIP] Elkan–Noto: pulearn chưa cài.")
        return np.zeros(len(X))

    n_pos = (y_pu == 1).sum()
    if n_pos < 3:
        print(f"  [WARN] Elkan–Noto: chỉ có {n_pos} mẫu dương — kết quả không đáng tin.")

    base_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    pu_clf = ElkanotoPuClassifier(estimator=base_clf, hold_out_ratio=0.2)
    pu_clf.fit(X, y_pu)

    # predict_proba trả về xác suất dương
    proba = pu_clf.predict_proba(X)[:, 1]
    return _normalize(proba)


# ──────────────────────────────────────────────────────────────
# Mô hình 2 — PU-Bagging (Mordelet & Vert 2014)
# ──────────────────────────────────────────────────────────────
def run_pu_bagging(X: np.ndarray, y_pu: np.ndarray, n_estimators: int = 100) -> np.ndarray:
    """
    PU-Bagging: mỗi vòng lặp lấy mẫu bootstrap từ tập không nhãn (giả âm),
    kết hợp với tất cả mẫu dương → huấn luyện bộ phân loại nhỏ.
    Trung bình đầu ra nhiều vòng lặp → điểm PU ổn định.

    Đây là cách tiếp cận mạnh nhất khi nhãn dương rất ít.
    """
    pos_idx = np.where(y_pu == 1)[0]
    unlabeled_idx = np.where(y_pu == 0)[0]
    n_pos = len(pos_idx)

    if n_pos == 0:
        print("  [WARN] PU-Bagging: không có mẫu dương.")
        return np.zeros(len(X))

    rng = np.random.default_rng(RANDOM_SEED)
    all_probas = np.zeros(len(X))

    for _ in range(n_estimators):
        # Lấy mẫu ngẫu nhiên từ không nhãn có kích thước = n_pos
        # (cân bằng lớp để mô hình không bị lệch)
        neg_sample_idx = rng.choice(unlabeled_idx, size=min(n_pos * 5, len(unlabeled_idx)), replace=False)

        idx_train = np.concatenate([pos_idx, neg_sample_idx])
        X_train = X[idx_train]
        y_train = np.concatenate([np.ones(n_pos), np.zeros(len(neg_sample_idx))])

        clf = RandomForestClassifier(
            n_estimators=50, random_state=int(rng.integers(1e6)),
            n_jobs=-1, class_weight="balanced"
        )
        clf.fit(X_train, y_train)
        all_probas += clf.predict_proba(X)[:, 1]

    avg_proba = all_probas / n_estimators
    return _normalize(avg_proba)


# ──────────────────────────────────────────────────────────────
# Mô hình 3 — Phân loại hai bước (Two-Step PU)
# ──────────────────────────────────────────────────────────────
def run_two_step(X: np.ndarray, y_pu: np.ndarray) -> np.ndarray:
    """
    Two-Step PU:
      Bước 1 — Dùng Spy technique: trộn một số mẫu dương vào tập không nhãn,
               huấn luyện classifier, lấy ngưỡng từ điểm spy để xác định "âm đáng tin".
      Bước 2 — Huấn luyện lại với {dương thật} + {âm đáng tin} → classifier cuối.

    Phù hợp khi tập không nhãn lớn và nhãn dương rất ít.
    """
    pos_idx = np.where(y_pu == 1)[0]
    unlabeled_idx = np.where(y_pu == 0)[0]
    n_pos = len(pos_idx)

    if n_pos < 2:
        print("  [WARN] Two-Step: không đủ mẫu dương cho bước 1.")
        return np.zeros(len(X))

    rng = np.random.default_rng(RANDOM_SEED)

    # ── Bước 1: Spy ──
    # Chọn 15% mẫu dương làm "spy" (nhúng vào tập không nhãn)
    n_spy = max(1, int(0.15 * n_pos))
    spy_idx = rng.choice(pos_idx, size=n_spy, replace=False)
    remaining_pos_idx = np.setdiff1d(pos_idx, spy_idx)

    # Tập train bước 1: không nhãn + spy (đều gán nhãn 0)
    step1_idx = np.concatenate([unlabeled_idx, spy_idx])
    y_step1 = np.concatenate([
        np.zeros(len(unlabeled_idx)),
        np.zeros(n_spy)          # spy cũng gán 0 (giả vờ không nhãn)
    ])

    clf1 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    X_remaining = X[remaining_pos_idx]
    X_step1 = X[step1_idx]
    clf1.fit(
        np.vstack([X_remaining, X_step1]),
        np.concatenate([np.ones(len(remaining_pos_idx)), y_step1])
    )

    # Ngưỡng = điểm thấp nhất của spy (bất kỳ điểm nào dưới ngưỡng này là âm đáng tin)
    spy_scores = clf1.predict_proba(X[spy_idx])[:, 1]
    threshold = np.percentile(spy_scores, 15)   # 15% dưới cùng của spy

    # Xác định "âm đáng tin" từ tập không nhãn
    unlabeled_scores = clf1.predict_proba(X[unlabeled_idx])[:, 1]
    reliable_neg_idx = unlabeled_idx[unlabeled_scores < threshold]

    if len(reliable_neg_idx) == 0:
        print("  [WARN] Two-Step: không tìm được âm đáng tin.")
        return np.zeros(len(X))

    # ── Bước 2: Classifier cuối ──
    step2_idx   = np.concatenate([pos_idx, reliable_neg_idx])
    y_step2     = np.concatenate([np.ones(n_pos), np.zeros(len(reliable_neg_idx))])

    clf2 = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED,
        class_weight="balanced", n_jobs=-1
    )
    clf2.fit(X[step2_idx], y_step2)

    proba = clf2.predict_proba(X)[:, 1]
    return _normalize(proba)


# ──────────────────────────────────────────────────────────────
# Tổ hợp (Ensemble) PU
# ──────────────────────────────────────────────────────────────
def ensemble_pu(scores_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Trung bình điểm từ các mô hình PU (đã chuẩn hóa [0,1]).
    Mô hình SKIP (toàn 0) vẫn được tính — cân nhắc loại bỏ nếu cần.
    """
    matrix = np.column_stack(list(scores_dict.values()))
    return matrix.mean(axis=1)


# ──────────────────────────────────────────────────────────────
# Hàm chính
# ──────────────────────────────────────────────────────────────
def run():
    print("=== Phần 5: PU Learning ===")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df_feat, X, feat_cols = _load_features()
    y_pu = _make_pu_labels(df_feat)

    n_pos = (y_pu == 1).sum()
    print(f"Mẫu dương (nhãn thật): {n_pos}  |  Không nhãn: {(y_pu == 0).sum()}")
    if n_pos < 5:
        print("[WARN] Nhãn dương rất ít — PU Learning sẽ không ổn định, cần dữ liệu tổng hợp.")

    # Chạy từng mô hình
    print("\nChạy Elkan–Noto...")
    sc_en = run_elkan_noto(X, y_pu)

    print("Chạy PU-Bagging...")
    sc_bag = run_pu_bagging(X, y_pu)

    print("Chạy Two-Step...")
    sc_ts = run_two_step(X, y_pu)

    # Ensemble
    scores_dict = {
        "elkan_noto":  sc_en,
        "pu_bagging":  sc_bag,
        "two_step":    sc_ts,
    }
    sc_ensemble = ensemble_pu(scores_dict)

    # Lưu kết quả
    meta_cols = ["tinh_ma", "tinh_ten", "mon", "nam", "dot", "n_thi", "la_mo_neo_2018"]
    df_out = df_feat[meta_cols].copy()
    df_out["y_pu_label"] = y_pu
    for name, sc in scores_dict.items():
        df_out[f"score_pu_{name}"] = sc
    df_out["score_pu_ensemble"] = sc_ensemble

    out_path = OUTPUTS / "scores_pu.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu: {out_path}")

    # Ghi nhận trung thực so với baseline
    baseline_path = OUTPUTS / "scores_baseline.csv"
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)
        # Kiểm tra ca 2018 top-10 theo PU vs baseline
        df_cmp = df_out.merge(df_base[["tinh_ma", "mon", "nam", "score_ensemble"]],
                              on=["tinh_ma", "mon", "nam"], suffixes=("", "_base"))
        df_2018 = df_cmp[df_cmp["nam"] == 2018].sort_values("score_pu_ensemble", ascending=False)
        anchor_in_top_pu   = df_2018.head(10)["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys()).sum()
        df_2018_base = df_cmp[df_cmp["nam"] == 2018].sort_values("score_ensemble", ascending=False)
        anchor_in_top_base = df_2018_base.head(10)["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys()).sum()

        print(f"\nTỉnh mỏ neo trong top-10 (2018):")
        print(f"  PU ensemble : {anchor_in_top_pu}/3")
        print(f"  Baseline    : {anchor_in_top_base}/3")
        if anchor_in_top_pu < anchor_in_top_base:
            print("  → PU KHÔNG vượt baseline trên tập thật — ghi nhận trung thực.")

    print("\nPhần 5 hoàn thành.")
    return df_out


if __name__ == "__main__":
    run()
