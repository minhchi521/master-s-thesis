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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel

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
    df = pd.read_csv(feat_path, dtype={"tinh_ma": str})
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

    df_syn = pd.read_csv(syn_path, dtype={"src_tinh_ma": str, "base_tinh_ma": str})
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
# Mô hình 4 — ScalePU (tái hiện theo mô tả thuật toán)
# ──────────────────────────────────────────────────────────────
# Nguồn: Dai, Y., Li, X., Wang, W., Li, C., Niu, G., & Sugiyama, M. (2026).
# "Positive-Unlabeled Learning with Extreme Scarcity of Labeled Positives." ICML 2026.
#
# LƯU Ý QUAN TRỌNG: nhóm tác giả chưa công bố mã nguồn tại thời điểm viết luận văn.
# Hàm dưới đây KHÔNG phải mã nguồn gốc — đây là bản tái hiện theo đúng hai cơ chế
# chính mô tả trong bài báo, áp dụng vào bài toán khan hiếm nhãn của luận văn:
#
#   (1) Variance regularization — thu hẹp không gian giả thuyết (giảm độ phức tạp
#       Rademacher) bằng cách phạt mạnh hệ số mô hình (L2 lớn) VÀ lấy trung bình
#       qua nhiều bootstrap trên tập dương ít ỏi để giảm phương sai của ước lượng.
#   (2) Geometric regularization — khuyến khích các mẫu dương gom cụm chặt: cộng
#       thêm vào điểm phân loại một số hạng đo độ gần với trọng tâm (centroid) của
#       tập dương đã biết, ép "vùng hiệu lực" (effective range) quanh các mẫu dương
#       trở nên chật hơn — đúng tinh thần "geometric regularization" của bài báo.
#
# Việc dùng bản tái hiện thay vì mã nguồn gốc phải được nêu rõ trong Chương 4/5
# của luận văn (Mục 7.2b của đề cương).
def run_scale_pu(
    X: np.ndarray,
    y_pu: np.ndarray,
    n_bootstrap: int = 60,
    geo_weight: float = 0.35,
    l2_strength: float = 2.0,
) -> np.ndarray:
    """
    ScalePU (tái hiện): variance regularization (bootstrap + L2 mạnh)
    + geometric regularization (phạt theo khoảng cách tới trọng tâm dương).

    geo_weight  — trọng số trộn giữa xác suất phân loại và số hạng hình học
                  (0 = bỏ qua hình học, chỉ dùng L2/bootstrap;
                   1 = chỉ dùng khoảng cách hình học).
    l2_strength — độ mạnh chính quy hóa L2 (C = 1/l2_strength trong LogisticRegression).
    """
    pos_idx = np.where(y_pu == 1)[0]
    unlabeled_idx = np.where(y_pu == 0)[0]
    n_pos = len(pos_idx)

    if n_pos == 0:
        print("  [WARN] ScalePU: không có mẫu dương.")
        return np.zeros(len(X))

    # Chuẩn hóa đặc trưng để khoảng cách hình học có ý nghĩa
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── (1) Variance regularization: bootstrap trên tập dương + L2 mạnh ──
    C = 1.0 / max(l2_strength, 1e-6)
    all_probas = np.zeros(len(X))
    n_used = 0
    for _ in range(n_bootstrap):
        # bootstrap có hoàn lại trên chính tập dương (n_pos rất nhỏ) để mô phỏng
        # nhiều "góc nhìn" khác nhau và giảm phương sai khi trung bình hóa
        boot_pos_idx = rng.choice(pos_idx, size=n_pos, replace=True)
        neg_sample_idx = rng.choice(
            unlabeled_idx, size=min(n_pos * 8, len(unlabeled_idx)), replace=False
        )
        idx_train = np.concatenate([boot_pos_idx, neg_sample_idx])
        y_train = np.concatenate([np.ones(n_pos), np.zeros(len(neg_sample_idx))])

        clf = LogisticRegression(
            C=C, max_iter=2000, random_state=int(rng.integers(1e6)),
            class_weight="balanced",
        )
        try:
            clf.fit(X_std[idx_train], y_train)
        except ValueError:
            continue  # bootstrap suy biến (toàn 1 lớp) — bỏ qua vòng này
        all_probas += clf.predict_proba(X_std)[:, 1]
        n_used += 1

    if n_used == 0:
        print("  [WARN] ScalePU: mọi vòng bootstrap đều suy biến.")
        return np.zeros(len(X))
    proba_var_reg = all_probas / n_used

    # ── (2) Geometric regularization: khoảng cách RBF tới trọng tâm dương ──
    centroid = X_std[pos_idx].mean(axis=0, keepdims=True)
    # băng thông RBF theo độ tán xạ nội tại của chính tập dương (đo bằng khoảng
    # cách trung bình giữa các mẫu dương với trọng tâm) — càng ít mẫu dương,
    # băng thông càng phải ước lượng thận trọng
    intra_spread = np.linalg.norm(X_std[pos_idx] - centroid, axis=1).mean()
    gamma = 1.0 / (2 * max(intra_spread, 1e-6) ** 2)
    geo_score = rbf_kernel(X_std, centroid, gamma=gamma).ravel()  # gần centroid → điểm cao

    # ── Kết hợp ──
    combined = (1 - geo_weight) * proba_var_reg + geo_weight * geo_score
    return _normalize(combined)


# ──────────────────────────────────────────────────────────────
# Mô hình 5 — PUICL-style (in-context, không huấn luyện lại)
# ──────────────────────────────────────────────────────────────
# Nguồn nguyên lý: Liu, S., Chang, Y., Cheng, M., Tian, Q., & Li, P. (2026).
# "In-Context Positive-Unlabeled Learning." arXiv:2605.05591.
#
# LƯU Ý QUAN TRỌNG: PUICL gốc là một transformer đã được huấn luyện trước trên
# hàng loạt bài toán PU tổng hợp; checkpoint gốc chưa được công bố công khai.
# Hàm dưới đây KHÔNG phải mô hình PUICL gốc — đây là một bộ phân loại
# "PUICL-style" giữ đúng tính chất ứng dụng cốt lõi mà PUICL hướng tới:
#   • nhận (tập dương, tập không nhãn) làm MỘT đầu vào duy nhất,
#   • KHÔNG cần huấn luyện lại / cập nhật gradient khi có dữ liệu (năm) mới,
#   • trả về xác suất ngay trong một lượt tính "suy luận có điều kiện".
#
# Cơ chế: mật độ tương đối kiểu Parzen-window (kernel density ratio) giữa
# P và U trong không gian đặc trưng đã chuẩn hóa — một dạng "attention" đơn
# giản: mỗi điểm không nhãn được so khớp (kernel similarity) với toàn bộ tập
# dương và một mẫu tham chiếu từ tập không nhãn, tương tự cách một mô hình
# in-context "chú ý" tới các ví dụ ngữ cảnh để suy ra nhãn cho điểm truy vấn.
# Đây là điểm khác biệt so với ScalePU (có bước huấn luyện/tối ưu tham số);
# PUICL-style hoàn toàn không có bước tối ưu — đúng phát biểu "one forward pass".
def run_puicl_style(
    X: np.ndarray,
    y_pu: np.ndarray,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """
    PUICL-style: suy luận có điều kiện trên (P, U) bằng tỷ lệ mật độ kernel,
    không có bước huấn luyện/tối ưu tham số — mô phỏng tính chất
    "một lượt forward, không huấn luyện lại" của PUICL gốc.
    """
    pos_idx = np.where(y_pu == 1)[0]
    unlabeled_idx = np.where(y_pu == 0)[0]
    n_pos = len(pos_idx)

    if n_pos == 0:
        print("  [WARN] PUICL-style: không có mẫu dương.")
        return np.zeros(len(X))

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Băng thông chung ước lượng theo quy tắc kiểu Silverman đơn giản,
    # dựa trên toàn bộ tập (không chỉ tập dương) vì tập dương quá nhỏ để
    # ước lượng băng thông ổn định — đây là một điều chỉnh thực dụng cần
    # nêu rõ khi báo cáo (khác với PUICL gốc, vốn học băng thông ngầm qua
    # pretraining trên hàng loạt cấu hình class-prior khác nhau).
    d = X_std.shape[1]
    n = X_std.shape[0]
    sigma = np.median(np.std(X_std, axis=0)) + 1e-6
    bandwidth = bandwidth_scale * sigma * (n ** (-1.0 / (d + 4)))
    gamma = 1.0 / (2 * max(bandwidth, 1e-6) ** 2)

    # "Ngữ cảnh": tập dương P và một mẫu con của tập không nhãn U làm proxy
    # cho phần "âm" trong ngữ cảnh (đúng tinh thần PU: U vẫn hỗn hợp dương/âm,
    # nhưng đóng vai trò nền tham chiếu để ước lượng tỷ lệ mật độ)
    rng = np.random.default_rng(RANDOM_SEED)
    ctx_u_idx = rng.choice(
        unlabeled_idx, size=min(len(unlabeled_idx), max(200, n_pos * 20)), replace=False
    )

    sim_to_pos = rbf_kernel(X_std, X_std[pos_idx], gamma=gamma).mean(axis=1)
    sim_to_u   = rbf_kernel(X_std, X_std[ctx_u_idx], gamma=gamma).mean(axis=1)

    # Tỷ lệ mật độ dương/không-nhãn (đã cộng epsilon tránh chia 0) →
    # điểm càng cao nếu một mẫu "giống ngữ cảnh dương" hơn hẳn "nền không nhãn"
    eps = 1e-8
    density_ratio = sim_to_pos / (sim_to_u + eps)

    return _normalize(density_ratio)


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

    print("Chạy ScalePU (tái hiện — xem ghi chú trong mã nguồn)...")
    sc_scale = run_scale_pu(X, y_pu)

    print("Chạy PUICL-style (tái hiện — xem ghi chú trong mã nguồn)...")
    sc_puicl = run_puicl_style(X, y_pu)

    # Ensemble — tách riêng "PU cổ điển" và "PU thế hệ mới" để so sánh công bằng
    # theo đúng CH4/GT4 của đề cương, thay vì trộn lẫn hết vào một ensemble duy nhất
    scores_dict = {
        "elkan_noto":  sc_en,
        "pu_bagging":  sc_bag,
        "two_step":    sc_ts,
        "scale_pu":    sc_scale,
        "puicl_style": sc_puicl,
    }
    classic_scores = {"elkan_noto": sc_en, "pu_bagging": sc_bag, "two_step": sc_ts}
    modern_scores  = {"scale_pu": sc_scale, "puicl_style": sc_puicl}

    sc_ensemble = ensemble_pu(scores_dict)
    sc_ensemble_classic = ensemble_pu(classic_scores)
    sc_ensemble_modern  = ensemble_pu(modern_scores)

    # Lưu kết quả
    meta_cols = ["tinh_ma", "tinh_ten", "mon", "nam", "dot", "n_thi", "la_mo_neo_2018"]
    df_out = df_feat[meta_cols].copy()
    df_out["y_pu_label"] = y_pu
    for name, sc in scores_dict.items():
        df_out[f"score_pu_{name}"] = sc
    df_out["score_pu_ensemble"] = sc_ensemble
    df_out["score_pu_ensemble_classic"] = sc_ensemble_classic   # Elkan-Noto/PU-bagging/Two-Step
    df_out["score_pu_ensemble_modern"]  = sc_ensemble_modern    # ScalePU/PUICL-style (CH4/GT4)

    out_path = OUTPUTS / "scores_pu.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu: {out_path}")

    # Ghi nhận trung thực so với baseline
    baseline_path = OUTPUTS / "scores_baseline.csv"
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path, dtype={"tinh_ma": str})
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

    # So sánh trung thực PU cổ điển vs PU thế hệ mới (CH4/GT4) trên ca 2018 thật.
    # Đây chỉ là một kiểm tra nhanh định tính; kết luận chính thức phải dựa trên
    # leave-one-province-out + tiêm nhiễu ở Phần 8 (evaluate.py), không dựa vào
    # khối này.
    df_2018_all = df_out[df_out["nam"] == 2018]
    if len(df_2018_all) > 0:
        top_classic = df_2018_all.sort_values("score_pu_ensemble_classic", ascending=False)
        top_modern  = df_2018_all.sort_values("score_pu_ensemble_modern",  ascending=False)
        n_classic = top_classic.head(10)["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys()).sum()
        n_modern  = top_modern.head(10)["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys()).sum()
        print(f"\nTỉnh mỏ neo trong top-10 (2018) — PU cổ điển vs PU thế hệ mới:")
        print(f"  PU cổ điển (Elkan-Noto/PU-bagging/Two-Step) : {n_classic}/3")
        print(f"  PU thế hệ mới (ScalePU/PUICL-style)         : {n_modern}/3")
        if n_modern < n_classic:
            print("  → PU thế hệ mới KHÔNG vượt PU cổ điển trên kiểm tra nhanh này — "
                  "ghi nhận trung thực, cần xem đầy đủ ở evaluate.py trước khi kết luận.")

    print("\nPhần 5 hoàn thành.")
    return df_out


if __name__ == "__main__":
    run()
