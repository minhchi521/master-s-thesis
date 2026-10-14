"""
pu_models.py — Phần 5: PU Learning (BẢN ĐÃ SỬA RÒ RỈ)
=====================================================
Khai thác ít nhãn dương (ca gian lận đã biết) + dữ liệu không nhãn.

Lưu ý trung thực: PU Learning với nhãn cực hiếm không đảm bảo vượt baseline.
Ghi lại cả trường hợp KHÔNG vượt — đó vẫn là kết quả khoa học.

Thay đổi so với bản đầu (xem AUDIT_ROI_RI.md):
  * Mỗi mô hình tách thành  fit_xxx(X_train, y_train) -> hàm dự đoán(X)  để có thể
    HUẤN LUYỆN trên một tập và CHẤM ĐIỂM trên tập khác chưa từng thấy. Bản đầu chỉ có
    run_xxx(X, y) huấn luyện rồi chấm lại đúng trên X (chấm điểm trong-mẫu).
  * Việc chuẩn hóa (StandardScaler) được học CHỈ từ dữ liệu huấn luyện.
  * Bộ đặc trưng PU bỏ nhóm "temporal" (xem features.py).
  * Ensemble dùng trung bình THỨ HẠNG (rank) thay vì min-max, để không phụ thuộc điểm
    cực trị của một lần chạy.
  * Các hàm run_xxx(X, y) cũ vẫn còn (chấm trong-mẫu) CHỈ để xuất danh sách theo dõi.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

try:
    from pulearn import ElkanotoPuClassifier       # pulearn >= 0.2 không còn WeightedUnlabelledPuClassifier
    HAS_PULEARN = True
except ImportError:
    HAS_PULEARN = False
    print("[WARN] pulearn chưa cài. Bỏ qua Elkan–Noto.")

from config import DATA_PROCESSED, OUTPUTS, RANDOM_SEED, ANCHOR_PROVINCES_2018
from features import PU_FEATURE_COLS, META_COLS

MODEL_NAMES = ["elkan_noto", "pu_bagging", "two_step", "scale_pu", "puicl_style"]
CLASSIC = ["elkan_noto", "pu_bagging", "two_step"]
MODERN = ["scale_pu", "puicl_style"]


# ──────────────────────────────────────────────────────────────
# Trợ giúp
# ──────────────────────────────────────────────────────────────
def rank01(scores):
    """Thứ hạng chuẩn hóa về (0,1] — dùng để trộn điểm các mô hình."""
    return rankdata(scores, method="average") / len(scores)


def _minmax(s):
    return MinMaxScaler().fit_transform(np.asarray(s).reshape(-1, 1)).ravel()


def load_features(cols=None):
    """Đọc dac_trung.csv. Mặc định chỉ lấy bộ đặc trưng PU (bỏ nhóm temporal)."""
    df = pd.read_csv(DATA_PROCESSED / "dac_trung.csv", dtype={"tinh_ma": str})
    cols = cols or PU_FEATURE_COLS
    return df, df[cols].fillna(0).values, cols


# giữ tên cũ cho module khác
_load_features = load_features


def make_pu_labels(df_feat):
    """+1 = 3 tỉnh mỏ neo năm 2018; 0 = không nhãn (KHÔNG có nghĩa là âm)."""
    y = np.zeros(len(df_feat), dtype=int)
    y[(df_feat["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys()) & (df_feat["nam"] == 2018)).values] = 1
    return y


_make_pu_labels = make_pu_labels


# ──────────────────────────────────────────────────────────────
# Mô hình 1 — Elkan–Noto (Elkan & Noto, KDD 2008)
# ──────────────────────────────────────────────────────────────
def fit_elkan_noto(X_tr, y_tr, seed=RANDOM_SEED):
    """pulearn giữ lại 20% TOÀN BỘ dòng (không phải 20% mẫu dương) để ước lượng c; với ~18 mẫu dương
    thì ~2% lần bốc không có mẫu dương nào → thử lại với hạt giống khác. Đây cũng là điểm yếu thật của
    Elkan–Noto khi nhãn dương cực hiếm: c được ước lượng từ chỉ vài mẫu dương."""
    if not HAS_PULEARN:
        raise RuntimeError("Cần cài pulearn để chạy Elkan–Noto (pip install pulearn).")
    sc = StandardScaler().fit(X_tr); Z = sc.transform(X_tr)
    for attempt in range(30):
        clf = ElkanotoPuClassifier(estimator=LogisticRegression(max_iter=1000, random_state=seed),
                                   hold_out_ratio=0.2, random_state=seed + 7919 * attempt)
        try:
            clf.fit(Z, y_tr)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("Elkan–Noto: không ước lượng được c sau 30 lần thử.")
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


# ──────────────────────────────────────────────────────────────
# Mô hình 2 — PU-Bagging (Mordelet & Vert, 2014)
# ──────────────────────────────────────────────────────────────
def fit_pu_bagging(X_tr, y_tr, seed=RANDOM_SEED, n_estimators=100):
    pos = np.where(y_tr == 1)[0]; un = np.where(y_tr == 0)[0]
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(n_estimators):
        neg = rng.choice(un, size=min(len(pos) * 5, len(un)), replace=False)
        idx = np.concatenate([pos, neg])
        yy = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        clf = RandomForestClassifier(n_estimators=50, random_state=int(rng.integers(1e6)),
                                     n_jobs=1, class_weight="balanced").fit(X_tr[idx], yy)
        models.append(clf)
    return lambda X: np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


# ──────────────────────────────────────────────────────────────
# Mô hình 3 — Two-Step với kỹ thuật "spy" (Liu et al., 2003)
# ──────────────────────────────────────────────────────────────
def fit_two_step(X_tr, y_tr, seed=RANDOM_SEED):
    pos = np.where(y_tr == 1)[0]; un = np.where(y_tr == 0)[0]
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(X_tr); Xs = sc.transform(X_tr)
    n_spy = max(1, int(0.15 * len(pos)))
    spy = rng.choice(pos, size=n_spy, replace=False)
    rest = np.setdiff1d(pos, spy)
    idx1 = np.concatenate([un, spy])
    clf1 = LogisticRegression(max_iter=1000, random_state=seed).fit(
        np.vstack([Xs[rest], Xs[idx1]]),
        np.concatenate([np.ones(len(rest)), np.zeros(len(idx1))]))
    thr = np.percentile(clf1.predict_proba(Xs[spy])[:, 1], 15)
    rel_neg = un[clf1.predict_proba(Xs[un])[:, 1] < thr]
    if len(rel_neg) == 0:
        rel_neg = un
    idx2 = np.concatenate([pos, rel_neg])
    clf2 = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced",
                                  n_jobs=1).fit(X_tr[idx2],
                                  np.concatenate([np.ones(len(pos)), np.zeros(len(rel_neg))]))
    return lambda X: clf2.predict_proba(X)[:, 1]


# ──────────────────────────────────────────────────────────────
# Mô hình 4 — ScalePU-style (tái hiện, KHÔNG phải mã gốc)
#   Dai, Li, Wang, Li, Niu, Sugiyama (ICML 2026) — Positive-Unlabeled Learning with
#   Extreme Scarcity of Labeled Positives. Bản này chỉ lấy cảm hứng từ hai cơ chế chính
#   (giảm phương sai + chính quy hóa hình học); phải nêu rõ là bản tái hiện trong luận văn.
# ──────────────────────────────────────────────────────────────
def fit_scale_pu(X_tr, y_tr, seed=RANDOM_SEED, n_bootstrap=60, geo_weight=0.35, l2_strength=2.0):
    pos = np.where(y_tr == 1)[0]; un = np.where(y_tr == 0)[0]
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(X_tr); Xs = sc.transform(X_tr)
    C = 1.0 / max(l2_strength, 1e-6)
    lrs = []
    for _ in range(n_bootstrap):
        bp = rng.choice(pos, size=len(pos), replace=True)
        neg = rng.choice(un, size=min(len(pos) * 8, len(un)), replace=False)
        idx = np.concatenate([bp, neg])
        yy = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        try:
            lrs.append(LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                          random_state=int(rng.integers(1e6))).fit(Xs[idx], yy))
        except ValueError:
            continue
    centroid = Xs[pos].mean(axis=0, keepdims=True)
    spread = np.linalg.norm(Xs[pos] - centroid, axis=1).mean()
    gamma = 1.0 / (2 * max(spread, 1e-6) ** 2)

    def predict(X):
        Z = sc.transform(X)
        p_var = np.mean([m.predict_proba(Z)[:, 1] for m in lrs], axis=0)
        geo = rbf_kernel(Z, centroid, gamma=gamma).ravel()
        return (1 - geo_weight) * p_var + geo_weight * geo
    return predict


# ──────────────────────────────────────────────────────────────
# Mô hình 5 — PUICL-style (tái hiện, KHÔNG phải mô hình gốc)
#   Liu, Chang, Cheng, Tian, Li (2026), In-Context Positive-Unlabeled Learning, arXiv:2605.05591.
#   Bản này là tỷ lệ mật độ kernel P/U, không có bước huấn luyện tham số.
# ──────────────────────────────────────────────────────────────
def fit_puicl_style(X_tr, y_tr, seed=RANDOM_SEED, bandwidth_scale=1.0):
    pos = np.where(y_tr == 1)[0]; un = np.where(y_tr == 0)[0]
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(X_tr); Xs = sc.transform(X_tr)
    d, n = Xs.shape[1], Xs.shape[0]
    sigma = np.median(np.std(Xs, axis=0)) + 1e-6
    bw = bandwidth_scale * sigma * (n ** (-1.0 / (d + 4)))
    gamma = 1.0 / (2 * max(bw, 1e-6) ** 2)
    ctx = rng.choice(un, size=min(len(un), max(200, len(pos) * 20)), replace=False)
    P, U = Xs[pos], Xs[ctx]

    def predict(X):
        Z = sc.transform(X)
        return rbf_kernel(Z, P, gamma=gamma).mean(axis=1) / (rbf_kernel(Z, U, gamma=gamma).mean(axis=1) + 1e-8)
    return predict


FIT_FUNCS = {
    "elkan_noto": fit_elkan_noto, "pu_bagging": fit_pu_bagging, "two_step": fit_two_step,
    "scale_pu": fit_scale_pu, "puicl_style": fit_puicl_style,
}


def ensemble_rank(scores_dict):
    """Trung bình thứ hạng các mô hình (không phụ thuộc thang điểm từng mô hình)."""
    return np.mean([rank01(v) for v in scores_dict.values()], axis=0)


# ──────────────────────────────────────────────────────────────
# Tương thích ngược: chấm điểm TRONG-MẪU (chỉ để xuất danh sách theo dõi)
# ──────────────────────────────────────────────────────────────
def _insample(fit_fn):
    def run_fn(X, y_pu, **kw):
        return _minmax(fit_fn(X, y_pu, **kw)(X))
    return run_fn

run_elkan_noto = _insample(fit_elkan_noto)
run_pu_bagging = _insample(fit_pu_bagging)
run_two_step = _insample(fit_two_step)
run_scale_pu = _insample(fit_scale_pu)
run_puicl_style = _insample(fit_puicl_style)
ensemble_pu = ensemble_rank


def run():
    """
    Xuất điểm PU để lập DANH SÁCH THEO DÕI. Đây là điểm TRONG-MẪU: 27 đơn vị mỏ neo là
    nhãn huấn luyện nên bị chấm cao hiển nhiên. KHÔNG dùng file này để báo cáo hiệu năng —
    hiệu năng lấy từ evaluate.py (LOPO nghiêm ngặt, chấm trên đơn vị chưa từng thấy).
    """
    print("=== Phần 5: PU Learning ===")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df_feat, X, feat_cols = load_features()
    y_pu = make_pu_labels(df_feat)
    print(f"Đặc trưng PU: {len(feat_cols)} (bỏ nhóm temporal) | dương: {(y_pu==1).sum()} | không nhãn: {(y_pu==0).sum()}")
    scores = {}
    for name in MODEL_NAMES:
        print(f"Chạy {name}...")
        scores[name] = _insample(FIT_FUNCS[name])(X, y_pu)
    df_out = df_feat[META_COLS].copy()
    df_out["y_pu_label"] = y_pu
    for k, v in scores.items():
        df_out[f"score_pu_{k}"] = v
    df_out["score_pu_ensemble"] = ensemble_rank(scores)
    df_out["score_pu_ensemble_classic"] = ensemble_rank({k: scores[k] for k in CLASSIC})
    df_out["score_pu_ensemble_modern"] = ensemble_rank({k: scores[k] for k in MODERN})
    out = OUTPUTS / "scores_pu.csv"
    df_out.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {out}  (điểm trong-mẫu — chỉ dùng cho danh sách theo dõi)")
    print("Phần 5 hoàn thành.")
    return df_out


if __name__ == "__main__":
    run()
