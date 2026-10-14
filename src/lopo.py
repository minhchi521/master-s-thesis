"""
lopo.py — Đánh giá Leave-One-Province-Out NGHIÊM NGẶT (không rò rỉ)
====================================================================
Thiết kế (xem AUDIT_ROI_RI.md):

  * Mỗi "fold" giấu hẳn 1 trong 3 tỉnh mỏ neo (9 đơn vị 2018 dương thật).
  * Tập kiểm tra âm = toàn bộ đơn vị 2018 của N_TEST_PROV tỉnh không phải mỏ neo, chọn
    ngẫu nhiên THEO TỈNH (không theo đơn vị), lặp N_REPEATS lần. Chọn theo tỉnh vì 9 môn
    của cùng một tỉnh chia sẻ đặc trưng cấp tỉnh (cross_subject_*), nếu tách theo đơn vị
    thì "anh em cùng tỉnh" nằm ở tập huấn luyện → rò rỉ.
  * Chế độ "strict": MỌI đơn vị của tập kiểm tra bị loại hoàn toàn khỏi huấn luyện (kể cả
    ở dạng không nhãn), chuẩn hóa học chỉ trên tập huấn luyện, chấm điểm ngoài-mẫu.
  * Chế độ "legacy": tái hiện cách làm cũ (huấn luyện trên tất cả, chấm lại trên chính các
    đơn vị đã huấn luyện) CHỈ để đo mức độ ảnh hưởng của lỗi.
"""
import sys, time, pathlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

import features as F
import pu_models as P
from config import ANCHOR_PROVINCES_2018, OUTPUTS, RANDOM_SEED
from data_prep import load_pho_diem

N_TEST_PROV = 18
N_REPEATS = 8
INTENSITIES = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
FPR_TARGET = 0.05
ANCHORS = list(ANCHOR_PROVINCES_2018)
# Tỉnh "giỏi hợp pháp" — MÃ THEO HỆ MÃ THI (khác hệ hành chính). Bản cũ dùng nhầm mã hành chính:
# 31/40/48 thực ra là Quảng Bình/Đắk Lắk/Đồng Nai, còn 79 không tồn tại trong dữ liệu.
CONTROL_PROVINCES = {"01": "Hà Nội", "02": "TP HCM", "03": "Hải Phòng", "04": "Đà Nẵng", "29": "Nghệ An"}


# ── số đo ─────────────────────────────────────────────────────
def metrics(y, s, rng):
    k = int(y.sum())
    order = np.lexsort((rng.random(len(s)), -s))          # chia hòa ngẫu nhiên, không thổi phồng precision@k
    top = np.zeros(len(s), bool); top[order[:k]] = True
    return dict(roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                precision_at_k=float((y[top] == 1).mean()), recall_at_k=float(y[top].sum() / k))


def fast_auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ── ma trận đặc trưng ─────────────────────────────────────────
def build_matrices(df_pho):
    raw_l = F.compute_all_features(df_pho, reference="mean_all", temporal="const2018")
    sc_l, _, _ = F.normalize_features(raw_l)
    raw_abs = F.compute_all_features(df_pho)
    stats = F.fit_relative_stats(raw_abs)
    raw_f = F.apply_relative_stats(raw_abs, stats)          # đặc trưng so sánh tương đối trong (môn, năm)
    sc_f, _, scaler = F.normalize_features(raw_f)
    meta = sc_f[F.META_COLS].copy()
    return dict(
        legacy=sc_l[F.LEGACY_FEATURE_COLS].fillna(0).values,
        fixed=sc_f[F.PU_FEATURE_COLS].fillna(0).values,
        meta=meta, raw_fixed=raw_f, scaler=scaler, rel_stats=stats,
    )


def iter_splits(meta, n_repeats=N_REPEATS, seed=RANDOM_SEED):
    is18 = (meta.nam == 2018).values; code = meta.tinh_ma.values
    non_anchor = sorted(set(code[is18]) - set(ANCHORS))
    for fi, ho in enumerate(ANCHORS):
        held = is18 & (code == ho)
        other = is18 & np.isin(code, [a for a in ANCHORS if a != ho])
        for r in range(n_repeats):
            rng = np.random.default_rng([seed, fi, r])
            tp = list(rng.choice(non_anchor, N_TEST_PROV, replace=False))
            yield dict(fold=ho, fi=fi, rep=r, held=held, other=other,
                       test=is18 & np.isin(code, tp), test_prov=tp)


# ── chạy một phiên bản (đặc trưng × quy trình) ────────────────
def run_variant(X, meta, protocol, tag, pert=None, keep=False, models=P.MODEL_NAMES):
    rows, units, sens, store = [], [], [], []
    t0 = time.time()
    for sp in iter_splits(meta):
        ev = np.where(sp["held"] | sp["test"])[0]
        y_ev = sp["held"][ev].astype(int)
        tr = np.where(~(sp["held"] | sp["test"]))[0] if protocol == "strict" else np.arange(len(X))
        y_tr = sp["other"][tr].astype(int)
        seed = 1000 * sp["fi"] + sp["rep"]
        rng = np.random.default_rng([RANDOM_SEED, 7, sp["fi"], sp["rep"]])
        preds, sc = {}, {}
        for m in models:
            preds[m] = P.FIT_FUNCS[m](X[tr], y_tr, seed=seed)
            sc[m] = preds[m](X[ev])
        sc["ensemble_classic"] = P.ensemble_rank({k: sc[k] for k in P.CLASSIC if k in sc})
        sc["ensemble_modern"] = P.ensemble_rank({k: sc[k] for k in P.MODERN if k in sc})
        sc["ensemble"] = P.ensemble_rank({k: sc[k] for k in models})
        for name, s in sc.items():
            r = metrics(y_ev, s, rng)
            rows.append(dict(variant=tag, protocol=protocol, fold=sp["fold"], rep=sp["rep"], model=name, **r))
        if keep:
            store.append((y_ev, {k: v.copy() for k, v in sc.items()}))
            k_top = int(y_ev.sum())
            for name in ["elkan_noto", "pu_bagging", "two_step", "scale_pu", "puicl_style",
                         "ensemble_classic", "ensemble_modern", "ensemble"]:
                if name not in sc:
                    continue
                order = np.lexsort((rng.random(len(ev)), -sc[name])); top = np.zeros(len(ev), bool); top[order[:k_top]] = True
                for j, i in enumerate(ev):
                    if y_ev[j] == 0:
                        units.append((tag, sp["fold"], sp["rep"], name, meta.tinh_ma.values[i], meta.mon.values[i], bool(top[j])))
        if pert is not None:       # đường cong độ nhạy: chấm ca tiêm nhiễu bằng chính các mô hình vừa huấn luyện
            neg_clean = {m: sc[m][y_ev == 0] for m in list(models) + ["ensemble_classic", "ensemble_modern", "ensemble"] if m in sc}
            ecdf = lambda ref, v: np.searchsorted(np.sort(ref), v, side="right") / len(ref)
            code = meta.tinh_ma.values
            test_rows = np.where(sp["test"])[0]
            for lam in INTENSITIES:
                Xp_all, idx_all = pert[(sp["fold"], lam)]
                pos_in = {i: j for j, i in enumerate(idx_all)}
                sel = np.array([pos_in[i] for i in test_rows]); Xp = Xp_all[sel]
                u_clean, u_pert = {}, {}
                for m in models:
                    u_clean[m] = ecdf(neg_clean[m], neg_clean[m]); u_pert[m] = ecdf(neg_clean[m], preds[m](Xp))
                u_real = {m: ecdf(neg_clean[m], sc[m][y_ev == 1]) for m in models}
                for ens, members in [("ensemble_classic", P.CLASSIC), ("ensemble_modern", P.MODERN), ("ensemble", list(models))]:
                    members = [x for x in members if x in models]
                    u_clean[ens] = np.mean([u_clean[x] for x in members], axis=0)
                    u_pert[ens] = np.mean([u_pert[x] for x in members], axis=0)
                    u_real[ens] = np.mean([u_real[x] for x in members], axis=0)
                for name in u_clean:
                    tau = np.quantile(u_clean[name], 1 - FPR_TARGET)
                    sens.append(dict(variant=tag, fold=sp["fold"], rep=sp["rep"], model=name, intensity=lam,
                                     recall_fake=float((u_pert[name] > tau).mean()),
                                     recall_real=float((u_real[name] > tau).mean())))
    print(f"  [{tag}] xong {len(rows)//8} lượt trong {time.time()-t0:.0f}s")
    return pd.DataFrame(rows), pd.DataFrame(units, columns=["variant", "fold", "rep", "model", "tinh_ma", "mon", "flagged"]), \
        pd.DataFrame(sens), store


# ── ca tiêm nhiễu cho từng (fold, cường độ) ───────────────────
def build_perturbed(df_pho, mats, intensities=INTENSITIES):
    """Chữ ký lấy từ CHÍNH tỉnh bị giấu (chỉ để TẠO ca kiểm tra, không dùng huấn luyện), tiêm vào mọi
    đơn vị 2018 không-mỏ-neo. Đặc trưng ca giả tính với tham chiếu quốc gia GỐC (sạch)."""
    df = df_pho.reset_index(drop=True)
    bins = F._bin_columns(df)
    raw = df[bins].values.astype(float); s = raw.sum(axis=1, keepdims=True)
    D = np.where(s > 0, raw / np.where(s > 0, s, 1), 0.0)
    ref = F.compute_reference(df)
    is18 = (df.nam == 2018).values
    na18 = np.where(is18 & ~df.tinh_ma.isin(ANCHORS).values)[0]
    scaler, cols_all = mats["scaler"], F.ALL_FEATURE_COLS
    pu_pos = [cols_all.index(c) for c in F.PU_FEATURE_COLS]
    out = {}
    for ho in ANCHORS:
        sig = {df.mon.values[i]: D[i] - ref[i] for i in np.where(is18 & (df.tinh_ma.values == ho))[0]}
        for lam in intensities:
            Dm = D.copy()
            for i in na18:
                v = np.clip(D[i] + lam * sig[df.mon.values[i]], 0, None)
                Dm[i] = v / v.sum() if v.sum() > 0 else D[i]
            dfm = df.copy(); dfm[bins] = Dm * df["n_thi"].values[:, None]
            fm = F.apply_relative_stats(F.compute_all_features(dfm, ref_override=ref), mats["rel_stats"])
            Xs = np.nan_to_num(scaler.transform(fm[cols_all]), nan=0.0)
            out[(ho, lam)] = (Xs[na18][:, pu_pos], na18)
    return out


# ── kiểm định hoán vị cho AUC trung bình theo lượt ────────────
def permutation_test(store, model, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.mean([fast_auc(y, sc[model]) for y, sc in store])
    null = np.empty(B)
    for b in range(B):
        null[b] = np.mean([fast_auc(rng.permutation(y), sc[model]) for y, sc in store])
    return obs, float((np.sum(null >= obs) + 1) / (B + 1)), float(null.mean()), float(np.quantile(null, 0.95))


# ── đường cơ sở không giám sát trên đúng các tập kiểm tra ─────
def run_baselines(X, meta, df_feat_fixed):
    import detectors as D_
    sc = {"isolation_forest": D_.run_isolation_forest(X), "lof": D_.run_lof(X), "dbscan": D_.run_dbscan(X),
          "divergence": D_.run_divergence_rank(df_feat_fixed)}
    sc["baseline_ensemble"] = P.ensemble_rank(sc)
    rows = []
    for sp in iter_splits(meta):
        ev = np.where(sp["held"] | sp["test"])[0]; y_ev = sp["held"][ev].astype(int)
        rng = np.random.default_rng([RANDOM_SEED, 9, sp["fi"], sp["rep"]])
        for name, s in sc.items():
            rows.append(dict(variant="baseline", protocol="unsupervised", fold=sp["fold"], rep=sp["rep"], model=name,
                             **metrics(y_ev, s[ev], rng)))
    return pd.DataFrame(rows)


def summarize(runs):
    g = runs.groupby(["variant", "model"])[["roc_auc", "pr_auc", "precision_at_k", "recall_at_k"]]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    fold = runs.groupby(["variant", "model", "fold"])[["roc_auc", "pr_auc"]].mean().unstack("fold")
    fold.columns = [f"{a}_fold{b}" for a, b in fold.columns]
    return out.join(fold).reset_index()
