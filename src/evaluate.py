"""
evaluate.py — Phần 8: Đánh giá & độ tin cậy (BẢN ĐÃ SỬA RÒ RỈ)
================================================================
Toàn bộ số liệu hiệu năng đều là NGOÀI-MẪU (xem lopo.py và AUDIT_ROI_RI.md):
  * LOPO nghiêm ngặt: giấu 1 trong 3 tỉnh mỏ neo + giấu 18 tỉnh âm (chọn theo tỉnh), lặp 8 lần.
  * Đường cong độ nhạy: ca tiêm nhiễu (chữ ký từ tỉnh bị giấu) chấm bằng mô hình KHÔNG thấy ca giả.
  * Đối chứng âm với mã tỉnh đúng; null = hoán vị nhãn / hoán vị tỉnh.
  * Kiểm định: hoán vị cho ROC-AUC; bootstrap phân tầng theo fold cho hiệu số độ nhạy.
Đầu ra: lopo_runs.csv, lopo_summary.csv, lopo_perm_test.csv, sensitivity_curve.csv/png,
        gt1_feature_sets.csv, ablation.csv, gt2_recall_at_fpr.csv, control_report.txt,
        ket_qua_nghiem_thu.txt, eval_results.csv
"""
import numpy as np
import pandas as pd

import features as F
import pu_models as P
import lopo as L
import controls as C
from config import OUTPUTS
from data_prep import load_pho_diem


def _boot_ci(values_by_fold, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    folds = list(values_by_fold)
    bs = [np.mean(np.concatenate([rng.choice(values_by_fold[f], len(values_by_fold[f])) for f in folds])) for _ in range(B)]
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def gt1_feature_sets(mats, model="elkan_noto"):
    cols, G, meta = F.PU_FEATURE_COLS, F.FEATURE_GROUPS, mats["meta"]
    sets = {"chỉ mean+median": ["mean", "median"], "hình dạng (5 thống kê)": G["shape"],
            "đuôi cao + chéo-môn": G["tail_high"] + G["cross_subject"],
            "đuôi cao + chéo-môn + lệch quốc gia": G["tail_high"] + G["cross_subject"] + G["divergence"],
            "tất cả đặc trưng PU (không temporal)": cols}
    def run_subset(keep):
        ix = [cols.index(c) for c in keep]; X = mats["fixed"][:, ix]; out = []
        for sp in L.iter_splits(meta):
            ev = np.where(sp["held"] | sp["test"])[0]; y = sp["held"][ev].astype(int)
            tr = np.where(~(sp["held"] | sp["test"]))[0]
            s = P.FIT_FUNCS[model](X[tr], sp["other"][tr].astype(int), seed=1000 * sp["fi"] + sp["rep"])(X[ev])
            out.append(dict(fold=sp["fold"], rep=sp["rep"], **L.metrics(y, s, np.random.default_rng([0, 5, sp["fi"], sp["rep"]]))))
        return pd.DataFrame(out)
    res = {k: run_subset(v) for k, v in sets.items()}
    full = res["tất cả đặc trưng PU (không temporal)"]
    rows = [dict(feature_set=k, roc_auc=v.roc_auc.mean(), pr_auc=v.pr_auc.mean(),
                 d_roc_vs_mean_median=(v.roc_auc - res["chỉ mean+median"].roc_auc).mean(),
                 ci_low=_boot_ci({f: (v.roc_auc - res["chỉ mean+median"].roc_auc)[v.fold == f].values for f in v.fold.unique()})[0],
                 ci_high=_boot_ci({f: (v.roc_auc - res["chỉ mean+median"].roc_auc)[v.fold == f].values for f in v.fold.unique()})[1])
            for k, v in res.items()]
    abl = []
    for g in ["tail_high", "shape", "divergence", "cross_subject", "bunching"]:
        r = run_subset([c for c in cols if c not in G[g]])
        abl.append(dict(group_removed=g, roc_auc=r.roc_auc.mean(), pr_auc=r.pr_auc.mean(),
                        d_roc=r.roc_auc.mean() - full.roc_auc.mean(), d_pr=r.pr_auc.mean() - full.pr_auc.mean()))
    abl.append(dict(group_removed="none (full)", roc_auc=full.roc_auc.mean(), pr_auc=full.pr_auc.mean(), d_roc=0.0, d_pr=0.0))
    return pd.DataFrame(rows), pd.DataFrame(abl)


def recall_at_fpr(y, s, fpr=0.05):
    return float((s[y == 1] > np.quantile(s[y == 0], 1 - fpr)).mean())


def gt2_table(store, mats):
    import detectors as D_
    meta = mats["meta"]; X = mats["fixed"]
    base = {"divergence": D_.run_divergence_rank(mats["raw_fixed"]), "isolation_forest": D_.run_isolation_forest(X)}
    rows = []
    for sp, (y, sc) in zip(L.iter_splits(meta), store):
        ev = np.where(sp["held"] | sp["test"])[0]
        r = dict(fold=sp["fold"], rep=sp["rep"])
        for m in P.MODEL_NAMES + ["ensemble_classic", "ensemble_modern", "ensemble"]:
            r[m] = recall_at_fpr(y, sc[m])
        for b, s in base.items():
            r[b] = recall_at_fpr(y, s[ev])
        rows.append(r)
    return pd.DataFrame(rows)


def timing(mats):
    import time
    meta = mats["meta"]; X = mats["fixed"]
    sp = next(L.iter_splits(meta)); tr = np.where(~(sp["held"] | sp["test"]))[0]; y = sp["other"][tr].astype(int)
    new = np.where((meta.nam == 2024).values)[0]       # mô phỏng "phổ điểm năm mới": 567 đơn vị
    rows = []
    for m in P.MODEL_NAMES:
        t0 = time.time(); f = P.FIT_FUNCS[m](X[tr], y, seed=0); t1 = time.time(); f(X[new]); t2 = time.time()
        rows.append(dict(model=m, fit_seconds=t1 - t0, score_new_year_seconds=t2 - t1))
    return pd.DataFrame(rows)


def run():
    print("=== Phần 8: Đánh giá & Độ tin cậy (LOPO nghiêm ngặt) ===")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df_pho = load_pho_diem()
    mats = L.build_matrices(df_pho); meta = mats["meta"]
    C.verify_control_codes(df_pho)

    print("Dựng ca tiêm nhiễu...")
    pert = L.build_perturbed(df_pho, mats)
    print("Chạy LOPO nghiêm ngặt (5 mô hình PU × 24 lượt)...")
    runs, units, sens, store = L.run_variant(mats["fixed"], meta, "strict", "fixed_strict", pert=pert, keep=True)
    base = L.run_baselines(mats["fixed"], meta, mats["raw_fixed"])
    all_runs = pd.concat([runs, base], ignore_index=True)
    all_runs.to_csv(OUTPUTS / "lopo_runs.csv", index=False, encoding="utf-8-sig")
    summ = L.summarize(all_runs); summ.to_csv(OUTPUTS / "lopo_summary.csv", index=False, encoding="utf-8-sig")
    units.to_csv(OUTPUTS / "lopo_unit_flags.csv", index=False, encoding="utf-8-sig")

    print("Kiểm định hoán vị...")
    perm = pd.DataFrame([dict(model=m, **dict(zip(["auc_obs", "p_perm", "auc_null_mean", "auc_null_p95"], L.permutation_test(store, m, B=1000))))
                         for m in P.MODEL_NAMES + ["ensemble_classic", "ensemble_modern", "ensemble"]])
    perm.to_csv(OUTPUTS / "lopo_perm_test.csv", index=False, encoding="utf-8-sig")

    sens.to_csv(OUTPUTS / "sensitivity_runs.csv", index=False, encoding="utf-8-sig")
    sv = sens.groupby(["model", "intensity"])[["recall_fake", "recall_real"]].mean().reset_index()
    sv.to_csv(OUTPUTS / "sensitivity_curve.csv", index=False, encoding="utf-8-sig")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in ["ensemble_classic", "ensemble_modern", "elkan_noto", "pu_bagging"]:
            t = sv[sv.model == m]; ax.plot(t.intensity, t.recall_fake, marker="o", label=m)
        ax.axhline(sv[sv.model == "ensemble_classic"].recall_real.iloc[0], ls="--", c="gray", label="ca thật bị giấu (ensemble_classic)")
        ax.set_xlabel("Cường độ tiêm nhiễu λ"); ax.set_ylabel("Độ nhạy tại tỷ lệ báo nhầm 5%"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUTPUTS / "sensitivity_curve.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("  [WARN] không vẽ được biểu đồ:", e)

    print("GT1 / ablation...")
    gt1, abl = gt1_feature_sets(mats); gt1.to_csv(OUTPUTS / "gt1_feature_sets.csv", index=False, encoding="utf-8-sig")
    abl.to_csv(OUTPUTS / "ablation.csv", index=False, encoding="utf-8-sig")

    print("GT2 (độ nhạy tại FPR 5%)...")
    gt2 = gt2_table(store, mats); gt2.to_csv(OUTPUTS / "gt2_recall_at_fpr.csv", index=False, encoding="utf-8-sig")
    tim = timing(mats); tim.to_csv(OUTPUTS / "chi_phi_ap_dung.csv", index=False, encoding="utf-8-sig")

    # Đối chứng âm (ngoài-mẫu)
    per, ctl = C.control_from_flags(units, "ensemble_classic")
    lines = ["BÁO CÁO ĐỐI CHỨNG ÂM (NGOÀI-MẪU, mô hình ensemble_classic, cờ = top-k mỗi lượt)", ""]
    for name, r in per.iterrows():
        lines.append(f"  {name:10s}: {int(r.n_flag):3d}/{int(r.n):3d} gắn cờ (FPR = {r.fpr:.1%})")
    lines += ["", f"  FPR trên toàn bộ đơn vị âm kiểm tra : {ctl['fpr_all']:.1%}",
              f"  FPR nhóm đối chứng (5 tỉnh)          : {ctl['fpr_ctrl']:.1%}",
              f"  Null hoán vị tỉnh: TB {ctl['null_mean']:.1%}, p95 {ctl['null_p95']:.1%}; tỷ lệ trung bình theo tỉnh của nhóm đối chứng {ctl['prov_mean_ctrl']:.1%}; p = {ctl['p_value']:.3f}"]
    (OUTPUTS / "control_report.txt").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))

    # Báo cáo nghiệm thu
    fs = runs; g = lambda m, c: fs[fs.model == m][c].mean()
    pc = fs[fs.model == "ensemble_classic"].set_index(["fold", "rep"]); pm = fs[fs.model == "ensemble_modern"].set_index(["fold", "rep"])
    rep = ["KẾT QUẢ ĐỊNH LƯỢNG (LOPO nghiêm ngặt, trung bình 24 lượt = 3 fold × 8 lần chia)", ""]
    for m in P.MODEL_NAMES + ["ensemble_classic", "ensemble_modern", "ensemble"]:
        rep.append(f"  {m:18s} ROC-AUC {g(m,'roc_auc'):.3f}  PR-AUC {g(m,'pr_auc'):.3f}  Recall@k {g(m,'recall_at_k'):.3f}")
    for m in ["divergence", "isolation_forest", "lof", "dbscan"]:
        b = base[base.model == m]; rep.append(f"  [baseline] {m:16s} ROC-AUC {b.roc_auc.mean():.3f}  PR-AUC {b.pr_auc.mean():.3f}  Recall@k {b.recall_at_k.mean():.3f}")
    rep += ["", "CH4/GT4 (mới − cổ điển, ensemble):"]
    for met in ["roc_auc", "pr_auc", "recall_at_k"]:
        d = pm[met] - pc[met]; rep.append(f"  {met}: {d.mean():+.3f}; mô hình mới hơn ở {(d>0).sum()}/{len(d)} lượt; theo fold {d.groupby('fold').mean().round(3).to_dict()}")
    rep += ["", "GT2 (độ nhạy tại FPR 5%, hiệu số so với baseline divergence):"]
    for m in ["elkan_noto", "pu_bagging", "ensemble_classic", "ensemble_modern", "ensemble"]:
        dd = gt2[m] - gt2["divergence"]; lo, hi = _boot_ci({f: dd[gt2.fold == f].values for f in gt2.fold.unique()})
        rep.append(f"  {m:18s} {dd.mean():+.3f}  CI95 [{lo:+.3f}, {hi:+.3f}]")
    (OUTPUTS / "ket_qua_nghiem_thu.txt").write_text("\n".join(rep), encoding="utf-8"); print("\n".join(rep))

    df_eval = all_runs.groupby(["variant", "model"])[["roc_auc", "pr_auc", "precision_at_k", "recall_at_k"]].mean().reset_index()
    df_eval.to_csv(OUTPUTS / "eval_results.csv", index=False, encoding="utf-8-sig")
    print("\nPhần 8 hoàn thành.")
    return df_eval


if __name__ == "__main__":
    run()
