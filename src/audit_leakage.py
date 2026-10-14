"""
audit_leakage.py — Đo mức ảnh hưởng của từng lỗi rò rỉ (chạy ~10 phút)
=======================================================================
Chạy 4 phiên bản trên CÙNG các tập kiểm tra LOPO (3 fold × 8 lần chia):
  legacy_legacy : đặc trưng cũ  + quy trình cũ  (tái hiện bản đầu)
  fixed_legacy  : đặc trưng mới + quy trình cũ  (hiệu ứng của sửa đặc trưng)
  legacy_strict : đặc trưng cũ  + quy trình mới (hiệu ứng của sửa quy trình)
  fixed_strict  : đặc trưng mới + quy trình mới (bản đã sửa — số liệu dùng cho luận văn)
Kết quả: outputs/audit_attribution.csv
Dùng:  python audit_leakage.py
"""
import numpy as np, pandas as pd
import lopo as L
from config import OUTPUTS
from data_prep import load_pho_diem

if __name__ == "__main__":
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    d = load_pho_diem(); mats = L.build_matrices(d); frames = []
    for tag in ["legacy_legacy", "fixed_legacy", "legacy_strict", "fixed_strict"]:
        feat, proto = tag.split("_")
        runs, *_ = L.run_variant(mats[feat], mats["meta"], proto, tag)
        frames.append(runs)
    R = pd.concat(frames)
    tab = R.groupby(["model", "variant"])[["roc_auc", "pr_auc"]].mean().unstack("variant").round(3)
    tab.to_csv(OUTPUTS / "audit_attribution.csv", encoding="utf-8-sig")
    R.to_csv(OUTPUTS / "audit_runs.csv", index=False, encoding="utf-8-sig")
    print(tab.to_string())
