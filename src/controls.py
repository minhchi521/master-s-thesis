"""
controls.py — Phần 6: Đối chứng âm + Mô hình null (BẢN ĐÃ SỬA)
================================================================
Hai lỗi của bản đầu:
  1. Danh sách tỉnh "giỏi hợp pháp" dùng MÃ HÀNH CHÍNH (01/31/79/40/48) trong khi dữ liệu thi dùng
     MÃ THI (01–64). Hệ quả: "Hải Phòng"→thực ra là Quảng Bình, "Nghệ An"→Đắk Lắk, "Đà Nẵng"→Đồng Nai,
     "TP.HCM" (79) không có trong dữ liệu. Bản này dùng đúng mã và KIỂM TRA tên khớp dữ liệu.
  2. "Mô hình null" hoán vị điểm rồi đếm tỷ lệ vượt ngưỡng cố định: tỷ lệ này BẤT BIẾN dưới hoán vị
     (độ lệch chuẩn luôn 0), nên không mang thông tin. Bản này dùng hai null có nghĩa:
       (a) hoán vị NHÃN trong từng lượt LOPO → phân phối null của ROC-AUC (lopo.permutation_test);
       (b) hoán vị TỈNH: tỷ lệ gắn cờ của nhóm đối chứng so với 5 tỉnh bất kỳ cùng cỡ.
Cờ gắn theo luật "top-k" (k = số ca dương) thay vì ngưỡng 0.7 trên điểm min-max (phụ thuộc cực trị).
"""
import numpy as np
import pandas as pd

from config import OUTPUTS, DATA_PROCESSED, RANDOM_SEED
from lopo import CONTROL_PROVINCES


def _names_from_data(df):
    return df.drop_duplicates("tinh_ma").set_index("tinh_ma")["tinh_ten"].to_dict()


def verify_control_codes(df_any):
    """Cảnh báo nếu mã tỉnh đối chứng không trỏ đúng tên tỉnh trong dữ liệu."""
    real = _names_from_data(df_any)
    norm = lambda s: s.lower().replace(".", "").replace(" ", "")
    bad = [(c, n, real.get(c)) for c, n in CONTROL_PROVINCES.items() if norm(real.get(c, "")) != norm(n)]
    for c, n, r in bad:
        print(f"  [CẢNH BÁO] mã {c}: đối chứng ghi '{n}' nhưng dữ liệu là '{r}'")
    return not bad


def control_from_flags(units, model="ensemble_classic", n_null=5000):
    """units: bảng lopo_unit_flags (mỗi dòng là một đơn vị âm trong một lượt, có cột flagged)."""
    u = units[units.model == model].copy()
    ctrl = u[u.tinh_ma.isin(CONTROL_PROVINCES)]
    per = ctrl.groupby("tinh_ma").flagged.agg(n_flag="sum", n="count").assign(fpr=lambda t: t.n_flag / t.n)
    per.index = [CONTROL_PROVINCES[c] for c in per.index]
    fpr_all = u.flagged.mean(); fpr_ctrl = ctrl.flagged.mean()
    rng = np.random.default_rng(RANDOM_SEED)
    prov_rate = u.groupby("tinh_ma").flagged.mean()
    others = prov_rate.drop(index=[c for c in CONTROL_PROVINCES if c in prov_rate.index]).values
    null = np.array([rng.choice(others, len(CONTROL_PROVINCES), replace=False).mean() for _ in range(n_null)])
    obs = prov_rate.reindex(list(CONTROL_PROVINCES)).dropna().mean()
    return per, dict(fpr_all=float(fpr_all), fpr_ctrl=float(fpr_ctrl), prov_mean_ctrl=float(obs),
                     null_mean=float(null.mean()), null_p95=float(np.quantile(null, .95)),
                     p_value=float((np.sum(null >= obs) + 1) / (n_null + 1)))


def run():
    """Kiểm tra TRONG-MẪU trên điểm triển khai (chỉ để rà nhanh): tỷ lệ đơn vị của các tỉnh đối chứng nằm
    trong top 5% điểm bất thường. Số liệu chính thức (ngoài-mẫu) do evaluate.py ghi vào control_report.txt."""
    print("=== Phần 6: Đối chứng âm (kiểm tra nhanh, trong-mẫu) ===")
    p = OUTPUTS / "scores_pu.csv"
    if not p.exists():
        print("  [SKIP] chưa có scores_pu.csv"); return
    df = pd.read_csv(p, dtype={"tinh_ma": str}); verify_control_codes(df)
    lines = ["KIỂM TRA NHANH ĐỐI CHỨNG ÂM (trong-mẫu, top 5% mỗi năm)"]
    for col in ["score_pu_ensemble_classic", "score_pu_ensemble"]:
        thr = df.groupby("nam")[col].transform(lambda s: s.quantile(0.95))
        df["_f"] = df[col] >= thr
        for code, name in CONTROL_PROVINCES.items():
            s = df[df.tinh_ma == code]
            lines.append(f"  {col:28s} {name:10s} {int(s._f.sum()):3d}/{len(s):3d} gắn cờ ({s._f.mean():.1%})")
    (OUTPUTS / "control_quick.txt").write_text("\n".join(lines), encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    run()
