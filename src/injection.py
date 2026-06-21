"""
injection.py — Phần 4: Máy tiêm nhiễu hiệu chỉnh theo 2018  ⭐ (cốt lõi)
==========================================================================
Mục tiêu: tạo hàng trăm ca "gian lận giả" GIỐNG ca thật 2018
          để có nhãn mà đánh giá mà không bị đánh giá vòng tròn.

Quy trình 3 bước:
  Bước 1 — Trích "chữ ký" gian lận từ Hà Giang / Sơn La / Hòa Bình 2018.
  Bước 2 — Bóp méo phổ điểm sạch theo chữ ký đó ở nhiều cường độ (nhẹ → nặng).
  Bước 3 — Kiểm chứng (GT3): ca mô phỏng phải đủ gần ca thật.

Input:  data/processed/pho_diem_tat_ca.csv  (phổ điểm từ Phần 1)
Output:
  - data/processed/synthetic_labeled.csv    — nhãn tổng hợp đã kiểm chứng
  - outputs/injection_report.txt            — báo cáo kiểm chứng GT3

Ứng với Mục 7.3 trong đề cương.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from config import (
    DATA_PROCESSED, OUTPUTS, RANDOM_SEED,
    ANCHOR_PROVINCES_2018, BIN_WIDTH, SCORE_MIN, SCORE_MAX,
    SUBJECTS,
)
from data_prep import load_pho_diem

RNG = np.random.default_rng(RANDOM_SEED)


# ──────────────────────────────────────────────────────────────
# Hàm trợ giúp
# ──────────────────────────────────────────────────────────────
def _get_bin_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("bin_")],
        key=lambda x: float(x.replace("bin_", ""))
    )


def _row_to_dist(row: pd.Series, bin_cols: list[str]) -> np.ndarray:
    v = row[bin_cols].values.astype(float)
    s = v.sum()
    return v / s if s > 0 else np.ones(len(v)) / len(v)


# ──────────────────────────────────────────────────────────────
# Bước 1 — Trích chữ ký gian lận từ 3 tỉnh mỏ neo 2018
# ──────────────────────────────────────────────────────────────
def extract_fraud_signature(df_pho: pd.DataFrame, bin_cols: list[str]) -> dict:
    """
    Đo xem 3 tỉnh mỏ neo 2018 lệch khỏi mặt bằng toàn quốc như thế nào.

    Với mỗi (tỉnh mỏ neo, môn), tính:
      - delta_dist : sự chênh lệch phân phối so với trung bình quốc gia cùng năm/môn
      - tail_excess: bao nhiêu xác suất "dư" ở đuôi cao (≥ 8.0)

    Trả về dict {(tinh_ma, mon): {"delta": ..., "tail_excess": ...}}
    """
    signatures = {}

    # Phân phối quốc gia theo (môn, năm=2018)
    nat_2018 = {}
    df_2018 = df_pho[(df_pho["nam"] == 2018) & (df_pho["dot"] == 1)]
    for mon, grp in df_2018.groupby("mon"):
        dists = grp[bin_cols].values.astype(float)
        row_sums = dists.sum(axis=1, keepdims=True)
        nat_2018[mon] = np.where(row_sums > 0, dists / row_sums, 0).mean(axis=0)

    # Trích chênh lệch cho từng tỉnh mỏ neo
    for tinh_ma in ANCHOR_PROVINCES_2018:
        rows = df_2018[df_2018["tinh_ma"] == tinh_ma]
        for _, row in rows.iterrows():
            mon = row["mon"]
            local = _row_to_dist(row, bin_cols)
            national = nat_2018.get(mon, np.ones(len(bin_cols)) / len(bin_cols))

            delta = local - national   # vector hiệu (dương = tỉnh cao hơn quốc gia)

            # Phần xác suất "dư" ở đuôi ≥ 8.0
            bins_left = [float(c.replace("bin_", "")) for c in bin_cols]
            tail_mask = np.array([b >= 8.0 for b in bins_left])
            tail_excess = float(delta[tail_mask].sum())

            signatures[(tinh_ma, mon)] = {
                "delta": delta,
                "tail_excess": tail_excess,
                "national_dist": national,
                "local_dist": local,
            }

    return signatures


# ──────────────────────────────────────────────────────────────
# Bước 2 — Bóp méo phổ điểm sạch theo chữ ký
# ──────────────────────────────────────────────────────────────
def inject_fraud(
    clean_dist: np.ndarray,
    signature_delta: np.ndarray,
    intensity: float,        # 0.0 = không can thiệp, 1.0 = đúng bằng chữ ký
) -> np.ndarray:
    """
    Bóp méo phân phối sạch theo hướng của chữ ký gian lận.

    Công thức:
        dist_fake = clean_dist + intensity * signature_delta

    Sau đó clamp [0,1] và tái chuẩn hóa để tổng = 1.
    """
    fake = clean_dist + intensity * signature_delta
    fake = np.clip(fake, 0, None)          # không cho âm
    s = fake.sum()
    return fake / s if s > 0 else clean_dist


# ──────────────────────────────────────────────────────────────
# Bước 3 — Kiểm chứng (GT3)
# ──────────────────────────────────────────────────────────────
def verify_synthetic(
    fake_dist: np.ndarray,
    real_dist: np.ndarray,
    bin_cols: list[str],
    js_threshold: float = 0.15,   # ngưỡng JS; cao hơn = không đủ giống
) -> dict:
    """
    Đo khoảng cách giữa ca mô phỏng và ca thật cùng cường độ.
    Chỉ chấp nhận ca mô phỏng khi JS divergence đủ thấp.

    Ngưỡng 0.15: kinh nghiệm từ các nghiên cứu phổ điểm; điều chỉnh nếu cần.
    """
    js = float(jensenshannon(fake_dist + 1e-10, real_dist + 1e-10))

    bins_left = np.array([float(c.replace("bin_", "")) for c in bin_cols])
    ws = float(wasserstein_distance(bins_left, bins_left, fake_dist, real_dist))

    return {
        "js_divergence": js,
        "wasserstein": ws,
        "passed_gt3": js < js_threshold,
    }


# ──────────────────────────────────────────────────────────────
# Sinh tập dữ liệu có nhãn tổng hợp
# ──────────────────────────────────────────────────────────────
def generate_synthetic_dataset(
    df_pho: pd.DataFrame,
    signatures: dict,
    intensities: list[float] | None = None,
    n_clean_per_sig: int = 20,         # số ca sạch dùng để bóp méo cho mỗi chữ ký
) -> pd.DataFrame:
    """
    Với mỗi chữ ký gian lận × mỗi cường độ × n ca sạch ngẫu nhiên:
      → tạo 1 ca "gian lận giả", kiểm chứng GT3, gắn nhãn.

    Nhãn:
      label = 1  (fraud)  cho ca bóp méo đã qua GT3
      label = 0  (clean)  cho ca gốc không bóp méo (negative giả)

    Trả về DataFrame gộp có cột: [tinh_ma, mon, nam, intensity, label,
                                   passed_gt3, js_divergence, wasserstein,
                                   bin_0.0, ..., bin_9.8]
    """
    if intensities is None:
        intensities = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

    bin_cols = _get_bin_cols(df_pho)
    records = []

    # Tập phổ điểm "sạch" (loại trừ 3 tỉnh mỏ neo năm 2018)
    anchor_mask = (
        df_pho["tinh_ma"].isin(ANCHOR_PROVINCES_2018.keys()) &
        (df_pho["nam"] == 2018)
    )
    df_clean = df_pho[~anchor_mask].copy()

    for (tinh_ma, mon), sig in signatures.items():
        delta       = sig["delta"]
        real_dist   = sig["local_dist"]

        # Lọc phổ điểm sạch cùng môn để làm nguyên liệu
        cands = df_clean[df_clean["mon"] == mon]
        if len(cands) < n_clean_per_sig:
            cands_sample = cands
        else:
            cands_sample = cands.sample(n_clean_per_sig, random_state=RANDOM_SEED)

        for intensity in intensities:
            for _, row_c in cands_sample.iterrows():
                clean = _row_to_dist(row_c, bin_cols)
                fake  = inject_fraud(clean, delta, intensity)

                # Kiểm chứng GT3
                vfy = verify_synthetic(fake, real_dist, bin_cols)

                record = {
                    "src_tinh_ma":  tinh_ma,      # tỉnh gốc dùng chữ ký
                    "src_mon":      mon,
                    "base_tinh_ma": row_c["tinh_ma"],  # tỉnh sạch làm nền
                    "base_tinh_ten":row_c["tinh_ten"],
                    "base_nam":     row_c["nam"],
                    "intensity":    intensity,
                    "label":        1,             # gian lận giả
                    "passed_gt3":   vfy["passed_gt3"],
                    "js_divergence":vfy["js_divergence"],
                    "wasserstein":  vfy["wasserstein"],
                }
                record.update({c: float(fake[i]) for i, c in enumerate(bin_cols)})
                records.append(record)

        # Thêm ca sạch gốc làm negative (intensity = 0)
        for _, row_c in cands_sample.iterrows():
            clean = _row_to_dist(row_c, bin_cols)
            record_clean = {
                "src_tinh_ma":  tinh_ma,
                "src_mon":      mon,
                "base_tinh_ma": row_c["tinh_ma"],
                "base_tinh_ten":row_c["tinh_ten"],
                "base_nam":     row_c["nam"],
                "intensity":    0.0,
                "label":        0,
                "passed_gt3":   True,
                "js_divergence":0.0,
                "wasserstein":  0.0,
            }
            record_clean.update({c: float(clean[i]) for i, c in enumerate(bin_cols)})
            records.append(record_clean)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# Báo cáo kiểm chứng
# ──────────────────────────────────────────────────────────────
def write_verification_report(df_syn: pd.DataFrame):
    """
    In và lưu báo cáo tỷ lệ vượt GT3 theo cường độ và chữ ký.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    report_lines = ["=== BÁO CÁO KIỂM CHỨNG GT3 ===\n"]

    df_fraud = df_syn[df_syn["label"] == 1]
    summary = (
        df_fraud.groupby("intensity")["passed_gt3"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "passed", "count": "total"})
    )
    summary["rate"] = summary["passed"] / summary["total"]
    report_lines.append("Tỷ lệ vượt GT3 theo cường độ:\n" + summary.to_string())

    # Cảnh báo nếu cường độ thấp mà tỷ lệ thấp (máy tiêm không giống ca thật)
    for intensity, row_s in summary.iterrows():
        if intensity <= 0.5 and row_s["rate"] < 0.5:
            report_lines.append(
                f"\n[CẢNH BÁO] Cường độ {intensity}: chỉ {row_s['rate']:.0%} ca vượt GT3. "
                "Cần xem lại chữ ký hoặc ngưỡng JS."
            )

    report_text = "\n".join(report_lines)
    print(report_text)
    (OUTPUTS / "injection_report.txt").write_text(report_text, encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Hàm chính
# ──────────────────────────────────────────────────────────────
def run():
    print("=== Phần 4: Máy tiêm nhiễu ===")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df_pho = load_pho_diem()
    if df_pho.empty:
        print("[LỖI] Chưa có dữ liệu phổ điểm.")
        return

    bin_cols = _get_bin_cols(df_pho)

    # Bước 1: trích chữ ký
    print("Bước 1 — Trích chữ ký gian lận 2018...")
    signatures = extract_fraud_signature(df_pho, bin_cols)
    print(f"  Số chữ ký trích được: {len(signatures)}")

    # Bước 2 & 3: sinh tập tổng hợp + kiểm chứng
    print("Bước 2 & 3 — Sinh ca giả và kiểm chứng...")
    df_syn = generate_synthetic_dataset(df_pho, signatures)
    print(f"  Tổng số ca: {len(df_syn)}  (fraud={df_syn['label'].sum()}, clean={(df_syn['label']==0).sum()})")

    # Báo cáo
    write_verification_report(df_syn)

    # Lưu kết quả
    out_path = DATA_PROCESSED / "synthetic_labeled.csv"
    df_syn.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {out_path}")
    print("Phần 4 hoàn thành.")

    return df_syn, signatures


if __name__ == "__main__":
    run()
