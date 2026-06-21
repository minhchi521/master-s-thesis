"""
controls.py — Phần 6: Đối chứng âm + mô hình null
====================================================
Mục tiêu: đảm bảo mô hình KHÔNG báo nhầm tỉnh giỏi hợp pháp;
          đo tỷ lệ báo nhầm nền (false positive rate).

Hai loại kiểm tra:
  A. Đối chứng âm thật  — tỉnh giỏi ổn định nhiều năm → mô hình KHÔNG được gắn cờ.
  B. Mô hình null/hoán vị — trộn ngẫu nhiên nhãn nhiều lần → FPR nền.

Input:  outputs/scores_baseline.csv + outputs/scores_pu.csv
Output: outputs/control_report.txt + outputs/fpr_null.csv

Ứng với Mục 7.4 trong đề cương.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DATA_PROCESSED, OUTPUTS, RANDOM_SEED, ANCHOR_PROVINCES_2018, TOPK_ALERT


# ──────────────────────────────────────────────────────────────
# Danh sách "giỏi hợp pháp" (đối chứng âm)
# Chọn tỉnh có nhiều trường chuyên, kết quả ổn định nhiều năm liên tiếp.
# Danh sách này nên được cập nhật theo dữ liệu thực tế.
# ──────────────────────────────────────────────────────────────
LEGITIMATE_STRONG_PROVINCES = {
    "01": "Hà Nội",          # trường chuyên hàng đầu, luôn top
    "31": "Hải Phòng",       # nhiều trường chuyên chất lượng
    "79": "TP. Hồ Chí Minh", # cạnh tranh cao, ổn định
    "40": "Nghệ An",         # chuyên Phan Bội Châu, luôn top tự nhiên
    "48": "Đà Nẵng",         # kết quả ổn định nhiều năm
}

# Ngưỡng: coi là "báo nhầm" khi điểm bất thường vượt ngưỡng này
ALERT_THRESHOLD = 0.7   # 70/100 điểm bất thường


# ──────────────────────────────────────────────────────────────
# A. Đối chứng âm: kiểm tra tỉnh giỏi hợp pháp
# ──────────────────────────────────────────────────────────────
def check_false_positive_on_legitimate(
    df_scores: pd.DataFrame,
    score_col: str = "score_ensemble",
    threshold: float = ALERT_THRESHOLD,
) -> pd.DataFrame:
    """
    Lọc các dòng thuộc tỉnh 'giỏi hợp pháp', kiểm tra tỷ lệ bị gắn cờ bất thường.

    Kết quả lý tưởng: 0% bị gắn cờ.
    Kết quả chấp nhận được: < 10%.
    Kết quả cần xem lại: > 20% (mô hình bị nhầm giỏi với gian lận).
    """
    df_legit = df_scores[
        df_scores["tinh_ma"].isin(LEGITIMATE_STRONG_PROVINCES.keys())
    ].copy()

    if df_legit.empty:
        print("  [WARN] Không tìm thấy tỉnh đối chứng âm trong dữ liệu.")
        return pd.DataFrame()

    # Gắn cờ khi vượt ngưỡng
    df_legit["flagged"] = df_legit[score_col] >= threshold
    df_legit["tinh_ten_check"] = df_legit["tinh_ma"].map(LEGITIMATE_STRONG_PROVINCES)

    summary = df_legit.groupby(["tinh_ma", "tinh_ten_check"])["flagged"].agg(
        n_total="count",
        n_flagged="sum"
    )
    summary["fpr"] = summary["n_flagged"] / summary["n_total"]

    return summary.reset_index()


# ──────────────────────────────────────────────────────────────
# B. Mô hình null — hoán vị nhãn ngẫu nhiên
# ──────────────────────────────────────────────────────────────
def run_null_model(
    df_scores: pd.DataFrame,
    score_col: str = "score_ensemble",
    n_permutations: int = 200,
    threshold: float = ALERT_THRESHOLD,
) -> dict:
    """
    Hoán vị ngẫu nhiên điểm bất thường nhiều lần.
    Đếm tỷ lệ mẫu ngẫu nhiên vượt ngưỡng → FPR nền.

    Nếu FPR thật ≈ FPR nền: mô hình không phân biệt được.
    Nếu FPR thật >> FPR nền: mô hình đang nhạy hơn ngẫu nhiên.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    scores = df_scores[score_col].values
    fpr_list = []

    for _ in range(n_permutations):
        shuffled = rng.permutation(scores)
        fpr = (shuffled >= threshold).mean()
        fpr_list.append(fpr)

    return {
        "null_fpr_mean": float(np.mean(fpr_list)),
        "null_fpr_std":  float(np.std(fpr_list)),
        "null_fpr_p95":  float(np.percentile(fpr_list, 95)),
        "real_fpr":      float((scores >= threshold).mean()),
    }


# ──────────────────────────────────────────────────────────────
# Tổng hợp và viết báo cáo
# ──────────────────────────────────────────────────────────────
def write_control_report(
    fp_legit: pd.DataFrame,
    null_stats: dict,
    model_name: str,
    report_path,
):
    """Viết báo cáo dạng text cho một mô hình."""
    lines = [f"\n{'='*60}", f"  Báo cáo đối chứng — mô hình: {model_name}", f"{'='*60}"]

    lines.append("\n[A] Đối chứng âm (tỉnh giỏi hợp pháp):")
    if fp_legit.empty:
        lines.append("  Không có dữ liệu.")
    else:
        for _, r in fp_legit.iterrows():
            flag = "✓ OK" if r["fpr"] < 0.1 else ("⚠ CẦN XEM" if r["fpr"] < 0.25 else "✗ SAI NGHIÊM TRỌNG")
            lines.append(
                f"  {r['tinh_ten_check']:20s}: {r['n_flagged']:3.0f}/{r['n_total']:3.0f} gắn cờ "
                f"(FPR={r['fpr']:.1%})  [{flag}]"
            )

    lines.append("\n[B] Mô hình null (hoán vị ngẫu nhiên):")
    lines.append(f"  FPR thật      : {null_stats['real_fpr']:.1%}")
    lines.append(f"  FPR null trung bình: {null_stats['null_fpr_mean']:.1%} ± {null_stats['null_fpr_std']:.1%}")
    lines.append(f"  FPR null p95  : {null_stats['null_fpr_p95']:.1%}")

    if null_stats["real_fpr"] > null_stats["null_fpr_p95"]:
        lines.append("  → FPR thật vượt ngưỡng ngẫu nhiên — mô hình đang ĐÁNH DẤU NHIỀU hơn cơ hội.")
    else:
        lines.append("  → FPR thật trong phạm vi ngẫu nhiên — không có tín hiệu rõ.")

    report_text = "\n".join(lines)
    print(report_text)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(report_text + "\n")


# ──────────────────────────────────────────────────────────────
# Hàm chính
# ──────────────────────────────────────────────────────────────
def run():
    print("=== Phần 6: Đối chứng âm + Mô hình null ===")
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    report_path = OUTPUTS / "control_report.txt"
    report_path.write_text("BÁO CÁO ĐỐI CHỨNG ÂM VÀ MÔ HÌNH NULL\n", encoding="utf-8")

    # Đọc điểm từ Phần 3 và Phần 5
    models_to_check = []
    base_path = OUTPUTS / "scores_baseline.csv"
    pu_path   = OUTPUTS / "scores_pu.csv"

    if base_path.exists():
        df_base = pd.read_csv(base_path)
        models_to_check.append(("Baseline Ensemble", df_base, "score_ensemble"))

    if pu_path.exists():
        df_pu = pd.read_csv(pu_path)
        models_to_check.append(("PU Ensemble", df_pu, "score_pu_ensemble"))

    if not models_to_check:
        print("[LỖI] Chưa có file điểm bất thường. Chạy detectors.run() và pu_models.run() trước.")
        return

    null_records = []

    for model_name, df_scores, score_col in models_to_check:
        if score_col not in df_scores.columns:
            print(f"  [SKIP] Cột {score_col} không có trong {model_name}.")
            continue

        # Kiểm tra đối chứng âm
        fp_legit = check_false_positive_on_legitimate(df_scores, score_col)

        # Mô hình null
        null_stats = run_null_model(df_scores, score_col)

        # Ghi báo cáo
        write_control_report(fp_legit, null_stats, model_name, report_path)

        null_records.append({"model": model_name, **null_stats})

    # Lưu thống kê null dạng CSV để dùng trong evaluate.py
    if null_records:
        pd.DataFrame(null_records).to_csv(
            OUTPUTS / "fpr_null.csv", index=False, encoding="utf-8-sig"
        )

    print(f"\nBáo cáo đã lưu: {report_path}")
    print("Phần 6 hoàn thành.")


if __name__ == "__main__":
    run()
