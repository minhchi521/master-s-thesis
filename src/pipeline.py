"""
pipeline.py — Phần 9: Pipeline đóng gói đầu-cuối
==================================================
Mục tiêu: một lệnh duy nhất chạy toàn bộ:
    phổ điểm mới → danh sách tỉnh theo dõi + phiếu sức khỏe

Ứng với Mục 8 trong đề cương.

Cách dùng:
    # Chạy toàn bộ (tất cả năm, tất cả môn)
    python pipeline.py

    # Chọn năm cụ thể
    python pipeline.py --years 2023 2024

    # Chọn môn
    python pipeline.py --subjects toan ngu_van

    # Chọn mô hình
    python pipeline.py --model baseline   # hoặc pu

    # Chạy từ một bước cụ thể (bỏ qua các bước đã có output)
    python pipeline.py --from-step 3
"""

import argparse
import time
import pathlib
import pandas as pd

from config import (
    DATA_PROCESSED, OUTPUTS, YEARS_MAIN, YEAR_HOLDOUT, SUBJECTS, TOPK_ALERT
)


# ──────────────────────────────────────────────────────────────
# Định nghĩa các bước pipeline và điều kiện skip
# ──────────────────────────────────────────────────────────────
STEPS = [
    {
        "id": 1,
        "name": "Phần 1 — Chuẩn bị dữ liệu",
        "module": "data_prep",
        "output_check": DATA_PROCESSED / "pho_diem_tat_ca.csv",
    },
    {
        "id": 2,
        "name": "Phần 2 — Thiết kế đặc trưng",
        "module": "features",
        "output_check": DATA_PROCESSED / "dac_trung.csv",
    },
    {
        "id": 3,
        "name": "Phần 3 — Baseline không giám sát",
        "module": "detectors",
        "output_check": OUTPUTS / "scores_baseline.csv",
    },
    {
        "id": 4,
        "name": "Phần 4 — Máy tiêm nhiễu",
        "module": "injection",
        "output_check": DATA_PROCESSED / "synthetic_labeled.csv",
    },
    {
        "id": 5,
        "name": "Phần 5 — PU Learning",
        "module": "pu_models",
        "output_check": OUTPUTS / "scores_pu.csv",
    },
    {
        "id": 6,
        "name": "Phần 6 — Đối chứng âm & Null model",
        "module": "controls",
        "output_check": OUTPUTS / "control_report.txt",
    },
    {
        "id": 7,
        "name": "Phần 7 — Giải thích XAI",
        "module": "explain",
        "output_check": OUTPUTS / "shap_summary.png",
    },
    {
        "id": 8,
        "name": "Phần 8 — Đánh giá & Độ tin cậy",
        "module": "evaluate",
        "output_check": OUTPUTS / "eval_results.csv",
    },
]


# ──────────────────────────────────────────────────────────────
# Chạy một bước
# ──────────────────────────────────────────────────────────────
def _run_step(step: dict, force: bool = False):
    """
    Chạy một module. Bỏ qua nếu output đã tồn tại và không bắt buộc chạy lại.
    """
    out_check = step.get("output_check")
    if not force and out_check and pathlib.Path(out_check).exists():
        print(f"  [SKIP] {step['name']} — output đã tồn tại: {out_check.name}")
        return True

    print(f"\n{'─'*60}")
    print(f"  Bắt đầu: {step['name']}")
    print(f"{'─'*60}")
    t0 = time.time()

    try:
        # Import động để tránh vòng tròn import khi khởi động
        import importlib
        mod = importlib.import_module(step["module"])
        mod.run()
        elapsed = time.time() - t0
        print(f"  ✓ Hoàn thành trong {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"  ✗ LỖI ở {step['name']}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────────
# Xuất danh sách theo dõi cuối cùng
# ──────────────────────────────────────────────────────────────
def export_watchlist(
    model: str = "baseline",
    top_k: int = TOPK_ALERT,
    years: list[int] | None = None,
    subjects: list[str] | None = None,
):
    """
    Đọc điểm bất thường, lọc theo năm/môn nếu cần, xuất top_k tỉnh đáng theo dõi.
    """
    score_file = OUTPUTS / ("scores_baseline.csv" if model == "baseline" else "scores_pu.csv")
    score_col  = "score_ensemble" if model == "baseline" else "score_pu_ensemble"

    if not score_file.exists():
        print(f"[WARN] Không tìm thấy file điểm: {score_file}")
        return

    df = pd.read_csv(score_file)

    # Lọc năm
    if years:
        df = df[df["nam"].isin(years)]
    # Lọc môn
    if subjects:
        df = df[df["mon"].isin(subjects)]

    if df.empty or score_col not in df.columns:
        print("[WARN] Không có dữ liệu phù hợp để xuất danh sách.")
        return

    df_top = df.nlargest(top_k, score_col)[[
        "tinh_ma", "tinh_ten", "mon", "nam", "dot", score_col, "n_thi"
    ]].copy()
    df_top.columns = ["Mã tỉnh", "Tên tỉnh", "Môn", "Năm", "Đợt",
                      "Điểm bất thường", "Số thí sinh"]
    df_top["Điểm bất thường"] = df_top["Điểm bất thường"].round(4)

    out_path = OUTPUTS / "watchlist.csv"
    df_top.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  DANH SÁCH THEO DÕI — Top {top_k} đơn vị bất thường nhất")
    print(f"  Mô hình: {model}  |  Năm: {years or 'tất cả'}  |  Môn: {subjects or 'tất cả'}")
    print(f"{'='*60}")
    print(df_top.to_string(index=False))
    print(f"\nĐã lưu: {out_path}")


# ──────────────────────────────────────────────────────────────
# Hàm chính
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline phát hiện bất thường phổ điểm THPT"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Danh sách năm cần xử lý (mặc định: 2018–2024)"
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Danh sách môn (mặc định: tất cả 9 môn)"
    )
    parser.add_argument(
        "--model", choices=["baseline", "pu"], default="baseline",
        help="Mô hình dùng để xuất watchlist (mặc định: baseline)"
    )
    parser.add_argument(
        "--from-step", type=int, default=1, dest="from_step",
        help="Bắt đầu từ bước nào (1–8, mặc định: 1)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bắt buộc chạy lại kể cả khi output đã tồn tại"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOPK_ALERT, dest="top_k",
        help=f"Số đơn vị trong danh sách theo dõi (mặc định: {TOPK_ALERT})"
    )
    args = parser.parse_args()

    OUTPUTS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PIPELINE PHÁT HIỆN BẤT THƯỜNG PHỔ ĐIỂM THPT")
    print("=" * 60)
    print(f"  Năm    : {args.years or YEARS_MAIN}")
    print(f"  Môn    : {args.subjects or SUBJECTS}")
    print(f"  Mô hình: {args.model}")
    print(f"  Từ bước: {args.from_step}")
    print("=" * 60)

    t_total = time.time()
    success_all = True

    # Chạy lần lượt các bước
    for step in STEPS:
        if step["id"] < args.from_step:
            print(f"  [BỎ QUA] {step['name']} (from-step={args.from_step})")
            continue

        # Bỏ qua Phần 5 (PU) nếu chọn model=baseline
        if step["id"] == 5 and args.model == "baseline":
            print(f"  [SKIP] {step['name']} (model=baseline, không cần PU)")
            continue

        ok = _run_step(step, force=args.force)
        if not ok:
            print(f"\n[DỪNG] Pipeline thất bại ở bước {step['id']}. Kiểm tra lỗi trên.")
            success_all = False
            break

    if success_all:
        # Xuất danh sách theo dõi
        export_watchlist(
            model=args.model,
            top_k=args.top_k,
            years=args.years,
            subjects=args.subjects,
        )

        elapsed_total = time.time() - t_total
        print(f"\n{'='*60}")
        print(f"  Pipeline hoàn thành trong {elapsed_total/60:.1f} phút")
        print(f"  Kết quả lưu tại: {OUTPUTS}/")
        print(f"{'='*60}")
    else:
        print("\nPipeline kết thúc sớm do lỗi.")


if __name__ == "__main__":
    main()
