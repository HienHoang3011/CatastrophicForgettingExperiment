import argparse
from src.core import (
    load_saved_datasets, train_model, 
    evaluate_reasoning, evaluate_general, delete_model, clean_memory
)
from transformers import AutoTokenizer

def main():
    # 1. SETUP ARGUMENTS (CẤU HÌNH BẰNG TERMINAL)
    parser = argparse.ArgumentParser(description="Hệ thống Huấn luyện và Đánh giá LLM chống Forgetting")
    
    # Cấu hình Model & Data
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace Model ID")
    parser.add_argument(
        "--dataset",
        type=str,
        default="AI-MO/NuminaMath-CoT",
        help="Đường dẫn file JSON local hoặc Hugging Face dataset ID"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Split khi dùng Hugging Face dataset (vd: train)"
    )
    parser.add_argument("--samples", type=int, default=50000, help="Số lượng mẫu muốn load (vd: 10000)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size khi chạy eval")
    parser.add_argument("--train-epochs", type=float, default=1.0, help="Số epoch train cho SFT/Steered")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate cho SFT/Steered")
    parser.add_argument("--eval-max-new-tokens", type=int, default=2048, help="Số token sinh tối đa khi evaluate_reasoning")
    parser.add_argument("--eval-preview-samples", type=int, default=10, help="Số mẫu in preview trong evaluate_reasoning")
    parser.add_argument(
        "--system-prompt",
        "--eval-system-prompt",
        dest="system_prompt",
        type=str,
        default=(
            "You are a rigorous math reasoning assistant. "
            "Reason step by step internally, then return only one final answer in the form \\boxed{...}. "
            "Do not include extra text after the boxed answer."
        ),
        help="System prompt dùng chung cho cả train formatting và evaluate_reasoning"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "sft", "steered", "sft-steered", "all"],
        help="Chọn chế độ chạy nhanh: baseline | sft | steered | sft-steered | all"
    )
    
    # Cấu hình Bật/Tắt Experiment (Mặc định là False, truyền cờ vào sẽ thành True)
    parser.add_argument("--run-baseline", action="store_true", help="Chỉ chạy Zero-shot Evaluation cho Base model")
    parser.add_argument("--run-sft", action="store_true", help="Chỉ chạy Standard SFT")
    parser.add_argument("--run-steered", action="store_true", help="Chỉ chạy Steered SFT (Thuật toán của bạn)")
    parser.add_argument("--run-all", action="store_true", help="Chạy toàn bộ 3 thí nghiệm")

    args = parser.parse_args()

    # Ưu tiên --mode nếu có, giữ tương thích ngược cho các cờ cũ
    if args.mode:
        args.run_baseline = args.mode in ["baseline", "all"]
        args.run_sft = args.mode in ["sft", "sft-steered", "all"]
        args.run_steered = args.mode in ["steered", "sft-steered", "all"]
    elif args.run_all:
        args.run_baseline = args.run_sft = args.run_steered = True

    # Check nếu user chạy script mà quên chọn experiment
    if not (args.run_baseline or args.run_sft or args.run_steered):
        print("❌ LỖI: Bạn chưa chọn Thí nghiệm nào để chạy!")
        print("💡 Gợi ý: Dùng --mode sft / --mode steered / --mode all")
        return
        
    # Luôn bật đánh giá mô hình gốc (baseline) để làm mốc so sánh trước finetune
    args.run_baseline = True

    # 2. CHUẨN BỊ DỮ LIỆU CHUNG
    print(f"🚀 Khởi chạy Pipeline với Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_ds, eval_ds, test_ds = load_saved_datasets(
        tokenizer,
        "processed_data/train.jsonl",
        "processed_data/eval.jsonl",
        "processed_data/test.jsonl",
        args.system_prompt
    )
    del tokenizer
    clean_memory()

    report = {}

    # 3. PHASE 1: TRAINING
    sft_dir = "./model_sft"
    steer_dir = "./model_steer"

    if args.run_sft:
        print("\n" + "="*50 + "\n[PHASE 1] TRAINING: STANDARD SFT\n" + "="*50)
        train_model(
            args.model,
            train_ds,
            eval_ds,
            sft_dir,
            use_steer=False,
            learning_rate=args.learning_rate,
            num_train_epochs=args.train_epochs,
        )

    if args.run_steered:
        print("\n" + "="*50 + "\n[PHASE 1] TRAINING: STEERED SFT\n" + "="*50)
        train_model(
            args.model,
            train_ds,
            eval_ds,
            steer_dir,
            use_steer=True,
            learning_rate=args.learning_rate,
            num_train_epochs=args.train_epochs,
        )

    # 4. PHASE 2: EVALUATION
    if args.run_baseline:
        print("\n" + "="*50 + "\n[PHASE 2] EVALUATION: BASELINE (ZERO-SHOT)\n" + "="*50)
        print("[INFO] Tiến hành đánh giá tập reasoning trên model gốc.")
        base_rsn = evaluate_reasoning(
            args.model,
            test_ds,
            args.batch_size,
            args.system_prompt,
            args.eval_max_new_tokens,
            args.eval_preview_samples,
        )
        base_hs, base_mmlu, base_gsm8k = evaluate_general(args.model)
        report["Baseline"] = {
            "Reasoning": base_rsn, "HellaSwag": base_hs, "MMLU": base_mmlu, "GSM8K": base_gsm8k
        }
        print(f"\n[KẾT QUẢ TRUNG GIAN - BASELINE]")
        print(f"Reasoning: {base_rsn:.2f}% | HellaSwag: {base_hs:.2f}% | MMLU: {base_mmlu:.2f}% | GSM8K: {base_gsm8k:.2f}%")

    if args.run_sft:
        print("\n" + "="*50 + "\n[PHASE 2] EVALUATION: STANDARD SFT\n" + "="*50)
        sft_rsn = evaluate_reasoning(
            sft_dir,
            test_ds,
            args.batch_size,
            args.system_prompt,
            args.eval_max_new_tokens,
            args.eval_preview_samples,
        )
        sft_hs, sft_mmlu, sft_gsm8k = evaluate_general(sft_dir)
        report["Standard SFT"] = {
            "Reasoning": sft_rsn, "HellaSwag": sft_hs, "MMLU": sft_mmlu, "GSM8K": sft_gsm8k
        }
        print(f"\n[KẾT QUẢ TRUNG GIAN - STANDARD SFT]")
        print(f"Reasoning: {sft_rsn:.2f}% | HellaSwag: {sft_hs:.2f}% | MMLU: {sft_mmlu:.2f}% | GSM8K: {sft_gsm8k:.2f}%")

    if args.run_steered:
        print("\n" + "="*50 + "\n[PHASE 2] EVALUATION: STEERED SFT\n" + "="*50)
        steer_rsn = evaluate_reasoning(
            steer_dir,
            test_ds,
            args.batch_size,
            args.system_prompt,
            args.eval_max_new_tokens,
            args.eval_preview_samples,
        )
        steer_hs, steer_mmlu, steer_gsm8k = evaluate_general(steer_dir)
        report["Steered SFT"] = {
            "Reasoning": steer_rsn, "HellaSwag": steer_hs, "MMLU": steer_mmlu, "GSM8K": steer_gsm8k
        }
        print(f"\n[KẾT QUẢ TRUNG GIAN - STEERED SFT]")
        print(f"Reasoning: {steer_rsn:.2f}% | HellaSwag: {steer_hs:.2f}% | MMLU: {steer_mmlu:.2f}% | GSM8K: {steer_gsm8k:.2f}%")

    # 4. IN BÁO CÁO CUỐI CÙNG DỰA TRÊN NHỮNG GÌ ĐÃ CHẠY
    print("\n\n" + "*"*80)
    print(f"FINAL REPORT (Model: {args.model} | Samples: {args.samples})")
    print("*"*80)
    
    experiments = list(report.keys())
    benchmarks = [
        ("Reasoning", "Reasoning Acc (%)"),
        ("HellaSwag", "HellaSwag (%)"),
        ("MMLU", "MMLU (%)"),
        ("GSM8K", "GSM8K (%)")
    ]
    
    header_cols = [f"{'Benchmark':<20}"] + [f"{exp:<15}" for exp in experiments]
    header_str = "| " + " | ".join(header_cols) + " |"
    sep_cols = ["-"*20] + ["-"*15 for _ in experiments]
    sep_str = "|" + "|".join(sep_cols) + "|"
    
    markdown_table = [
        f"# Báo cáo Kết quả (Model: {args.model} | Samples: {args.samples})\n",
        header_str,
        sep_str
    ]
    
    print(header_str)
    print(sep_str)
    
    for bm_key, bm_name in benchmarks:
        row_cols = [f"{bm_name:<20}"]
        for exp in experiments:
            val = report[exp].get(bm_key, "N/A")
            val_str = f"{val:<15.2f}" if isinstance(val, (int, float)) else f"{val:<15}"
            row_cols.append(val_str)
        row_str = "| " + " | ".join(row_cols) + " |"
        print(row_str)
        markdown_table.append(row_str)
    print("*"*80)

    # Ghi ra file result.md
    with open("result.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_table) + "\n")
    print(f"[INFO] Đã lưu báo cáo chi tiết vào file result.md")

if __name__ == "__main__":
    main()