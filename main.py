import argparse
from src.core import (
    prepare_data, train_model, 
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
    parser.add_argument("--samples", type=int, default=30000, help="Số lượng mẫu muốn load (vd: 10000)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size khi chạy eval")
    parser.add_argument("--train-epochs", type=float, default=1.0, help="Số epoch train cho SFT/Steered")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate cho SFT/Steered")
    parser.add_argument("--eval-max-new-tokens", type=int, default=2048, help="Số token sinh tối đa khi evaluate_reasoning")
    parser.add_argument("--eval-preview-samples", type=int, default=2, help="Số mẫu in preview trong evaluate_reasoning")
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

    # 2. CHUẨN BỊ DỮ LIỆU CHUNG
    print(f"🚀 Khởi chạy Pipeline với Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_ds, test_ds = prepare_data(
        tokenizer,
        args.dataset,
        args.samples,
        args.dataset_split,
        args.system_prompt
    )
    del tokenizer
    clean_memory()

    report = {}

    # 3. THỰC THI CÁC THÍ NGHIỆM ĐƯỢC CHỌN
    if args.run_baseline:
        print("\n" + "="*50 + "\n[1] EXPERIMENT: BASELINE (ZERO-SHOT)\n" + "="*50)
        base_rsn = evaluate_reasoning(
            args.model,
            test_ds,
            args.batch_size,
            args.system_prompt,
            args.eval_max_new_tokens,
            args.eval_preview_samples,
        )
        base_hs, base_mmlu = evaluate_general(args.model)
        report["Baseline"] = {"Reasoning": base_rsn, "HellaSwag": base_hs, "MMLU": base_mmlu}

    if args.run_sft:
        print("\n" + "="*50 + "\n[2] EXPERIMENT: STANDARD SFT\n" + "="*50)
        sft_dir = "./model_sft"
        train_model(
            args.model,
            train_ds,
            test_ds,
            sft_dir,
            use_steer=False,
            learning_rate=args.learning_rate,
            num_train_epochs=args.train_epochs,
        )
        sft_rsn = evaluate_reasoning(
            sft_dir,
            test_ds,
            args.batch_size,
            args.system_prompt,
            args.eval_max_new_tokens,
            args.eval_preview_samples,
        )
        sft_hs, sft_mmlu = evaluate_general(sft_dir)
        report["Standard SFT"] = {"Reasoning": sft_rsn, "HellaSwag": sft_hs, "MMLU": sft_mmlu}
        delete_model(sft_dir)

    if args.run_steered:
        print("\n" + "="*50 + "\n[3] EXPERIMENT: STEERED SFT\n" + "="*50)
        steer_dir = "./model_steer"
        train_model(
            args.model,
            train_ds,
            test_ds,
            steer_dir,
            use_steer=True,
            learning_rate=args.learning_rate,
            num_train_epochs=args.train_epochs,
        )
        steer_rsn = evaluate_reasoning(
            steer_dir,
            test_ds,
            args.batch_size,
            args.system_prompt,
            args.eval_max_new_tokens,
            args.eval_preview_samples,
        )
        steer_hs, steer_mmlu = evaluate_general(steer_dir)
        report["Steered SFT"] = {"Reasoning": steer_rsn, "HellaSwag": steer_hs, "MMLU": steer_mmlu}
        delete_model(steer_dir)

    # 4. IN BÁO CÁO CUỐI CÙNG DỰA TRÊN NHỮNG GÌ ĐÃ CHẠY
    print("\n\n" + "*"*60)
    print(f"FINAL REPORT (Model: {args.model} | Samples: {args.samples})")
    print("*"*60)
    print(f"| {'Experiment':<15} | {'Reasoning Acc (%)':<17} | {'HellaSwag (%)':<15} | {'MMLU (%)':<10} |")
    print(f"|{'-'*17}|{'-'*19}|{'-'*17}|{'-'*12}|")
    for exp, scores in report.items():
        print(f"| {exp:<15} | {scores['Reasoning']:<17.2f} | {scores['HellaSwag']:<15.2f} | {scores['MMLU']:<10.2f} |")
    print("*"*60)

if __name__ == "__main__":
    main()