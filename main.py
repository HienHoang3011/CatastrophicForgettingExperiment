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
    parser.add_argument("--dataset", type=str, default="Open-Reasoner-Zero/data/orz_math_57k_collected.json", help="Đường dẫn file JSON")
    parser.add_argument("--samples", type=int, default=30000, help="Số lượng mẫu muốn load (vd: 30000)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size khi chạy eval")
    
    # Cấu hình Bật/Tắt Experiment (Mặc định là False, truyền cờ vào sẽ thành True)
    parser.add_argument("--run-baseline", action="store_true", help="Chỉ chạy Zero-shot Evaluation cho Base model")
    parser.add_argument("--run-sft", action="store_true", help="Chỉ chạy Standard SFT")
    parser.add_argument("--run-steered", action="store_true", help="Chỉ chạy Steered SFT (Thuật toán của bạn)")
    parser.add_argument("--run-all", action="store_true", help="Chạy toàn bộ 3 thí nghiệm")

    args = parser.parse_args()

    # Nếu người dùng chọn --run-all, tự động bật 3 cờ kia lên
    if args.run_all:
        args.run_baseline = args.run_sft = args.run_steered = True

    # Check nếu user chạy script mà quên chọn experiment
    if not (args.run_baseline or args.run_sft or args.run_steered):
        print("❌ LỖI: Bạn chưa chọn Thí nghiệm nào để chạy!")
        print("💡 Gợi ý: Thêm cờ --run-baseline, --run-steered, hoặc --run-all")
        return

    # 2. CHUẨN BỊ DỮ LIỆU CHUNG
    print(f"🚀 Khởi chạy Pipeline với Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_ds, test_ds = prepare_data(tokenizer, args.dataset, args.samples)
    del tokenizer
    clean_memory()

    report = {}

    # 3. THỰC THI CÁC THÍ NGHIỆM ĐƯỢC CHỌN
    if args.run_baseline:
        print("\n" + "="*50 + "\n[1] EXPERIMENT: BASELINE (ZERO-SHOT)\n" + "="*50)
        base_rsn = evaluate_reasoning(args.model, test_ds, args.batch_size)
        base_hs, base_mmlu = evaluate_general(args.model)
        report["Baseline"] = {"Reasoning": base_rsn, "HellaSwag": base_hs, "MMLU": base_mmlu}

    if args.run_sft:
        print("\n" + "="*50 + "\n[2] EXPERIMENT: STANDARD SFT\n" + "="*50)
        sft_dir = "./model_sft"
        train_model(args.model, train_ds, sft_dir, use_steer=False)
        sft_rsn = evaluate_reasoning(sft_dir, test_ds, args.batch_size)
        sft_hs, sft_mmlu = evaluate_general(sft_dir)
        report["Standard SFT"] = {"Reasoning": sft_rsn, "HellaSwag": sft_hs, "MMLU": sft_mmlu}
        delete_model(sft_dir)

    if args.run_steered:
        print("\n" + "="*50 + "\n[3] EXPERIMENT: STEERED SFT\n" + "="*50)
        steer_dir = "./model_steer"
        train_model(args.model, train_ds, steer_dir, use_steer=True)
        steer_rsn = evaluate_reasoning(steer_dir, test_ds, args.batch_size)
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