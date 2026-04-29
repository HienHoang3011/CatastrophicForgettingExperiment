import argparse
import os
from src.core import evaluate_general, evaluate_reasoning
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="So sánh kết quả đánh giá General và Reasoning giữa model gốc và model sau finetune")
    
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace Model ID (Base model)")
    parser.add_argument("--finetuned-model", type=str, default="./model_sft", help="Đường dẫn model sau finetune")
    parser.add_argument("--output", type=str, default="result_compare.md", help="File lưu kết quả")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size khi chạy eval reasoning")
    parser.add_argument("--eval-max-new-tokens", type=int, default=2048, help="Số token sinh tối đa")
    parser.add_argument("--eval-preview-samples", type=int, default=5, help="Số mẫu preview")
    
    args = parser.parse_args()
    
    print("\n[DATA] Đang load tập test từ 'saved_test_dataset.jsonl'...")
    try:
        test_dataset = load_dataset("json", data_files="saved_test_dataset.jsonl", split="train")
        print(f"[DATA] Load thành công {len(test_dataset)} mẫu.")
    except Exception as e:
        print(f"❌ Lỗi khi load test dataset: {e}")
        print("💡 Gợi ý: Chắc chắn file saved_test_dataset.jsonl tồn tại trong thư mục hiện tại.")
        return

    results = {}
    
    # ==========================
    # ĐÁNH GIÁ MODEL GỐC
    # ==========================
    print("\n" + "="*60)
    print("📊 Đánh giá MODEL GỐC (Baseline)...")
    print("="*60)
    try:
        base_rsn = evaluate_reasoning(
            args.base_model,
            test_dataset,
            args.batch_size,
            None,
            args.eval_max_new_tokens,
            args.eval_preview_samples
        )
        hs_base, mmlu_base, gsm8k_base, aime_base, acp_base, acp_hard_base = evaluate_general(args.base_model)
        results["Baseline"] = {
            "Reasoning": base_rsn, "HellaSwag": hs_base, "MMLU": mmlu_base,
            "GSM8K": gsm8k_base, "AIME": aime_base, "ACP Bench": acp_base, "ACP Hard": acp_hard_base
        }
        print(f"✅ Hoàn thành đánh giá Baseline!")
    except Exception as e:
        print(f"❌ Lỗi Baseline: {e}")
        return
    
    # ==========================
    # ĐÁNH GIÁ MODEL FINETUNE
    # ==========================
    print("\n" + "="*60)
    print("📊 Đánh giá MODEL SAU FINETUNE...")
    print("="*60)
    try:
        if not os.path.exists(args.finetuned_model):
            print(f"❌ Lỗi: Không tìm thấy model tại {args.finetuned_model}")
            return
            
        fine_rsn = evaluate_reasoning(
            args.finetuned_model,
            test_dataset,
            args.batch_size,
            None,
            args.eval_max_new_tokens,
            args.eval_preview_samples
        )
        hs_fine, mmlu_fine, gsm8k_fine, aime_fine, acp_fine, acp_hard_fine = evaluate_general(args.finetuned_model)
        results["Finetuned"] = {
            "Reasoning": fine_rsn, "HellaSwag": hs_fine, "MMLU": mmlu_fine,
            "GSM8K": gsm8k_fine, "AIME": aime_fine, "ACP Bench": acp_fine, "ACP Hard": acp_hard_fine
        }
        print(f"✅ Hoàn thành đánh giá Finetuned!")
    except Exception as e:
        print(f"❌ Lỗi Finetuned: {e}")
        return
    
    # ==========================
    # BÁO CÁO KẾT QUẢ
    # ==========================
    print("\n\n" + "*"*80)
    print("FINAL REPORT - SO SÁNH KẾT QUẢ")
    print("*"*80)
    
    benchmarks = [
        ("Reasoning", "Reasoning Acc (%)"),
        ("HellaSwag", "HellaSwag (%)"),
        ("MMLU", "MMLU (%)"),
        ("GSM8K", "GSM8K (%)"),
        ("AIME", "AIME (%)"),
        ("ACP Bench", "ACP Bench (%)"),
        ("ACP Hard", "ACP Bench Hard (%)")
    ]
    
    experiments = ["Baseline", "Finetuned"]
    
    header_cols = [f"{'Benchmark':<20}"] + [f"{exp:<15}" for exp in experiments] + ["Change"]
    header_str = "| " + " | ".join(header_cols) + " |"
    sep_cols = ["-"*20] + ["-"*15 for _ in experiments] + ["-"*15]
    sep_str = "|" + "|".join(sep_cols) + "|"
    
    markdown_table = [
        "# Báo cáo So Sánh (Baseline vs Finetuned)\n",
        header_str,
        sep_str
    ]
    
    print(header_str)
    print(sep_str)
    
    for bm_key, bm_name in benchmarks:
        row_cols = [f"{bm_name:<20}"]
        b_val = results["Baseline"][bm_key]
        f_val = results["Finetuned"][bm_key]
        change = f_val - b_val
        
        row_cols.append(f"{b_val:<15.2f}")
        row_cols.append(f"{f_val:<15.2f}")
        row_cols.append(f"{change:<+15.2f}")
        
        row_str = "| " + " | ".join(row_cols) + " |"
        print(row_str)
        markdown_table.append(row_str)
        
    print("*"*80)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_table) + "\n")
        
    print(f"\n✅ Đã lưu bảng kết quả vào: {args.output}")

if __name__ == "__main__":
    main()
