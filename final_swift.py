import argparse
import os
import json
import subprocess
import glob
from src.core import (
    prepare_data, delete_model, clean_memory, _extract_final_answer, _default_math_system_prompt
)
from transformers import AutoTokenizer

def train_model_swift(
    base_model_path,
    train_dataset,
    output_dir,
    system_prompt,
    learning_rate=5e-6,
    num_train_epochs=1.0,
    max_lora_rank=64,
    max_length=8192
):
    print(f"\n[SWIFT SFT] Chuẩn bị dữ liệu cho ms-swift...")
    train_file = "temp_train_swift.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for i in range(len(train_dataset)):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": train_dataset['instruction'][i]},
                {"role": "assistant", "content": train_dataset['output'][i]}
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"[SWIFT SFT] Bắt đầu huấn luyện với ms-swift (LoRA rank: {max_lora_rank}, Max len: {max_length})...")
    cmd = [
        "swift", "sft",
        "--model_id_or_path", base_model_path,
        "--dataset", train_file,
        "--output_dir", output_dir,
        "--sft_type", "lora",
        "--lora_rank", str(max_lora_rank),
        "--max_length", str(max_length),
        "--learning_rate", str(learning_rate),
        "--num_train_epochs", str(num_train_epochs),
        "--save_only_model", "true"
    ]
    print(f"[SWIFT SFT] Lệnh: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("❌ LỖI: Không tìm thấy lệnh 'swift'. Vui lòng cài đặt: pip install ms-swift")
        raise
    
    # ms-swift thường lưu checkpoint bên trong thư mục dạng output_dir/<tên_model>/v0.../checkpoint-xxx
    checkpoints = glob.glob(os.path.join(output_dir, "*", "*", "checkpoint-*"))
    if not checkpoints:
        checkpoints = glob.glob(os.path.join(output_dir, "*", "checkpoint-*"))
    if checkpoints:
        latest = sorted(checkpoints, key=os.path.getmtime)[-1]
        print(f"[SWIFT SFT] Checkpoint mới nhất: {latest}")
        return latest
    return output_dir

def evaluate_reasoning_swift(
    base_model_path,
    lora_path,
    test_dataset,
    system_prompt,
    max_lora_rank=64,
    max_tokens=7680,
    top_p=1.0,
    temperature=0.0,
    max_num_seqs=64,
    gpu_memory_utilization=0.85,
    max_model_len=8192,
    preview_samples=2
):
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        import torch
    except ImportError:
        raise ImportError("Vui lòng cài đặt vllm: pip install vllm")

    print(f"\n[EVAL] Running vLLM Reasoning Inference on {len(test_dataset)} samples...")
    llm = LLM(
        model=base_model_path,
        enable_lora=bool(lora_path),
        max_lora_rank=max_lora_rank if lora_path else 16,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count() or 1
    )
    sampling_params = SamplingParams(max_tokens=max_tokens, top_p=top_p, temperature=temperature)
    tokenizer = llm.get_tokenizer()

    prompts = []
    for q in test_dataset['instruction']:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    lora_req = LoRARequest("adapter_1", 1, lora_path) if lora_path else None
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)

    correct = 0
    total = len(test_dataset)
    shown = 0

    for i, out in enumerate(outputs):
        gen_text = out.outputs[0].text
        gen_final = _extract_final_answer(gen_text)
        target_final = test_dataset['final_answer'][i]
        is_match = bool(target_final) and (target_final == gen_final)
        if is_match: correct += 1

        if shown < preview_samples:
            shown += 1
            print(f"\n[PREVIEW {shown}] Q: {test_dataset['instruction'][i][:100]}...")
            print(f"Pred: {gen_text[:100]}... | Final: {gen_final} | Target: {target_final} | Match: {is_match}")

    del llm
    clean_memory()
    return (correct / total) * 100 if total > 0 else 0.0

def evaluate_general_swift(base_model_path, lora_path=None):
    import lm_eval
    model_args = f"pretrained={base_model_path},dtype=bfloat16,trust_remote_code=True"
    if lora_path:
        model_args += f",peft={lora_path}"
    
    print(f"\n[EVAL] Running lm_eval for {base_model_path} + LoRA: {lora_path}...")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["hellaswag", "mmlu", "gsm8k"],
        device="cuda:0",
        batch_size="auto"
    )
    hs_acc = results["results"]["hellaswag"].get("acc_norm,none", results["results"]["hellaswag"].get("acc_norm", 0.0))
    mmlu_acc = results["results"]["mmlu"].get("acc,none", results["results"]["mmlu"].get("acc", 0.0))
    gsm8k_acc = results["results"]["gsm8k"].get("exact_match,strict-match", results["results"]["gsm8k"].get("exact_match", results["results"]["gsm8k"].get("acc", 0.0)))
    
    clean_memory()
    return hs_acc * 100, mmlu_acc * 100, gsm8k_acc * 100


def main():
    parser = argparse.ArgumentParser(description="Hệ thống Huấn luyện và Đánh giá LLM dùng ms-swift (SFT)")
    
    # Cấu hình Model & Data (giống final.py)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--dataset", type=str, default="AI-MO/NuminaMath-CoT")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--eval-preview-samples", type=int, default=2)
    parser.add_argument(
        "--system-prompt", dest="system_prompt", type=str,
        default=(
            "You are a rigorous math reasoning assistant. "
            "Reason step by step internally, then return only one final answer in the form \\boxed{...}. "
            "Do not include extra text after the boxed answer."
        )
    )

    # Các cấu hình cụ thể cho SFT bằng Swift và Inference bằng vLLM theo yêu cầu
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=7680)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)

    # Chế độ chạy
    parser.add_argument("--mode", type=str, choices=["baseline", "sft", "all"], help="baseline | sft | all")
    parser.add_argument("--run-baseline", action="store_true")
    parser.add_argument("--run-sft", action="store_true")

    args = parser.parse_args()

    if args.mode:
        args.run_baseline = args.mode in ["baseline", "all"]
        args.run_sft = args.mode in ["sft", "all"]
    elif args.run_baseline or args.run_sft:
        pass
    else:
        # Default nếu không truyền gì
        args.run_baseline = True
        args.run_sft = True

    # Luôn đánh giá base model làm mốc
    args.run_baseline = True

    print(f"🚀 Khởi chạy Pipeline với Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_ds, test_ds = prepare_data(
        tokenizer, args.dataset, args.samples, args.dataset_split, args.system_prompt
    )
    del tokenizer
    clean_memory()

    report = {}
    sft_dir = "./model_sft_swift"
    sft_lora_path = None

    # PHASE 1: TRAINING VỚI SWIFT
    if args.run_sft:
        print("\n" + "="*50 + "\n[PHASE 1] TRAINING: STANDARD SFT (SWIFT)\n" + "="*50)
        sft_lora_path = train_model_swift(
            base_model_path=args.model,
            train_dataset=train_ds,
            output_dir=sft_dir,
            system_prompt=args.system_prompt,
            learning_rate=args.learning_rate,
            num_train_epochs=args.train_epochs,
            max_lora_rank=args.max_lora_rank,
            max_length=args.max_model_len
        )

    # PHASE 2: EVALUATION VỚI VLLM
    if args.run_baseline:
        print("\n" + "="*50 + "\n[PHASE 2] EVALUATION: BASELINE (ZERO-SHOT)\n" + "="*50)
        base_hs, base_mmlu, base_gsm8k = evaluate_general_swift(args.model, lora_path=None)
        report["Baseline"] = {"Reasoning": "N/A", "HellaSwag": base_hs, "MMLU": base_mmlu, "GSM8K": base_gsm8k}

    if args.run_sft and sft_lora_path:
        print("\n" + "="*50 + "\n[PHASE 2] EVALUATION: STANDARD SFT (SWIFT)\n" + "="*50)
        sft_rsn = evaluate_reasoning_swift(
            base_model_path=args.model,
            lora_path=sft_lora_path,
            test_dataset=test_ds,
            system_prompt=args.system_prompt,
            max_lora_rank=args.max_lora_rank,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            preview_samples=args.eval_preview_samples
        )
        sft_hs, sft_mmlu, sft_gsm8k = evaluate_general_swift(args.model, lora_path=sft_lora_path)
        report["Swift SFT"] = {"Reasoning": sft_rsn, "HellaSwag": sft_hs, "MMLU": sft_mmlu, "GSM8K": sft_gsm8k}
        # Có thể delete nếu không cần giữ lại, nhưng ms-swift tạo ra checkpoint nhẹ
        # delete_model(sft_dir)

    # BÁO CÁO CUỐI CÙNG
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

    with open("result_swift.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_table) + "\n")
    print(f"[INFO] Đã lưu báo cáo chi tiết vào file result_swift.md")

if __name__ == "__main__":
    main()
