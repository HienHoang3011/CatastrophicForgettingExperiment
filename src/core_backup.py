# src/core.py
import os
import json
import math
import re
import torch
import shutil
import gc
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import lm_eval

def _import_trl_or_raise():
    try:
        from trl import SFTTrainer, SFTConfig
    except Exception as e:
        raise RuntimeError(
            "Failed to import TRL training components. Baseline/evaluation can run without TRL, "
            "but SFT training needs compatible versions of trl and transformers."
        ) from e
    return SFTTrainer, SFTConfig


def _build_steered_trainer_class(sft_trainer_cls):
    class SteeredTrainer(sft_trainer_cls):
        def __init__(self, *args, x_factor=0.2, **kwargs):
            super().__init__(*args, **kwargs)
            self.margin = math.log(1 + x_factor)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            mask = shift_labels != -100
            active_logits = shift_logits[mask]
            active_labels = shift_labels[mask]

            with torch.no_grad():
                target_logits = active_logits.clone()
                temp_logits = target_logits.clone()
                batch_idx = torch.arange(target_logits.size(0), device=target_logits.device)
                temp_logits[batch_idx, active_labels] = float('-inf')

                max_other_logits, _ = torch.max(temp_logits, dim=-1)
                current_gt_logits = target_logits[batch_idx, active_labels]

                target_gt_logits = torch.max(current_gt_logits, max_other_logits + self.margin)
                target_logits[batch_idx, active_labels] = target_gt_logits

                target_probs = F.softmax(target_logits, dim=-1)

            log_probs = F.log_softmax(active_logits, dim=-1)
            loss = F.kl_div(log_probs, target_probs, reduction='batchmean')

            return (loss, outputs) if return_outputs else loss

    return SteeredTrainer

def _extract_final_answer(text):
    if not text:
        return ""

    boxed_contents = []
    needle = "\\boxed{"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break

        i = idx + len(needle)
        depth = 1
        chunk = []
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            if depth > 0:
                chunk.append(ch)
            i += 1

        if depth == 0:
            boxed_contents.append("".join(chunk).strip())
            start = i
        else:
            # Unbalanced braces: stop parsing boxed blocks and fallback below.
            break

    if boxed_contents:
        return boxed_contents[-1]

    # Box-only policy: if no \boxed{...} is found, treat as empty answer.
    return ""


def _extract_from_numina_row(row):
    user_text = row.get("problem", "") or ""
    assistant_text = row.get("solution", "") or ""

    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user" and not user_text:
                user_text = content
            elif role == "assistant" and content:
                assistant_text = content

    return user_text, assistant_text


def _default_math_system_prompt():
    return (
        "You are a rigorous math reasoning assistant. "
        "Reason step by step internally, then return only one final answer in the form \\boxed{...}. "
        "Do not include extra text after the boxed answer."
    )


def prepare_data(tokenizer, dataset_path, max_samples, dataset_split="train", system_prompt=None):
    print(f"\n[DATA] Loading {max_samples} samples from {dataset_path}...")

    if system_prompt is None:
        system_prompt = _default_math_system_prompt()

    formatted_data = {"instruction": [], "output": [], "final_answer": [], "source": []}
    skipped_no_box = 0

    def add_sample(instruction, output_text, source="unknown"):
        nonlocal skipped_no_box
        final_answer = _extract_final_answer(output_text)
        if final_answer:
            formatted_data["instruction"].append(instruction)
            formatted_data["output"].append(output_text)
            formatted_data["final_answer"].append(final_answer)
            formatted_data["source"].append(source)
            return True

        skipped_no_box += 1
        return False

    from src.data_filter import get_filtered_dataset
    hf_dataset = get_filtered_dataset(dataset_path, dataset_split, max_samples)
    for row in hf_dataset:
        if len(formatted_data["instruction"]) >= max_samples:
            break
            
        # Dùng hàm trích xuất gốc từ core.py để tương thích với cấu trúc của Numina hoặc JSON custom
        question, solution = _extract_from_numina_row(row)
        
        # Thử lấy từ các format khác nếu hàm trên không lấy được
        if not question or not solution:
            conv = row.get("conversations", []) if isinstance(row, dict) else []
            for turn in conv:
                if isinstance(turn, dict):
                    if turn.get("from") == "human":
                        question = turn.get("value", "")
                    elif turn.get("from") == "assistant":
                        solution = turn.get("value", "")
                        if not solution and "ground_truth" in turn:
                            solution = str(turn["ground_truth"].get("value", ""))
                            
        source_val = str(row.get("source", "unknown"))
        if question and solution:
            add_sample(question, solution, source_val)

    if not formatted_data["instruction"]:
        raise ValueError(
            "No valid samples were loaded. Check dataset path/repo and expected columns."
        )

    if skipped_no_box > 0:
        print(f"[DATA] Skipped {skipped_no_box} samples without \\boxed{{...}} in target solution.")

    raw_dataset = Dataset.from_dict(formatted_data)
    
    # Chia train/test dựa trên phân phối Stratified theo "source"
    from collections import defaultdict
    import random
    
    def build_stratified_index_order(labels, batch_size, seed):
        """Approximate nemotron-master's stratified batching over effective batches."""
        by_label = defaultdict(list)
        for idx, label in enumerate(labels):
            by_label[label].append(idx)

        rng = random.Random(seed)
        for idx_list in by_label.values():
            rng.shuffle(idx_list)

        n_batches = max(1, math.ceil(len(labels) / batch_size))
        batches = [[] for _ in range(n_batches)]
        batch_order = list(range(n_batches))
        rng.shuffle(batch_order)

        assigned = 0
        for label in sorted(by_label.keys()):
            for idx in by_label[label]:
                batches[batch_order[assigned % n_batches]].append(idx)
                assigned += 1

        order = [idx for batch in batches for idx in batch]
        if len(order) != len(labels):
            raise ValueError("Stratified order size mismatch")
        return order
        
    labels = formatted_data["source"]
    # Chia dữ liệu để stratify (giả định dùng batch_size=32 để xáo trộn tốt)
    stratified_order = build_stratified_index_order(labels, batch_size=32, seed=42)
    
    split_idx = int(len(stratified_order) * 0.95)
    train_indices = stratified_order[:split_idx]
    test_indices = stratified_order[split_idx:]
    
    # Cắt tập test tối đa 1500 mẫu nhưng vẫn giữ phân phối (do mảng order đã được stratify dàn trải)
    if len(test_indices) > 1500:
        test_indices = test_indices[:1500]
        
    train_dataset = raw_dataset.select(train_indices)
    test_dataset = raw_dataset.select(test_indices)
    
    # Lưu lại tập test để tái sử dụng sau này
    print("[DATA] Đang lưu tập test (đã stratify) ra file 'saved_test_dataset.jsonl'...")
    test_dataset.to_json("saved_test_dataset.jsonl", force_ascii=False)

    def format_train(examples):
        texts = []
        for q, a in zip(examples['instruction'], examples['output']):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    train_dataset = train_dataset.map(format_train, batched=True)
    test_dataset = test_dataset.map(format_train, batched=True)

    return train_dataset, test_dataset

def evaluate_reasoning(
    model_path,
    test_dataset,
    batch_size=16,
    eval_system_prompt=None,
    eval_max_new_tokens=2048,
    preview_samples=5,
):
    print(f"\n[EVAL] Running Batched Reasoning Inference on {len(test_dataset)} samples...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    import os
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_peft:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        ).eval()

    correct = 0
    total = len(test_dataset)
    shown_preview = 0

    if eval_system_prompt is None:
        eval_system_prompt = _default_math_system_prompt()

    for i in tqdm(range(0, total, batch_size), desc="Inferencing"):
        batch = test_dataset[i : i + batch_size]
        prompts = []
        for q in batch['instruction']:
            messages = [
                {"role": "system", "content": eval_system_prompt},
                {"role": "user", "content": q} 
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=eval_max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None
            )
            
        input_len = inputs.input_ids.shape[-1]
        for j in range(len(prompts)):
            gen_text = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            gen_final = _extract_final_answer(gen_text)

            target_final = batch['final_answer'][j] if 'final_answer' in batch else ""
            is_match = bool(target_final) and (target_final == gen_final)

            if is_match:
                correct += 1

            if shown_preview < preview_samples:
                shown_preview += 1
                print("\n[PREVIEW SAMPLE]", shown_preview)
                print("Q:", batch['instruction'][j])
                print("Pred:", gen_text)
                print("Pred Final:", gen_final)
                print("Target Final:", target_final)
                print("Match:", is_match)

    del model
    del tokenizer
    clean_memory()
    return (correct / total) * 100

def evaluate_general(model_path):
    print(f"\n[EVAL] Running lm_eval (HellaSwag, MMLU, GSM8K) for {model_path}...")
    import os
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_peft:
        import json
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            config = json.load(f)
        base = config.get("base_model_name_or_path", "")
        model_args = f"pretrained={base},peft={model_path},dtype=bfloat16,trust_remote_code=True"
    else:
        model_args = f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True"

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

def train_model(base_model, train_dataset, eval_dataset, output_dir, use_steer=False, learning_rate=5e-6, num_train_epochs=1):
    print(f"\n[TRAIN] Starting Training (Steered={use_steer}). Output: {output_dir}")
    SFTTrainer, SFTConfig = _import_trl_or_raise()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cuda", 
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing on the model
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    from peft import LoraConfig
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0,
        target_modules="all-linear",
        task_type="CAUSAL_LM"
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",  
        max_length=8192,               
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=16, 
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        bf16=True,
        gradient_checkpointing=True,  # CRITICAL: Reduces memory by ~30-40%
        logging_steps=10,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=50,
        per_device_eval_batch_size=2,
        save_strategy="no",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # save_total_limit=1,
        save_only_model=True,
        optim="adamw_torch_fused",
        report_to="none"
    )

    trainer_class = _build_steered_trainer_class(SFTTrainer) if use_steer else SFTTrainer
    trainer_kwargs = {"x_factor": 1.0} if use_steer else {}

    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        peft_config=peft_config,
        **trainer_kwargs
    )

    trainer.train()
    trainer.save_model(output_dir)
    
    del trainer
    del model
    del tokenizer
    clean_memory()

def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()

def delete_model(path):
    if os.path.exists(path):
        print(f"\n[CLEANUP] Deleting {path} to free up disk space...")
        shutil.rmtree(path)