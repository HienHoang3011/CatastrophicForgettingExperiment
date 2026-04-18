# src/core.py
import os
import json
import math
import torch
import shutil
import gc
import torch.nn.functional as F
from tqdm import tqdm
from datasets import Dataset
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

def prepare_data(tokenizer, dataset_path, max_samples):
    print(f"\n[DATA] Loading {max_samples} samples from {dataset_path}...")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    formatted_data = {"instruction": [], "output": []}
    
    # Lấy linh hoạt số lượng sample
    for conv in raw_list[:max_samples]:
        human_text = ""
        assistant_text = ""
        for turn in conv:
            if turn.get("from") == "human":
                human_text = turn.get("value", "")
            elif turn.get("from") == "assistant":
                assistant_text = turn.get("value", "")
                if not assistant_text and "ground_truth" in turn:
                    assistant_text = str(turn["ground_truth"].get("value", ""))
                    
        formatted_data["instruction"].append(human_text)
        formatted_data["output"].append(assistant_text)

    raw_dataset = Dataset.from_dict(formatted_data)
    split_ds = raw_dataset.train_test_split(test_size=0.2, seed=42)
    
    train_dataset = split_ds['train']
    test_dataset = split_ds['test']

    def format_train(examples):
        texts = []
        for q, a in zip(examples['instruction'], examples['output']):
            messages = [
                {"role": "system", "content": "You are a logical reasoning assistant."},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    train_dataset = train_dataset.map(format_train, batched=True)
    return train_dataset, test_dataset

def evaluate_reasoning(model_path, test_dataset, batch_size=16):
    print(f"\n[EVAL] Running Batched Reasoning Inference on {len(test_dataset)} samples...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda", 
        attn_implementation="flash_attention_2", trust_remote_code=True
    ).eval()

    correct = 0
    total = len(test_dataset)

    for i in tqdm(range(0, total, batch_size), desc="Inferencing"):
        batch = test_dataset[i : i + batch_size]
        prompts = []
        for q in batch['instruction']:
            # ZERO SYSTEM PROMPT AS REQUESTED
            constrained_q = q + "\n\nAnswer with only the final result in one short line."
            messages = [{"role": "user", "content": constrained_q}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
            )
            
        input_len = inputs.input_ids.shape[-1]
        for j in range(len(prompts)):
            gen_text = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True).lower()
            if batch['output'][j].strip().lower() in gen_text:
                correct += 1

    del model
    del tokenizer
    clean_memory()
    return (correct / total) * 100

def evaluate_general(model_path):
    print(f"\n[EVAL] Running lm_eval (HellaSwag, MMLU) for {model_path}...")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
        tasks=["hellaswag", "mmlu"],
        device="cuda:0",
        batch_size="auto"
    )
    
    hs_acc = results["results"]["hellaswag"].get("acc_norm,none", results["results"]["hellaswag"].get("acc_norm"))
    mmlu_acc = results["results"]["mmlu"].get("acc,none", results["results"]["mmlu"].get("acc"))
    
    clean_memory()
    return hs_acc * 100, mmlu_acc * 100

def train_model(base_model, train_dataset, output_dir, use_steer=False):
    print(f"\n[TRAIN] Starting Training (Steered={use_steer}). Output: {output_dir}")
    SFTTrainer, SFTConfig = _import_trl_or_raise()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cuda", 
        attn_implementation="flash_attention_2", trust_remote_code=True
    )

    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",  
        max_length=4096,               
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=8,
        learning_rate=2e-5,            
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=False,  
        logging_steps=10,
        save_strategy="no",       # CHỐNG TRÀN DISK: Không lưu giữa chừng
        save_only_model=True,     # CHỐNG TRÀN DISK: Bỏ Optimizer states (Tiết kiệm 13GB)
        optim="adamw_torch_fused",
        report_to="none"
    )

    trainer_class = _build_steered_trainer_class(SFTTrainer) if use_steer else SFTTrainer
    trainer_kwargs = {"x_factor": 1.0} if use_steer else {}

    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
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