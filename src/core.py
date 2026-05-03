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


def load_saved_datasets(tokenizer, train_path, eval_path, test_path, system_prompt=None):
    print(f"\n[DATA] Tải dữ liệu đã chuẩn bị từ: {train_path}, {eval_path}, {test_path}...")

    if system_prompt is None:
        system_prompt = _default_math_system_prompt()

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    eval_dataset = load_dataset("json", data_files=eval_path, split="train")
    test_dataset = load_dataset("json", data_files=test_path, split="train")

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

    print("[DATA] Áp dụng chat template...")
    train_dataset = train_dataset.map(format_train, batched=True)
    eval_dataset = eval_dataset.map(format_train, batched=True)
    test_dataset = test_dataset.map(format_train, batched=True)

    return train_dataset, eval_dataset, test_dataset

def evaluate_reasoning(
    model_path,
    test_dataset,
    batch_size=16,
    eval_system_prompt=None,
    eval_max_new_tokens=2048
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
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
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
            
            def is_math_match(g, t):
                if not g or not t: return False
                
                # 1. Làm sạch cơ bản
                def clean(s):
                    return s.lower().replace(" ", "").replace("$", "").replace(",", "").rstrip(".")
                cg, ct = clean(g), clean(t)
                if cg == ct: return True
                
                # 2. So sánh tương đương toán học bằng sympy
                try:
                    import sympy
                    if sympy.simplify(sympy.sympify(g) - sympy.sympify(t)) == 0:
                        return True
                except:
                    pass
                
                # 3. Chấp nhận nếu Target nằm trọn vẹn trong Predicted
                import re
                try:
                    if re.search(r'(?<!\d)' + re.escape(ct) + r'(?!\d)', cg):
                        return True
                except:
                    if ct in cg:
                        return True
                        
                return False

            is_match = is_math_match(gen_final, target_final)

            if is_match:
                correct += 1

            # In chi tiết từng mẫu ra terminal (theo yêu cầu debug)
            print(f"\n[SAMPLE {i + j + 1}/{total} | MODEL: {model_path}]")
            print("Q:", batch['instruction'][j])
            print("Pred (Raw):\n", gen_text)
            print("-" * 20)
            print("Pred (Filtered):", gen_final)
            print("Target Final:", target_final)
            print("Result:", "✅ ĐÚNG" if is_match else "❌ SAI")

    del model
    del tokenizer
    clean_memory()
    return (correct / total) * 100

def _evaluate_general_internal(model_path, queue):
    try:
        import lm_eval
        
        # Monkey patch lm_eval Registry để sửa lỗi đăng ký trùng lặp khi load nhiều task
        from lm_eval.api.registry import Registry
        original_register = Registry.register
        def safe_register(self, *aliases, target=None):
            def decorator(obj):
                try:
                    # Lấy decorator gốc và truyền obj vào
                    dec = original_register(self, *aliases, target=target)
                    return dec(obj) if callable(dec) else obj
                except ValueError as e:
                    if "already registered" in str(e):
                        return obj
                    raise e
            
            if target is not None:
                try:
                    return original_register(self, *aliases, target=target)
                except ValueError as e:
                    if "already registered" in str(e):
                        return lambda x: x
                    raise e
            return decorator
            
        Registry.register = safe_register

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
            batch_size=16,
            gen_kwargs={"max_gen_toks": 2048}
        )
        
        hs_acc = results["results"]["hellaswag"].get("acc_norm,none", results["results"]["hellaswag"].get("acc_norm", 0.0))
        mmlu_acc = results["results"]["mmlu"].get("acc,none", results["results"]["mmlu"].get("acc", 0.0))
        gsm8k_acc = results["results"]["gsm8k"].get("exact_match,strict-match", results["results"]["gsm8k"].get("exact_match", results["results"]["gsm8k"].get("acc", 0.0)))
        
        clean_memory()
        res = (hs_acc * 100, mmlu_acc * 100, gsm8k_acc * 100)
        queue.put(("SUCCESS", res))
    except Exception as e:
        queue.put(("ERROR", str(e)))

def evaluate_general(model_path):
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=_evaluate_general_internal, args=(model_path, queue))
    p.start()
    p.join()
    
    if not queue.empty():
        status, result = queue.get()
        if status == "ERROR":
            raise Exception(result)
        return result
    else:
        raise Exception("Tiến trình đánh giá bị gián đoạn đột ngột.")

def evaluate_accuracy_in_memory(model, tokenizer, test_dataset, batch_size=4, max_new_tokens=1500):
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    correct = 0
    total = len(test_dataset)
    import torch
    from tqdm import tqdm
    
    system_prompt = _default_math_system_prompt()
    for i in tqdm(range(0, total, batch_size), desc="Generative Eval during Train"):
        batch = test_dataset[i : i + batch_size]
        prompts = []
        for q in batch['instruction']:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None
            )
            
        input_len = inputs.input_ids.shape[-1]
        for j in range(len(prompts)):
            gen_text = tokenizer.decode(outputs[j][input_len:], skip_special_tokens=True)
            gen_final = _extract_final_answer(gen_text)
            target_final = batch['final_answer'][j] if 'final_answer' in batch else ""
            
            def is_math_match(g, t):
                if not g or not t: return False
                
                # 1. Làm sạch cơ bản
                def clean(s):
                    return s.lower().replace(" ", "").replace("$", "").replace(",", "").rstrip(".")
                cg, ct = clean(g), clean(t)
                if cg == ct: return True
                
                # 2. So sánh tương đương toán học bằng sympy (VD: 1/2 == 0.5, x+y == y+x)
                try:
                    import sympy
                    if sympy.simplify(sympy.sympify(g) - sympy.sympify(t)) == 0:
                        return True
                except:
                    pass
                
                # 3. Chấp nhận nếu Target nằm trọn vẹn trong Predicted (có chặn viền từ - tránh lỗi 1 in 19)
                import re
                try:
                    # Tránh lỗi escape cho chuỗi toán học
                    if re.search(r'(?<!\d)' + re.escape(ct) + r'(?!\d)', cg):
                        return True
                except:
                    if ct in cg: # Bất đắc dĩ fallback
                        return True
                        
                return False

            if is_math_match(gen_final, target_final):
                correct += 1
                
    model.train()
    tokenizer.padding_side = original_padding_side
    return correct / total if total > 0 else 0.0

def plot_training_history(log_history, output_path, title="Training Curve"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] Không tìm thấy thư viện matplotlib. Bỏ qua việc vẽ đồ thị.")
        return

    steps = []
    loss = []
    eval_steps = []
    eval_acc = []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            loss.append(entry["loss"])
        if "eval_accuracy" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_acc.append(entry["eval_accuracy"])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Training Loss', color=color)
    if steps and loss:
        ax1.plot(steps, loss, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    if eval_steps and eval_acc:
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Eval Accuracy (%)', color=color)
        ax2.plot(eval_steps, [acc * 100 for acc in eval_acc], color=color, marker='o', label='Eval Acc')
        ax2.tick_params(axis='y', labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\n[INFO] Đã lưu đồ thị huấn luyện tại: {output_path}")

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
        task_type="CAUSAL_LM",
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
        eval_steps=0.25,
        per_device_eval_batch_size=2,
        save_strategy="no",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # save_total_limit=1,
        save_only_model=True,
        optim="adamw_torch_fused",
        report_to="none",
    )


    from transformers import DataCollatorForLanguageModeling
    
    class CustomDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
        def __init__(self, response_template, tokenizer, *args, **kwargs):
            super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
            self.response_template = response_template

        def torch_call(self, examples):
            batch = super().torch_call(examples)
            for i in range(len(batch["labels"])):
                label = batch["labels"][i]
                response_len = len(self.response_template)
                for j in range(len(label) - response_len + 1):
                    if label[j : j + response_len].tolist() == self.response_template:
                        batch["labels"][i, : j + response_len] = -100
                        break
            return batch

    try:
        # Tối ưu hóa đặc biệt cho Qwen để tránh lỗi mã hóa chuỗi
        response_template_str = "<|im_start|>assistant\n"
        response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
        print(f"[TRAIN] Sử dụng Custom DataCollatorForCompletionOnlyLM với template IDs: {response_template_ids} (Qwen format)")
        data_collator = CustomDataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)
    except Exception as e:
        print(f"[WARNING] Lỗi thiết lập DataCollator: {e}. Sẽ chạy mặc định.")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def _build_custom_trainer_class(base_class):
        class CustomTrainer(base_class):
            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                # 1. Tính toán Loss bình thường
                metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
                
                # 2. Sinh văn bản (Generation) để đo Accuracy (TẠM TẮT ĐỂ THỬ NGHIỆM NHANH)
                # ds_to_eval = eval_dataset if eval_dataset is not None else self.eval_dataset
                # if ds_to_eval is not None:
                #     print(f"\n[EVAL] Đo Accuracy sinh text trên {len(ds_to_eval)} mẫu...")
                #     acc = evaluate_accuracy_in_memory(self.model, self.processing_class, ds_to_eval, batch_size=4, max_new_tokens=1500)
                #     metrics[f"{metric_key_prefix}_accuracy"] = acc
                #     print(f"[EVAL] Accuracy: {acc*100:.2f}%")
                #     
                #     # Inject vào log_history để hàm plot_training_history có dữ liệu vẽ
                #     if len(self.state.log_history) > 0:
                #         self.state.log_history[-1][f"{metric_key_prefix}_accuracy"] = acc
                        
                return metrics
        return CustomTrainer

    base_trainer_cls = _build_steered_trainer_class(SFTTrainer) if use_steer else SFTTrainer
    trainer_class = _build_custom_trainer_class(base_trainer_cls)
    trainer_kwargs = {"x_factor": 1.0} if use_steer else {}

    trainer = trainer_class(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        peft_config=peft_config,
        data_collator=data_collator,
        **trainer_kwargs
    )

    trainer.train()
    trainer.save_model(output_dir)
    
    # Vẽ đồ thị training
    plot_name = "steered" if use_steer else "sft"
    plot_path = f"{plot_name}_training_curve.png"
    plot_training_history(trainer.state.log_history, plot_path, title=f"Training Curve ({plot_name.upper()})")
    
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