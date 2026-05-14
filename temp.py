import argparse
import os
import gc
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from huggingface_hub import HfApi, create_repo

def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()

def _import_trl_or_raise():
    try:
        from trl import SFTTrainer, SFTConfig
    except Exception as e:
        raise RuntimeError("Failed to import TRL training components.") from e
    return SFTTrainer, SFTConfig


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

def train_model(base_model, train_dataset, output_dir, learning_rate=5e-6, num_train_epochs=1):
    print(f"\n[TRAIN] Starting Training. Output: {output_dir}")
    SFTTrainer, SFTConfig = _import_trl_or_raise()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cuda", 
        trust_remote_code=True
    )
    
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
        gradient_checkpointing=True,
        logging_steps=10,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="no",  # Đã bỏ eval
        save_strategy="no",
        save_only_model=True,
        optim="adamw_torch_fused",
        report_to="none",
    )

    try:
        response_template_str = "<|im_start|>assistant\n"
        response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
        print(f"[TRAIN] Sử dụng Custom DataCollatorForCompletionOnlyLM với template IDs: {response_template_ids}")
        data_collator = CustomDataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)
    except Exception as e:
        print(f"[WARNING] Lỗi thiết lập DataCollator: {e}. Sẽ chạy mặc định.")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    
    del trainer
    del model
    del tokenizer
    clean_memory()

def main():
    parser = argparse.ArgumentParser(description="Hệ thống Huấn luyện và Đẩy lên HuggingFace")
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Đường dẫn file JSONL local đã qua preprocess")
    parser.add_argument("--samples", type=int, default=50000, help="Số lượng mẫu muốn load")
    parser.add_argument("--train-epochs", type=float, default=1.0, help="Số epoch train")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are an expert psychological assistant. "
            "Analyze the scenario carefully, considering psychological principles and concepts, "
            "and provide a detailed explanation for your conclusion."
        ),
        help="System prompt dùng chung cho train formatting"
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Đẩy model lên HuggingFace Hub sau khi train")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token")
    parser.add_argument("--hub-model-id", type=str, help="Tên repo trên HuggingFace Hub")

    args = parser.parse_args()

    if args.push_to_hub and (not args.hf_token or not args.hub_model_id):
        print("❌ LỖI: Bạn đã chọn --push-to-hub nhưng chưa cung cấp --hf-token hoặc --hub-model-id!")
        return

    print(f"🚀 Khởi chạy Pipeline với Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[DATA] Tải dữ liệu từ file JSONL: {args.dataset}")
    # Không cần chia 80/20 nữa vì không có bước eval
    train_ds = load_dataset("json", data_files=args.dataset, split="train")
    
    if args.samples and args.samples < len(train_ds):
        train_ds = train_ds.select(range(args.samples))
        
    def format_train(examples):
        texts = []
        for q, a in zip(examples['instruction'], examples['output']):
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    print("[DATA] Áp dụng chat template...")
    train_ds = train_ds.map(format_train, batched=True)

    temp_tokenizer_dir = "./temp_tokenizer"
    tokenizer.save_pretrained(temp_tokenizer_dir)

    del tokenizer
    clean_memory()

    sft_dir = "./model_sft"

    print("\n" + "="*50 + "\n[TRAINING] SFT\n" + "="*50)
    train_model(
        args.model,
        train_ds,
        sft_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.train_epochs,
    )
    
    if args.push_to_hub:
        print(f"\n[PUSH TO HUB] Pushing model to {args.hub_model_id}...")
        try:
            api = HfApi(token=args.hf_token)
            repo_id = args.hub_model_id
            create_repo(repo_id=repo_id, token=args.hf_token, exist_ok=True)
            
            import shutil
            for filename in os.listdir(temp_tokenizer_dir):
                shutil.copy2(os.path.join(temp_tokenizer_dir, filename), sft_dir)
                
            api.upload_folder(
                folder_path=sft_dir,
                repo_id=repo_id,
                repo_type="model",
            )
            print("✅ Đã push model thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi push model: {e}")

    if os.path.exists(temp_tokenizer_dir):
        import shutil
        shutil.rmtree(temp_tokenizer_dir)

if __name__ == "__main__":
    main()
