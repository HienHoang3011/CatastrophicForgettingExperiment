import os
import math
import argparse
from datasets import Dataset
from src.data_filter import get_filtered_dataset
from src.core import _extract_final_answer
from collections import defaultdict
import random

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

def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu và lưu thành train, eval, test")
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
    parser.add_argument("--samples", type=int, default=50000, help="Số lượng mẫu muốn load (vd: 50000)")
    parser.add_argument("--output-dir", type=str, default="processed_data", help="Thư mục lưu dữ liệu")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n[DATA PREP] Tải {args.samples} mẫu từ {args.dataset}...")
    hf_dataset = get_filtered_dataset(args.dataset, args.dataset_split, args.samples)
    
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

    for row in hf_dataset:
        if len(formatted_data["instruction"]) >= args.samples:
            break
            
        question, solution = _extract_from_numina_row(row)
        
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
        raise ValueError("Không có mẫu nào hợp lệ. Kiểm tra lại dataset.")

    if skipped_no_box > 0:
        print(f"[DATA PREP] Đã bỏ qua {skipped_no_box} mẫu không có \\boxed{{...}}.")

    raw_dataset = Dataset.from_dict(formatted_data)
    
    labels = formatted_data["source"]
    stratified_order = build_stratified_index_order(labels, batch_size=32, seed=42)
    
    # Chia train 90%, eval 5%, test 5%
    n_total = len(stratified_order)
    train_end = int(n_total * 0.90)
    eval_end = int(n_total * 0.95)
    
    train_indices = stratified_order[:train_end]
    eval_indices = stratified_order[train_end:eval_end]
    test_indices = stratified_order[eval_end:]
    
    # Giới hạn eval và test tối đa 500 mẫu nhưng giữ nguyên phân phối (stratified)
    def get_stratified_subset(indices, target_size):
        if len(indices) <= target_size:
            return indices
        from collections import defaultdict
        import random
        
        by_label = defaultdict(list)
        for idx in indices:
            by_label[labels[idx]].append(idx)
            
        rng = random.Random(42)
        sampled = []
        for label, idx_list in by_label.items():
            k = int(round(len(idx_list) / len(indices) * target_size))
            k = min(k, len(idx_list))
            sampled.extend(rng.sample(idx_list, k))
            
        # Bù trừ sai số do làm tròn
        while len(sampled) < target_size:
            remaining = list(set(indices) - set(sampled))
            if not remaining: break
            sampled.append(rng.choice(remaining))
        while len(sampled) > target_size:
            sampled.pop(rng.randint(0, len(sampled)-1))
            
        rng.shuffle(sampled)
        return sampled

    eval_indices = get_stratified_subset(eval_indices, 500)
    test_indices = get_stratified_subset(test_indices, 500)
    train_dataset = raw_dataset.select(train_indices)
    eval_dataset = raw_dataset.select(eval_indices)
    test_dataset = raw_dataset.select(test_indices)
    
    train_path = os.path.join(args.output_dir, "train.jsonl")
    eval_path = os.path.join(args.output_dir, "eval.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    
    print(f"\n[DATA PREP] Đang lưu dữ liệu vào {args.output_dir}...")
    train_dataset.to_json(train_path, force_ascii=False)
    eval_dataset.to_json(eval_path, force_ascii=False)
    test_dataset.to_json(test_path, force_ascii=False)
    
    print(f"[DATA PREP] Hoàn tất!")
    print(f" - Train: {len(train_dataset)} mẫu ({train_path})")
    print(f" - Eval:  {len(eval_dataset)} mẫu ({eval_path})")
    print(f" - Test:  {len(test_dataset)} mẫu ({test_path})")

if __name__ == "__main__":
    main()
