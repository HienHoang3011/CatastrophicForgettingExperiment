# src/data_filter.py
from datasets import load_dataset, Dataset
import pandas as pd
import re

def _normalize_text(text: str) -> str:
    if not text: return ""
    return re.sub(r'\s+', '', str(text).lower())

def get_filtered_dataset(dataset_path: str, split: str, max_samples: int):
    """
    Tải tập dữ liệu gốc, lọc bỏ các mẫu có source là gsm8k, 
    loại bỏ trùng lặp theo problem/answer và lấy max_samples theo đúng phân phối.
    """
    print(f"\n[DATA FILTER] Đang tải {dataset_path}...")
    import os
    if os.path.exists(dataset_path):
        ds_main = load_dataset("json", data_files=dataset_path, split="train")
    else:
        ds_main = load_dataset(dataset_path, split=split)
    
    # 1. Lọc bỏ các mẫu có source gsm8k trực tiếp từ dataset
    def filter_source(example):
        source = str(example.get('source', '')).lower()
        if 'gsm8k' in source:
            return False
        return True
    
    ds_no_gsm8k = ds_main.filter(filter_source, num_proc=8, desc="Lọc bỏ source gsm8k")
    print(f"[DATA FILTER] Lọc bỏ gsm8k: Còn lại {len(ds_no_gsm8k)}/{len(ds_main)} mẫu.")
    
    # 2. Xử lý trùng lặp (Dùng Pandas để nhanh gọn và tiện lợi)
    print(f"[DATA FILTER] Chuyển đổi sang Pandas để kiểm tra trùng lặp...")
    df = ds_no_gsm8k.to_pandas()
    
    questions = []
    answers = []
    
    # Trích xuất q và a với tốc độ cao
    for p, s, m in zip(df.get('problem', [None]*len(df)), df.get('solution', [None]*len(df)), df.get('messages', [None]*len(df))):
        q = p if p else ""
        a = s if s else ""
        
        if not q or not a:
            if isinstance(m, list) or hasattr(m, '__iter__'):
                for msg in m:
                    # Kiểm tra msg có phải dict không, do pandas có thể parse thành object
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "user" and not q: 
                            q = content
                        elif role == "assistant" and content: 
                            a = content
                            
        questions.append(q)
        answers.append(a)
        
    df['norm_q'] = [_normalize_text(q) for q in questions]
    df['norm_a'] = [_normalize_text(a) for a in answers]
    
    # Xóa dòng có problem rỗng hoặc answer rỗng (tránh bị lọc trùng lặp sai)
    df = df[(df['norm_q'] != "") & (df['norm_a'] != "")]
    
    # Lọc trùng lặp theo norm_q trước, sau đó lọc theo norm_a
    len_before = len(df)
    df = df.drop_duplicates(subset=['norm_q'])
    df = df.drop_duplicates(subset=['norm_a'])
    
    df = df.drop(columns=['norm_q', 'norm_a'])
    print(f"[DATA FILTER] Đã lọc trùng lặp: Loại bỏ {len_before - len(df)} mẫu. Còn lại {len(df)} mẫu.")

    # 3. Lấy mẫu theo phân phối của cột source
    if max_samples >= len(df):
        print(f"[DATA FILTER] Số mẫu yêu cầu ({max_samples}) >= Số mẫu hiện có ({len(df)}). Lấy toàn bộ & Shuffle.")
        sampled_df = df.sample(frac=1.0, random_state=42)
    else:
        print(f"[DATA FILTER] Lấy {max_samples} mẫu theo phân phối chuẩn của dữ liệu còn lại...")
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            proportions = source_counts / len(df)
            
            sampled_dfs = []
            for source_val, prop in proportions.items():
                # Tính số lượng cần lấy cho từng source
                n_samples_for_source = int(prop * max_samples)
                df_source = df[df['source'] == source_val]
                n_samples_for_source = min(n_samples_for_source, len(df_source))
                
                sampled_dfs.append(df_source.sample(n=n_samples_for_source, random_state=42))
                
            sampled_df = pd.concat(sampled_dfs)
            
            # Nếu thiếu do làm tròn, lấy thêm ngẫu nhiên
            shortfall = max_samples - len(sampled_df)
            if shortfall > 0:
                remaining_df = df.drop(sampled_df.index)
                if not remaining_df.empty:
                    extra_samples = remaining_df.sample(n=min(shortfall, len(remaining_df)), random_state=42)
                    sampled_df = pd.concat([sampled_df, extra_samples])
            
            # Shuffle lần cuối
            sampled_df = sampled_df.sample(frac=1.0, random_state=42)
        else:
            # Fallback nếu không có cột source
            sampled_df = df.sample(n=max_samples, random_state=42)
            
    print(f"[DATA FILTER] Hoàn tất lấy {len(sampled_df)} mẫu. Chuyển lại về HuggingFace Dataset...")
    return Dataset.from_pandas(sampled_df, preserve_index=False)
