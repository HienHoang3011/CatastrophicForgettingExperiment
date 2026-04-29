from datasets import load_dataset
from collections import Counter
import pandas as pd

def print_dataset_stats(dataset_name="AI-MO/NuminaMath-CoT", split="train"):
    print(f"Đang tải dataset: {dataset_name} (split: {split})...")
    print("Quá trình này có thể mất vài phút nếu chưa tải trước đó.\n")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Lỗi khi tải dataset: {e}")
        return

    print("="*60)
    print("1. THÔNG TIN CƠ BẢN")
    print("="*60)
    print(f"- Tổng số mẫu (rows)     : {len(ds):,}")
    print(f"- Các trường (columns)   : {', '.join(ds.column_names)}")
    
    # In ra ví dụ độ dài của text
    print("\n" + "="*60)
    print("2. THỐNG KÊ CÁC TRƯỜNG DỮ LIỆU (SUBJECT / SOURCE)")
    print("="*60)
    
    # Bộ NuminaMath thường lưu thể loại / nguồn bài toán ở cột 'source'
    subject_col = None
    if 'source' in ds.column_names:
        subject_col = 'source'
    elif 'subject' in ds.column_names:
        subject_col = 'subject'
    elif 'type' in ds.column_names:
        subject_col = 'type'
        
    if subject_col:
        print(f"Phân loại theo cột: '{subject_col}'")
        counts = Counter(ds[subject_col])
        
        # Tạo bảng thống kê
        stats_df = pd.DataFrame.from_dict(counts, orient='index', columns=['Số lượng'])
        stats_df['Tỉ lệ (%)'] = (stats_df['Số lượng'] / len(ds) * 100).round(2)
        stats_df = stats_df.sort_values(by='Số lượng', ascending=False)
        
        print("-" * 50)
        print(f"{'Loại (Subject/Source)':<30} | {'Số lượng':<10} | {'Tỉ lệ (%)'}")
        print("-" * 50)
        for index, row in stats_df.iterrows():
            print(f"{str(index):<30} | {row['Số lượng']:<10,} | {row['Tỉ lệ (%)']}%")
        print("-" * 50)
        print(f"Tổng số phân loại khác nhau: {len(stats_df)}")
    else:
        print(f"Không tìm thấy trường nào chứa subject/source.")

    print("\n" + "="*60)
    print("3. KIỂM TRA TRÙNG LẶP (DUPLICATES)")
    print("="*60)
    
    # Kiểm tra trùng lặp trên cột problem / question (câu hỏi)
    question_col = None
    if 'problem' in ds.column_names:
        question_col = 'problem'
    elif 'question' in ds.column_names:
        question_col = 'question'
        
    if question_col:
        # Đếm duplicates
        problems = ds[question_col]
        df = pd.DataFrame({question_col: problems})
        
        total_dupes = df.duplicated(subset=[question_col]).sum()
        
        print(f"- Đã kiểm tra trùng lặp dựa trên cột: '{question_col}'")
        print(f"- Số lượng câu hỏi bị trùng lặp chính xác: {total_dupes:,} mẫu")
        print(f"- Tỉ lệ trùng lặp: {(total_dupes / len(ds) * 100):.2f}%")
        
        if total_dupes > 0:
            # Thống kê câu nào lặp nhiều nhất
            print("\n- Top 3 câu hỏi lặp nhiều nhất:")
            top_dupes = df[question_col].value_counts().head(3)
            for text, count in top_dupes.items():
                print(f"  + Xuất hiện {count} lần: {str(text)[:100]}...")
    else:
        print(f"Không tìm thấy cột 'problem' hoặc 'question' để kiểm tra trùng lặp.")
        
    print("\n" + "="*60)
    print("HOÀN TẤT THỐNG KÊ")
    print("="*60)

if __name__ == "__main__":
    print_dataset_stats()
