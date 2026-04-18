# LLM Steered Evaluation Pipeline 🚀

Một framework huấn luyện và đánh giá LLM linh hoạt, được thiết kế đặc biệt để đo lường và giảm thiểu hiện tượng **Catastrophic Forgetting** (Lãng quên kiến thức nền tảng) khi thực hiện Full Fine-Tuning trên các tác vụ suy luận (Reasoning) dài.

Framework này được tối ưu hóa đặc biệt cho GPU **NVIDIA H100 (80GB VRAM)**, kết hợp Flash Attention 2 và Batched Inference tốc độ cao.

---

## 🌟 Tính Năng Nổi Bật

- **Steered SFT (Custom Loss):** Tích hợp thuật toán tinh chỉnh phân phối Logit (Margin-based Logit Steering) để giữ lại các kiến thức cũ trong quá trình học tác vụ mới.
- **Tối Ưu VRAM & Tốc Độ:** Sử dụng `bfloat16`, vô hiệu hóa Gradient Checkpointing (phù hợp cho H100) và áp dụng **Flash Attention 2** để tối đa hóa Throughput.
- **Batched Generation Inference:** Đánh giá hàng nghìn mẫu dữ liệu sinh chuỗi dài (Chain-of-Thought) trong thời gian siêu tốc bằng tính năng padding/batching thông minh.
- **Quản Lý Ổ Cứng Thông Minh:** Tự động xóa các Optimizer States rác (tiết kiệm ~13GB mỗi model 3B) và dọn dẹp các mô hình trung gian ngay sau khi đánh giá xong để chạy tốt trên các Cloud Storage nhỏ (ví dụ: ổ cứng 20GB).
- **CLI Linh Hoạt:** Cho phép bật/tắt tùy ý từng giai đoạn thử nghiệm (Base, SFT, Steered) chỉ bằng một dòng lệnh.

---

## 🛠 Yêu Cầu Hệ Thống

- Hệ điều hành: Linux (Ubuntu)
- GPU: NVIDIA H100, A100 hoặc RTX 4090 (Khuyến nghị VRAM >= 24GB. Pipeline mặc định đang thiết kế chạy cực nhanh cho 80GB VRAM).
- Python: `3.10+`
- Trình quản lý gói: `uv` (Nhanh hơn pip rất nhiều)

---

## ⚙️ Cài Đặt

**1. Cài đặt `uv` (Nếu chưa có):**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**2. Khởi tạo và tải dự án:**
```bash
git clone https://github.com/QuangNguyen711/CatastrophicForgettingExperiment.git
cd CatastrophicForgettingExperiment
git clone https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero.git

# 1. Tạo môi trường ảo
uv init --python 3.10
source .venv/bin/activate

# 2.1 Cài dependency chính theo bộ version ổn định (Nếu chưa có, hoặc muốn cài lại từ đầu)
uv add --index https://pypi.org/simple --index https://download.pytorch.org/whl/cu128 \
   'torch==2.8.0' \
   'torchvision==0.23.0' \
   'torchaudio==2.8.0' \
   'transformers==4.57.6' \
   'peft>=0.11,<0.19' \
   'trl>=0.15,<0.25' \
   'deepspeed>=0.14' \
   'vllm==0.11.0'

uv add datasets accelerate bitsandbytes scikit-learn "lm_eval[hf]" "torchao<0.8.0"

# 2.2 Nếu đã có project.toml với uv.lock rồi thì không chạy 2.1
uv sync --no-install-package flash-attn

# 3. Cài FlashAttention
uv add flash-attn==2.8.3 --no-build-isolation
```

---

## 🚀 Hướng Dẫn Sử Dụng (CLI)

Bộ điều khiển trung tâm nằm tại file `main.py`. Bạn có thể tùy biến mọi tham số đầu vào.

Mặc định pipeline hiện dùng dataset Hugging Face: `AI-MO/NuminaMath-CoT` (split `train`).

### Các Lệnh Phổ Biến

**1. Chạy Toàn Bộ 3 Thực Nghiệm (Zero-shot -> SFT -> Steered SFT)**
Đây là luồng chính để xuất ra báo cáo cuối cùng. Quá trình này sẽ huấn luyện, đánh giá và tự động xóa model trung gian.
```bash
uv run main.py --run-all
```

**2. Chỉ Chạy Đánh Giá Base Model (Không Train)**
```bash
uv run main.py --mode baseline
```

**3. Chỉ Chạy Standard SFT + Eval (Không chạy Baseline)**
```bash
uv run main.py --mode sft
```

**4. Chỉ Chạy Huấn Luyện & Đánh Giá Thuật Toán Steered (Custom Loss)**
```bash
uv run main.py --mode steered
```

**5. Chạy SFT + Steered SFT + Eval (Bỏ qua Baseline)**
```bash
uv run main.py --mode sft-steered
```

**6. Chạy Thử Nghiệm Nhanh (Debug Mode)**
Dùng model nhỏ gọn 1.5B, chỉ lấy 1.000 sample và đẩy Batch Size lên 32 để test xem code có chạy mượt không trước khi chạy thật.
```bash
uv run main.py --model Qwen/Qwen2.5-1.5B-Instruct --samples 1000 --batch-size 32 --mode all
```

**7. Chạy với NuminaMath-CoT (chỉ rõ dataset HF nếu muốn override)**
```bash
uv run main.py --dataset AI-MO/NuminaMath-CoT --dataset-split train --mode sft
```

**8. Quay lại dataset JSON local cũ**
```bash
uv run main.py --dataset Open-Reasoner-Zero/data/orz_math_57k_collected.json --mode sft
```

Bạn vẫn có thể dùng cờ cũ (`--run-sft`, `--run-steered`, `--run-all`) để tương thích với script cũ.

---

## 📋 Cấu Trúc Dự Án

```text
qwen-steered-eval/
├── data_repo/                  # Folder chứa dataset tải từ Github
├── src/
│   ├── __init__.py
│   └── core.py                 # Chứa Logic Train, Custom Loss, Eval & Clean VRAM
├── main.py                     # CLI điều khiển trung tâm
├── pyproject.toml              # File quản lý thư viện của uv
├── uv.lock                     # Khóa phiên bản thư viện
└── README.md                   # File tài liệu bạn đang đọc
```

---

## 📊 Hệ Thống Đánh Giá (Evaluation Suite)

Pipeline tự động thực hiện 2 bài kiểm tra để lấy số liệu:

1. **In-distribution (Reasoning Accuracy):** - Đánh giá khả năng giải toán logic trên tập test (Split 20% từ dataset huấn luyện).
   - Prompt format: Chỉ mớm câu hỏi thuần túy (Không dùng System Prompt hướng dẫn suy luận) để kiểm tra mô hình có thực sự tự học được Chain-of-Thought hay không.
2. **Out-of-distribution (Catastrophic Forgetting):**
   - Đánh giá kiến thức nền tảng thông qua thư viện `lm_eval`.
   - Các task mặc định: `HellaSwag` (Common sense reasoning) và `MMLU` (Massive Multitask Language Understanding).

---

## ⚠️ Khắc Phục Sự Cố (Troubleshooting)

- **Lỗi OOM (Out of Memory):** Nếu bạn chạy trên GPU yếu hơn H100, hãy giảm `batch_size` (trong `main.py`) và giảm `per_device_train_batch_size` (trong `src/core.py`). Bật lại `gradient_checkpointing=True`.
- **Lỗi FlashAttention/flash-attn binary (undefined symbol, import error):** Nếu môi trường CUDA/PyTorch không khớp, hãy vào `src/core.py` và dùng `attn_implementation="eager"` (an toàn nhất) hoặc `"sdpa"` thay vì `"flash_attention_2"`.