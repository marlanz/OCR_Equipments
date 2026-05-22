import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import torch
from paddleocr import PaddleOCRVL

# --- Change this to your image path ---
sample_img = "./tests/test_files/eq3.jpg"
# --------------------------------------

# VL-1.5 needs the doc-parser extra (not just ocr-core):
#   pip install "paddlex[ocr]>=3.5.0,<3.6.0"
#   pip install "transformers>=5.8.0" torch torchvision
# For GPU + fp16, install CUDA PyTorch (CPU torch cannot use device="gpu:0"):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

device = "gpu:0" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("PaddleOCR-VL-1.5 via Transformers backend (native fp16)")
print("=" * 60)
print(f"Input: {sample_img}")
print(f"Device: {device} (cuda available: {torch.cuda.is_available()})\n")

if device == "cpu":
    print(
        "WARNING: Running on CPU. For GPU/fp16, reinstall PyTorch with CUDA support "
        "(see install notes at top of this script).\n"
    )

# Use Transformers engine - it handles fp16 natively for VLMs
# This avoids the Paddle fp32 VRAM ceiling issue on 4GB cards
pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    engine="transformers",          # HuggingFace transformers - proper fp16 support
    device=device,
    use_doc_orientation_classify=False,  # Disable to save VRAM
    use_doc_unwarping=False,             # Disable to save VRAM
    use_layout_detection=True,
    # lang='vi'
)

print("Running VL-1.5 inference...")
result = pipeline.predict(sample_img)

os.makedirs("output_vl", exist_ok=True)
print("\n=== VL-1.5 Results ===")
for res in result:
    res.print()
    res.save_to_json("output_vl")
    res.save_to_markdown("output_vl")

print("\n[DONE] Results saved to output_vl/")
print("  - Markdown: full document as clean Markdown")
print("  - JSON: structured layout + text data")
