import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

from paddleocr import PaddleOCRVL

# --- Change this to your image path ---
sample_img = "./tests/test_files/ocr_eq2.jpg"
# --------------------------------------

print("=" * 60)
print("PaddleOCR-VL-1.5 via Transformers backend (native fp16)")
print("=" * 60)
print(f"Input: {sample_img}\n")

# Use Transformers engine - it handles fp16 natively for VLMs
# This avoids the Paddle fp32 VRAM ceiling issue on 4GB cards
pipeline = PaddleOCRVL(
    pipeline_version="v1.5",
    engine="transformers",          # HuggingFace transformers - proper fp16 support
    device="gpu:0",
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
