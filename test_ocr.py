from paddleocr import PaddleOCR
import sys
import os

# Set UTF-8 output so Vietnamese characters display correctly in Windows terminal
sys.stdout.reconfigure(encoding="utf-8")

# --- Change this to your Vietnamese image path ---
sample_img = "./tests/test_files/eq3.jpg"
# -------------------------------------------------

print(f"Running Vietnamese OCR on: {sample_img}")
print("Using model: latin_PP-OCRv5_mobile_rec (covers Vietnamese)\n")

ocr = PaddleOCR(
    lang="vi",                              # <-- KEY: use Vietnamese/Latin model
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    engine="paddle",
)

result = ocr.predict(sample_img)

print("=== OCR Results ===")
for res in result:
    res.print()
    os.makedirs("output", exist_ok=True)
    res.save_to_img("output")
    res.save_to_json("output")

print("\n[DONE] Saved annotated image and JSON to the 'output' folder.")
print("       Open output/ to see the visualized result.")
