import sys
import os
import uuid
import json
import shutil
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PaddleOCR VL-1.5 API",
    description="Upload an image and get back structured OCR results as JSON + Markdown.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output_vl")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Pipeline (loaded once at startup) ─────────────────────────────────────────
device = "gpu:0" if torch.cuda.is_available() else "cpu"
pipeline = None  # lazy-loaded on first request to keep startup fast


def get_pipeline():
    global pipeline
    if pipeline is None:
        from paddleocr import PaddleOCRVL
        pipeline = PaddleOCRVL(
            pipeline_version="v1.5",
            engine="transformers",
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
        )
    return pipeline


# ── Response schema ────────────────────────────────────────────────────────────
class OCRResponse(BaseModel):
    request_id: str
    filename: str
    device: str
    pages: int
    json_path: str
    markdown_path: str
    results: list  # raw per-page dicts from PaddleOCR


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "docs": "/docs",
    }

@app.get("/debug", tags=["Debug"])
def debug_versions():
    import importlib
    versions = {}
    for pkg in ["torch", "torchvision", "transformers", "paddlex", "paddlepaddle"]:
        try:
            m = importlib.import_module(pkg)
            versions[pkg] = getattr(m, "__version__", "unknown")
        except Exception as e:
            versions[pkg] = f"NOT FOUND: {e}"

    # Check if AutoImageProcessor is importable directly
    try:
        from transformers import AutoImageProcessor
        versions["AutoImageProcessor"] = "OK"
    except Exception as e:
        versions["AutoImageProcessor"] = f"FAILED: {e}"

    # Check paddlex's own transformers path
    try:
        import paddlex
        paddlex_path = paddlex.__file__
        versions["paddlex_path"] = paddlex_path
    except Exception as e:
        versions["paddlex_path"] = f"FAILED: {e}"

    return versions

@app.post("/ocr", response_model=OCRResponse, tags=["OCR"])
async def run_ocr(
    file: UploadFile = File(..., description="Image file to process (jpg, png, bmp, tiff, webp)"),
    save_files: bool = Query(True, description="Whether to persist JSON and Markdown outputs to disk"),
):
    """
    Run PaddleOCR VL-1.5 on an uploaded image.

    Returns:
    - **results**: structured OCR data (bounding boxes, text, confidence)
    - **json_path**: path to the saved JSON output file
    - **markdown_path**: path to the saved Markdown output file
    """
    allowed_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_exts)}",
        )

    request_id = uuid.uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f"{request_id}{ext}"
    out_dir = OUTPUT_DIR / request_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save upload to disk
    try:
        with upload_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Run OCR
    try:
        pl = get_pipeline()
        raw_results = pl.predict(str(upload_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR pipeline error: {e}")

    # Collect results and save outputs
    results_list = []
    json_path = ""
    markdown_path = ""

    for i, res in enumerate(raw_results):
        # save_to_json / save_to_markdown write files named after the input image
        if save_files:
            res.save_to_json(str(out_dir))
            res.save_to_markdown(str(out_dir))

        # Convert result to a plain dict for the response
        try:
            # PaddleOCR results expose .json property or can be serialised via json module
            page_dict = json.loads(json.dumps(res.json, default=str)) if hasattr(res, "json") else {}
        except Exception:
            page_dict = {}
        results_list.append(page_dict)

    # Locate saved files (PaddleOCR names them after the original stem)
    stem = upload_path.stem
    json_files = list(out_dir.glob("*.json"))
    md_files = list(out_dir.glob("*.md"))
    json_path = str(json_files[0]) if json_files else ""
    markdown_path = str(md_files[0]) if md_files else ""

    return OCRResponse(
        request_id=request_id,
        filename=file.filename,
        device=device,
        pages=len(results_list),
        json_path=json_path,
        markdown_path=markdown_path,
        results=results_list,
    )


@app.get("/result/{request_id}/json", tags=["Download"])
def download_json(request_id: str):
    """Download the raw JSON output for a previous request."""
    out_dir = OUTPUT_DIR / request_id
    files = list(out_dir.glob("*.json")) if out_dir.exists() else []
    if not files:
        raise HTTPException(status_code=404, detail="JSON output not found for this request_id")
    return FileResponse(files[0], media_type="application/json", filename=files[0].name)


@app.get("/result/{request_id}/markdown", tags=["Download"])
def download_markdown(request_id: str):
    """Download the Markdown output for a previous request."""
    out_dir = OUTPUT_DIR / request_id
    files = list(out_dir.glob("*.md")) if out_dir.exists() else []
    if not files:
        raise HTTPException(status_code=404, detail="Markdown output not found for this request_id")
    return FileResponse(files[0], media_type="text/markdown", filename=files[0].name)


@app.get("/results", tags=["History"])
def list_results():
    """List all past request IDs."""
    ids = [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    return {"count": len(ids), "request_ids": sorted(ids, reverse=True)}