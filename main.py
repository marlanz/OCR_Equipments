"""
main.py — PaddleOCR-VL-1.5 FastAPI Inference Service

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Install deps:
    pip install "paddlex[ocr]>=3.5.0,<3.6.0"
    pip install "transformers>=5.8.0" torch torchvision
    pip install fastapi uvicorn[standard] python-multipart pydantic
    # GPU (optional):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
"""

import asyncio
import io
import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any  # kept for potential future use

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paddleocr_vl")

# ---------------------------------------------------------------------------
# Global singleton — loaded once at startup, reused across all requests
# ---------------------------------------------------------------------------
_pipeline = None
_startup_time: float = 0.0
_device: str = "cpu"


def _resolve_device() -> str:
    """Mirror the device selection logic in test_vl15.py."""
    return "gpu:0" if torch.cuda.is_available() else "cpu"


def _load_pipeline():
    """Load the PaddleOCR-VL-1.5 pipeline exactly as in test_vl15.py."""
    global _pipeline, _device
    from paddleocr import PaddleOCRVL  # imported here so startup error is clear

    _device = _resolve_device()
    logger.info("Loading PaddleOCR-VL-1.5 | device=%s", _device)

    if _device == "cpu":
        logger.warning(
            "Running on CPU. For GPU/fp16 reinstall PyTorch with CUDA support."
        )

    try:
        _pipeline = PaddleOCRVL(
            pipeline_version="v1.5",
            engine="transformers",           # HuggingFace backend — proper fp16 support
            device=_device,
            use_doc_orientation_classify=False,  # disabled to save VRAM
            use_doc_unwarping=False,             # disabled to save VRAM
            use_layout_detection=True,
        )
        logger.info("Pipeline ready.")
    except Exception as exc:
        logger.exception("Failed to load PaddleOCR-VL-1.5 pipeline")
        raise exc


# ---------------------------------------------------------------------------
# Lifespan — load model on startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_time
    _startup_time = time.time()
    _load_pipeline()
    yield
    logger.info("Shutting down — releasing pipeline.")
    # PaddleOCRVL has no explicit close(); Python GC handles it.


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PaddleOCR-VL-1.5 API",
    description="Document OCR & layout understanding via PaddleOCR Vision-Language 1.5",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "health", "description": "Service liveness & readiness"},
        {"name": "inference", "description": "OCR / document understanding endpoints"},
    ],
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class PageResult(BaseModel):
    """Serialisable representation of a single page's OCR output."""
    markdown: str | None = None
    key_value: dict[str, str] | None = None


class PredictResponse(BaseModel):
    status: str = "ok"
    results: list[PageResult]
    processing_time_seconds: float


# ---------------------------------------------------------------------------
# Helper: parse markdown text into key/value pairs
# ---------------------------------------------------------------------------
def _parse_markdown_to_kv(markdown: str) -> dict[str, str]:
    """
    Parse OCR markdown output into a flat key/value dict.

    Rules:
    - Lines are split by \\n
    - Each line is split on the FIRST ":" to get key and value
    - The first line has no key, so it is stored as "machine_name"
    - Lines with no ":" are appended to the previous key's value

    Example input:
        MÁY HÀN CO2 TÂN THÀNH -TTC-500T
        Mã MMTB : B22400814
        Model : TTC-500T

    Example output:
        {
            "machine_name": "MÁY HÀN CO2 TÂN THÀNH -TTC-500T",
            "Mã MMTB": "B22400814",
            "Model": "TTC-500T",
        }
    """
    kv: dict[str, str] = {}
    last_key: str | None = None

    for i, raw_line in enumerate(markdown.splitlines()):
        line = raw_line.strip()
        if not line:
            continue

        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            # First line might itself contain a colon (e.g. a model code).
            # If i == 0 and the key looks like a name (not a label), treat
            # the whole line as machine_name. Otherwise use key:value split.
            if i == 0 and not kv:
                kv["machine_name"] = line
                last_key = "machine_name"
            else:
                kv[key] = value
                last_key = key
        else:
            # No colon — first line becomes machine_name, subsequent
            # keyless lines are appended to the previous key.
            if not kv:
                kv["machine_name"] = line
                last_key = "machine_name"
            elif last_key:
                kv[last_key] = (kv[last_key] + " " + line).strip()

    return kv


# ---------------------------------------------------------------------------
# Helper: extract result data from PaddleOCRVL output objects
# ---------------------------------------------------------------------------
def _extract_result(res) -> PageResult:
    """
    Convert a single PaddleOCRVL result object into a serialisable dict.
    The result objects expose .print() / .save_to_json() / .save_to_markdown().
    We capture them in-memory instead of writing to disk.
    """
    markdown_text: str | None = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            res.save_to_markdown(tmp_dir)
            md_files = list(Path(tmp_dir).glob("*.md"))
            if md_files:
                markdown_text = md_files[0].read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not extract markdown: %s", exc)

    kv = _parse_markdown_to_kv(markdown_text) if markdown_text else None
    return PageResult(markdown=markdown_text, key_value=kv)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """Liveness + readiness check."""
    return HealthResponse(
        status="ok" if _pipeline is not None else "loading",
        model_loaded=_pipeline is not None,
        device=_device,
        uptime_seconds=round(time.time() - _startup_time, 2),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Run VL-1.5 OCR on an uploaded image or a local file path",
)
async def predict(
    file: UploadFile | None = File(default=None, description="Image file to process"),
    image_path: str | None = Form(default=None, description="Server-side file path (alternative to upload)"),
):
    """
    Accepts either:
    - `file`       — a multipart image upload (JPEG, PNG, TIFF, BMP, PDF page, etc.)
    - `image_path` — an absolute path to a file already on the server

    Returns OCR results as structured JSON and Markdown per page/region.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if file is None and not image_path:
        raise HTTPException(
            status_code=422,
            detail="Provide either a `file` upload or an `image_path` form field.",
        )

    t0 = time.time()

    # Resolve the input path — write upload to a temp file if necessary
    tmp_file_path: str | None = None
    try:
        if file is not None:
            suffix = Path(file.filename or "upload").suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_file_path = tmp.name
            input_path = tmp_file_path
        else:
            input_path = image_path
            if not os.path.exists(input_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"image_path not found on server: {input_path}",
                )

        logger.info("Running inference on: %s", input_path)

        # --- Inference (run off-thread to avoid blocking event loop) ---
        def run_inference_and_extraction():
            with torch.no_grad():
                raw_results = list(_pipeline.predict(input_path))
            return [_extract_result(res) for res in raw_results]

        page_results = await asyncio.to_thread(run_inference_and_extraction)

        elapsed = round(time.time() - t0, 3)
        logger.info("Inference done in %.3fs — %d page(s)", elapsed, len(page_results))

        return PredictResponse(
            status="ok",
            results=page_results,
            processing_time_seconds=elapsed,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


@app.post(
    "/predict/path",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Run VL-1.5 OCR on a server-side file path (JSON body)",
)
async def predict_by_path(body: dict):
    """
    JSON body convenience endpoint:
        { "image_path": "/absolute/path/to/image.jpg" }
    Useful for server-to-server calls where the file is already on disk.
    """
    image_path = body.get("image_path")
    if not image_path:
        raise HTTPException(status_code=422, detail="`image_path` required in body.")

    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"File not found: {image_path}")

    t0 = time.time()
    try:
        # --- Inference (run off-thread to avoid blocking event loop) ---
        def run_inference_and_extraction():
            with torch.no_grad():
                raw_results = list(_pipeline.predict(image_path))
            return [_extract_result(res) for res in raw_results]

        page_results = await asyncio.to_thread(run_inference_and_extraction)
        elapsed = round(time.time() - t0, 3)

        return PredictResponse(
            status="ok",
            results=page_results,
            processing_time_seconds=elapsed,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)