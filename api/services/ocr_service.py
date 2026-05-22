import time
import asyncio
import numpy as np
import logging
from typing import Dict, List, Union
from PIL import Image
from paddleocr import PaddleOCR
from paddleocr import PaddleOCRVL
from api.core.config import settings

logger = logging.getLogger(__name__)

class OCRService:
    """
    A service that wraps the PaddleOCR engine in a singleton pattern
    to reuse model instances across requests and handle language caching.
    """
    _instances: Dict[str, PaddleOCR] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get_engine(cls, lang: str) -> PaddleOCR:
        """
        Retrieves or initializes a PaddleOCR instance for the requested language.
        """
        lang = lang.lower().strip()
        if lang not in cls._instances:
            async with cls._lock:
                if lang not in cls._instances:
                    logger.info(f"Initializing PaddleOCR model instance for language: '{lang}' (use_gpu={settings.USE_GPU})...")
                    
                    # Model loading can block the event loop, so run it in the default thread pool executor
                    loop = asyncio.get_running_loop()
                    try:
                        # Let's pass configuration parameters from environment settings
                        engine = await loop.run_in_executor(
                            None,
                            # lambda: PaddleOCR(
                            #     lang=lang,
                            #     device="gpu" if settings.USE_GPU else "cpu",
                            #     use_doc_orientation_classify=False,
                            #     use_doc_unwarping=False,
                            #     use_textline_orientation=False,
                            #     engine="paddle"
                            # )
                            lambda: PaddleOCRVL(
                                pipeline_version="v1.5",
                                engine="transformers",          # HuggingFace transformers - proper fp16 support
                                device='cpu',
                                use_doc_orientation_classify=False,  # Disable to save VRAM
                                use_doc_unwarping=False,             # Disable to save VRAM
                                use_layout_detection=True,
                             # lang='vi'
                            )
                        )
                        cls._instances[lang] = engine
                        logger.info(f"PaddleOCR model instance for language: '{lang}' loaded successfully.")
                    except Exception as e:
                        logger.error(f"Failed to initialize PaddleOCR engine for language '{lang}': {e}")
                        raise RuntimeError(f"Failed to initialize OCR engine for language '{lang}'") from e
        return cls._instances[lang]

    @classmethod
    async def run_ocr(cls, image: Union[str, np.ndarray, Image.Image], lang: str = None) -> dict:
        """
        Processes an image using the loaded PaddleOCR model for the specified language.
        Runs OCR inference in a background thread to prevent blocking the asyncio event loop.
        """
        lang = lang or settings.DEFAULT_LANG
        engine = await cls.get_engine(lang)

        # Convert input image to a format expected by PaddleOCR (BGR numpy array)
        if isinstance(image, Image.Image):
            # Convert PIL RGB to BGR numpy array
            img_rgb = np.array(image.convert("RGB"))
            img_bgr = img_rgb[..., ::-1]
            input_data = np.ascontiguousarray(img_bgr)
        elif isinstance(image, np.ndarray):
            input_data = image
        elif isinstance(image, str):
            input_data = image
        else:
            raise ValueError("Unsupported image type for OCR service")

        start_time = time.perf_counter()
        
        # Run inference in a thread pool executor to keep FastAPI async loop free
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: engine.predict(input_data)
            )
        except Exception as e:
            logger.error(f"OCR inference execution failed: {e}")
            raise RuntimeError(f"OCR engine prediction failed: {str(e)}") from e

        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000

        clean_texts = []
        confidences = []
        text_lines = []

        if results and len(results) > 0:
            for result in results:
                # Handle direct dictionary/object subscripting safely
                try:
                    texts = result["rec_texts"]
                    scores = result["rec_scores"]
                    
                    # Prioritize polygon coordinates (dt_polys/rec_polys) over flat boxes (rec_boxes)
                    boxes = None
                    for key in ["dt_polys", "rec_polys", "rec_boxes"]:
                        if key in result:
                            boxes = result[key]
                            break
                    if boxes is None:
                        boxes = []
                except (TypeError, KeyError, IndexError) as e:
                    logger.warning(f"Could not parse fields directly from OCR result object: {e}")
                    continue

                for i, text in enumerate(texts):
                    if text and text.strip():
                        # Extract confidence score
                        conf = 0.0
                        if i < len(scores):
                            try:
                                conf = float(scores[i])
                            except (TypeError, ValueError):
                                pass
                        
                        clean_texts.append(text.strip())
                        confidences.append(conf)
                        
                        # Extract coordinates
                        bbox = []
                        if i < len(boxes):
                            box_val = boxes[i]
                            # Bounding box is typically: numpy array shape (4, 2) or (4,)
                            if hasattr(box_val, "tolist"):
                                bbox_raw = box_val.tolist()
                            else:
                                bbox_raw = list(box_val)
                            
                            # Handle coordinates nesting list formatting: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                            try:
                                if len(bbox_raw) == 4 and any(isinstance(pt, (list, tuple, np.ndarray)) for pt in bbox_raw):
                                    bbox = [[float(coord) for coord in pt] for pt in bbox_raw]
                                elif len(bbox_raw) == 4 and all(isinstance(pt, (int, float)) for pt in bbox_raw):
                                    # Convert [x_min, y_min, x_max, y_max] to 4-point polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                    x_min, y_min, x_max, y_max = bbox_raw
                                    bbox = [
                                        [float(x_min), float(y_min)],
                                        [float(x_max), float(y_min)],
                                        [float(x_max), float(y_max)],
                                        [float(x_min), float(y_max)]
                                    ]
                                else:
                                    # Fallback flat list mapped to pairs
                                    bbox = [[float(coord)] for coord in bbox_raw]
                            except Exception as ex:
                                logger.warning(f"Coordinate parser fallback: {ex}")
                                bbox = bbox_raw
                        
                        text_lines.append({
                            "text": text.strip(),
                            "confidence": round(conf, 4),
                            "bbox": bbox
                        })

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "text": "\n".join(clean_texts),
            "confidence": round(avg_confidence, 4),
            "text_lines": text_lines,
            "processing_time_ms": round(processing_time_ms, 2)
        }
