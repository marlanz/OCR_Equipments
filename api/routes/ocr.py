import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from api.schemas.response import OCRResponse, BatchOCRResponse, OCRUrlRequest, OCRBase64Request
from api.services.ocr_service import OCRService
from api.utils.image import (
    validate_image_file,
    load_image_from_bytes,
    decode_base64_image,
    download_image_from_url
)
from api.utils.rate_limit import rate_limiter

router = APIRouter(prefix="/ocr", tags=["OCR"])
logger = logging.getLogger(__name__)

@router.post("/image", response_model=OCRResponse, dependencies=[Depends(rate_limiter)])
async def ocr_image(
    file: UploadFile = File(..., description="The image file to process"),
    lang: Optional[str] = Form(None, description="Optional OCR language code (e.g. vi, en, ch)")
):
    """
    Upload an image file and perform OCR processing on it.
    """
    validate_image_file(file)
    try:
        content = await file.read()
        img = load_image_from_bytes(content)
        res = await OCRService.run_ocr(img, lang)
        return OCRResponse(success=True, result=res)
    except HTTPException as e:
        # Re-raise HTTPExceptions so they bypass general status 500 handler
        raise e
    except Exception as e:
        logger.error(f"OCR Image endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )

@router.post("/base64", response_model=OCRResponse, dependencies=[Depends(rate_limiter)])
async def ocr_base64(
    request: OCRBase64Request
):
    """
    Submit a base64 encoded image string for OCR extraction.
    """
    try:
        img = decode_base64_image(request.base64_str)
        res = await OCRService.run_ocr(img, request.lang)
        return OCRResponse(success=True, result=res)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"OCR Base64 endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )

@router.post("/url", response_model=OCRResponse, dependencies=[Depends(rate_limiter)])
async def ocr_url(
    request: OCRUrlRequest
):
    """
    Submit a remote image URL to download and process.
    """
    try:
        img = await download_image_from_url(str(request.url))
        res = await OCRService.run_ocr(img, request.lang)
        return OCRResponse(success=True, result=res)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"OCR URL endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}"
        )

@router.post("/batch", response_model=BatchOCRResponse, dependencies=[Depends(rate_limiter)])
async def ocr_batch(
    files: List[UploadFile] = File(..., description="List of image files to process"),
    lang: Optional[str] = Form(None, description="Optional OCR language code")
):
    """
    Process multiple uploaded image files in batch. Individual file failures are caught
    and reported in the list structure without failing the entire request.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided."
        )

    results = []
    for file in files:
        try:
            validate_image_file(file)
            content = await file.read()
            img = load_image_from_bytes(content)
            res = await OCRService.run_ocr(img, lang)
            results.append(OCRResponse(success=True, result=res))
        except HTTPException as e:
            logger.warning(f"File {file.filename} failed validation or loading: {e.detail}")
            results.append(OCRResponse(
                success=False,
                error=f"File {file.filename}: {e.detail}",
                result=None
            ))
        except Exception as e:
            logger.error(f"Batch file {file.filename} failed during processing: {e}")
            results.append(OCRResponse(
                success=False,
                error=f"File {file.filename}: Processing failed - {str(e)}",
                result=None
            ))

    return BatchOCRResponse(success=True, results=results)
