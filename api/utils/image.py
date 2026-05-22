import io
import base64
import httpx
import logging
from PIL import Image, ImageOps
from fastapi import HTTPException, UploadFile, status
from api.core.config import settings

logger = logging.getLogger(__name__)

def validate_image_file(file: UploadFile):
    """
    Validates file extension against allowed types.
    """
    filename = file.filename or ""
    if "." not in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File has no extension."
        )
    ext = filename.split(".")[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension: {ext}. Allowed extensions: {settings.ALLOWED_EXTENSIONS}"
        )

def check_image_bytes_size(image_bytes: bytes):
    """
    Enforces maximum upload file size.
    """
    if len(image_bytes) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds maximum limit of {settings.MAX_FILE_SIZE / (1024 * 1024):.1f} MB."
        )

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Parses image bytes and runs verification tests. Auto-transposes rotation from EXIF metadata.
    """
    check_image_bytes_size(image_bytes)
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verifies the file integrity
        
        # PIL.Image.verify() closes the file pointer; reopen to perform pixel operations
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # Load pixel data to trigger decoding
        
        # Correct orientation based on EXIF tag (e.g. smartphone shots taken sideways)
        img = ImageOps.exif_transpose(img)
        return img
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or corrupted image data."
        )

def decode_base64_image(base64_str: str) -> Image.Image:
    """
    Decodes a base64 encoded image string.
    """
    try:
        # Strip header if present (e.g., data:image/png;base64,...)
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        
        # Decode base64 bytes
        img_bytes = base64.b64decode(base64_str)
        return load_image_from_bytes(img_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 string or failed to decode."
        )

async def download_image_from_url(url: str) -> Image.Image:
    """
    Downloads image from url with configured timeout.
    """
    try:
        timeout = httpx.Timeout(settings.TIMEOUT_SECONDS)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to download image from URL. HTTP status code {response.status_code}."
                )
            
            content_type = response.headers.get("content-type", "")
            if "image" not in content_type:
                logger.warning(f"Downloaded URL content-type is '{content_type}', proceeding with byte checks.")
                
            return load_image_from_bytes(response.content)
    except httpx.RequestError as e:
        logger.error(f"HTTP request error downloading image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to fetch image from URL: network or timeout error."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading image from URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching the image from URL."
        )
