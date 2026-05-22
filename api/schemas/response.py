from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class OCRTextLine(BaseModel):
    text: str = Field(..., description="Extracted text line content")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0")
    bbox: List[List[float]] = Field(..., description="Bounding box polygon coordinates: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]")

class OCRResult(BaseModel):
    text: str = Field(..., description="All extracted text joined by newlines")
    confidence: float = Field(..., description="Average confidence score")
    text_lines: List[OCRTextLine] = Field(..., description="List of individual text lines with boxes and scores")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class OCRResponse(BaseModel):
    success: bool = Field(True, description="True if OCR was successful")
    result: Optional[OCRResult] = Field(None, description="OCR extraction results")
    error: Optional[str] = Field(None, description="Error message if success is False")

class BatchOCRResponse(BaseModel):
    success: bool = Field(True, description="True if batch processing completed")
    results: List[OCRResponse] = Field(..., description="List of individual OCR responses")

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Status of the API service")
    gpu_available: bool = Field(..., description="Whether a GPU is active for PaddleOCR")
    version: str = Field(..., description="API version")

class VersionResponse(BaseModel):
    version: str = Field(..., description="API version")
    title: str = Field(..., description="API title")

class OCRUrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="The URL of the image to process")
    lang: Optional[str] = Field(None, description="Optional language parameter (e.g. 'vi', 'en', 'ch')")

class OCRBase64Request(BaseModel):
    base64_str: str = Field(..., description="Base64 encoded string of the image")
    lang: Optional[str] = Field(None, description="Optional language parameter")
