import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """
    Test that the health endpoint returns service status details.
    """
    with patch("api.services.ocr_service.OCRService.get_engine", new_callable=AsyncMock) as mock_get_engine:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["gpu_available"] is False
        assert "version" in data

def test_version_endpoint():
    """
    Test that the version endpoint returns API metadata.
    """
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0"
    assert data["title"] == "PaddleOCR API Service"

@patch("api.services.ocr_service.OCRService.run_ocr", new_callable=AsyncMock)
def test_ocr_image_success(mock_run_ocr):
    """
    Test OCR image upload returns correct results.
    """
    mock_run_ocr.return_value = {
        "text": "Hello World",
        "confidence": 0.95,
        "text_lines": [
            {
                "text": "Hello World",
                "confidence": 0.95,
                "bbox": [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]
            }
        ],
        "processing_time_ms": 120.0
    }
    
    # 1x1 transparent GIF bytes
    gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    
    response = client.post(
        "/ocr/image",
        files={"file": ("test.gif", gif_bytes, "image/gif")},
        data={"lang": "vi"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"]["text"] == "Hello World"
    assert data["result"]["confidence"] == 0.95
    assert len(data["result"]["text_lines"]) == 1

def test_ocr_image_invalid_extension():
    """
    Test that invalid file extensions return status 400.
    """
    response = client.post(
        "/ocr/image",
        files={"file": ("test.txt", b"some text content", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Unsupported file extension" in data["error"]

def test_ocr_base64_malformed():
    """
    Test that malformed base64 strings return status 400.
    """
    response = client.post(
        "/ocr/base64",
        json={"base64_str": "invalid_base64_string", "lang": "en"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Failed to decode" in data["error"] or "Invalid base64 string" in data["error"]

@patch("api.services.ocr_service.OCRService.run_ocr", new_callable=AsyncMock)
def test_ocr_base64_success(mock_run_ocr):
    """
    Test OCR processing from valid base64 image strings.
    """
    mock_run_ocr.return_value = {
        "text": "Base64 Text",
        "confidence": 0.88,
        "text_lines": [],
        "processing_time_ms": 50.0
    }
    
    # 1x1 base64 GIF representation
    base64_gif = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    response = client.post(
        "/ocr/base64",
        json={"base64_str": base64_gif, "lang": "en"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"]["text"] == "Base64 Text"

@patch("httpx.AsyncClient.get")
@patch("api.services.ocr_service.OCRService.run_ocr", new_callable=AsyncMock)
def test_ocr_url_success(mock_run_ocr, mock_get):
    """
    Test OCR downloading and processing images from URL.
    """
    mock_run_ocr.return_value = {
        "text": "URL Text",
        "confidence": 0.90,
        "text_lines": [],
        "processing_time_ms": 60.0
    }
    
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.content = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    mock_response.headers = {"content-type": "image/gif"}
    mock_get.return_value = mock_response
    
    response = client.post(
        "/ocr/url",
        json={"url": "http://example.com/test.gif", "lang": "en"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["result"]["text"] == "URL Text"

@patch("api.services.ocr_service.OCRService.run_ocr", new_callable=AsyncMock)
def test_batch_ocr(mock_run_ocr):
    """
    Test processing multiple image uploads in a single batch request.
    """
    mock_run_ocr.return_value = {
        "text": "Batch Text",
        "confidence": 0.92,
        "text_lines": [],
        "processing_time_ms": 40.0
    }
    
    gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    
    response = client.post(
        "/ocr/batch",
        files=[
            ("files", ("image1.gif", gif_bytes, "image/gif")),
            ("files", ("image2.gif", gif_bytes, "image/gif"))
        ],
        data={"lang": "en"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["results"]) == 2
    assert data["results"][0]["success"] is True
    assert data["results"][0]["result"]["text"] == "Batch Text"
