import time
import logging
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.core.config import settings
from api.routes import ocr
from api.schemas.response import HealthResponse, VersionResponse
from api.services.ocr_service import OCRService

# Setup logger configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request response time and diagnostic logging middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    
    # Request tracing log
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        f"IP: {client_ip} | Method: {request.method} | Path: {request.url.path} | "
        f"Status: {response.status_code} | Process Time: {process_time:.4f}s"
    )
    return response

# Standard JSON error response handler for HTTPException
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail, "result": None}
    )

# Clean, structured output formatting for Pydantic RequestValidationErrors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        loc = " -> ".join(str(l) for l in error.get("loc", []))
        msg = error.get("msg", "")
        errors.append(f"{loc}: {msg}")
    
    detail = "; ".join(errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "error": f"Validation failed: {detail}", "result": None}
    )

# Safety net for unhandled exceptions to prevent leaking server diagnostics
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception occurred: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "An unexpected error occurred on the server.", "result": None}
    )

# Include OCR endpoints router
app.include_router(ocr.router)

# Health & Version status endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    gpu_available = settings.USE_GPU
    service_status = "healthy"
    try:
        # Pre-initialize or confirm engine readiness on health checks
        await OCRService.get_engine(settings.DEFAULT_LANG)
    except Exception as e:
        logger.error(f"Health check model initialization warning: {e}")
        service_status = "degraded"

    return HealthResponse(
        status=service_status,
        gpu_available=gpu_available,
        version=settings.API_VERSION
    )

@app.get("/version", response_model=VersionResponse, tags=["System"])
async def get_version():
    return VersionResponse(
        version=settings.API_VERSION,
        title=settings.API_TITLE
    )
