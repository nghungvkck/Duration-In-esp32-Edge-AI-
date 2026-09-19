"""
FastAPI main entry point.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
import traceback
from app.logger import logger
from app.service import PipelineService


BACKEND_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BACKEND_DIR / 'data' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
MAX_UPLOAD_SIZE_MB = 50


app = FastAPI(title="Durian Audio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = PipelineService()


@app.get("/")
def root():
    return {"message": "Durian Audio API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# ENDPOINT 1: Upload + summary
# ============================================================
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload audio file.
    Returns summary (metadata + list segments) — NO Mel matrix.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format: {ext}")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(500, f"Cannot save: {e}")

    size_mb = save_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        save_path.unlink()
        raise HTTPException(400, f"File too large: {size_mb:.1f}MB")

    logger.info(f"Nhận file: {file.filename} ({size_mb:.2f}MB)")

    try:
        result = service.analyze_summary(str(save_path))
        result['file_name'] = file.filename
        result['stored_name'] = unique_name
        logger.info(f"Xử lý xong: {file.filename}")
        return result
    except Exception as e:
        logger.error(f"Lỗi xử lý {file.filename}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(status_code=500, content={"error": str(exc)})


# ============================================================
# ENDPOINT 2: Segment detail
# ============================================================
@app.get("/api/segment/{stored_name}/{segment_index}")
def get_segment(stored_name: str, segment_index: int):
    """
    Get full data for ONE segment (including Mel matrix).
    """
    filepath = UPLOAD_DIR / stored_name
    if not filepath.exists():
        raise HTTPException(404, f"File not found: {stored_name}")

    logger.info(f"Lấy segment {segment_index} của {stored_name}")

    try:
        result = service.analyze_segment(str(filepath), segment_index)
    except Exception as e:
        logger.error(f"Lỗi segment {segment_index}/{stored_name}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Pipeline error: {e}")

    if not result.get('success'):
        raise HTTPException(400, result.get('message', 'Unknown error'))

    return result