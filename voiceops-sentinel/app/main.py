"""
app/main.py
FastAPI application entry point for VoiceOps Sentinel.

API surface
───────────
  POST /transcribe          – Upload audio file, receive transcript JSON
  GET  /health              – Liveness probe
  GET  /docs                – Swagger UI (FastAPI built-in)
  GET  /redoc               – ReDoc UI (FastAPI built-in)

Design decisions
────────────────
* File validation happens BEFORE any I/O-heavy preprocessing (fail fast).
* Preprocessing & transcription run in a thread pool executor so the
  asyncio event loop is never blocked by CPU/IO-bound ffmpeg/API work.
* Temporary preprocessed WAV files are cleaned up in a ``finally`` block
  to prevent disk accumulation even on errors.
* All HTTP errors use RFC 7807-style JSON bodies via ``ErrorResponse``.
* Startup / shutdown hooks configure structured logging once at boot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiofiles
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.preprocessor import preprocess_audio
from app.schemas import ErrorResponse, TranscriptionResult
from app.transcriber import transcribe
from app.summarizer import CallSummarizer
from app.latency_tracker import LatencyTracker

# Singleton summarizer (loaded once at startup)
_summarizer = CallSummarizer()

# ── Load .env before anything else ───────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────


def _configure_logging() -> None:
    """
    Configure root logger with both console and rotating file handlers.

    Log files are written to the ``LOG_DIR`` directory (default: ./logs)
    with a date-stamped filename: transcription_YYYYMMDD.log
    """
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"transcription_{today}.log"

    # Structured log format includes timestamp, level, module, and message
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

    logging.basicConfig(
        level=log_level,
        format=fmt,
        handlers=[
            logging.StreamHandler(),                          # Console
            logging.FileHandler(log_file, encoding="utf-8"),  # Daily file
        ],
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.info("Logging initialised: level=%s, file=%s", log_level, log_file)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_BYTES", 26_214_400))  # 25 MB
ALLOWED_EXTENSIONS: set = {".mp3", ".wav", ".flac"}
ALLOWED_CONTENT_TYPES: set = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/x-flac",
    "application/octet-stream",  # some clients send generic type
}

logger = logging.getLogger(__name__)

# Thread pool for blocking preprocessor / transcriber calls
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="voiceops")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise logging and thread pool lifespan events."""
    _configure_logging()
    from app.database import init_db
    init_db()
    logger.info(
        "VoiceOps Sentinel starting up | ASR_BACKEND=%s | MAX_FILE_SIZE=%d bytes",
        os.getenv("ASR_BACKEND", "whisper"),
        MAX_FILE_SIZE_BYTES,
    )
    yield
    logger.info("VoiceOps Sentinel shutting down…")
    _executor.shutdown(wait=True)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VoiceOps Sentinel – Transcription API",
    description=(
        "Real-Time Call Intelligence System.\n\n"
        "Upload customer support audio files (mp3/wav/flac) and receive "
        "structured transcripts with timestamped segments, language detection, "
        "optional Word Error Rate scoring, LLM-powered call summarisation, "
        "sentiment analysis, and PII redaction. (Week 2: Intelligence Layer)"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# Allow all origins in development; restrict in production via env
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Static file serving with no-cache for JS/CSS ─────────────────────────────
import os
import mimetypes
from fastapi.responses import FileResponse

frontend_dir = Path(__file__).parent.parent / "frontend"
if not frontend_dir.exists():
    frontend_dir = Path("frontend")
    os.makedirs(frontend_dir, exist_ok=True)


@app.get("/static/{filepath:path}", include_in_schema=False)
async def serve_static(filepath: str):
    """Serve frontend static files. Adds no-cache for JS and CSS."""
    file_path = frontend_dir / filepath
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Static file not found: {filepath}")

    mime_type, _ = mimetypes.guess_type(str(file_path))
    headers: dict = {}
    # Prevent browser caching JS and CSS so updates are instant
    if filepath.endswith(".js") or filepath.endswith(".css"):
        headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        headers["Pragma"] = "no-cache"
        headers["Expires"] = "0"

    return FileResponse(str(file_path), media_type=mime_type or "application/octet-stream", headers=headers)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: validate uploaded file
# ─────────────────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    """
    Validate an uploaded audio file before processing.

    Checks:
      1. Filename is not empty.
      2. File extension is in the allowed set (.mp3, .wav, .flac).
      3. Content-Type header is a recognised audio MIME type (best-effort).

    Note: File size is validated after spooling to disk because
    ``UploadFile`` does not expose content-length reliably.

    Args:
        file: FastAPI UploadFile object from the incoming request.

    Raises:
        HTTPException 400: If any validation check fails.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            "Unexpected content-type '%s' for file '%s' – proceeding anyway",
            content_type, file.filename,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Home Page",
    tags=["UI"],
)
async def serve_home() -> HTMLResponse:
    """Serve the VoiceOps Sentinel Landing/Home Page UI."""
    frontend_dir = Path(__file__).parent.parent / "frontend"
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
        index_path = Path("frontend/index.html")
    if not index_path.exists():
        index_path = Path(__file__).parent / "static" / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>index.html not found</h1>",
            status_code=status.HTTP_404_NOT_FOUND
        )
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Main Dashboard Page",
    tags=["UI"],
)
async def serve_dashboard_page() -> HTMLResponse:
    """Serve the VoiceOps Sentinel Main Dashboard Page UI."""
    frontend_dir = Path(__file__).parent.parent / "frontend"
    dash_path = frontend_dir / "dashboard.html"
    if not dash_path.exists():
        dash_path = Path("frontend/dashboard.html")
    if not dash_path.exists():
        return HTMLResponse(
            content="<h1>dashboard.html not found. Please verify the frontend files exist.</h1>",
            status_code=status.HTTP_404_NOT_FOUND
        )
    return HTMLResponse(content=dash_path.read_text(encoding="utf-8"))


@app.get(
    "/health",
    summary="Liveness probe",
    tags=["Monitoring"],
)
async def health_check() -> dict:
    """
    Liveness endpoint for orchestrators (Docker, Kubernetes, ALB).

    Returns:
        JSON with status ``ok`` and the current UTC timestamp.
    """
    return {
        "status": "ok",
        "service": "voiceops-sentinel",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "asr_backend": os.getenv("ASR_BACKEND", "whisper"),
    }


@app.post(
    "/transcribe",
    response_model=TranscriptionResult,
    status_code=status.HTTP_200_OK,
    summary="Transcribe an audio file",
    tags=["Transcription"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        500: {"model": ErrorResponse, "description": "Transcription failed"},
        503: {"model": ErrorResponse, "description": "ASR backend unavailable"},
    },
)
async def transcribe_audio(
    file: UploadFile = File(
        ...,
        description="Audio file to transcribe (mp3, wav, or flac; max 25 MB)",
    ),
    language: Optional[str] = Form(
        None,
        description="Optional BCP-47 language hint (e.g. 'en', 'es'). "
                    "Leave blank for auto-detection.",
    ),
    reference_text: Optional[str] = Form(
        None,
        description="Optional ground-truth text. When provided, WER is computed "
                    "and included in the response.",
    ),
) -> TranscriptionResult:
    """
    **Transcribe an audio file** using OpenAI Whisper (or Deepgram fallback).

    ### Input
    - **file**: Audio file in mp3, wav, or flac format (≤ 25 MB).
    - **language** *(optional)*: BCP-47 language code for language-hint mode.
    - **reference_text** *(optional)*: Ground-truth transcript for WER scoring.

    ### Processing pipeline
    1. Validate file format & size.
    2. Preprocess: convert to 16 kHz mono WAV, normalise loudness, strip silence.
    3. Transcribe via the configured ASR backend (Whisper or Deepgram).
    4. Return structured JSON with timestamped segments.

    ### Response
    Structured `TranscriptionResult` JSON with `job_id`, `segments`,
    `full_transcript`, `duration_seconds`, `language`, and optional `wer_score`.
    """
    # ── 1. Validate ───────────────────────────────────────────────────────────
    _validate_upload(file)
    original_filename = file.filename  # type: ignore[assignment]

    logger.info("File received: name=%s, content_type=%s", original_filename, file.content_type)

    # ── Latency tracker (Week 2) ──────────────────────────────────────────────
    tracker = LatencyTracker()

    # ── 2. Spool to temp file on disk ─────────────────────────────────────────
    # We use a temp directory so we can safely clean up regardless of outcome.
    tmp_dir = Path(tempfile.mkdtemp(prefix="voiceops_"))
    tmp_input = tmp_dir / original_filename
    preprocessed_wav: Optional[Path] = None

    try:
        # Write uploaded bytes to disk asynchronously
        async with aiofiles.open(tmp_input, "wb") as f:
            content = await file.read()
            await f.write(content)

        # ── 3. Size check (post-spool; more reliable than Content-Length) ─────
        file_size = tmp_input.stat().st_size
        logger.info("File spooled: name=%s, size=%d bytes", original_filename, file_size)

        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File size {file_size:,} bytes exceeds the "
                    f"{MAX_FILE_SIZE_BYTES:,} byte limit (25 MB)."
                ),
            )

        # ── 4. Preprocess (blocking – runs in thread pool) ───────────────────
        loop = asyncio.get_event_loop()

        def _run_preprocess() -> tuple[Path, float]:
            """Blocking wrapper for thread pool execution."""
            return preprocess_audio(
                input_path=tmp_input,
                output_dir=tmp_dir / "processed",
            )

        try:
            tracker.start("preprocess")
            preprocessed_wav, duration_seconds = await loop.run_in_executor(
                _executor, _run_preprocess
            )
            tracker.stop("preprocess")
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            logger.exception("Preprocessing failed for %s", original_filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio preprocessing failed: {exc}",
            ) from exc

        # ── 5. Transcribe (blocking – runs in thread pool) ───────────────────
        def _run_transcribe() -> TranscriptionResult:
            """Blocking wrapper for thread pool execution."""
            return transcribe(
                audio_path=preprocessed_wav,  # type: ignore[arg-type]
                duration_seconds=duration_seconds,
                original_filename=original_filename,
                language=language,
                reference_text=reference_text,
            )

        try:
            tracker.start("transcribe")
            result = await loop.run_in_executor(_executor, _run_transcribe)
            tracker.stop("transcribe")
        except EnvironmentError as exc:
            # Missing API key
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"ASR backend unavailable: {exc}",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Transcription failed for %s", original_filename)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed after retries: {exc}",
            ) from exc

        # ── 6. Intelligence Layer — Summarisation (Week 2) ───────────────────
        def _run_intelligence() -> dict:
            """Run LLM summariser in thread pool (may call OpenAI API)."""
            return _summarizer.summarize(
                transcript=result.full_transcript,
                language=result.language or "en",
            )

        tracker.start("intelligence")
        summary_data = await loop.run_in_executor(_executor, _run_intelligence)
        tracker.stop("intelligence")

        # Populate Week 2 fields on the result
        result.summary = summary_data.get("summary", "")
        result.summary_issue = summary_data.get("issue", "")
        result.summary_resolution = summary_data.get("resolution", "")
        result.summary_follow_up = summary_data.get("follow_up", "None")
        result.summary_engine = summary_data.get("engine", "extractive")

        # Attach latency report
        latency_report = tracker.build_report()
        result.latency_report = latency_report.to_dict()
        latency_report.log(str(result.job_id))

        logger.info(
            "Transcription complete: job_id=%s, segments=%d, duration=%.2fs, wer=%s",
            result.job_id, len(result.segments), result.duration_seconds,
            f"{result.wer_score:.4f}" if result.wer_score is not None else "N/A",
        )

        return result

    finally:
        # ── 6. Cleanup temp files ─────────────────────────────────────────────
        # Always remove the temp directory to prevent disk accumulation.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.debug("Temp directory cleaned up: %s", tmp_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Calls & Stats Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/calls",
    summary="Get all processed call records",
    tags=["Calls"],
)
async def list_calls() -> list[dict]:
    """Retrieve all processed call records stored in SQLite database."""
    from app.database import get_all_calls
    return get_all_calls()


@app.get(
    "/calls/{job_id}",
    summary="Get details of a single call record",
    tags=["Calls"],
)
async def get_call(job_id: str) -> dict:
    """Retrieve the full details of a specific call record by ID."""
    from app.database import get_call_by_id
    call_record = get_call_by_id(job_id)
    if not call_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call record with ID '{job_id}' not found."
        )
    return call_record


@app.get(
    "/stats",
    summary="Get aggregated call statistics",
    tags=["Calls"],
)
async def get_stats() -> dict:
    """Compute and return overall dashboard metrics across all calls."""
    from app.database import get_call_stats
    return get_call_stats()


@app.delete(
    "/calls/{job_id}",
    summary="Delete a call record",
    tags=["Calls"],
)
async def delete_call(job_id: str) -> dict:
    """Delete a call record from database by ID."""
    from app.database import delete_call_by_id
    deleted = delete_call_by_id(job_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call record with ID '{job_id}' not found."
        )
    return {"status": "deleted", "job_id": job_id}
