"""
app/schemas.py
Pydantic v2 models for request / response validation.

Every field has an explicit type hint + Field description so that
FastAPI can auto-generate rich OpenAPI documentation.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Segment-level result (matches Whisper verbose_json output)
# ─────────────────────────────────────────────────────────────────────────────
class TranscriptSegment(BaseModel):
    """A single timestamped chunk of the transcript."""

    id: int = Field(..., description="Zero-based segment index")
    start: float = Field(..., ge=0.0, description="Segment start time in seconds")
    end: float = Field(..., ge=0.0, description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score [0, 1]; None if provider does not return it",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level transcription result returned by POST /transcribe
# ─────────────────────────────────────────────────────────────────────────────
class TranscriptionResult(BaseModel):
    """Full transcription response envelope."""

    job_id: UUID = Field(..., description="Unique job identifier (UUIDv4)")
    audio_file: str = Field(..., description="Original uploaded filename")
    duration_seconds: float = Field(
        ..., ge=0.0, description="Audio duration in seconds"
    )
    language: str = Field(
        ..., description="Detected or specified BCP-47 language code (e.g. 'en')"
    )
    segments: List[TranscriptSegment] = Field(
        default_factory=list, description="Timestamped transcript segments"
    )
    full_transcript: str = Field(
        ..., description="Concatenated plain-text transcript"
    )
    wer_score: Optional[float] = Field(
        None,
        ge=0.0,
        description="Word Error Rate vs. reference (if reference was provided)",
    )
    processed_at: datetime = Field(
        ..., description="UTC timestamp when processing completed"
    )
    asr_backend: str = Field(
        default="whisper", description="ASR engine used ('whisper' or 'deepgram')"
    )


# ─────────────────────────────────────────────────────────────────────────────
# WER evaluation report (used by wer_evaluator.py)
# ─────────────────────────────────────────────────────────────────────────────
class WERScenarioResult(BaseModel):
    """Result for a single noise-scenario WER test."""

    scenario: str = Field(..., description="Human-readable scenario name")
    hypothesis: str = Field(..., description="Transcribed text (from ASR)")
    reference: str = Field(..., description="Ground-truth reference text")
    wer: float = Field(..., ge=0.0, description="Word Error Rate [0, 1+]")
    passed: bool = Field(
        ..., description="True when WER is within the acceptable threshold"
    )


class WERReport(BaseModel):
    """Aggregated WER evaluation report across all scenarios."""

    scenarios: List[WERScenarioResult]
    average_wer: float = Field(..., ge=0.0)
    overall_passed: bool


# ─────────────────────────────────────────────────────────────────────────────
# HTTP error response (returned by FastAPI exception handlers)
# ─────────────────────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
    error_code: Optional[str] = None
