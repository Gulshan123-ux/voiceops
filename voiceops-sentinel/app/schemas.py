"""
app/schemas.py
Pydantic v2 models for request / response validation.

Every field has an explicit type hint + Field description so that
FastAPI can auto-generate rich OpenAPI documentation.

Week 2 additions:
  - ``summary``            : LLM/extractive call summary
  - ``summary_issue``      : Structured detected issue
  - ``summary_resolution`` : Detected resolution outcome
  - ``summary_follow_up``  : Outstanding follow-up action
  - ``summary_engine``     : 'gpt' | 'extractive'
  - ``latency_report``     : Per-stage pipeline latency breakdown
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Union
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
    speaker: Optional[str] = Field(
        None, description="Speaker identifier (e.g. 'Speaker A' or 'Speaker B')"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level transcription result returned by POST /transcribe
# ─────────────────────────────────────────────────────────────────────────────
class TranscriptionResult(BaseModel):
    """Full transcription response envelope."""

    job_id: Union[UUID, str] = Field(..., description="Unique job identifier (UUIDv4)")
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
    redacted_transcript: str = Field(
        default="", description="PII-redacted transcript"
    )
    sentiment: str = Field(
        default="Neutral", description="Overall sentiment (Positive / Negative / Neutral)"
    )
    sentiment_score: float = Field(
        default=0.0, description="Sentiment confidence score (0-100)"
    )
    wer_score: Optional[float] = Field(
        None,
        ge=0.0,
        description="Word Error Rate vs. reference (if reference was provided)",
    )
    action_items: List[str] = Field(
        default_factory=list, description="Extracted customer support action items"
    )
    speakers: dict[str, List[TranscriptSegment]] = Field(
        default_factory=dict, description="Segments grouped by speaker (Agent / Customer)"
    )
    flagged: bool = Field(
        default=False, description="Whether the call triggered manager alerts"
    )
    is_flagged: bool = Field(
        default=False, description="Whether the call triggered manager alerts (legacy)"
    )
    flag_reasons: List[str] = Field(
        default_factory=list, description="Reasons for flagging the call"
    )
    processed_at: Union[datetime, str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        description="ISO timestamp when processing completed"
    )
    asr_backend: str = Field(
        default="whisper", description="ASR engine used ('whisper', 'deepgram', 'mock')"
    )

    # ── Week 2: Intelligence Layer ───────────────────────────────────────────
    summary: str = Field(
        default="",
        description="LLM or extractive summary of the call (2-4 sentences)",
    )
    summary_issue: str = Field(
        default="",
        description="Primary customer issue detected in the call",
    )
    summary_resolution: str = Field(
        default="",
        description="Resolution / outcome of the call",
    )
    summary_follow_up: str = Field(
        default="None",
        description="Outstanding follow-up action required after the call",
    )
    summary_engine: str = Field(
        default="extractive",
        description="Engine used to produce summary: 'gpt' or 'extractive'",
    )
    latency_report: Optional[dict] = Field(
        default=None,
        description="Per-stage pipeline latency breakdown (ms) for this job",
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
