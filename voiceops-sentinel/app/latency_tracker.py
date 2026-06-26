"""
app/latency_tracker.py
Intelligence Layer Latency Tracker — Week 2: VoiceOps Sentinel.

Tracks end-to-end pipeline latency per request and logs latency metrics
for each processing stage (preprocessing, transcription, intelligence).

Latency target (per project spec):
    - Intelligence output (summary + sentiment) available < 3 seconds
      after audio ends / transcription is complete.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Latency threshold (seconds) for the intelligence layer
INTELLIGENCE_LATENCY_TARGET_S: float = 3.0


@dataclass
class PipelineLatencyReport:
    """
    Captures timing for each stage of the VoiceOps processing pipeline.

    Stages:
        preprocess  — pydub audio normalisation + format conversion
        transcribe  — Whisper / Deepgram ASR call
        intelligence — summarisation + sentiment analysis
        total       — wall-clock time from request receipt to response
    """

    preprocess_ms: float = 0.0
    transcribe_ms: float = 0.0
    intelligence_ms: float = 0.0
    total_ms: float = 0.0
    intelligence_ok: bool = True  # True if intelligence_ms < target
    stages: Dict[str, float] = field(default_factory=dict)

    @property
    def intelligence_latency_s(self) -> float:
        return self.intelligence_ms / 1000.0

    def to_dict(self) -> dict:
        return {
            "preprocess_ms": round(self.preprocess_ms, 1),
            "transcribe_ms": round(self.transcribe_ms, 1),
            "intelligence_ms": round(self.intelligence_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "intelligence_within_target": self.intelligence_ok,
            "target_s": INTELLIGENCE_LATENCY_TARGET_S,
        }

    def log(self, job_id: str) -> None:
        """Emit a structured latency log line."""
        status = "✅ OK" if self.intelligence_ok else "⚠️ SLOW"
        logger.info(
            "LATENCY job_id=%s preprocess=%.0fms transcribe=%.0fms "
            "intelligence=%.0fms total=%.0fms [%s]",
            job_id,
            self.preprocess_ms,
            self.transcribe_ms,
            self.intelligence_ms,
            self.total_ms,
            status,
        )


class LatencyTracker:
    """
    Context-manager–style stage timer.

    Usage::

        tracker = LatencyTracker()
        with tracker.stage("preprocess"):
            ...
        with tracker.stage("transcribe"):
            ...
        report = tracker.build_report()
    """

    def __init__(self) -> None:
        self._stage_start: Dict[str, float] = {}
        self._stage_end: Dict[str, float] = {}
        self._wall_start: float = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────────────
    # Simple manual start / stop interface (easier for thread-pool contexts)
    # ──────────────────────────────────────────────────────────────────────────

    def start(self, stage: str) -> None:
        """Record the start time for a named stage."""
        self._stage_start[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """
        Record the end time for a stage and return elapsed ms.
        Returns 0.0 if ``start`` was never called for this stage.
        """
        now = time.perf_counter()
        self._stage_end[stage] = now
        start = self._stage_start.get(stage)
        if start is None:
            return 0.0
        return (now - start) * 1000.0

    def elapsed_ms(self, stage: str) -> float:
        """Return elapsed time in ms for a completed stage."""
        start = self._stage_start.get(stage)
        end = self._stage_end.get(stage)
        if start is None or end is None:
            return 0.0
        return (end - start) * 1000.0

    def build_report(self) -> PipelineLatencyReport:
        """Construct a :class:`PipelineLatencyReport` from recorded stages."""
        wall_total_ms = (time.perf_counter() - self._wall_start) * 1000.0
        intel_ms = self.elapsed_ms("intelligence")
        report = PipelineLatencyReport(
            preprocess_ms=self.elapsed_ms("preprocess"),
            transcribe_ms=self.elapsed_ms("transcribe"),
            intelligence_ms=intel_ms,
            total_ms=wall_total_ms,
            intelligence_ok=(intel_ms / 1000.0) <= INTELLIGENCE_LATENCY_TARGET_S,
            stages={k: round(self.elapsed_ms(k), 1) for k in self._stage_start},
        )
        return report


def check_latency_ok(intelligence_ms: float) -> bool:
    """Return True if the intelligence layer met its latency target."""
    return (intelligence_ms / 1000.0) <= INTELLIGENCE_LATENCY_TARGET_S
