"""
tests/test_intelligence_layer.py
Unit tests for VoiceOps Sentinel Week 2 — Intelligence Layer.

Tests cover:
  - CallSummarizer (extractive path — no API key needed)
  - LatencyTracker stage recording and report generation
  - Latency threshold assertions
  - Schema Week 2 field presence
"""

from __future__ import annotations

import time

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# CallSummarizer tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCallSummarizer:
    """Tests for the CallSummarizer extractive fallback (no API key needed)."""

    def _get_summarizer(self, monkeypatch):
        """Return a summarizer with no API key so extractive path is used."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from app.summarizer import CallSummarizer
        return CallSummarizer()

    def test_empty_transcript(self, monkeypatch):
        """Empty input should return safe defaults with 0 latency."""
        summarizer = self._get_summarizer(monkeypatch)
        result = summarizer.summarize("")
        assert result["summary"] == "No transcript content to summarize."
        assert result["latency_ms"] == 0.0
        assert result["engine"] == "extractive"

    def test_non_empty_transcript_has_all_keys(self, monkeypatch):
        """Every result dict must have all required keys."""
        summarizer = self._get_summarizer(monkeypatch)
        transcript = (
            "Agent: Hello, thank you for calling support. "
            "Customer: I need a refund for my last billing charge. "
            "Agent: I understand. I will process the refund for you right away."
        )
        result = summarizer.summarize(transcript)
        for key in ("summary", "issue", "resolution", "follow_up", "latency_ms", "engine"):
            assert key in result, f"Missing key: {key}"

    def test_extractive_engine_selected_without_api_key(self, monkeypatch):
        """Extractive engine must be used when OPENAI_API_KEY is absent."""
        summarizer = self._get_summarizer(monkeypatch)
        result = summarizer.summarize("Customer called about a billing issue.")
        assert result["engine"] == "extractive"

    def test_issue_detection_refund(self, monkeypatch):
        """Issue detector should identify billing/refund topics."""
        summarizer = self._get_summarizer(monkeypatch)
        result = summarizer.summarize(
            "Customer: My bill has an unexpected charge. "
            "I want a refund immediately."
        )
        # Issue should not be 'N/A' for a clear billing mention
        assert result["issue"] != ""

    def test_follow_up_callback_detection(self, monkeypatch):
        """Follow-up detector should catch 'call you back' patterns."""
        summarizer = self._get_summarizer(monkeypatch)
        result = summarizer.summarize(
            "Agent: I will call you back within 24 hours with an update."
        )
        assert "callback" in result["follow_up"].lower() or "call" in result["follow_up"].lower()

    def test_latency_ms_is_float(self, monkeypatch):
        """latency_ms must be a float >= 0."""
        summarizer = self._get_summarizer(monkeypatch)
        result = summarizer.summarize("Any text here.")
        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0.0

    def test_summarize_within_latency_target(self, monkeypatch):
        """
        Extractive summarizer must complete in < 3 seconds (intelligence latency target).
        """
        summarizer = self._get_summarizer(monkeypatch)
        transcript = " ".join(["Customer needs help with order."] * 50)
        result = summarizer.summarize(transcript)
        # The latency_ms reported by the summarizer itself
        assert result["latency_ms"] < 3000.0, (
            f"Intelligence layer exceeded 3s target: {result['latency_ms']}ms"
        )

    def test_long_transcript_does_not_crash(self, monkeypatch):
        """Very long transcripts should be safely truncated, not crash."""
        summarizer = self._get_summarizer(monkeypatch)
        long_text = "Agent: Hello. Customer: I have a problem. " * 500
        result = summarizer.summarize(long_text)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_resolution_detection_positive(self, monkeypatch):
        """Resolution should detect positive resolution keywords."""
        summarizer = self._get_summarizer(monkeypatch)
        result = summarizer.summarize(
            "Agent: The issue has been resolved. I have processed your request."
        )
        assert result["resolution"] != "N/A"

    def test_hinglish_transcript(self, monkeypatch):
        """Hinglish transcripts should not crash and return valid results."""
        summarizer = self._get_summarizer(monkeypatch)
        result = summarizer.summarize(
            "Customer: Mera bill galat hai, refund chahiye. "
            "Agent: Aapka request process ho raha hai."
        )
        assert isinstance(result["summary"], str)


# ─────────────────────────────────────────────────────────────────────────────
# LatencyTracker tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLatencyTracker:
    """Tests for the LatencyTracker stage timer."""

    def test_basic_stage_recording(self):
        """start/stop should record non-zero elapsed time."""
        from app.latency_tracker import LatencyTracker
        tracker = LatencyTracker()
        tracker.start("preprocess")
        time.sleep(0.01)  # 10ms sleep
        ms = tracker.stop("preprocess")
        assert ms >= 10.0, f"Expected >= 10ms, got {ms:.1f}ms"

    def test_build_report_has_all_fields(self):
        """build_report() must return a PipelineLatencyReport with all fields."""
        from app.latency_tracker import LatencyTracker
        tracker = LatencyTracker()

        for stage in ("preprocess", "transcribe", "intelligence"):
            tracker.start(stage)
            time.sleep(0.001)
            tracker.stop(stage)

        report = tracker.build_report()
        assert report.preprocess_ms >= 0
        assert report.transcribe_ms >= 0
        assert report.intelligence_ms >= 0
        assert report.total_ms >= 0

    def test_intelligence_within_target_flag(self):
        """intelligence_ok should be True for a fast intelligence stage."""
        from app.latency_tracker import LatencyTracker, INTELLIGENCE_LATENCY_TARGET_S
        tracker = LatencyTracker()
        tracker.start("intelligence")
        time.sleep(0.001)  # 1ms << 3s target
        tracker.stop("intelligence")
        report = tracker.build_report()
        assert report.intelligence_ok is True

    def test_latency_report_to_dict(self):
        """to_dict() should return all required dict keys."""
        from app.latency_tracker import LatencyTracker
        tracker = LatencyTracker()
        tracker.start("preprocess")
        tracker.stop("preprocess")
        tracker.start("intelligence")
        tracker.stop("intelligence")
        report = tracker.build_report()
        d = report.to_dict()
        required_keys = {
            "preprocess_ms", "transcribe_ms", "intelligence_ms",
            "total_ms", "intelligence_within_target", "target_s"
        }
        assert required_keys.issubset(set(d.keys()))

    def test_unstarted_stage_returns_zero(self):
        """elapsed_ms for an unstarted stage should return 0.0."""
        from app.latency_tracker import LatencyTracker
        tracker = LatencyTracker()
        assert tracker.elapsed_ms("nonexistent_stage") == 0.0

    def test_check_latency_ok_helper(self):
        """check_latency_ok utility function should work correctly."""
        from app.latency_tracker import check_latency_ok
        assert check_latency_ok(500.0) is True    # 0.5s < 3s
        assert check_latency_ok(2999.9) is True   # just under 3s
        assert check_latency_ok(3001.0) is False  # over 3s


# ─────────────────────────────────────────────────────────────────────────────
# Schema field presence tests
# ─────────────────────────────────────────────────────────────────────────────


class TestWeek2SchemaFields:
    """Week 2 intelligence fields must be present in TranscriptionResult."""

    def test_transcription_result_has_summary_fields(self):
        """TranscriptionResult schema must include all Week 2 fields."""
        from app.schemas import TranscriptionResult
        import uuid

        result = TranscriptionResult(
            job_id=str(uuid.uuid4()),
            audio_file="test.wav",
            duration_seconds=10.0,
            language="en",
            full_transcript="Hello world",
        )
        # Week 2 fields should exist with defaults
        assert hasattr(result, "summary")
        assert hasattr(result, "summary_issue")
        assert hasattr(result, "summary_resolution")
        assert hasattr(result, "summary_follow_up")
        assert hasattr(result, "summary_engine")
        assert hasattr(result, "latency_report")

    def test_week2_fields_default_values(self):
        """Week 2 fields should have sensible default values."""
        from app.schemas import TranscriptionResult
        import uuid

        result = TranscriptionResult(
            job_id=str(uuid.uuid4()),
            audio_file="test.wav",
            duration_seconds=5.0,
            language="en",
            full_transcript="Test",
        )
        assert result.summary == ""
        assert result.summary_issue == ""
        assert result.summary_resolution == ""
        assert result.summary_follow_up == "None"
        assert result.summary_engine == "extractive"
        assert result.latency_report is None

    def test_latency_report_can_be_dict(self):
        """latency_report field should accept a dict."""
        from app.schemas import TranscriptionResult
        import uuid

        result = TranscriptionResult(
            job_id=str(uuid.uuid4()),
            audio_file="test.wav",
            duration_seconds=5.0,
            language="en",
            full_transcript="Test",
            latency_report={
                "preprocess_ms": 150.0,
                "transcribe_ms": 1200.0,
                "intelligence_ms": 800.0,
                "total_ms": 2200.0,
                "intelligence_within_target": True,
                "target_s": 3.0,
            },
        )
        assert result.latency_report is not None
        assert result.latency_report["intelligence_within_target"] is True
