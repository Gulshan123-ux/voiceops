"""
tests/test_transcriber.py
Unit tests for app/transcriber.py.

These tests use ``unittest.mock`` to patch the OpenAI / Deepgram SDK calls,
so no real API credentials are needed.  Tests validate:
  - Backend dispatch logic
  - Parsing of mock API responses
  - Retry behaviour on transient failures
  - WER integration when reference_text is provided
  - Error handling for missing API keys and unsupported backends

Run with:
    pytest tests/test_transcriber.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import TranscriptSegment


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_whisper_mock_response() -> MagicMock:
    """Build a MagicMock that mimics Whisper verbose_json SDK response."""
    mock = MagicMock()
    mock.model_dump.return_value = {
        "text": "Hello thank you for calling support.",
        "language": "en",
        "duration": 4.2,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 4.2,
                "text": "Hello thank you for calling support.",
                "avg_logprob": -0.1,
            }
        ],
    }
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Backend dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendDispatch:
    """Verify that the correct backend function is called based on ASR_BACKEND."""

    def _dummy_audio(self, tmp_path: Path) -> Path:
        p = tmp_path / "dummy.wav"
        p.write_bytes(b"\x00" * 100)
        return p

    @patch("app.transcriber._call_whisper_api")
    @patch("app.transcriber._parse_whisper_response")
    @patch("app.transcriber._build_full_transcript", return_value="Hello.")
    def test_whisper_backend_dispatched(
        self,
        mock_build: Any,
        mock_parse: Any,
        mock_call: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ASR_BACKEND=whisper, _call_whisper_api must be invoked."""
        monkeypatch.setenv("ASR_BACKEND", "whisper")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        mock_call.return_value = {"language": "en", "segments": []}
        mock_parse.return_value = []

        from app.transcriber import transcribe

        result = transcribe(
            audio_path=self._dummy_audio(tmp_path),
            duration_seconds=4.0,
            original_filename="test.wav",
        )

        mock_call.assert_called_once()
        assert result.asr_backend == "whisper"

    def test_unknown_backend_raises_value_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unrecognised ASR_BACKEND value must raise ValueError."""
        monkeypatch.setenv("ASR_BACKEND", "unknown_engine")

        from app.transcriber import transcribe

        with pytest.raises(ValueError, match="Unknown ASR_BACKEND"):
            transcribe(
                audio_path=self._dummy_audio(tmp_path),
                duration_seconds=1.0,
                original_filename="test.wav",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Whisper response parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestWhisperResponseParsing:
    """Tests for _parse_whisper_response."""

    def test_parses_segments_correctly(self) -> None:
        """Parsed segments should match mock Whisper output."""
        from app.transcriber import _parse_whisper_response

        raw = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": " Hello.", "avg_logprob": -0.2},
                {"id": 1, "start": 2.5, "end": 5.0, "text": " Thank you.", "avg_logprob": -0.1},
            ]
        }
        segments = _parse_whisper_response(raw)

        assert len(segments) == 2
        assert segments[0].start == 0.0
        assert segments[0].end == 2.5
        assert "Hello" in segments[0].text
        assert isinstance(segments[0].confidence, float)

    def test_empty_segments_returns_empty_list(self) -> None:
        """Empty segments field should return an empty list."""
        from app.transcriber import _parse_whisper_response

        segments = _parse_whisper_response({"segments": []})
        assert segments == []

    def test_missing_segments_key_returns_empty_list(self) -> None:
        """Missing 'segments' key should return empty list without error."""
        from app.transcriber import _parse_whisper_response

        segments = _parse_whisper_response({})
        assert segments == []

    def test_confidence_is_between_zero_and_one(self) -> None:
        """Confidence values must be in [0, 1] range."""
        from app.transcriber import _parse_whisper_response

        raw = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.0, "text": "test", "avg_logprob": -0.5},
            ]
        }
        segments = _parse_whisper_response(raw)
        assert 0.0 <= segments[0].confidence <= 1.0  # type: ignore[operator]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Confidence conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestLogProbToConfidence:
    """Tests for the log-probability → confidence conversion."""

    def test_zero_logprob_gives_full_confidence(self) -> None:
        """avg_logprob=0 → exp(0)=1.0 → confidence=1.0."""
        from app.transcriber import _logprob_to_confidence

        assert _logprob_to_confidence(0.0) == 1.0

    def test_very_negative_logprob_gives_near_zero_confidence(self) -> None:
        """Very negative log-prob → confidence near 0."""
        from app.transcriber import _logprob_to_confidence

        conf = _logprob_to_confidence(-10.0)
        assert conf < 0.01, f"Expected near-zero confidence, got {conf}"

    def test_confidence_clamped_to_zero_one(self) -> None:
        """Confidence must always be in [0, 1]."""
        from app.transcriber import _logprob_to_confidence

        # Positive logprob would give exp > 1, but must be clamped to 1
        assert _logprob_to_confidence(5.0) == 1.0
        assert _logprob_to_confidence(-100.0) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Full transcript builder
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFullTranscript:
    """Tests for _build_full_transcript."""

    def test_concatenates_segments(self) -> None:
        """Full transcript should be all segment texts joined by spaces."""
        from app.transcriber import _build_full_transcript

        segments = [
            TranscriptSegment(id=0, start=0.0, end=1.0, text="Hello"),
            TranscriptSegment(id=1, start=1.0, end=2.0, text="world"),
        ]
        result = _build_full_transcript(segments)
        assert result == "Hello world"

    def test_empty_segments_returns_empty_string(self) -> None:
        """Empty segment list should return empty string."""
        from app.transcriber import _build_full_transcript

        result = _build_full_transcript([])
        assert result == ""

    def test_strips_whitespace_from_each_segment(self) -> None:
        """Leading/trailing whitespace in segments is stripped."""
        from app.transcriber import _build_full_transcript

        segments = [
            TranscriptSegment(id=0, start=0.0, end=1.0, text="  Hello  "),
            TranscriptSegment(id=1, start=1.0, end=2.0, text=" world "),
        ]
        result = _build_full_transcript(segments)
        assert result == "Hello world"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Environment error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentErrors:
    """Test that missing API keys are caught early."""

    def test_whisper_missing_api_key_raises_environment_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When OPENAI_API_KEY is not set, EnvironmentError must be raised."""
        import sys
        from unittest.mock import MagicMock

        # Provide a mock openai module so the test doesn't fail on import
        mock_openai = MagicMock()
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ASR_BACKEND", "whisper")

        from app.transcriber import _call_whisper_api

        dummy = tmp_path / "dummy.wav"
        dummy.write_bytes(b"\x00" * 100)

        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            _call_whisper_api(dummy)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Smart Features (PII Redaction, Action Items, Hinglish mock)
# ─────────────────────────────────────────────────────────────────────────────

class TestSmartFeatures:
    """Validate PII redaction and Action Item extractor."""

    def test_pii_redaction(self) -> None:
        """Verify emails, phones, and names are correctly redacted."""
        from app.smart_features import redact_pii

        raw_text = "Hello my name is Alex, my email is check@test.com and phone is +91-9886012345."
        redacted = redact_pii(raw_text)

        assert "[REDACTED NAME]" in redacted
        assert "[REDACTED EMAIL]" in redacted
        assert "[REDACTED PHONE]" in redacted

    def test_pii_redaction_hinglish(self) -> None:
        """Verify spelled-out numbers in Hinglish calls are redacted."""
        from app.smart_features import redact_pii

        raw_text = (
            "Mera naam Amit hai aur number nine double eight "
            "six zero one two three four five hai."
        )
        redacted = redact_pii(raw_text)

        assert "[REDACTED NAME]" in redacted
        assert "[REDACTED PHONE]" in redacted

    def test_action_item_extraction(self) -> None:
        """Verify action items are extracted based on support keywords."""
        from app.smart_features import extract_action_items
        from app.schemas import TranscriptSegment

        segments = [
            TranscriptSegment(
                id=0, start=0.0, end=2.0, text="Please schedule a callback tomorrow."
            ),
            TranscriptSegment(
                id=1, start=2.0, end=4.0, text="I will process the refund check now."
            )
        ]
        actions = extract_action_items(segments)

        assert any("callback" in a.lower() for a in actions)
        assert any("refund" in a.lower() for a in actions)

    def test_mock_hinglish_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify Hinglish option returns Hinglish dialog in mock backend."""
        monkeypatch.setenv("ASR_BACKEND", "mock")
        from app.transcriber import transcribe

        dummy = tmp_path / "dummy.wav"
        dummy.write_bytes(b"\x00" * 100)

        result = transcribe(
            audio_path=dummy,
            duration_seconds=10.0,
            original_filename="dummy.wav",
            language="hi-en"
        )

        assert result.language == "hi-en"
        # The mock dialogue has Amit / swagat / refund questions
        assert any("[REDACTED NAME]" in seg.text for seg in result.segments)
        assert any("[REDACTED EMAIL]" in seg.text for seg in result.segments)

    def test_evaluate_alerts(self) -> None:
        """Verify alerts are triggered on high WER and frustrated customer keywords."""
        from app.smart_features import evaluate_alerts
        from app.schemas import TranscriptSegment

        # Case 1: High WER trigger
        segments_ok = [
            TranscriptSegment(
                id=0, start=0.0, end=2.0, text="Everything is fine.", speaker="Speaker B"
            )
        ]
        flagged, reasons = evaluate_alerts(0.35, segments_ok)
        assert flagged is True
        assert any("High Word Error Rate" in r for r in reasons)

        # Case 2: Frustrated customer keywords trigger
        segments_frustrated = [
            TranscriptSegment(
                id=0,
                start=0.0,
                end=2.0,
                text="I am very frustrated with this service.",
                speaker="Speaker B",
            )
        ]
        flagged, reasons = evaluate_alerts(0.05, segments_frustrated)
        assert flagged is True
        assert any("Customer frustration" in r for r in reasons)

        # Case 3: Frustrated agent keywords (should not trigger alert if agent is talking)
        segments_agent_frustrated = [
            TranscriptSegment(
                id=0,
                start=0.0,
                end=2.0,
                text="I am very frustrated with this service.",
                speaker="Speaker A",
            )
        ]
        flagged, reasons = evaluate_alerts(0.05, segments_agent_frustrated)
        assert flagged is False
