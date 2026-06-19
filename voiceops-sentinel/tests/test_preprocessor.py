"""
tests/test_preprocessor.py
Unit tests for app/preprocessor.py using pytest.

These tests use pydub's in-memory audio generation to avoid needing
real audio files on disk. A minimal 1-second sine-wave WAV is created
programmatically and used as the test fixture.

Run with:
    pytest tests/test_preprocessor.py -v
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Guard: skip entire module if pydub / ffmpeg is not available
pydub = pytest.importorskip("pydub", reason="pydub not installed")
AudioSegment = pydub.AudioSegment


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_audio_dir() -> Generator[Path, None, None]:
    """Create and tear down a temporary directory for audio test files."""
    d = Path(tempfile.mkdtemp(prefix="voiceops_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def sample_wav(tmp_audio_dir: Path) -> Path:
    """
    Generate a 1-second 44.1 kHz stereo WAV using pydub's sine generator.

    This gives us a real audio file to test the preprocessing pipeline
    without requiring any external audio files in the repo.
    """
    try:
        from pydub.generators import Sine
    except ImportError:
        pytest.skip("pydub.generators not available")

    audio = Sine(440).to_audio_segment(duration=1000)  # 1 s at 440 Hz
    wav_path = tmp_audio_dir / "test_sample.wav"
    audio.export(str(wav_path), format="wav")
    return wav_path


@pytest.fixture(scope="module")
def output_dir(tmp_audio_dir: Path) -> Path:
    """Output directory for preprocessed files."""
    d = tmp_audio_dir / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessAudio:
    """Integration tests for the full preprocess_audio pipeline."""

    def test_preprocess_returns_wav_path(
        self, sample_wav: Path, output_dir: Path
    ) -> None:
        """Preprocessed output must be a .wav file."""
        from app.preprocessor import preprocess_audio

        out_path, duration = preprocess_audio(sample_wav, output_dir)
        assert out_path.suffix == ".wav", f"Expected .wav output, got {out_path.suffix}"

    def test_preprocess_output_exists(
        self, sample_wav: Path, output_dir: Path
    ) -> None:
        """The preprocessed WAV file must exist on disk after processing."""
        from app.preprocessor import preprocess_audio

        out_path, _ = preprocess_audio(sample_wav, output_dir)
        assert out_path.exists(), f"Output file not found: {out_path}"

    def test_preprocess_returns_positive_duration(
        self, sample_wav: Path, output_dir: Path
    ) -> None:
        """Duration returned must be > 0 seconds."""
        from app.preprocessor import preprocess_audio

        _, duration = preprocess_audio(sample_wav, output_dir)
        assert duration > 0.0, f"Duration should be positive, got {duration}"

    def test_output_is_mono(self, sample_wav: Path, output_dir: Path) -> None:
        """Output WAV must be mono (1 channel)."""
        from app.preprocessor import preprocess_audio

        out_path, _ = preprocess_audio(sample_wav, output_dir)
        result_audio = AudioSegment.from_wav(str(out_path))
        assert result_audio.channels == 1, (
            f"Expected 1 channel (mono), got {result_audio.channels}"
        )

    def test_output_sample_rate_is_16khz(
        self, sample_wav: Path, output_dir: Path
    ) -> None:
        """Output WAV must be resampled to 16 000 Hz."""
        from app.preprocessor import preprocess_audio

        out_path, _ = preprocess_audio(sample_wav, output_dir)
        result_audio = AudioSegment.from_wav(str(out_path))
        assert result_audio.frame_rate == 16_000, (
            f"Expected 16 000 Hz sample rate, got {result_audio.frame_rate}"
        )

    def test_output_sample_width_is_16bit(
        self, sample_wav: Path, output_dir: Path
    ) -> None:
        """Output WAV must have 16-bit sample width (2 bytes)."""
        from app.preprocessor import preprocess_audio

        out_path, _ = preprocess_audio(sample_wav, output_dir)
        result_audio = AudioSegment.from_wav(str(out_path))
        assert result_audio.sample_width == 2, (
            f"Expected 2-byte (16-bit) sample width, got {result_audio.sample_width}"
        )


class TestInputValidation:
    """Test that invalid inputs are rejected before processing."""

    def test_unsupported_extension_raises_value_error(
        self, tmp_audio_dir: Path, output_dir: Path
    ) -> None:
        """Files with unsupported extensions must raise ValueError."""
        from app.preprocessor import preprocess_audio

        bad_file = tmp_audio_dir / "audio.ogg"
        bad_file.write_bytes(b"\x00" * 100)  # Dummy content

        with pytest.raises(ValueError, match="Unsupported file extension"):
            preprocess_audio(bad_file, output_dir)

    def test_missing_file_raises_file_not_found(
        self, tmp_audio_dir: Path, output_dir: Path
    ) -> None:
        """Missing audio files must raise FileNotFoundError."""
        from app.preprocessor import preprocess_audio

        missing = tmp_audio_dir / "nonexistent.wav"
        with pytest.raises(FileNotFoundError):
            preprocess_audio(missing, output_dir)


class TestGetAudioDuration:
    """Tests for the standalone get_audio_duration helper."""

    def test_returns_positive_float(self, sample_wav: Path) -> None:
        """Duration of a 1-second WAV should be close to 1.0 s."""
        from app.preprocessor import get_audio_duration

        duration = get_audio_duration(sample_wav)
        assert isinstance(duration, float), "Duration must be a float"
        # Pydub sine generator creates exactly 1 000 ms → 1.0 s
        assert abs(duration - 1.0) < 0.05, f"Expected ~1.0s, got {duration}"
