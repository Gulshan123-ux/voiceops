"""
app/preprocessor.py
Audio normalisation & format conversion before sending to the ASR engine.

Design decisions
────────────────
* We use pydub (wraps ffmpeg) because it supports every common audio
  container and codec without requiring callers to know ffmpeg flags.
* Target format: 16 kHz, mono, 16-bit PCM WAV — the exact spec preferred
  by Whisper. Deepgram also handles this well.
* Silence stripping uses pydub's built-in silence detector to remove
  leading/trailing pauses > 2 s (reduces API token waste and noise).
* All operations are synchronous; the FastAPI route calls this in a
  thread pool executor so the event-loop is never blocked.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Tuple

from pydub import AudioSegment
from pydub.silence import detect_leading_silence

# ── Module-level logger (structured via app-wide logging config) ──────────────
logger = logging.getLogger(__name__)

# ── Target audio specification ────────────────────────────────────────────────
TARGET_SAMPLE_RATE: int = 16_000   # Hz – Whisper's preferred sample rate
TARGET_CHANNELS: int = 1           # Mono
TARGET_SAMPLE_WIDTH: int = 2       # 16-bit PCM (2 bytes per sample)

# Silence detection thresholds
SILENCE_THRESHOLD_DBFS: int = -40  # dBFS – below this is considered silence
MAX_SILENCE_MS: int = 2_000        # Strip silence runs longer than 2 s


def _load_audio(input_path: Path) -> AudioSegment:
    """
    Load an audio file into a pydub AudioSegment.

    pydub delegates to ffmpeg under the hood, so any format ffmpeg
    understands (mp3, wav, flac, ogg, …) is supported automatically.

    Args:
        input_path: Absolute path to the source audio file.

    Returns:
        Loaded AudioSegment object.

    Raises:
        ValueError: If the file extension is not in the allowed list.
        FileNotFoundError: If the path does not exist.
        RuntimeError: If ffmpeg is not installed or loading fails.
    """
    allowed_extensions = {".mp3", ".wav", ".flac"}
    suffix = input_path.suffix.lower()

    if suffix not in allowed_extensions:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    try:
        logger.debug("Loading audio from %s (format=%s)", input_path, suffix.lstrip("."))
        audio = AudioSegment.from_file(str(input_path), format=suffix.lstrip("."))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load audio '{input_path}': {exc}. "
            "Ensure ffmpeg is installed (brew install ffmpeg)."
        ) from exc

    return audio


def _convert_to_target_spec(audio: AudioSegment) -> AudioSegment:
    """
    Normalise audio to Whisper's preferred specification.

    Steps performed:
      1. Set sample rate to 16 kHz (``set_frame_rate``).
      2. Convert to mono by mixing down all channels (``set_channels``).
      3. Ensure 16-bit sample width (``set_sample_width``).

    Args:
        audio: Source AudioSegment (any spec).

    Returns:
        Normalised AudioSegment in 16 kHz mono 16-bit PCM format.
    """
    logger.debug(
        "Converting audio: %d Hz, %d ch, %d-bit → %d Hz, %d ch, %d-bit",
        audio.frame_rate, audio.channels, audio.sample_width * 8,
        TARGET_SAMPLE_RATE, TARGET_CHANNELS, TARGET_SAMPLE_WIDTH * 8,
    )
    audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
    audio = audio.set_channels(TARGET_CHANNELS)
    audio = audio.set_sample_width(TARGET_SAMPLE_WIDTH)
    return audio


def _normalize_loudness(audio: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    """
    Apply loudness normalisation so that all files have a consistent volume.

    We target -20 dBFS (a common broadcast standard).  Very quiet recordings
    (e.g. phone calls) benefit the most; already-loud files are attenuated
    rather than clipped.

    Args:
        audio: Input AudioSegment.
        target_dbfs: Target loudness in dBFS (default -20.0).

    Returns:
        Volume-adjusted AudioSegment.
    """
    delta = target_dbfs - audio.dBFS
    logger.debug(
        "Normalising loudness: current=%.1f dBFS, target=%.1f dBFS, delta=%.1f dB",
        audio.dBFS, target_dbfs, delta,
    )
    return audio.apply_gain(delta)


def _strip_silence(audio: AudioSegment) -> AudioSegment:
    """
    Remove leading and trailing silence runs longer than MAX_SILENCE_MS.

    Uses pydub's ``detect_leading_silence`` which scans sample-by-sample
    and returns the millisecond position where actual audio begins.

    Args:
        audio: Input AudioSegment.

    Returns:
        AudioSegment with silence stripped from both ends.
    """
    original_duration_ms = len(audio)

    # ── Detect leading silence ────────────────────────────────────────────────
    leading_silence_ms = detect_leading_silence(
        audio, silence_threshold=SILENCE_THRESHOLD_DBFS
    )

    # ── Detect trailing silence (reverse the segment, then do the same) ───────
    trailing_silence_ms = detect_leading_silence(
        audio.reverse(), silence_threshold=SILENCE_THRESHOLD_DBFS
    )

    # Only strip if silence exceeds the threshold
    start_trim = leading_silence_ms if leading_silence_ms > MAX_SILENCE_MS else 0
    end_trim = (
        len(audio) - trailing_silence_ms
        if trailing_silence_ms > MAX_SILENCE_MS
        else len(audio)
    )

    trimmed = audio[start_trim:end_trim]
    saved_ms = original_duration_ms - len(trimmed)

    logger.debug(
        "Silence strip: leading=%d ms, trailing=%d ms, trimmed=%d ms, "
        "original=%d ms, result=%d ms",
        leading_silence_ms, trailing_silence_ms, saved_ms,
        original_duration_ms, len(trimmed),
    )
    return trimmed


def preprocess_audio(input_path: Path, output_dir: Path) -> Tuple[Path, float]:
    """
    Full preprocessing pipeline for a single audio file.

    Pipeline steps:
      1. Load (supports mp3 / wav / flac via ffmpeg).
      2. Convert to 16 kHz mono 16-bit PCM.
      3. Normalise loudness to -20 dBFS.
      4. Strip leading/trailing silence > 2 s.
      5. Export as a temporary WAV file ready for the ASR API.

    Args:
        input_path: Path to the source audio file.
        output_dir: Directory where the preprocessed WAV will be written.
                    Created automatically if it does not exist.

    Returns:
        Tuple of:
          - ``output_path``: Path to the preprocessed WAV file.
          - ``duration_seconds``: Duration of the processed audio in seconds.

    Raises:
        ValueError: On unsupported extension.
        FileNotFoundError: If source file is missing.
        RuntimeError: If ffmpeg is unavailable or processing fails.
    """
    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preprocessing started: %s", input_path.name)

    # Step 1: Load
    audio = _load_audio(input_path)

    # Step 2: Convert format
    audio = _convert_to_target_spec(audio)

    # Step 3: Normalise loudness
    audio = _normalize_loudness(audio)

    # Step 4: Strip silence
    audio = _strip_silence(audio)

    # Step 5: Export to WAV
    output_filename = f"preprocessed_{input_path.stem}.wav"
    output_path = output_dir / output_filename
    audio.export(str(output_path), format="wav")

    duration_seconds = len(audio) / 1_000.0  # pydub works in milliseconds
    elapsed = time.perf_counter() - t_start

    logger.info(
        "Preprocessing complete: file=%s, duration=%.2fs, "
        "preprocessing_time=%.3fs, output=%s",
        input_path.name, duration_seconds, elapsed, output_path.name,
    )

    return output_path, duration_seconds


def get_audio_duration(file_path: Path) -> float:
    """
    Return the duration of an audio file in seconds without full preprocessing.

    Useful for quick validation (e.g., pre-flight size check) before
    committing to the full pipeline.

    Args:
        file_path: Path to audio file.

    Returns:
        Duration in seconds (float).
    """
    audio = AudioSegment.from_file(str(file_path))
    return len(audio) / 1_000.0
