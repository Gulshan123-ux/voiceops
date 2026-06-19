# 🎙️ VoiceOps Sentinel
### Real-Time Call Intelligence System — Week 1: Transcription Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Whisper API](https://img.shields.io/badge/ASR-Whisper--1-orange.svg)](https://openai.com/research/whisper)
[![WER Target](https://img.shields.io/badge/WER-<15%%20clean%20|%20<30%%20noisy-brightgreen.svg)](#wer-evaluation)

A **production-grade audio transcription pipeline** built for customer support call centers. Ingests mp3/wav/flac audio, preprocesses it for optimal ASR performance, transcribes via OpenAI Whisper (with Deepgram fallback), and returns structured JSON with timestamped segments.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Server](#-running-the-server)
- [API Usage](#-api-usage)
- [WER Evaluation](#-wer-evaluation)
- [Running Tests](#-running-tests)
- [Project Structure](#-project-structure)
- [Linting](#-linting)
- [Sample Output](#-sample-output)

---

## 🏗️ Architecture

```
Audio File (mp3/wav/flac)
        │
        ▼
┌───────────────────┐
│   FastAPI Layer   │  POST /transcribe  (validation, size check)
└────────┬──────────┘
         │ async (ThreadPoolExecutor)
         ▼
┌───────────────────┐
│  Preprocessor     │  pydub + ffmpeg
│  ─────────────    │  • Format → 16kHz mono 16-bit WAV
│  • Normalize      │  • Loudness normalize to -20 dBFS
│  • Strip silence  │  • Strip silence > 2s from ends
└────────┬──────────┘
         │
         ▼
┌───────────────────┐     ┌────────────────────┐
│  Transcriber      │────▶│  OpenAI Whisper-1  │ (primary)
│  (ASR Dispatch)   │     └────────────────────┘
│                   │     ┌────────────────────┐
│  ASR_BACKEND env  │────▶│  Deepgram Nova-2   │ (fallback)
└────────┬──────────┘     └────────────────────┘
         │ Retry: 3 attempts, exponential backoff (1→8s)
         ▼
┌───────────────────┐
│  TranscriptionResult │  Structured JSON response
│  • job_id (UUID)  │  • segments [ {id, start, end, text, confidence} ]
│  • duration_secs  │  • full_transcript
│  • language       │  • wer_score (optional)
│  • processed_at   │
└───────────────────┘
```

---

## 📦 Prerequisites

| Dependency | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://python.org) |
| ffmpeg | Latest | `brew install ffmpeg` |
| OpenAI API Key | — | [platform.openai.com](https://platform.openai.com) |
| Deepgram API Key | — | [deepgram.com](https://deepgram.com) *(optional)* |

> **Note:** ffmpeg is required by pydub for audio decoding. Without it, mp3/flac loading will fail.

---

## 🚀 Installation

```bash
# 1. Clone / navigate to the project
cd voiceops-sentinel

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ffmpeg (macOS)
brew install ffmpeg

# 5. Copy environment template
cp .env.example .env
# → Edit .env and add your API keys
```

---

## ⚙️ Configuration

Edit `.env` (never commit this file):

```dotenv
# Primary ASR
OPENAI_API_KEY=sk-your-key-here

# Fallback ASR (activate by setting ASR_BACKEND=deepgram)
DEEPGRAM_API_KEY=your-deepgram-key

# Backend selection: "whisper" (default) | "deepgram"
ASR_BACKEND=whisper

# File size limit (bytes): default 25 MB
MAX_FILE_SIZE_BYTES=26214400

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
```

---

## ▶️ Running the Server

```bash
# Development (auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc UI:**   http://localhost:8000/redoc
- **Health:**     http://localhost:8000/health

---

## 🔌 API Usage

### `POST /transcribe`

Upload an audio file and receive a structured transcript.

#### cURL Example

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample_audio/clean_call.wav" \
  -F "language=en"
```

#### With reference text for WER scoring

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample_audio/clean_call.wav" \
  -F "language=en" \
  -F "reference_text=Hello thank you for calling support"
```

#### Python (httpx)

```python
import httpx

with open("sample_audio/call.wav", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/transcribe",
        files={"file": ("call.wav", f, "audio/wav")},
        data={"language": "en"},
    )

result = response.json()
print(result["full_transcript"])
```

#### Request Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | `UploadFile` | ✅ | Audio file (.mp3, .wav, .flac, max 25 MB) |
| `language` | `string` | ❌ | BCP-47 language hint (e.g. `en`, `es`, `fr`) |
| `reference_text` | `string` | ❌ | Ground-truth text for WER calculation |

#### HTTP Status Codes

| Code | Meaning |
|---|---|
| `200` | Transcription successful |
| `400` | Invalid file format or missing filename |
| `413` | File exceeds 25 MB limit |
| `415` | Unsupported media type |
| `500` | Preprocessing or transcription failed |
| `503` | ASR backend unavailable (missing API key) |

---

## 📊 Sample Output

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "audio_file": "customer_call.wav",
  "duration_seconds": 142.5,
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "Hello, thank you for calling support.",
      "confidence": 0.9512
    },
    {
      "id": 1,
      "start": 4.2,
      "end": 9.1,
      "text": "My name is Alex. How can I help you today?",
      "confidence": 0.9231
    }
  ],
  "full_transcript": "Hello, thank you for calling support. My name is Alex. How can I help you today?",
  "wer_score": null,
  "processed_at": "2026-06-17T08:00:00.000000+00:00",
  "asr_backend": "whisper"
}
```

---

## 📈 WER Evaluation

The WER (Word Error Rate) evaluator tests transcription accuracy across 3 simulated noise scenarios:

| Scenario | Simulation Strategy | Target WER |
|---|---|---|
| Background Office Noise | Word substitutions from confusion dictionary (~10%) | < 30% |
| Accented Speech | Phoneme-adjacent word replacements (~15%) | < 30% |
| Phone Call Quality (Low Bitrate) | Word deletions + filler insertion (~20%) | < 30% |

### Run the WER Report

```bash
# Run full WER test suite with verbose output
pytest tests/test_wer.py -v

# See formatted WER report (no API calls needed)
python -c "
from app.wer_evaluator import run_wer_test_suite, print_wer_report
report = run_wer_test_suite()
print_wer_report(report)
"
```

### Sample WER Report Output

```
────────────────────────────────────────────────────────────────────────────────
  VoiceOps Sentinel – WER Evaluation Report
────────────────────────────────────────────────────────────────────────────────
  Scenario                                     WER  Threshold    Status
────────────────────────────────────────────────────────────────────────────────
  Background Office Noise                    8.33%        30%   ✅ PASS
  Accented Speech Simulation                 9.09%        30%   ✅ PASS
  Phone Call Quality (Low Bitrate)          14.29%        30%   ✅ PASS
────────────────────────────────────────────────────────────────────────────────
  Average WER                               10.57%        30%   ✅ PASS
────────────────────────────────────────────────────────────────────────────────
```

---

## 🧪 Running Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install -r requirements.txt

# Run all tests
pytest -v

# Run only WER tests (no API calls)
pytest tests/test_wer.py -v

# Run only preprocessor tests
pytest tests/test_preprocessor.py -v

# Run only transcriber tests (mocked)
pytest tests/test_transcriber.py -v

# Run with coverage
pytest --cov=app --cov-report=term-missing
```

---

## 📁 Project Structure

```
voiceops-sentinel/
├── app/
│   ├── __init__.py          # Package metadata
│   ├── main.py              # FastAPI app + /transcribe + /health endpoints
│   ├── transcriber.py       # Whisper & Deepgram ASR engines + retry logic
│   ├── preprocessor.py      # pydub audio normalization & format conversion
│   ├── wer_evaluator.py     # jiwer WER/CER + noise simulation test suite
│   └── schemas.py           # Pydantic v2 request/response models
├── tests/
│   ├── __init__.py
│   ├── test_transcriber.py  # Mocked ASR unit tests
│   ├── test_preprocessor.py # Audio processing unit tests
│   └── test_wer.py          # WER calculation & scenario tests
├── sample_audio/            # Place test .mp3/.wav/.flac files here
│   └── README.md
├── logs/                    # Auto-created; daily rotating log files
├── conftest.py              # Shared pytest fixtures
├── .env.example             # Environment template (copy → .env)
├── requirements.txt         # Pinned Python dependencies
└── README.md                # This file
```

---

## 🔍 Linting

```bash
# Run flake8 on all source files
flake8 app/ tests/ --max-line-length=99

# Expected output: no errors
```

---

## 🔐 Security Notes

1. **Never commit `.env`** — it is listed in `.gitignore`
2. API keys are loaded exclusively from environment variables via `python-dotenv`
3. Temporary audio files are written to isolated temp directories and cleaned up after each request
4. File size is validated server-side (not relying on client `Content-Length`)

---

## 🗺️ Roadmap (Weeks 2–4)

| Week | Feature |
|---|---|
| Week 2 | Intelligence Layer: LLM-based summarization + sentiment analysis |
| Week 3 | Diarization (Pyannote) + PII redaction (spaCy) |
| Week 4 | Dashboard UI + live audio streaming + time-synced audio player |

---

*Built by VoiceOps Sentinel Team | Infotact Solutions AI R&D Wing*
