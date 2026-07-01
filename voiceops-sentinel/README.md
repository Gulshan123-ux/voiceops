# 🎙️ VoiceOps Sentinel
### Real-Time Call Intelligence System — Week 2: Intelligence Layer & Advanced Search

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Whisper API](https://img.shields.io/badge/ASR-Whisper--1-orange.svg)](https://openai.com/research/whisper)
[![WER Target](https://img.shields.io/badge/WER-<15%25%20clean%20|%20<30%25%20noisy-brightgreen.svg)](#-wer-evaluation)

VoiceOps Sentinel is a **production-grade audio transcription and call intelligence pipeline** designed for customer support operations. The system validates and preprocesses audio uploads, runs multi-engine speech recognition (Whisper / Deepgram fallback), applies Presidio PII redaction, extracts actionable follow-ups, structures speaker-labeled transcripts, generates call summaries, and measures stage latency against strict targets.

The platform now features an **advanced search engine, dynamic metadata tagging, custom audio playback speeds, and a premium Light/Dark theme toggle**, backed by robust SQLite persistence and auto-migration schemas.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [New Advanced Features](#-new-advanced-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Server](#-running-the-server)
- [API Reference](#-api-reference)
- [WER Evaluation](#-wer-evaluation)
- [Running Tests](#-running-tests)
- [Project Structure](#-project-structure)
- [Security & Best Practices](#-security--best-practices)

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
┌─────────────────────────────────────────────────────────┐
│  Smart Processing & Redaction (app/smart_features.py)  │
│    • PII Redaction: Presidio (names, emails, phone numbers)│
│    • speaker classification (Agent/Customer channels)   │
│    • quality alerts & customer frustration indicators   │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Intelligence Layer (app/summarizer.py)                 │
│    • CallSummarizer: GPT-3.5 structured Issue/Resolution │
│    • Extractive fallback: TF-IDF local sentence ranker  │
│                                                          │
│  LatencyTracker (app/latency_tracker.py)                │
│    • Per-stage timer: preprocess / transcribe / intel   │
│    • Target: intelligence stage runtime < 3.0s          │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  SQLite DB & Operations Dashboard                       │
│    • Stores metadata, summaries, segments, latency logs │
│    • Dynamic frontend charts (Chart.js) + sync audio    │
│    • NEW: Dynamic search service & tags service modules │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ New Advanced Features

1. **Advanced Database & Backend Search Service**:
   - Dynamic query builder filtering transcripts, filenames, sentiments, alerts/flagged status, custom tag metadata, WER scores, and call durations.
   - Persisted SQLite intelligence schema upgrades running auto-migrations on application launch.
2. **Dynamic Tag Management Service**:
   - Multi-tag additions and deletions saved directly to the database calls store.
3. **Responsive Operations Control**:
   - Expanded filters drawer on the dashboard for deep search queries.
   - Variable playback speed controls (`0.5x` up to `2.0x`) matching call listener requirements.
   - Gorgeous premium Light/Dark theme toggle with dynamic Chart.js color adaptation.

---

## 📦 Prerequisites

| Dependency | Version | Install |
|---|---|---|
| Python | 3.11+ | [python.org](https://python.org) |
| ffmpeg | Latest | `brew install ffmpeg` |
| OpenAI API Key | — | [platform.openai.com](https://platform.openai.com) |
| Deepgram API Key | — | [deepgram.com](https://deepgram.com) *(optional)* |

---

## 🚀 Installation

```bash
# 1. Navigate to the project directory
cd voiceops-sentinel

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env
# → Edit .env and add your API keys
```

---

## ⚙️ Configuration

Edit `.env` (this file is excluded from git):

```dotenv
# Primary ASR Configuration
OPENAI_API_KEY=sk-your-key-here

# Fallback ASR Configuration
DEEPGRAM_API_KEY=your-deepgram-key
ASR_BACKEND=whisper  # "whisper" (default) or "deepgram"

# System Limits & Diagnostics
MAX_FILE_SIZE_BYTES=26214400
LOG_LEVEL=INFO
LOG_DIR=logs
```

---

## ▶️ Running the Server

Since port `8000` is reserved by secondary services (e.g. StatBot Pro), VoiceOps Sentinel runs on port **`8001`**:

```bash
# Development (with auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

- **Operations Dashboard:** http://localhost:8001/dashboard
- **Swagger Documentation:** http://localhost:8001/docs
- **Health Diagnostics:** http://localhost:8001/health

---

## 🔌 API Reference

### 1. `POST /transcribe`
Uploads and transcribes audio files, executes PII redaction, and processes Week 2 intelligence metadata.

- **Request**:
  - `file`: Audio file (`.mp3`, `.wav`, `.flac`, max 25 MB)
  - `language` *(optional)*: BCP-47 language code (e.g. `en`, `hi`)
  - `reference_text` *(optional)*: Reference text for computing Word Error Rate.
- **cURL Example**:
  ```bash
  curl -X POST http://localhost:8001/transcribe \
    -F "file=@sample_audio/clean_call.wav" \
    -F "reference_text=Hello thank you for calling support"
  ```

### 2. `GET /calls`
Returns an array of call history records. Accepts advanced filtering parameters:
- `q`: Search query string matching filenames and transcripts.
- `sentiment`: Filter by sentiment string (`Positive`, `Negative`, `Neutral`).
- `flagged`: Boolean string (`true`, `false`) matching flagged/alert calls.
- `tag`: Filter calls containing a specific tag badge.
- `wer_min` / `wer_max`: Word Error Rate range values (0.0 to 1.0).
- `duration_min` / `duration_max`: Duration range values in seconds.

### 3. `GET /calls/{job_id}`
Returns details for a single call record.

### 4. `DELETE /calls/{job_id}`
Permanently removes a call record and its associated audio file.

### 5. `GET /stats`
Returns aggregated analytics metrics for the Operations Dashboard.

### 6. `GET /calls/{job_id}/audio`
Retrieves the preprocessed WAV audio file.

### 7. `POST /calls/{job_id}/tags`
Adds a unique tag to the specified call record.
- **Form Data**:
  - `tag`: The tag name to add.

### 8. `DELETE /calls/{job_id}/tags/{tag}`
Removes a tag from the specified call record.

---

## 📊 Sample Response (with Intelligence Details)

```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "audio_file": "customer_call.wav",
  "duration_seconds": 12.4,
  "language": "en",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.1,
      "text": "Hello, my phone is +91-[REDACTED PHONE]. I need a refund.",
      "speaker": "Customer",
      "confidence": 0.941
    }
  ],
  "full_transcript": "Hello, my phone is +91-[REDACTED PHONE]. I need a refund.",
  "wer_score": 0.083,
  "processed_at": "2026-06-26T05:00:00Z",
  "asr_backend": "whisper",
  "flagged": true,
  "sentiment": "Negative",
  "sentiment_score": -0.85,
  "summary": "Customer called requesting a refund for billing discrepancies.",
  "summary_issue": "Refund request",
  "summary_resolution": "Agent processing transaction",
  "summary_follow_up": "None",
  "summary_engine": "extractive",
  "tags": ["refund", "frustrated"],
  "latency_report": {
    "preprocess_ms": 120.4,
    "transcribe_ms": 1105.1,
    "intelligence_ms": 420.2,
    "total_ms": 1645.7,
    "intelligence_within_target": true,
    "target_s": 3.0
  }
}
```

---

## 📈 WER Evaluation

Verify transcription accuracy across 3 predefined noise scenarios:

```bash
# See formatted WER report (completely local - no API credentials needed)
python -c "
from app.wer_evaluator import run_wer_test_suite, print_wer_report
report = run_wer_test_suite()
print_wer_report(report)
"
```

| Scenario | Target Threshold | Simulation Strategy |
|---|---|---|
| Office Noise | < 30.0% WER | Word substitutions (~10%) |
| Accented Speech | < 30.0% WER | Confused phonetic replacements (~15%) |
| Phone Call Quality | < 30.0% WER | Frame-rate deletions (~20%) |

---

## 🧪 Running Tests

A comprehensive suite of **80 tests** covers the transcription dispatchers, preprocessing conversions, Presidio redactions, summarization, latency timer modules, advanced queries, and tags.

```bash
# Run pytest globally using the venv interpreter
.venv/bin/pytest -v
```

---

## 📁 Project Structure

```
voiceops-sentinel/
├── app/
│   ├── main.py              # FastAPI app routing & lifespans
│   ├── transcriber.py       # ASR engines + backoff retries
│   ├── preprocessor.py      # Audio conversions & loudness normalization
│   ├── smart_features.py    # PII redactor, alerts, and action items
│   ├── summarizer.py        # CallSummarizer (GPT-3.5 + extractive fallback)
│   ├── latency_tracker.py   # Stage microsecond timing & benchmarks
│   ├── schemas.py           # Pydantic validation models
│   ├── database.py          # SQLite connections and migrations
│   ├── search/
│   │   └── search_service.py # Parameterized multi-field SQL query builder
│   └── tags/
│       └── tags_service.py  # Unique tag additions/removals logic
├── tests/
│   ├── test_transcriber.py  # Mocked transcriber tests
│   ├── test_preprocessor.py # Preprocessor conversions verification
│   ├── test_wer.py          # jiwer metrics unit tests
│   ├── test_intelligence.py # Summarization and latency unit tests
│   └── test_advanced_features.py # Search & tag backend test suite
├── frontend/
│   ├── index.html           # Landing layout Page (Theme toggle)
│   ├── dashboard.html       # Operations Workspace layout (Search panel)
│   ├── app.js               # Audio speed controls, tag binders, and theme toggle logic
│   └── style.css            # Dark/Glassmorphic dashboard theme overrides
├── requirements.txt         # Pinned packages list
└── README.md                # System documentation
```

---

## 🔐 Security & Best Practices

1. **Environment Separation**: API keys are loaded via `.env` files and never committed to version control.
2. **Resource Management**: Temp directories containing uploaded audio binaries are immediately purged upon completion or HTTP failure.
3. **Cache Invalidation**: Custom Static File responses set strict `Cache-Control` headers, bypassing local cached copies to deliver updated scripts instantly.
