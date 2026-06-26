"""
app/summarizer.py
LLM Summarizer module for VoiceOps Sentinel — Week 2: Intelligence Layer.

Generates concise, actionable summaries of customer support call transcripts.
Primary: OpenAI GPT-3.5-turbo (or GPT-4o-mini if configured).
Fallback: Extractive / rule-based summarizer when API is unavailable.

Latency target: summary available < 3 seconds after transcription ends.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CallSummarizer:
    """
    Generates a concise call summary from a customer support transcript.

    Strategy:
      1. If OPENAI_API_KEY is set, call GPT-3.5-turbo (max_tokens=200) for
         a structured summary: Issue, Resolution, Follow-up.
      2. Fallback to extractive summarisation using keyword scoring.

    Attributes:
        api_key: OpenAI API key loaded from environment.
        model: GPT model to use (default: gpt-3.5-turbo).
    """

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.model: str = os.getenv("SUMMARY_MODEL", "gpt-3.5-turbo")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def summarize(self, transcript: str, language: str = "en") -> dict:
        """
        Summarize a call transcript.

        Args:
            transcript: Full plain-text call transcript.
            language: BCP-47 language code hint (e.g. 'en', 'hi').

        Returns:
            dict with keys:
              - ``summary``       (str)   : 2-4 sentence human-readable summary.
              - ``issue``         (str)   : Detected main customer issue.
              - ``resolution``    (str)   : Detected resolution / outcome.
              - ``follow_up``     (str)   : Any outstanding follow-up needed.
              - ``latency_ms``    (float) : Time taken to produce summary (ms).
              - ``engine``        (str)   : 'gpt' | 'extractive'.
        """
        if not transcript.strip():
            return self._empty_result()

        t0 = time.perf_counter()

        result = self._try_gpt(transcript, language)
        if result is None:
            result = self._extractive_summary(transcript)

        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "Summary generated: engine=%s latency=%.1fms",
            result.get("engine"),
            result["latency_ms"],
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # GPT path
    # ──────────────────────────────────────────────────────────────────────────

    def _try_gpt(self, transcript: str, language: str) -> Optional[dict]:
        """Attempt GPT summarisation; return None on any failure."""
        if not self.api_key:
            return None

        try:
            import openai  # lazy import — avoid startup cost if unused

            client = openai.OpenAI(api_key=self.api_key)
            prompt = self._build_prompt(transcript, language)

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250,
            )
            raw = response.choices[0].message.content.strip()
            parsed = self._parse_gpt_response(raw)
            parsed["engine"] = "gpt"
            return parsed

        except Exception as exc:
            logger.warning("GPT summarisation failed (%s). Using extractive fallback.", exc)
            return None

    @staticmethod
    def _build_prompt(transcript: str, language: str) -> str:
        """Build a structured prompt for GPT summarisation."""
        # Truncate very long transcripts to avoid token overflow
        truncated = transcript[:3000] if len(transcript) > 3000 else transcript
        lang_note = (
            "The transcript may contain Hinglish (Hindi + English) phrases."
            if language in ("hi", "hinglish")
            else ""
        )
        return (
            "You are a quality-assurance assistant for a customer support call center. "
            f"{lang_note}\n\n"
            "Analyse the following call transcript and respond ONLY in this exact format "
            "(no extra text):\n\n"
            "ISSUE: <one-line description of the customer's main issue>\n"
            "RESOLUTION: <one-line description of how it was resolved, or 'Unresolved'>\n"
            "FOLLOW_UP: <one-line outstanding action, or 'None'>\n"
            "SUMMARY: <2-3 sentence human-readable summary of the call>\n\n"
            f"Transcript:\n\"\"\"\n{truncated}\n\"\"\""
        )

    @staticmethod
    def _parse_gpt_response(raw: str) -> dict:
        """Extract structured fields from GPT response text."""
        def _extract(label: str) -> str:
            pattern = rf"^{label}:\s*(.+)$"
            m = re.search(pattern, raw, re.MULTILINE | re.IGNORECASE)
            return m.group(1).strip() if m else "N/A"

        return {
            "issue": _extract("ISSUE"),
            "resolution": _extract("RESOLUTION"),
            "follow_up": _extract("FOLLOW_UP"),
            "summary": _extract("SUMMARY"),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Extractive / rule-based fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _extractive_summary(self, transcript: str) -> dict:
        """
        Lightweight extractive summariser.
        Scores sentences by keyword density and picks the top 3.
        """
        issue = self._detect_issue(transcript)
        resolution = self._detect_resolution(transcript)
        follow_up = self._detect_follow_up(transcript)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if len(s.strip()) > 20]
        scored = sorted(sentences, key=lambda s: self._sentence_score(s), reverse=True)
        top = scored[:3]
        summary = " ".join(top) if top else transcript[:300]

        return {
            "issue": issue,
            "resolution": resolution,
            "follow_up": follow_up,
            "summary": summary,
            "engine": "extractive",
        }

    @staticmethod
    def _sentence_score(sentence: str) -> int:
        """Score a sentence by presence of high-value support keywords."""
        keywords = [
            "issue", "problem", "refund", "cancel", "help", "resolve", "error",
            "billing", "account", "complaint", "thank", "sorry", "understood",
            "escalate", "callback", "email", "confirm",
        ]
        lower = sentence.lower()
        return sum(1 for kw in keywords if kw in lower)

    @staticmethod
    def _detect_issue(text: str) -> str:
        patterns = [
            (r"(?i)(issue|problem|concern)\s+(?:is|with)?\s*:?\s*(.{10,80})", 2),
            (r"(?i)I\s+(?:am|have|want)\s+(?:a\s+)?(?:problem|issue)\s+(?:with)?\s*(.{10,80})", 1),
            (r"(?i)(?:my|the)\s+(bill|order|account|charge|payment|refund)\b", 1),
        ]
        for pat, grp in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(grp).strip().rstrip(".").capitalize()
        return "Customer support inquiry"

    @staticmethod
    def _detect_resolution(text: str) -> str:
        positive_patterns = [
            r"(?i)(resolved|fixed|processed|confirmed|approved|done|completed)",
            r"(?i)(I\s+(?:will|have)\s+(?:refund|process|send|escalate|update))",
            r"(?i)(we(?:'ve|\s+have)\s+(?:resolved|fixed|updated))",
        ]
        for pat in positive_patterns:
            if re.search(pat, text):
                return "Issue acknowledged and action initiated by agent"
        return "Unresolved — follow-up required"

    @staticmethod
    def _detect_follow_up(text: str) -> str:
        patterns = [
            (r"(?i)(call\s+(?:you\s+)?back|callback)", "Agent to call customer back"),
            (r"(?i)(send\s+(?:an?\s+)?email|email\s+(?:you|confirmation))", "Send confirmation email to customer"),
            (r"(?i)(escalat)", "Escalate to senior support / manager"),
            (r"(?i)(refund\s+will|processing\s+refund)", "Process refund — check within 3-5 business days"),
        ]
        for pat, action in patterns:
            if re.search(pat, text):
                return action
        return "None"

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> dict:
        return {
            "issue": "N/A",
            "resolution": "N/A",
            "follow_up": "None",
            "summary": "No transcript content to summarize.",
            "latency_ms": 0.0,
            "engine": "extractive",
        }
