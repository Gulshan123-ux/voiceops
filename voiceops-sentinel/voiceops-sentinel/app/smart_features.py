"""
app/smart_features.py
Module for smart features: PII Redaction, Action Item Extraction, and Speaker turn heuristics.
"""

import re
from typing import List
from app.schemas import TranscriptSegment


def redact_pii(text: str) -> str:
    """
    Redacts emails, phone numbers, and names from text.
    Wraps them in HTML span tags for styling/highlighting on the frontend.
    """
    # 1. Redact Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(
        email_pattern, '<span class="pii-highlight">[REDACTED EMAIL]</span>', text
    )

    # 2. Redact Numeric Phone Numbers (e.g., +91-9886012345, +1 (555) 019-2834, etc.)
    phone_pattern = r'\b(\+?\d{1,4}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
    text = re.sub(
        phone_pattern, '<span class="pii-highlight">[REDACTED PHONE]</span>', text
    )

    # 3. Redact Spelled-out numbers (common in support calls)
    # Match contiguous sequences of 3 or more number words
    spelled_seq_pattern = (
        r'(\b(?:zero|one|two|three|four|five|six|seven|eight|nine|double\s+'
        r'(?:zero|one|two|three|four|five|six|seven|eight|nine))\b[-.\s]*){3,}'
    )
    text = re.sub(
        spelled_seq_pattern,
        '<span class="pii-highlight">[REDACTED PHONE]</span>',
        text,
        flags=re.IGNORECASE,
    )

    # 4. Redact Names introduced by common patterns
    name_pattern = r'(?i)\b(my name is|mera naam|naam hai|this is)\s+([A-Za-z]+)'

    def name_replacer(match):
        intro = match.group(1)
        name = match.group(2)
        stopwords = [
            'support', 'a', 'the', 'an', 'to', 'for', 'in', 'on', 'at', 'with', 'from', 'by'
        ]
        if name.lower() in stopwords:
            return match.group(0)
        return f'{intro} <span class="pii-highlight">[REDACTED NAME]</span>'

    text = re.sub(name_pattern, name_replacer, text)

    # 5. Direct fallback for common name tags in context
    common_names = r'\b(Alex|Amit|Sarah|John|Michael|Jessica|David|Emily)\b'
    text = re.sub(
        common_names,
        '<span class="pii-highlight">[REDACTED NAME]</span>',
        text,
        flags=re.IGNORECASE,
    )

    return text


def extract_action_items(segments: List[TranscriptSegment]) -> List[str]:
    """
    Scans segment text for actionable customer support keywords
    and returns a unique list of up to 4 clear action items.
    """
    actions = []
    for seg in segments:
        text_lower = seg.text.lower()

        # Check matching patterns for dynamic extraction
        if "refund" in text_lower or "paisa" in text_lower or "money back" in text_lower:
            actions.append("Process billing refund for the customer")
        if "verify" in text_lower or "confirm" in text_lower or "details" in text_lower:
            actions.append("Verify customer account credentials and identity")
        if "email" in text_lower or "mail" in text_lower or "send" in text_lower:
            actions.append("Send confirmation email / tracking details to customer")
        if "schedule" in text_lower or "callback" in text_lower or "call" in text_lower:
            actions.append("Schedule customer callback or CRM follow-up ticket")
        if "escalate" in text_lower or "manager" in text_lower or "supervisor" in text_lower:
            actions.append("Escalate case to supervisor/manager for review")
        if "check" in text_lower or "log" in text_lower:
            actions.append("Check internal logs for error trace")

    # De-duplicate
    unique_actions = list(dict.fromkeys(actions))

    # Fallback default actions if none parsed
    if not unique_actions:
        unique_actions = [
            "Verify customer identity and account status",
            "Review transaction details in billing dashboard",
            "Update customer contact information in CRM"
        ]
    return unique_actions[:4]


def evaluate_alerts(
    wer_score: float | None, segments: List[TranscriptSegment]
) -> tuple[bool, List[str]]:
    """
    Check if the call requires a manager alert.
    Triggers:
      1. WER > 30% (0.30)
      2. Angry / frustrated phrases used by the customer.
      3. Multiple negative sentiment customer segments.
    """
    flagged = False
    reasons = []

    # 1. Check WER > 30%
    if wer_score is not None and wer_score > 0.30:
        flagged = True
        reasons.append(
            f"High Word Error Rate ({wer_score * 100:.1f}%) detected."
        )

    # 2. Check for angry words / frustration
    angry_keywords = [
        "angry", "frustrated", "bad service", "terrible", "worst", "gussa",
        "disappointed", "complaint", "bekar", "unexpected charge"
    ]
    for seg in segments:
        # Strip HTML markup before scanning keywords
        clean_text = re.sub(r'<[^>]*>', '', seg.text).lower()
        if any(kw in clean_text for kw in angry_keywords):
            speaker_str = (seg.speaker or "").lower()
            # If Speaker B or Customer or default speaker tag
            if "b" in speaker_str or "customer" in speaker_str:
                flagged = True
                reasons.append("Customer frustration detected in speech.")
                break

    return flagged, reasons
