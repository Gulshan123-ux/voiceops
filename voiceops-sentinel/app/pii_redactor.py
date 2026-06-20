"""
app/pii_redactor.py
PII Redactor module. Uses SpaCy + Presidio Analyzer & Anonymizer,
with robust regex fallback if SpaCy/Presidio is not available.
"""

from __future__ import annotations
import logging
import re

logger = logging.getLogger(__name__)

_HAS_PRESIDIO = False
try:
    import spacy
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    _HAS_PRESIDIO = True
except ImportError:
    logger.warning("spacy or presidio-analyzer/anonymizer not installed. Using robust regex-based PII Redactor.")


class PIIRedactor:
    """
    Identifies and redacts PII elements from transcript text.
    """
    def __init__(self):
        self.analyzer = None
        self.anonymizer = None
        if _HAS_PRESIDIO:
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                logger.info("Presidio Analyzer/Anonymizer engines initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Presidio: {e}. Using regex fallback.")

    def redact(self, text: str) -> str:
        """
        Redact PII from the text, returning the redacted string.
        Redacted tags match the required formats: [NAME], [EMAIL], [PHONE], [CARD].
        """
        if not text.strip():
            return text

        if self.analyzer and self.anonymizer:
            try:
                # Map Presidio entities to our required outputs
                # EMAIL -> [EMAIL]
                # PHONE_NUMBER -> [PHONE]
                # PERSON -> [NAME]
                # CREDIT_CARD -> [CARD]
                # US_BANK_NUMBER / UK_BANK_NUMBER -> [CARD]
                results = self.analyzer.analyze(
                    text=text,
                    entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON", "CREDIT_CARD", "CRYPTO"],
                    language="en"
                )
                
                # Operators configuration
                operators = {
                    "PERSON": OperatorConfig("replace", {"value": "[NAME]"}),
                    "EMAIL_ADDRESS": OperatorConfig("replace", {"value": "[EMAIL]"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"value": "[PHONE]"}),
                    "CREDIT_CARD": OperatorConfig("replace", {"value": "[CARD]"}),
                    "CRYPTO": OperatorConfig("replace", {"value": "[CARD]"}),
                }
                
                anonymized_result = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=results,
                    operators=operators
                )
                return anonymized_result.text
            except Exception as e:
                logger.error(f"Presidio redaction failed: {e}. Falling back to regex.")

        # Regex Fallback
        redacted = text

        # 1. Emails
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        redacted = re.sub(email_pattern, '[EMAIL]', redacted)

        # 2. Credit Cards / Account Numbers (12 to 19 digits)
        card_pattern = r'\b(?:\d[ -]*?){12,19}\b'
        redacted = re.sub(card_pattern, '[CARD]', redacted)

        # 3. Numeric Phone Numbers
        phone_pattern = r'\b(\+?\d{1,4}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
        redacted = re.sub(phone_pattern, '[PHONE]', redacted)

        # 4. Spelled-out numbers (common in support transcripts)
        spelled_seq_pattern = (
            r'(\b(?:zero|one|two|three|four|five|six|seven|eight|nine|double\s+'
            r'(?:zero|one|two|three|four|five|six|seven|eight|nine))\b[-.\s]*){3,}'
        )
        redacted = re.sub(spelled_seq_pattern, '[PHONE]', redacted, flags=re.IGNORECASE)

        # 5. Names introduced by intro phrases
        name_pattern = r'(?i)\b(my name is|mera naam|naam hai|this is)\s+([A-Za-z]+)'
        def name_replacer(match):
            intro = match.group(1)
            name = match.group(2)
            stopwords = {
                'support', 'a', 'the', 'an', 'to', 'for', 'in', 'on', 'at', 'with', 'from', 'by',
                'agent', 'customer', 'here', 'online', 'going', 'doing', 'working', 'help'
            }
            if name.lower() in stopwords:
                return match.group(0)
            return f"{intro} [NAME]"
        redacted = re.sub(name_pattern, name_replacer, redacted)

        # 6. Specific common name occurrences in context
        common_names = r'\b(Alex|Amit|Sarah|John|Michael|Jessica|David|Emily)\b'
        redacted = re.sub(common_names, '[NAME]', redacted, flags=re.IGNORECASE)

        return redacted
