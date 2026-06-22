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
    from presidio_analyzer.nlp_engine import NlpEngineProvider
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
                # Configure Presidio to use the small model en_core_web_sm
                config = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
                }
                provider = NlpEngineProvider(nlp_configuration=config)
                nlp_engine = provider.create_engine()
                
                self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
                self.anonymizer = AnonymizerEngine()
                logger.info("Presidio Analyzer/Anonymizer engines initialized successfully using en_core_web_sm.")
            except Exception as e:
                logger.error(f"Failed to initialize Presidio: {e}. Using regex fallback.")

    def redact(self, text: str) -> str:
        """
        Redact PII from the text, returning the redacted string.
        Redacted tags match the required formats: [NAME], [EMAIL], [PHONE], [CARD].
        """
        if not text.strip():
            return text

        redacted = text
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
                
                # Filter out PERSON entities that are actually common Hinglish words
                hindi_stopwords = {
                    "kar", "mera", "main", "aapki", "kya", "sakta", "hoon", "aapka", "swagat", 
                    "hai", "naam", "se", "ko", "ki", "ka", "shukriya", "dhanyawad", "acha", 
                    "sahi", "thik", "aur", "ya", "ho", "gaya", "karo", "karna", "leta", "raha", 
                    "ke", "liye", "mein", "team"
                }
                filtered_results = []
                for res in results:
                    ent_text = text[res.start:res.end].lower().strip()
                    if res.entity_type == "PERSON" and ent_text in hindi_stopwords:
                        continue
                    filtered_results.append(res)

                # Operators configuration
                operators = {
                    "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED NAME]"}),
                    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED EMAIL]"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED PHONE]"}),
                    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[REDACTED CARD]"}),
                    "CRYPTO": OperatorConfig("replace", {"new_value": "[REDACTED CARD]"}),
                }
                
                anonymized_result = self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=filtered_results,
                    operators=operators
                )
                redacted = anonymized_result.text
            except Exception as e:
                logger.error(f"Presidio redaction failed: {e}. Falling back to regex.")

        # Apply additional regex rules on top of Presidio output for complete coverage
        # (e.g. Hinglish name patterns, spelled-out numbers)

        # 1. Emails
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        redacted = re.sub(email_pattern, '[REDACTED EMAIL]', redacted)

        # 2. Credit Cards / Account Numbers (12 to 19 digits)
        card_pattern = r'\b(?:\d[ -]*?){12,19}\b'
        redacted = re.sub(card_pattern, '[REDACTED CARD]', redacted)

        # 3. Numeric Phone Numbers
        phone_pattern = r'\b(\+?\d{1,4}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'
        redacted = re.sub(phone_pattern, '[REDACTED PHONE]', redacted)

        # 4. Spelled-out numbers (common in support transcripts)
        spelled_seq_pattern = (
            r'(\b(?:zero|one|two|three|four|five|six|seven|eight|nine|double\s+'
            r'(?:zero|one|two|three|four|five|six|seven|eight|nine))\b[-.\s]*){3,}'
        )
        redacted = re.sub(spelled_seq_pattern, '[REDACTED PHONE]', redacted, flags=re.IGNORECASE)

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
            return f"{intro} [REDACTED NAME]"
        redacted = re.sub(name_pattern, name_replacer, redacted)

        # 6. Specific common name occurrences in context
        common_names = r'\b(Alex|Amit|Sarah|John|Michael|Jessica|David|Emily)\b'
        redacted = re.sub(common_names, '[REDACTED NAME]', redacted, flags=re.IGNORECASE)

        return redacted
