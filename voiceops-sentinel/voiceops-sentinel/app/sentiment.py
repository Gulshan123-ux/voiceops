"""
app/sentiment.py
Sentiment analyzer module. Uses Twitter-RoBERTa-base sentiment model,
with robust rule-based + lexical analysis fallback if transformers/torch is unavailable.
"""

from __future__ import annotations
import logging
import os
import re

logger = logging.getLogger(__name__)

# Try importing transformers and torch
_HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("transformers/torch not installed. Using robust lexical fallback for sentiment analysis.")


class SentimentAnalyzer:
    """
    Analyzes sentiment of transcripts and segments.
    """
    def __init__(self):
        self.pipeline = None
        if _HAS_TRANSFORMERS:
            try:
                # Force local files only to prevent downloading during unit tests or runtime hangs
                model_name = "cardiffnlp/twitter-roberta-base-sentiment"
                tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
                # Use CPU by default to avoid CUDA setup issues
                self.pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                logger.info("Loaded RoBERTa sentiment analysis pipeline successfully.")
            except Exception as e:
                logger.warning(f"Could not load RoBERTa pipeline locally: {e}. Falling back to lexical engine.")

    def analyze_text(self, text: str) -> tuple[str, float]:
        """
        Analyze a single string of text.
        Returns:
            Tuple of (SentimentLabel, ConfidencePercentage)
            SentimentLabel: "Positive" | "Negative" | "Neutral"
            ConfidencePercentage: float between 0.0 and 100.0
        """
        if not text.strip():
            return "Neutral", 100.0

        if self.pipeline:
            try:
                # RoBERTa mapping: LABEL_0 -> Negative, LABEL_1 -> Neutral, LABEL_2 -> Positive
                res = self.pipeline(text[:512])[0]
                label_map = {
                    "LABEL_0": "Negative",
                    "LABEL_1": "Neutral",
                    "LABEL_2": "Positive"
                }
                label = label_map.get(res["label"], "Neutral")
                score = round(res["score"] * 100.0, 2)
                return label, score
            except Exception as e:
                logger.error(f"Transformers sentiment run failed: {e}. Using lexical fallback.")

        # Lexical Fallback
        text_lower = text.lower()
        
        # Word lists
        positive_words = {
            "good", "great", "excellent", "awesome", "perfect", "satisfied", "thank",
            "thanks", "helpful", "resolved", "solving", "happy", "wonderful", "nice",
            "shukriya", "dhanyawad", "acha", "sahi", "thik"
        }
        negative_words = {
            "bad", "terrible", "worst", "angry", "frustrated", "gussa", "disappointed",
            "disappointing", "cancel", "refund", "waste", "useless", "annoyed", "annoying",
            "complaint", "bekar", "kharab", "radd", "chahat", "faltu", "error", "broken"
        }

        # Simple score
        words = re.findall(r'\b\w+\b', text_lower)
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)

        # Confidence heuristic
        total_hits = pos_count + neg_count
        if total_hits == 0:
            return "Neutral", 80.0

        diff = pos_count - neg_count
        confidence = min(100.0, 50.0 + (abs(diff) / total_hits) * 50.0)

        if diff > 0:
            return "Positive", round(confidence, 2)
        elif diff < 0:
            return "Negative", round(confidence, 2)
        else:
            return "Neutral", 70.0

    def analyze_call_sentiment(self, segments: list[dict]) -> tuple[str, float, bool]:
        """
        Aggregates segment sentiments.
        Returns:
            Tuple of (overall_sentiment, average_confidence, is_flagged)
            is_flagged: True if negative segments make up > 70% of non-neutral segments,
            or if severe customer anger is detected.
        """
        if not segments:
            return "Neutral", 100.0, False

        pos_count = 0
        neg_count = 0
        neu_count = 0
        total_conf = 0.0

        # Check for angry customer trigger keywords in customer turns
        angry_keywords = {"angry", "frustrated", "gussa", "terrible", "worst", "cancel my account"}
        customer_angry_detected = False

        for seg in segments:
            text = seg.get("text", "")
            speaker = seg.get("speaker", "")
            
            label, score = self.analyze_text(text)
            total_conf += score

            if label == "Positive":
                pos_count += 1
            elif label == "Negative":
                neg_count += 1
                # If speaker is customer (Speaker B)
                if "customer" in speaker.lower() or "b" in speaker.lower():
                    clean_text = re.sub(r'<[^>]*>', '', text).lower()
                    if any(kw in clean_text for kw in angry_keywords):
                        customer_angry_detected = True
            else:
                neu_count += 1

        avg_conf = round(total_conf / len(segments), 2) if segments else 100.0
        
        # Overall label
        if pos_count > neg_count:
            overall = "Positive"
        elif neg_count > pos_count:
            overall = "Negative"
        else:
            overall = "Neutral"

        # Flag call if negative > 70% of non-neutral segments, or direct customer anger
        non_neutral = pos_count + neg_count
        neg_ratio = (neg_count / non_neutral) if non_neutral > 0 else 0.0
        
        is_flagged = (neg_ratio > 0.70) or customer_angry_detected
        
        return overall, avg_conf, is_flagged
