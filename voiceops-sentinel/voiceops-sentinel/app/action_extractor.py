"""
app/action_extractor.py
Action Item Extractor module. Uses OpenAI GPT-3.5-turbo (model) to extract action items,
with robust heuristic fallback if API fails or is unconfigured.
"""

from __future__ import annotations
import logging
import os
import json
import re

logger = logging.getLogger(__name__)


class ActionExtractor:
    """
    Extracts action items from customer service call transcripts.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

    def extract(self, transcript: str) -> list[str]:
        """
        Extract follow-up action items from a transcript.
        Returns:
            List of action item strings.
        """
        if not transcript.strip():
            return []

        # If API key is present, try OpenAI GPT-3.5-turbo
        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                
                prompt = (
                    "You are a VoiceOps quality assurance assistant. Analyze the customer support "
                    "call transcript below and extract key follow-up action items (tasks/agreed actions) "
                    "for the agent or system.\n\n"
                    "Transcript:\n"
                    f"\"{transcript}\"\n\n"
                    "Return ONLY a JSON array of strings containing the action items, like:\n"
                    "[\"Process refund of $45\", \"Follow up with email within 24 hours\"]\n"
                    "No conversational intro, no formatting other than the valid JSON array."
                )
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150
                )
                
                content = response.choices[0].message.content.strip()
                # Clean markdown code blocks if any
                if content.startswith("```"):
                    content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content)
                
                actions = json.loads(content)
                if isinstance(actions, list):
                    return [str(a) for a in actions]
            except Exception as e:
                logger.error(f"OpenAI action item extraction failed: {e}. Using heuristic fallback.")

        # Heuristic/Keyword Fallback
        actions = []
        text_lower = transcript.lower()

        # Check matching patterns for dynamic extraction
        if "refund" in text_lower or "paisa" in text_lower or "money back" in text_lower or "return" in text_lower:
            actions.append("Process billing refund for the customer")
        if "verify" in text_lower or "confirm" in text_lower or "details" in text_lower or "check my" in text_lower:
            actions.append("Verify customer account credentials and identity")
        if "email" in text_lower or "mail" in text_lower or "send" in text_lower:
            actions.append("Send confirmation email / tracking details to customer")
        if "schedule" in text_lower or "callback" in text_lower or "call you back" in text_lower:
            actions.append("Schedule customer callback or CRM follow-up ticket")
        if "escalate" in text_lower or "manager" in text_lower or "supervisor" in text_lower or "boss" in text_lower:
            actions.append("Escalate case to supervisor/manager for review")
        if "check" in text_lower or "log" in text_lower or "system" in text_lower:
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
