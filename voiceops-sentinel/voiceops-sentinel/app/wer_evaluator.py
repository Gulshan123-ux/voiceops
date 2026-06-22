"""
app/wer_evaluator.py
Word Error Rate (WER) calculation + noise-scenario test suite.

What is WER?
────────────
WER = (S + D + I) / N
  where S=substitutions, D=deletions, I=insertions, N=words in reference.

We use the ``jiwer`` library which implements this correctly with
text normalisation (lowercase, strip punctuation) via ``Compose``
transforms.

Noise scenarios
───────────────
Because we may not have a live Whisper API during unit-testing, we
*simulate* transcription errors programmatically by introducing
controlled mutations into the hypothesis text:

  a) Office noise  → random word substitutions (≈10% of words)
  b) Accented speech → phoneme-adjacent substitutions (≈15% of words)
  c) Phone quality → word deletions + noise tokens (≈20% degradation)

These simulations let the test suite validate the WER calculator itself
and set realistic pass/fail thresholds without consuming API credits.
"""

from __future__ import annotations

import logging
import random
from typing import List

import jiwer

from app.schemas import WERReport, WERScenarioResult

logger = logging.getLogger(__name__)

# ── WER thresholds ─────────────────────────────────────────────────────────────
WER_THRESHOLD_CLEAN: float = 0.15   # 15 % – clean audio target
WER_THRESHOLD_NOISY: float = 0.30   # 30 % – noisy audio target

# ── jiwer text normalisation pipeline ─────────────────────────────────────────
# This pipeline is applied to BOTH reference and hypothesis before comparison
# to avoid spurious errors from punctuation, casing, or extra whitespace.
_TRANSFORMS = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemovePunctuation(),
    jiwer.ReduceToListOfListOfWords(),
])


# ─────────────────────────────────────────────────────────────────────────────
# Core WER computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis strings.

    Both strings are normalised (lowercased, punctuation removed, stripped)
    before comparison so that superficial differences do not inflate WER.

    Args:
        reference:  Ground-truth transcript string.
        hypothesis: ASR-generated transcript string.

    Returns:
        WER as a float in range [0.0, ∞).  Values above 1.0 are valid
        (more insertions/deletions than reference words).

    Raises:
        ValueError: If reference is empty (WER is undefined for empty refs).
    """
    if not reference.strip():
        raise ValueError("Reference text cannot be empty for WER computation.")

    wer_value = jiwer.wer(
        reference,
        hypothesis,
        reference_transform=_TRANSFORMS,
        hypothesis_transform=_TRANSFORMS,
    )
    logger.debug("WER: %.4f (ref_len=%d words)", wer_value, len(reference.split()))
    return round(float(wer_value), 4)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) as a supplementary metric.

    CER is useful for languages where word boundaries are ambiguous,
    or as a secondary signal when WER is misleadingly high.

    Args:
        reference:  Ground-truth string.
        hypothesis: ASR-generated string.

    Returns:
        CER as a float in range [0.0, ∞).
    """
    if not reference.strip():
        raise ValueError("Reference text cannot be empty for CER computation.")
    return round(float(jiwer.cer(reference, hypothesis)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Noise simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_office_noise(text: str, error_rate: float = 0.10) -> str:
    """
    Simulate ASR errors caused by background office noise.

    Strategy: Randomly substitute ~10% of words with phonetically plausible
    but wrong alternatives drawn from a small confusion dictionary.

    Args:
        text:       Clean reference transcript.
        error_rate: Fraction of words to corrupt (default 10%).

    Returns:
        Corrupted hypothesis string.
    """
    # Common office-noise ASR confusions (real-world examples)
    _OFFICE_CONFUSIONS = {
        "hello": "yellow",
        "support": "report",
        "calling": "falling",
        "account": "amount",
        "number": "member",
        "refund": "respond",
        "payment": "shipment",
        "please": "peace",
        "order": "older",
        "issue": "tissue",
    }

    words = text.split()
    corrupted: List[str] = []
    for word in words:
        lower = word.lower()
        # Apply known confusion if available, else random deletion
        if random.random() < error_rate:
            if lower in _OFFICE_CONFUSIONS:
                corrupted.append(_OFFICE_CONFUSIONS[lower])
            else:
                # Drop word entirely to simulate noise masking
                pass  # intentional skip (deletion)
        else:
            corrupted.append(word)
    return " ".join(corrupted)


def _simulate_accented_speech(text: str, error_rate: float = 0.15) -> str:
    """
    Simulate ASR errors from accented speech (e.g. South Asian or European accents).

    Strategy: Substitute ~15% of words with phoneme-adjacent alternatives
    representing common accent-induced ASR mistakes.

    Args:
        text:       Clean reference transcript.
        error_rate: Fraction of words to corrupt (default 15%).

    Returns:
        Corrupted hypothesis string.
    """
    _ACCENT_CONFUSIONS = {
        "the": "da",
        "this": "dis",
        "that": "dat",
        "three": "tree",
        "thirty": "dirty",
        "very": "wery",
        "value": "walue",
        "have": "hav",
        "with": "wid",
        "think": "tink",
        "customer": "costumer",
        "service": "serwice",
        "problem": "broblem",
    }

    words = text.split()
    corrupted: List[str] = []
    for word in words:
        lower = word.lower()
        if random.random() < error_rate and lower in _ACCENT_CONFUSIONS:
            corrupted.append(_ACCENT_CONFUSIONS[lower])
        else:
            corrupted.append(word)
    return " ".join(corrupted)


def _simulate_phone_quality(text: str, deletion_rate: float = 0.20) -> str:
    """
    Simulate ASR errors from low-bitrate phone audio.

    Phone calls at 8 kHz / G.711 codec lose high-frequency detail.
    Strategy: Delete ~20% of words randomly (dropouts / packet loss effects),
    and occasionally insert filler tokens ('uh', 'um') to mimic codec artifacts.

    Args:
        text:          Clean reference transcript.
        deletion_rate: Fraction of words to delete (default 20%).

    Returns:
        Corrupted hypothesis string.
    """
    words = text.split()
    corrupted: List[str] = []
    for word in words:
        if random.random() < deletion_rate:
            # Simulate dropout: skip word, sometimes insert artifact filler
            if random.random() < 0.2:
                corrupted.append("uh")
        else:
            corrupted.append(word)
    return " ".join(corrupted)


# ─────────────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────────────

# Ground-truth reference texts (simulate real customer support call snippets)
_REFERENCE_TEXTS = [
    (
        "Hello thank you for calling customer support my name is Alex how can I help you today "
        "I would like to check the status of my order number 12345 "
        "please hold while I look that up for you"
    ),
    (
        "I understand you are having trouble with your account "
        "let me verify your identity first can you please provide your phone number "
        "yes my number is 555-0199 and I have been waiting for a refund for three weeks"
    ),
    (
        "Thank you for your patience I can see that your payment was processed "
        "however there is a delay in the shipping "
        "we will send you an email with the tracking number within 24 hours "
        "is there anything else I can help you with today"
    ),
]

_SCENARIO_CONFIGS = [
    {
        "name": "Background Office Noise",
        "simulator": _simulate_office_noise,
        "threshold": WER_THRESHOLD_NOISY,
        "seed": 42,
    },
    {
        "name": "Accented Speech Simulation",
        "simulator": _simulate_accented_speech,
        "threshold": WER_THRESHOLD_NOISY,
        "seed": 7,
    },
    {
        "name": "Phone Call Quality (Low Bitrate)",
        "simulator": _simulate_phone_quality,
        "threshold": WER_THRESHOLD_NOISY,
        "seed": 99,
    },
]


def run_wer_test_suite() -> WERReport:
    """
    Execute WER evaluation across all noise scenarios and return a report.

    Each scenario takes one of the reference texts, applies a noise
    simulation function to generate a synthetic hypothesis, then computes WER.

    This function is designed to run without any API calls (pure Python),
    making it safe for CI environments without API credentials.

    Returns:
        WERReport containing per-scenario results and the overall average WER.
    """
    results: List[WERScenarioResult] = []

    for i, config in enumerate(_SCENARIO_CONFIGS):
        random.seed(config["seed"])  # Deterministic for reproducibility
        reference = _REFERENCE_TEXTS[i % len(_REFERENCE_TEXTS)]
        hypothesis = config["simulator"](reference)

        wer = compute_wer(reference=reference, hypothesis=hypothesis)
        passed = wer <= config["threshold"]

        result = WERScenarioResult(
            scenario=config["name"],
            reference=reference,
            hypothesis=hypothesis,
            wer=wer,
            passed=passed,
        )
        results.append(result)

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(
            "WER test [%s]: WER=%.2f%% (threshold=%.0f%%) %s",
            config["name"], wer * 100, config["threshold"] * 100, status,
        )

    avg_wer = round(sum(r.wer for r in results) / len(results), 4)
    overall_passed = all(r.passed for r in results)

    logger.info(
        "WER test suite complete: avg_wer=%.2f%%, overall=%s",
        avg_wer * 100, "PASS" if overall_passed else "FAIL",
    )

    return WERReport(
        scenarios=results,
        average_wer=avg_wer,
        overall_passed=overall_passed,
    )


def print_wer_report(report: WERReport) -> None:
    """
    Pretty-print the WER report to stdout in a human-readable tabular format.

    Args:
        report: WERReport produced by ``run_wer_test_suite``.
    """
    line = "─" * 80
    print(f"\n{line}")
    print("  VoiceOps Sentinel – WER Evaluation Report")
    print(line)
    print(f"  {'Scenario':<40} {'WER':>8}  {'Threshold':>9}  {'Status':>8}")
    print(line)

    for r in report.scenarios:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        is_noisy = (
            "noise" in r.scenario.lower()
            or "phone" in r.scenario.lower()
            or "accent" in r.scenario.lower()
        )
        threshold_label = (
            f"{WER_THRESHOLD_NOISY * 100:.0f}%"
            if is_noisy
            else f"{WER_THRESHOLD_CLEAN * 100:.0f}%"
        )
        print(
            f"  {r.scenario:<40} {r.wer * 100:>7.2f}%  {threshold_label:>9}  {status:>8}"
        )

    print(line)
    overall = "✅ PASS" if report.overall_passed else "❌ FAIL"
    print(f"  {'Average WER':<40} {report.average_wer * 100:>7.2f}%  {'30%':>9}  {overall:>8}")
    print(f"{line}\n")
