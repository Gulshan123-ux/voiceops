"""
tests/test_wer.py
WER evaluation test suite — runs without any API credentials.

Run with:
    pytest tests/test_wer.py -v

All tests use the built-in noise simulators (no live Whisper/Deepgram calls),
so this suite runs safely in CI environments.

Test structure
──────────────
1. Unit tests for ``compute_wer`` correctness.
2. Unit tests for ``compute_cer`` correctness.
3. Integration-style tests for each noise scenario (office, accent, phone).
4. A final test that prints the full formatted WER report.
"""

from __future__ import annotations

import pytest

from app.wer_evaluator import (
    WER_THRESHOLD_NOISY,
    _simulate_accented_speech,
    _simulate_office_noise,
    _simulate_phone_quality,
    compute_cer,
    compute_wer,
    print_wer_report,
    run_wer_test_suite,
)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: compute_wer
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeWER:
    """Tests for the core WER calculation function."""

    def test_perfect_match_returns_zero(self) -> None:
        """Identical reference and hypothesis should give WER = 0.0."""
        wer = compute_wer(
            reference="hello thank you for calling support",
            hypothesis="hello thank you for calling support",
        )
        assert wer == 0.0, f"Expected WER=0.0, got {wer}"

    def test_completely_wrong_hypothesis(self) -> None:
        """Totally different hypothesis should give WER > 0."""
        wer = compute_wer(
            reference="hello thank you for calling",
            hypothesis="cat dog bird fish tree",
        )
        assert wer > 0.0, "WER should be positive for mismatched transcripts"

    def test_case_insensitive(self) -> None:
        """WER should not change due to casing differences."""
        wer = compute_wer(
            reference="Hello Thank You",
            hypothesis="hello thank you",
        )
        assert wer == 0.0, "Case differences should not affect WER"

    def test_punctuation_ignored(self) -> None:
        """Punctuation should be stripped before WER computation."""
        wer = compute_wer(
            reference="Hello, thank you!",
            hypothesis="Hello thank you",
        )
        assert wer == 0.0, "Punctuation should be ignored in WER"

    def test_single_word_deletion(self) -> None:
        """Deleting one word from a 4-word reference = 25% WER."""
        # Reference: 4 words  |  Hypothesis: 3 words (1 deletion)
        wer = compute_wer(
            reference="one two three four",
            hypothesis="one two three",
        )
        # WER = 1 deletion / 4 reference words = 0.25
        assert abs(wer - 0.25) < 0.01, f"Expected ~0.25 WER, got {wer}"

    def test_empty_reference_raises(self) -> None:
        """compute_wer must raise ValueError for empty reference."""
        with pytest.raises(ValueError, match="Reference text cannot be empty"):
            compute_wer(reference="", hypothesis="some text")

    def test_returns_float(self) -> None:
        """WER must always return a float."""
        wer = compute_wer(reference="hello world", hypothesis="hello world")
        assert isinstance(wer, float)

    def test_wer_bounded_below_zero(self) -> None:
        """WER is always ≥ 0."""
        wer = compute_wer(reference="hello world", hypothesis="hello world extra")
        assert wer >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: compute_cer
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeCER:
    """Tests for the supplementary CER metric."""

    def test_perfect_match(self) -> None:
        """Identical strings → CER = 0.0."""
        cer = compute_cer("hello", "hello")
        assert cer == 0.0

    def test_empty_reference_raises(self) -> None:
        """compute_cer must raise ValueError for empty reference."""
        with pytest.raises(ValueError, match="Reference text cannot be empty"):
            compute_cer(reference="", hypothesis="some text")

    def test_cer_less_than_wer_for_partial_substitution(self) -> None:
        """CER is typically lower than WER for minor character-level errors."""
        ref = "hello world"
        hyp = "helo world"  # one character dropped
        cer = compute_cer(ref, hyp)
        wer = compute_wer(ref, hyp)
        # CER < WER is the expected relationship here
        assert cer < wer, f"Expected CER ({cer}) < WER ({wer})"


# ─────────────────────────────────────────────────────────────────────────────
# Noise simulator tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseSimulators:
    """Validate that noise simulators produce deterministically corrupted output."""

    _REFERENCE = (
        "hello thank you for calling customer support "
        "how can I help you today I need a refund please"
    )

    def test_office_noise_changes_text(self) -> None:
        """Office noise simulator should produce output different from input."""
        import random
        random.seed(42)
        corrupted = _simulate_office_noise(self._REFERENCE, error_rate=0.30)
        assert corrupted != self._REFERENCE, "Office noise simulator produced no change"

    def test_accented_speech_changes_text(self) -> None:
        """Accent simulator should mutate at least some known confusion words."""
        import random
        random.seed(7)
        # The reference must contain accent-confusion words
        ref = "the customer has this problem with the value of three items"
        corrupted = _simulate_accented_speech(ref, error_rate=1.0)  # 100% rate forces changes
        assert corrupted != ref, "Accent simulator produced no change"

    def test_phone_quality_shortens_text(self) -> None:
        """Phone quality simulator should produce fewer words (deletions)."""
        import random
        random.seed(99)
        corrupted = _simulate_phone_quality(self._REFERENCE, deletion_rate=0.90)
        original_words = len(self._REFERENCE.split())
        corrupted_words = len(corrupted.split())
        assert corrupted_words < original_words, (
            f"Expected fewer words after phone simulation: {corrupted_words} vs {original_words}"
        )

    def test_simulators_return_strings(self) -> None:
        """All simulators must return string types."""
        import random
        random.seed(0)
        assert isinstance(_simulate_office_noise(self._REFERENCE), str)
        assert isinstance(_simulate_accented_speech(self._REFERENCE), str)
        assert isinstance(_simulate_phone_quality(self._REFERENCE), str)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario-level WER tests (core requirement)
# ─────────────────────────────────────────────────────────────────────────────

class TestWERScenarios:
    """
    Per-scenario WER tests.

    Each test validates:
      - WER is computed (not None/error)
      - WER is within acceptable threshold for the noise type
      - WERScenarioResult.passed flag is consistent with threshold
    """

    @pytest.fixture(scope="class")
    @classmethod
    def report(cls):  # noqa: ANN201
        """Run the full WER suite once per class for efficiency."""
        return run_wer_test_suite()

    def test_scenario_count(self, report) -> None:  # noqa: ANN001
        """There must be exactly 3 noise scenarios in the report."""
        assert len(report.scenarios) == 3, (
            f"Expected 3 scenarios, got {len(report.scenarios)}"
        )

    def test_office_noise_scenario_present(self, report) -> None:  # noqa: ANN001
        """Office noise scenario must be included."""
        names = [s.scenario for s in report.scenarios]
        assert any("Office" in n for n in names), f"Office scenario missing from {names}"

    def test_accented_speech_scenario_present(self, report) -> None:  # noqa: ANN001
        """Accented speech scenario must be included."""
        names = [s.scenario for s in report.scenarios]
        assert any("Accent" in n for n in names), f"Accent scenario missing from {names}"

    def test_phone_quality_scenario_present(self, report) -> None:  # noqa: ANN001
        """Phone quality scenario must be included."""
        names = [s.scenario for s in report.scenarios]
        assert any("Phone" in n for n in names), f"Phone scenario missing from {names}"

    def test_all_scenarios_have_valid_wer(self, report) -> None:  # noqa: ANN001
        """Every scenario's WER must be a non-negative float."""
        for scenario in report.scenarios:
            assert isinstance(scenario.wer, float), (
                f"WER for '{scenario.scenario}' is not a float: {scenario.wer!r}"
            )
            assert scenario.wer >= 0.0, (
                f"WER for '{scenario.scenario}' is negative: {scenario.wer}"
            )

    def test_all_scenarios_pass_noisy_threshold(self, report) -> None:  # noqa: ANN001
        """All noise scenarios must have WER ≤ 30%."""
        for scenario in report.scenarios:
            assert scenario.wer <= WER_THRESHOLD_NOISY, (
                f"Scenario '{scenario.scenario}' WER={scenario.wer:.2%} "
                f"exceeds noisy threshold {WER_THRESHOLD_NOISY:.0%}"
            )

    def test_scenario_passed_flag_consistent(self, report) -> None:  # noqa: ANN001
        """The 'passed' flag must match the WER vs threshold comparison."""
        for scenario in report.scenarios:
            expected_passed = scenario.wer <= WER_THRESHOLD_NOISY
            assert scenario.passed == expected_passed, (
                f"Scenario '{scenario.scenario}': passed={scenario.passed} "
                f"but WER={scenario.wer:.2%} vs threshold {WER_THRESHOLD_NOISY:.0%}"
            )

    def test_average_wer_computed_correctly(self, report) -> None:  # noqa: ANN001
        """Average WER should match the mean of individual scenario WERs."""
        expected_avg = round(
            sum(s.wer for s in report.scenarios) / len(report.scenarios), 4
        )
        assert abs(report.average_wer - expected_avg) < 1e-4, (
            f"average_wer={report.average_wer} does not match computed mean {expected_avg}"
        )

    def test_overall_passed_flag(self, report) -> None:  # noqa: ANN001
        """overall_passed must be True iff all individual scenarios passed."""
        expected = all(s.passed for s in report.scenarios)
        assert report.overall_passed == expected


# ─────────────────────────────────────────────────────────────────────────────
# Full report printing test
# ─────────────────────────────────────────────────────────────────────────────

class TestWERReportOutput:
    """Validate the human-readable report generation."""

    def test_print_report_runs_without_error(self, capsys) -> None:  # noqa: ANN001
        """print_wer_report must execute without raising exceptions."""
        report = run_wer_test_suite()
        print_wer_report(report)  # Should not raise

        captured = capsys.readouterr()
        assert "WER Evaluation Report" in captured.out, (
            "Report header not found in output"
        )
        assert "Average WER" in captured.out, "Average WER row missing from report"

    def test_report_contains_all_scenario_names(self, capsys) -> None:  # noqa: ANN001
        """Report output must mention every scenario name."""
        report = run_wer_test_suite()
        print_wer_report(report)
        captured = capsys.readouterr()

        for scenario in report.scenarios:
            # Check at least the first word of the scenario name is present
            first_word = scenario.scenario.split()[0]
            assert first_word in captured.out, (
                f"Scenario '{scenario.scenario}' not found in report output"
            )
