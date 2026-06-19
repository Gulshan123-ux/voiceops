"""
conftest.py
Shared pytest fixtures and configuration for the entire test suite.
"""

from __future__ import annotations

import os
import pytest


@pytest.fixture(autouse=True)
def mock_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Set safe default environment variables for all tests.

    This prevents tests from accidentally reading the developer's real .env
    and ensures consistent behaviour across all environments (CI/CD, local).
    """
    # Only set defaults if not already set (allows individual tests to override)
    monkeypatch.setenv("ASR_BACKEND", os.getenv("ASR_BACKEND", "whisper"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")  # Suppress INFO logs during tests
    monkeypatch.setenv("LOG_DIR", "/tmp/voiceops_test_logs")
