from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db, insert_call, get_db_connection
from app.search import search_calls
from app.tags import add_tag, remove_tag

@pytest.fixture(autouse=True)
def setup_db():
    """Reset the database before each test."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM calls")
    conn.commit()
    conn.close()

def test_insert_and_retrieve_intelligence():
    """Verify that intelligence and tag fields are persisted and retrieved correctly."""
    job_id = "test-job-id-1"
    insert_call(
        job_id=job_id,
        filename="billing_call.wav",
        duration=15.5,
        transcript="I want a refund for the extra charges.",
        redacted_transcript="I want a refund for the extra charges.",
        sentiment="Negative",
        sentiment_score=85.0,
        wer_score=0.05,
        action_items=["Process refund"],
        flagged=True,
        segments=[],
        summary="Customer billing dispute",
        summary_issue="Refund request",
        summary_resolution="Pending action",
        summary_follow_up="Agent callback",
        summary_engine="extractive",
        latency_report={"preprocess_ms": 100.0, "total_ms": 1200.0},
        tags=["billing", "high-priority"]
    )

    from app.database import get_call_by_id
    call = get_call_by_id(job_id)
    assert call is not None
    assert call["summary"] == "Customer billing dispute"
    assert call["summary_issue"] == "Refund request"
    assert call["tags"] == ["billing", "high-priority"]
    assert call["latency_report"]["preprocess_ms"] == 100.0

def test_search_calls_filters():
    """Test search_calls function with different filtering parameters."""
    insert_call(
        job_id="job-1",
        filename="sales_call.wav",
        duration=25.0,
        transcript="Do you have any discounts on annual subscriptions?",
        redacted_transcript="Do you have any discounts on annual subscriptions?",
        sentiment="Positive",
        sentiment_score=90.0,
        wer_score=0.01,
        action_items=[],
        flagged=False,
        summary="Sales discount inquiry",
        tags=["sales"]
    )
    insert_call(
        job_id="job-2",
        filename="tech_issue.wav",
        duration=45.0,
        transcript="My screen is black and I cannot login.",
        redacted_transcript="My screen is black and I cannot login.",
        sentiment="Negative",
        sentiment_score=95.0,
        wer_score=0.15,
        action_items=["Escalate to Tier 2"],
        flagged=True,
        summary="Login failure",
        tags=["tech-support", "broken"]
    )

    # 1. Search by text query
    results = search_calls(query="discounts")
    assert len(results) == 1
    assert results[0]["job_id"] == "job-1"

    # 2. Search by sentiment
    results = search_calls(sentiment="Negative")
    assert len(results) == 1
    assert results[0]["job_id"] == "job-2"

    # 3. Search by flagging status
    results = search_calls(flagged=True)
    assert len(results) == 1
    assert results[0]["job_id"] == "job-2"

    # 4. Search by tags
    results = search_calls(tag="tech-support")
    assert len(results) == 1
    assert results[0]["job_id"] == "job-2"

    # 5. Search by WER range
    results = search_calls(wer_max=0.05)
    assert len(results) == 1
    assert results[0]["job_id"] == "job-1"

def test_tags_add_remove():
    """Verify add_tag and remove_tag logic."""
    job_id = "tag-job"
    insert_call(
        job_id=job_id,
        filename="call.wav",
        duration=10.0,
        transcript="Hello support",
        redacted_transcript="Hello support",
        sentiment="Neutral",
        sentiment_score=50.0,
        wer_score=None,
        action_items=[],
        flagged=False
    )

    tags = add_tag(job_id, "urgent")
    assert "urgent" in tags

    tags = add_tag(job_id, "resolved")
    assert "resolved" in tags
    assert len(tags) == 2

    tags = remove_tag(job_id, "urgent")
    assert "urgent" not in tags
    assert "resolved" in tags
    assert len(tags) == 1

def test_api_search_and_tags_endpoints():
    """Verify integration of search and tags with HTTP client."""
    client = TestClient(app)
    job_id = "api-job"
    insert_call(
        job_id=job_id,
        filename="test_api.wav",
        duration=12.0,
        transcript="API test call",
        redacted_transcript="API test call",
        sentiment="Positive",
        sentiment_score=80.0,
        wer_score=0.02,
        action_items=[],
        flagged=False
    )

    # Test HTTP Search
    res = client.get("/calls?q=test&sentiment=Positive")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["job_id"] == job_id

    # Test HTTP Add Tag
    res = client.post(f"/calls/{job_id}/tags", data={"tag": "api-test"})
    assert res.status_code == 200
    assert "api-test" in res.json()

    # Test HTTP Delete Tag
    res = client.delete(f"/calls/{job_id}/tags/api-test")
    assert res.status_code == 200
    assert "api-test" not in res.json()
