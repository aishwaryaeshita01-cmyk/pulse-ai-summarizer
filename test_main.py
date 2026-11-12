import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from groq import GroqError
from main import app

client = TestClient(app)


def _mock_chat_completion(content_dict: dict):
    """Helper to build a mock Groq chat completion response."""
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = json.dumps(content_dict)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "pulse-ai-summarizer"}


@patch("main.groq_client.chat.completions.create")
def test_model_decommissioned_automatic_fallback(mock_create):
    """Simulates primary model failing with GroqError and verifies fallback succeeds."""
    valid_payload = {
        "executive_summary": "System passed automated resilience verification.",
        "key_takeaways": ["Primary model failed gracefully", "Fallback model succeeded"],
        "action_items": ["Deploy to staging"],
        "sentiment": "Positive",
    }

    # First call raises GroqError (e.g., 400 model_decommissioned); second call succeeds
    mock_create.side_effect = [
        GroqError("Error code: 400 - model_decommissioned"),
        _mock_chat_completion(valid_payload),
    ]

    response = client.post(
        "/api/v1/summarize",
        json={"transcript_text": "Test transcript.", "tier": "high_tier_reasoning"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["summarization_model"] == "qwen/qwen3.8-27b"
    assert data["analysis"]["sentiment"] == "Positive"
    assert mock_create.call_count == 2


@patch("main.groq_client.chat.completions.create")
def test_schema_validation_triggers_fallback(mock_create):
    """Simulates primary model violating Pydantic schema and verifies fallback recovers."""
    broken_payload = {"wrong_field": "Missing required executive_summary and takeaways"}
    valid_payload = {
        "executive_summary": "Recovered after schema mismatch.",
        "key_takeaways": ["Schema validation enforced"],
        "action_items": [],
        "sentiment": "Neutral",
    }

    mock_create.side_effect = [
        _mock_chat_completion(broken_payload),
        _mock_chat_completion(valid_payload),
    ]

    response = client.post(
        "/api/v1/summarize",
        json={"transcript_text": "Test transcript.", "tier": "fast_summary"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summarization_model"] == "openai/gpt-oss-20b"
    assert data["analysis"]["executive_summary"] == "Recovered after schema mismatch."


def test_reject_unsupported_audio_format():
    """Ensures non-audio files are rejected with 415 before calling Groq."""
    response = client.post(
        "/api/v1/process-audio",
        files={"file": ("notes.txt", b"plain text content", "text/plain")},
        data={"tier": "high_tier_reasoning"},
    )
    assert response.status_code == 415
    assert "Unsupported file format" in response.json()["detail"]
