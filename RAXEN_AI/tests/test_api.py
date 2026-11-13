from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from raxen_ai import api


class FakeWorkflow:
    def __init__(self) -> None:
        self.request = None

    def invoke(self, request: dict[str, object]) -> dict[str, str]:
        self.request = request
        return {"agent": "adrian", "response": "Let's plan your build."}

    def stream(self, request: dict[str, object], stream_mode: str):
        self.request = request
        assert stream_mode == "custom"
        yield {"type": "handoff", "agent": "adrian"}
        yield {"type": "delta", "content": "Let's "}
        yield {"type": "delta", "content": "plan your build."}
        yield {"type": "done", "agent": "adrian"}


def test_health_endpoint() -> None:
    response = TestClient(api.app).get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frontend_is_served_from_the_api_origin() -> None:
    response = TestClient(api.app).get("/")

    assert response.status_code == 200
    assert "<title>RAXEN AI" in response.text


def test_public_password_protects_frontend_and_api(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_ACCESS_PASSWORD", "a-long-test-password")
    client = TestClient(api.app)

    page_response = client.get("/")
    health_response = client.get("/api/health")
    authenticated_response = client.get(
        "/api/health",
        auth=("raxen", "a-long-test-password"),
    )

    assert page_response.status_code == 401
    assert page_response.headers["www-authenticate"].startswith("Basic")
    assert health_response.status_code == 401
    assert authenticated_response.status_code == 200


def test_chat_endpoint_passes_conversation_and_returns_agent(monkeypatch) -> None:
    workflow = FakeWorkflow()
    monkeypatch.setattr(api, "_get_workflow", lambda: workflow)
    client = TestClient(api.app)

    response = client.post(
        "/api/chat",
        json={
            "messages": [
                {"role": "user", "content": "I need a PC."},
                {"role": "assistant", "content": "What will you use it for?"},
            ],
            "prompt": "A custom build for gaming.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "agent": "adrian",
        "response": "Let's plan your build.",
    }
    assert workflow.request is not None
    assert workflow.request["user_input"] == "A custom build for gaming."
    assert isinstance(workflow.request["messages"][0], HumanMessage)
    assert isinstance(workflow.request["messages"][1], AIMessage)


def test_chat_endpoint_rejects_blank_message(monkeypatch) -> None:
    monkeypatch.setattr(api, "_get_workflow", FakeWorkflow)

    response = TestClient(api.app).post(
        "/api/chat",
        json={"messages": [], "prompt": "  "},
    )

    assert response.status_code == 422


def test_streaming_chat_endpoint_emits_server_sent_events(monkeypatch) -> None:
    workflow = FakeWorkflow()
    monkeypatch.setattr(api, "_get_workflow", lambda: workflow)

    response = TestClient(api.app).post(
        "/api/chat/stream",
        json={"messages": [], "prompt": "I need a custom PC."},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"type": "handoff", "agent": "adrian"}' in response.text
    assert '"type": "delta"' in response.text
    assert "\"content\": \"Let's \"" in response.text
    assert 'data: {"type": "done", "agent": "adrian"}' in response.text
