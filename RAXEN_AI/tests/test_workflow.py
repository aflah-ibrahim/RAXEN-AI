import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from raxen_ai.agents import (
    ADRIAN_WORKSTATION_OPTIONS,
    ADRIAN_PROMPT,
    CONCIERGE_PROMPT,
    ETHAN_PROMPT,
    HANDOFF_LIMIT,
    HANDOFF_PROTOCOL,
    ROUTER_PROMPT,
    build_workflow,
)


class FakeChatModel:
    def __init__(self, routes: list[str], decisions: list[str] | None = None) -> None:
        self.routes = iter(routes)
        self.decisions = iter(decisions or [])
        self.calls: list[list[object]] = []

    def bind(self, **kwargs: object) -> "FakeChatModel":
        assert kwargs == {"response_format": {"type": "json_object"}}
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        self.calls.append(messages)
        if isinstance(messages[0], SystemMessage) and messages[0].content == ROUTER_PROMPT:
            return AIMessage(content=next(self.routes))
        return AIMessage(
            content=next(
                self.decisions,
                json.dumps({"action": "respond", "response": "A prompt-based response."}),
            )
        )

    def stream(self, messages: list[object]):
        response = self.invoke(messages)
        for character in response.content:
            yield AIMessageChunk(content=character)


@pytest.mark.parametrize(
    ("route", "expected"),
    [("ADRIAN", "adrian"), ("CONCIERGE", "concierge"), ("ETHAN", "ethan")],
)
def test_routes_to_each_agent(route: str, expected: str) -> None:
    workflow = build_workflow(FakeChatModel([route]))

    result = workflow.invoke({"messages": [], "user_input": "Help me choose."})

    assert result["agent"] == expected
    assert result["response"] == "A prompt-based response."
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)


@pytest.mark.parametrize("route", ["", "UNKNOWN"])
def test_invalid_router_output_raises_clear_error(route: str) -> None:
    with pytest.raises(ValueError, match="Router returned"):
        build_workflow(FakeChatModel([route])).invoke(
            {"messages": [], "user_input": "Help me."}
        )


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ('{"agent":"adrian"}', "adrian"),
        ('{"agent":"concierge"}', "concierge"),
        ('{"agent":"ethan"}', "ethan"),
        ("```json\n{\"agent\":\"adrian\"}\n```", "adrian"),
    ],
)
def test_router_parses_structured_route(route: str, expected: str) -> None:
    result = build_workflow(FakeChatModel([route])).invoke(
        {"messages": [], "user_input": "Help me."}
    )

    assert result["agent"] == expected


def test_router_rejects_conversational_text_instead_of_misrouting() -> None:
    with pytest.raises(ValueError, match="invalid specialist decision"):
        build_workflow(FakeChatModel(["Great!"])).invoke(
            {"messages": [], "user_input": "Help me."}
        )


def test_next_turn_receives_the_prior_conversation() -> None:
    model = FakeChatModel(["ADRIAN", "ETHAN"])
    workflow = build_workflow(model)
    first = workflow.invoke({"messages": [], "user_input": "I need a VFX workstation."})

    second = workflow.invoke(
        {"messages": first["messages"], "user_input": "I'm ready to purchase."}
    )

    assert second["agent"] == "ethan"
    second_agent_messages = model.calls[3]
    assert any(
        isinstance(message, HumanMessage) and message.content == "I need a VFX workstation."
        for message in second_agent_messages
    )
    assert any(
        isinstance(message, AIMessage) and message.content == "A prompt-based response."
        for message in second_agent_messages
    )
    assert len(second["messages"]) == 4


def test_adrian_receives_in_memory_workstation_options_and_sample_prices() -> None:
    model = FakeChatModel(["ADRIAN"])

    build_workflow(model).invoke(
        {"messages": [], "user_input": "Recommend a workstation."}
    )

    adrian_prompt = model.calls[1][0].content
    assert len(ADRIAN_WORKSTATION_OPTIONS) == 3
    for option in ADRIAN_WORKSTATION_OPTIONS:
        assert option["name"] in adrian_prompt
        assert f"${option['sample_price_usd']:,} USD sample estimate" in adrian_prompt
        assert option["configuration"] in adrian_prompt
    assert "not verified RAXEN products" in adrian_prompt


def test_workstation_options_are_not_injected_into_other_specialist_prompts() -> None:
    model = FakeChatModel(["CONCIERGE"])

    build_workflow(model).invoke(
        {"messages": [], "user_input": "I need a monitor."}
    )

    assert "RAXEN Studio Base" not in model.calls[1][0].content


@pytest.mark.parametrize(
    ("initial_agent", "next_agent"),
    [
        ("concierge", "adrian"),
        ("concierge", "ethan"),
        ("adrian", "concierge"),
        ("adrian", "ethan"),
        ("ethan", "concierge"),
        ("ethan", "adrian"),
    ],
)
def test_every_agent_can_hand_off_to_another_specialist(
    initial_agent: str, next_agent: str
) -> None:
    model = FakeChatModel(
        [initial_agent.upper()],
        [
            json.dumps(
                {
                    "action": "handoff",
                    "agent": next_agent,
                    "reason": "This specialist is better suited to the request.",
                }
            ),
            json.dumps(
                {"action": "respond", "response": "Handled by the receiving specialist."}
            ),
        ],
    )
    workflow = build_workflow(model)

    result = workflow.invoke({"messages": [], "user_input": "Please help me with this."})

    assert result["agent"] == next_agent
    assert result["response"] == "Handled by the receiving specialist."
    assert len(result["messages"]) == 2
    stream_model = FakeChatModel(
        [initial_agent.upper()],
        [
            json.dumps(
                {
                    "action": "handoff",
                    "agent": next_agent,
                    "reason": "This specialist is better suited to the request.",
                }
            ),
            json.dumps(
                {"action": "respond", "response": "Handled by the receiving specialist."}
            ),
        ],
    )
    events = list(
        build_workflow(stream_model).stream(
            {"messages": [], "user_input": "Please help me with this."},
            stream_mode="custom",
        )
    )
    assert events[0] == {"type": "handoff", "agent": next_agent}
    assert events[-1] == {"type": "done", "agent": next_agent}
    assert HANDOFF_PROTOCOL in model.calls[1][0].content
    assert model.calls[2][0].content.startswith(
        {
            "adrian": ADRIAN_PROMPT,
            "concierge": CONCIERGE_PROMPT,
            "ethan": ETHAN_PROMPT,
        }[next_agent]
    )
    assert model.calls[2][1].content.startswith("Internal handoff context:")


def test_invalid_agent_handoff_is_reported() -> None:
    model = FakeChatModel(
        ["CONCIERGE"],
        [json.dumps({"action": "handoff", "agent": "concierge", "reason": "Retry."})],
    )

    with pytest.raises(ValueError, match="cannot hand off to itself"):
        build_workflow(model).invoke({"messages": [], "user_input": "Help me."})


def test_specialist_plain_text_is_returned_as_the_reply() -> None:
    model = FakeChatModel(["ADRIAN"], ["Hello! How can I help with your build?"])

    result = build_workflow(model).invoke(
        {"messages": [], "user_input": "I need help with a build."}
    )

    assert result["response"] == "Hello! How can I help with your build?"
    assert result["agent"] == "adrian"


def test_unexpected_action_with_response_falls_back_to_answer() -> None:
    model = FakeChatModel(
        ["ADRIAN"],
        [
            json.dumps(
                {
                    "action": "proposed",
                    "response": "This is a proposed configuration.",
                }
            )
        ],
    )

    result = build_workflow(model).invoke(
        {"messages": [], "user_input": "Can you propose a PC build?"}
    )

    assert result["response"] == "This is a proposed configuration."
    assert result["agent"] == "adrian"


def test_specialist_response_streams_decoded_text_deltas() -> None:
    response = json.dumps(
        {"action": "respond", "response": 'A build for $2,000 is "possible".'}
    )
    model = FakeChatModel(["ADRIAN"], [response])

    events = list(
        build_workflow(model).stream(
            {"messages": [], "user_input": "Can you help with a build?"},
            stream_mode="custom",
        )
    )

    deltas = [event["content"] for event in events if event["type"] == "delta"]
    assert "".join(deltas) == 'A build for $2,000 is "possible".'
    assert events[-1] == {"type": "done", "agent": "adrian"}


def test_specialist_accepts_json_in_a_markdown_code_fence() -> None:
    model = FakeChatModel(
        ["CONCIERGE"],
        [
            '```json\n{"action":"handoff","agent":"adrian",'
            '"reason":"The customer needs a PC build."}\n```',
            json.dumps({"action": "respond", "response": "Let's plan the build."}),
        ],
    )

    result = build_workflow(model).invoke(
        {"messages": [], "user_input": "I need a custom build."}
    )

    assert result["agent"] == "adrian"
    assert result["response"] == "Let's plan the build."


def test_empty_specialist_response_has_a_clear_error() -> None:
    model = FakeChatModel(["ADRIAN"], [""])

    with pytest.raises(ValueError, match="empty response"):
        build_workflow(model).invoke(
            {"messages": [], "user_input": "I need help with a build."}
        )


def test_handoff_loop_is_bounded() -> None:
    targets = ["adrian", "ethan", "adrian", "ethan"]
    model = FakeChatModel(
        ["CONCIERGE"],
        [
            json.dumps(
                {
                    "action": "handoff",
                    "agent": target,
                    "reason": "Another specialist should answer.",
                }
            )
            for target in targets
        ],
    )

    with pytest.raises(ValueError, match="maximum handoff limit"):
        build_workflow(model).invoke({"messages": [], "user_input": "Help me."})

    assert HANDOFF_LIMIT == 3


def test_role_prompts_include_core_boundaries() -> None:
    assert "label" in ADRIAN_PROMPT
    assert "proposed" in ADRIAN_PROMPT
    assert "Do not invent" in CONCIERGE_PROMPT
    assert "Never ask for payment credentials" in ETHAN_PROMPT
    assert "do not promise a callback time" in ETHAN_PROMPT