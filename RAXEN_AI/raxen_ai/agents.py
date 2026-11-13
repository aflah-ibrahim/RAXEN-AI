import json
import re
from typing import Literal

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.config import get_stream_writer
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict


ROUTER_PROMPT = """You route RAXEN ecommerce conversations to one specialist.
Return exactly one JSON object with an "agent" field set to "adrian",
"concierge", or "ethan". Do not include other fields or prose.

Choose Adrian for any full PC, gaming PC, workstation, custom build, PC build
planning, or system-level technical advice. This takes priority even when the
customer is just starting to describe what they need.
Choose Concierge for accessories and other IT products, including monitors,
peripherals, printers, storage, networking, cables, adapters, and hubs.
Choose Ethan when the customer has chosen a product and is ready to purchase,
asks about purchase preparation, or needs checkout guidance.

Choose based on the actual request and whole conversation, including follow-up
answers and changes in intent. Route full-PC and custom-build discovery to
Adrian, accessory discovery to Concierge, and purchase or checkout guidance to
Ethan. Do not keep a customer with a specialist who cannot best answer the latest
question. If intent is unclear, choose CONCIERGE so the assistant can clarify.
Example: {"agent":"adrian"}"""

HANDOFF_LIMIT = 3
AgentName = Literal["adrian", "concierge", "ethan"]

HANDOFF_PROTOCOL = """You may either answer the customer or hand the conversation to
the specialist who is better suited to resolve it. Use the full conversation,
not just the latest message. Hand off when the request belongs to another role,
when the conversation changes topic, or when another specialist must handle a
part of the customer's unresolved request. Do not ask the customer to repeat
information already in the conversation.

Return exactly one JSON object and no other text. The "action" value must be
exactly "respond" or "handoff"; words such as "proposed" belong only in the
customer-facing response text.
To answer: {"action":"respond","response":"Your customer-facing reply"}
To hand off: {"action":"handoff","agent":"adrian|concierge|ethan","reason":"Brief internal context for the next specialist"}

On handoff, do not answer the customer yet. The next specialist will respond
using the complete conversation and your handoff context. Never hand off to
yourself. After three handoffs, you must answer rather than hand off again."""

ADRIAN_PROMPT = """You are Adrian Marcus, RAXEN's Workstation Architect. You advise on
gaming PCs, professional workstations, and custom configurations. Be technically
confident, consultative, and approachable; adapt detail to the customer's
knowledge. Understand workload, applications, budget, performance goals, and
practical constraints before recommending when those details matter. Compare
suitable prebuilts and custom options fairly, explain relevant trade-offs, and
use only facts stated in the conversation. Never invent products or specifications.
Do not claim a custom build is compatible unless checks are confirmed; label an
unchecked configuration as proposed. Ask a concise question when key information
is missing. Product selection remains the customer's decision. If the request is
primarily for an accessory or purchase/checkout support, hand it to the relevant
specialist instead of trying to handle it yourself."""

CONCIERGE_PROMPT = """You are RAXEN's friendly, efficient Product Concierge for accessories
and IT products. Identify the category and purpose, then ask only the questions
needed to distinguish options. Compare features and compatibility only using
facts stated in the conversation. Do not invent products, specifications, prices,
stock, promotions, or compatibility. Route full-PC, workstation, custom-build,
and system-level compatibility questions to Adrian; do not keep asking discovery
questions about those requests. Suggest complementary items only when they have a
clear practical benefit, without pressure. Hand off purchase or checkout requests
to Ethan."""

ETHAN_PROMPT = """You are Ethan Cole, RAXEN's Purchase & Customer Experience Advisor.
Be reassuring, transparent, and commercially aware. Confirm the chosen product,
variant, and quantity when relevant. Use only purchase details explicitly
provided in the conversation; never invent prices, stock, discounts, gifts, or
policy terms. Explain that the customer completes payment through RAXEN's secure
website checkout. Never ask for payment credentials or claim a cart, order, or
callback action succeeded. Offer relevant optional add-ons without pressure and
respect a customer's decision to decline. Offer human follow-up when requested or
when an issue cannot be resolved, but do not promise a callback time or claim a
request was submitted without confirmation. If product or technical selection
remains unresolved, hand it to Adrian or the Product Concierge rather than
guessing."""

AGENT_PROMPTS = {
    "adrian": ADRIAN_PROMPT,
    "concierge": CONCIERGE_PROMPT,
    "ethan": ETHAN_PROMPT,
}


class WorkflowState(TypedDict):
    messages: list[BaseMessage]
    user_input: str
    agent: NotRequired[AgentName]
    response: NotRequired[str]
    handoffs: NotRequired[int]
    handoff_reason: NotRequired[str]


class WorkstationOption(TypedDict):
    name: str
    use_case: str
    configuration: str
    sample_price_usd: int


ADRIAN_WORKSTATION_OPTIONS: tuple[WorkstationOption, ...] = (
    {
        "name": "RAXEN Studio Base",
        "use_case": "general productivity, light content creation, and entry-level GPU work",
        "configuration": "Ryzen 7 7700, GeForce RTX 4060 8 GB, 32 GB DDR5, 1 TB NVMe SSD",
        "sample_price_usd": 1499,
    },
    {
        "name": "RAXEN Creator",
        "use_case": "video editing, 3D work, and demanding multitasking",
        "configuration": "Ryzen 9 7900, GeForce RTX 4070 SUPER 12 GB, 64 GB DDR5, 2 TB NVMe SSD",
        "sample_price_usd": 2499,
    },
    {
        "name": "RAXEN Compute",
        "use_case": "heavy rendering, simulation, and GPU-intensive workloads",
        "configuration": "Ryzen 9 7950X, GeForce RTX 4080 SUPER 16 GB, 96 GB DDR5, 2 TB NVMe SSD",
        "sample_price_usd": 3499,
    },
)


class RespondDecision(TypedDict):
    action: Literal["respond"]
    response: str


class HandoffDecision(TypedDict):
    action: Literal["handoff"]
    agent: AgentName
    reason: str


AgentDecision = RespondDecision | HandoffDecision


def _content(response: object) -> str:
    value = getattr(response, "content", response)
    return value if isinstance(value, str) else str(value)


def _parse_agent_name(value: object) -> AgentName:
    if value == "adrian":
        return "adrian"
    if value == "concierge":
        return "concierge"
    if value == "ethan":
        return "ethan"
    raise ValueError(f"Unknown specialist: {value!r}")


def _format_workstation_options() -> str:
    options = "\n".join(
        f"- {option['name']} — ${option['sample_price_usd']:,} USD sample estimate; "
        f"best for {option['use_case']}. Configuration: {option['configuration']}."
        for option in ADRIAN_WORKSTATION_OPTIONS
    )
    return (
        "In-memory example workstation options (illustrative configurations and "
        "sample USD estimates only; not verified RAXEN products, current pricing, "
        "or inventory):\n"
        f"{options}\n"
        "You may use these examples when relevant, clearly identify their prices "
        "as sample estimates, and ask about workload and budget before recommending. "
        "Do not imply availability or that these are confirmed RAXEN listings."
    )


def _parse_route(response: object) -> AgentName:
    content = _content(response).strip()
    if not content:
        raise ValueError("Router returned an empty specialist label.")

    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()

    try:
        route = json.loads(content)
    except json.JSONDecodeError as error:
        label = content.strip("`*.,:;").lower()
        if label in AGENT_PROMPTS:
            return _parse_agent_name(label)
        raise ValueError("Router returned an invalid specialist decision.") from error

    if not isinstance(route, dict):
        raise ValueError("Router decision must be a JSON object.")
    return _parse_agent_name(route.get("agent"))


def _partial_response(content: str) -> str | None:
    match = re.search(r'"response"\s*:\s*"', content)
    if not match:
        return None

    start = match.end()
    escaped = False
    safe_end = 0
    index = start
    while index < len(content):
        char = content[index]
        if escaped:
            if char == "u":
                if index + 4 >= len(content):
                    break
                hex_digits = content[index + 1 : index + 5]
                if not all(digit in "0123456789abcdefABCDEF" for digit in hex_digits):
                    break
                index += 4
            escaped = False
            safe_end = index - start + 1
        elif char == "\\":
            escaped = True
        elif char == '"':
            safe_end = index - start
            break
        else:
            safe_end = index - start + 1
        index += 1

    try:
        decoded = json.loads(f'"{content[start : start + safe_end]}"')
    except json.JSONDecodeError:
        return None

    if decoded and 0xD800 <= ord(decoded[-1]) <= 0xDBFF:
        decoded = decoded[:-1]
    return decoded


def _parse_agent_decision(response: object, current_agent: AgentName) -> AgentDecision:
    content = _content(response).strip()
    if not content:
        raise ValueError("Specialist returned an empty response.")

    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            content = "\n".join(lines[1:-1]).strip()

    try:
        decision = json.loads(content)
    except json.JSONDecodeError as error:
        if content.startswith(("{", "[")):
            raise ValueError("Specialist returned invalid handoff JSON.") from error
        return {"action": "respond", "response": content}

    if not isinstance(decision, dict):
        raise ValueError("Specialist decision must be a JSON object.")

    action = decision.get("action")
    if action == "respond":
        reply = decision.get("response")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("Specialist response must include non-empty response text.")
        return {"action": action, "response": reply}

    if action == "handoff":
        target = _parse_agent_name(decision.get("agent"))
        reason = decision.get("reason")
        if target == current_agent:
            raise ValueError(f"{current_agent} cannot hand off to itself.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Specialist handoff must include a reason.")
        return {"action": action, "agent": target, "reason": reason}

    reply = decision.get("response")
    if isinstance(reply, str) and reply.strip():
        return {"action": "respond", "response": reply}

    raise ValueError(f"Specialist returned an unknown action: {action!r}")


def build_workflow(model: BaseChatModel):
    """Build the in-memory, prompt-only RAXEN agent graph."""

    router_model = model.bind(response_format={"type": "json_object"})

    def route(
        state: WorkflowState,
    ) -> Command[AgentName]:
        prompt_messages = [
            SystemMessage(content=ROUTER_PROMPT),
            *state["messages"],
            HumanMessage(content=state["user_input"]),
        ]
        agent = _parse_route(router_model.invoke(prompt_messages))
        return Command(update={"agent": agent, "handoffs": 0}, goto=agent)

    def make_agent_node(agent: AgentName):
        agent_model = model.bind(response_format={"type": "json_object"})

        def respond(
            state: WorkflowState,
        ) -> Command[AgentName | Literal["__end__"]]:
            system_prompt = f"{AGENT_PROMPTS[agent]}\n\n{HANDOFF_PROTOCOL}"
            if agent == "adrian":
                system_prompt += f"\n\n{_format_workstation_options()}"
            if state.get("handoffs", 0) >= HANDOFF_LIMIT:
                system_prompt += "\nYou have reached the handoff limit. Respond now."
            prompt_messages = [SystemMessage(content=system_prompt)]
            if state.get("handoff_reason"):
                prompt_messages.append(
                    SystemMessage(
                        content=f"Internal handoff context: {state['handoff_reason']}"
                    )
                )
            prompt_messages.extend([
                *state["messages"],
                HumanMessage(content=state["user_input"]),
            ])
            writer = get_stream_writer()
            response_content = ""
            streamed_response = ""
            for chunk in agent_model.stream(prompt_messages):
                response_content += _content(chunk)
                partial = _partial_response(response_content)
                if partial is not None and partial.startswith(streamed_response):
                    delta = partial[len(streamed_response) :]
                    if delta:
                        writer({"type": "delta", "content": delta})
                    streamed_response = partial

            decision = _parse_agent_decision(
                AIMessageChunk(content=response_content), agent
            )
            if decision["action"] == "handoff":
                if state.get("handoffs", 0) >= HANDOFF_LIMIT:
                    raise ValueError("Specialist exceeded the maximum handoff limit.")
                target = decision["agent"]
                writer({"type": "handoff", "agent": target})
                return Command(
                    update={
                        "agent": target,
                        "handoffs": state.get("handoffs", 0) + 1,
                        "handoff_reason": decision["reason"],
                    },
                    goto=target,
                )
            reply = decision["response"]
            if not streamed_response:
                writer({"type": "delta", "content": reply})
            writer({"type": "done", "agent": agent})
            return Command(
                update={
                    "agent": agent,
                    "response": reply,
                    "handoff_reason": "",
                    "messages": [
                        *state["messages"],
                        HumanMessage(content=state["user_input"]),
                        AIMessage(content=reply),
                    ],
                },
                goto=END,
            )

        return respond

    graph = StateGraph(WorkflowState)
    destinations = (*AGENT_PROMPTS, END)
    graph.add_node("router", route, destinations=tuple(AGENT_PROMPTS))
    for agent in AGENT_PROMPTS:
        graph.add_node(agent, make_agent_node(agent), destinations=destinations)

    graph.add_edge(START, "router")

    return graph.compile()