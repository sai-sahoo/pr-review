"""A hand-written ReAct loop: think -> act -> observe, until the model submits.

LangChain ships a prebuilt version (langchain.agents.create_agent). We write
it ourselves once so nothing about "agents" is magic.
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.graph.prompts import SPECIALIST_PROMPTS
from app.graph.tools import make_read_file
from app.llm import get_model
from app.schemas import Finding, PullRequest, Review, Specialist

MAX_TOOL_ROUNDS = 4  # after this many rounds of reading, the model must submit


def run_specialist(pr: PullRequest, focus: Specialist, diff_text: str) -> tuple[list[Finding], list[str]]:
    """Run one specialist agent. Returns its findings and a log of its tool calls."""
    read_file = make_read_file(pr)
    tools = {read_file.name: read_file}

    model = get_model("smart")
    # A Pydantic class passed to bind_tools becomes a tool too: calling
    # "Review" with valid arguments *is* submitting the final answer.
    # tool_choice="any": the model must call some tool every turn
    # (read more, or submit), so it can't drift into plain chat.
    explore = model.bind_tools([read_file, Review], tool_choice="any")
    finish = model.bind_tools([Review], tool_choice="Review")  # out of rounds: submit now

    messages = [SystemMessage(SPECIALIST_PROMPTS[focus]), HumanMessage(diff_text)]
    log: list[str] = []

    for round_no in range(MAX_TOOL_ROUNDS + 1):
        llm = explore if round_no < MAX_TOOL_ROUNDS else finish
        ai = llm.invoke(messages)  # THINK: the model decides what to do next
        messages.append(ai)  # its tool request becomes part of the conversation

        for call in ai.tool_calls:
            if call["name"] == "Review":
                findings = Review.model_validate(call["args"]).findings
                log.append(f"{focus}: submitted {len(findings)} findings after {round_no} tool rounds")
                return findings, log

        for call in ai.tool_calls:  # ACT: run every tool the model asked for
            tool = tools.get(call["name"])
            result = tool.invoke(call["args"]) if tool else f"error: unknown tool {call['name']!r}"
            args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
            log.append(f"{focus}: {call['name']}({args})")
            # OBSERVE: the result goes back in, linked to the request by its id
            messages.append(ToolMessage(result, tool_call_id=call["id"]))

    # Unreachable in practice: the final round forces a Review call.
    log.append(f"{focus}: gave up without submitting")
    return [], log
