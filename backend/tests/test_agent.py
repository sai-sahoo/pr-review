"""The ReAct loop, driven turn by turn by a scripted fake model."""

from langchain_core.messages import ToolMessage

from app.graph.agent import MAX_TOOL_ROUNDS, run_specialist
from fakes import scripted, tool_call
from sample_pr import PATH, PAYMENTS_PY, finding, make_pr


def test_agent_reads_a_file_then_submits(fake_llm, fake_github):
    pr = make_pr()
    fake_github(pr, files={PATH: PAYMENTS_PY})
    read = tool_call("read_file", path=PATH, start_line=1, end_line=20)
    llm = fake_llm(scripted(
        read,
        tool_call("Review", findings=[finding().model_dump()]),
    ))

    findings, log = run_specialist(pr, "correctness", "<diff>")

    assert findings == [finding()]
    assert log == [
        "correctness: read_file(path='app/payments.py', start_line=1, end_line=20)",
        "correctness: submitted 1 findings after 1 tool rounds",
    ]
    # OBSERVE: the second call saw the file, linked to the request by its id.
    observation = llm.calls[1].messages[-1]
    assert isinstance(observation, ToolMessage)
    assert observation.tool_call_id == read.tool_calls[0]["id"]
    assert "  10 def refund(order: Order, amount: int) -> int:" in observation.content


def test_tool_errors_become_observations_not_crashes(fake_llm, fake_github):
    pr = make_pr()
    fake_github(pr, files={})  # every path is a 404
    llm = fake_llm(scripted(
        tool_call("read_file", path="missing.py"),
        tool_call("delete_repo"),  # a tool that doesn't exist
        tool_call("Review", findings=[]),
    ))

    findings, _ = run_specialist(pr, "security", "<diff>")

    assert findings == []
    # Each error went back to the model as text, so it could change course.
    assert llm.calls[1].messages[-1].content == "error: missing.py not found at abc123"
    assert llm.calls[2].messages[-1].content == "error: unknown tool 'delete_repo'"


def test_last_round_forces_a_submit(fake_llm, fake_github):
    pr = make_pr()
    fake_github(pr, files={PATH: PAYMENTS_PY})
    # A model that never wants to stop reading...
    reads = [tool_call("read_file", path=PATH) for _ in range(MAX_TOOL_ROUNDS)]
    llm = fake_llm(scripted(*reads, tool_call("Review", findings=[])))

    run_specialist(pr, "maintainability", "<diff>")

    # ...gets MAX_TOOL_ROUNDS free turns, then one turn where Review is forced.
    assert [c.tool_choice for c in llm.calls] == ["any"] * MAX_TOOL_ROUNDS + ["Review"]
