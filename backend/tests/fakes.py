"""Test doubles: a chat model that answers from a script instead of an API.

Our agent code only ever does three things with a model: bind_tools(),
with_structured_output() (which LangChain builds on bind_tools), and invoke().
A fake that supports those can stand in for OpenAI everywhere, for free,
with the same answer every run.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# A responder decides the fake's reply, given the conversation so far.
Responder = Callable[[list[BaseMessage]], AIMessage]


@dataclass
class Call:
    """One request the fake received, kept so tests can inspect it."""

    messages: list[BaseMessage]
    tool_choice: str | None  # what our code forced, e.g. "any" or "Review"


class FakeChatModel(BaseChatModel):
    respond: Responder
    calls: list[Call] = []
    tool_choice: str | None = None

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # A real model sends the tool schemas to the API. The fake ignores them,
        # because the script decides which tool gets "called". We only remember
        # tool_choice so tests can check what the code forced.
        # model_copy is shallow: the copy shares the same `calls` list.
        return self.model_copy(update={"tool_choice": tool_choice})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # Copy the list: the agent keeps appending to its own after this call.
        self.calls.append(Call(list(messages), self.tool_choice))
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages))])


_ids = itertools.count(1)


def tool_call(name: str, **args) -> AIMessage:
    """An AI turn that calls one tool: what a real reply looks like once parsed."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def scripted(*replies: AIMessage) -> Responder:
    """Answer with these replies in order, one per call."""
    queue = list(replies)

    def respond(messages: list[BaseMessage]) -> AIMessage:
        assert queue, "the model was called more times than the test scripted"
        return queue.pop(0)

    return respond


def by_system_prompt(routes: dict[str, Responder]) -> Responder:
    """Pick the responder by the system prompt, i.e. by which agent is asking.

    Parallel specialists call the model in any order, so one shared queue
    would be flaky. Routing by prompt gives each agent its own script.
    """

    def respond(messages: list[BaseMessage]) -> AIMessage:
        system = messages[0].content
        assert system in routes, f"unexpected agent called the model: {system[:60]!r}"
        return routes[system](messages)

    return respond
