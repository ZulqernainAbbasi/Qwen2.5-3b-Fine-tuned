from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import config
from agent import graph
from backend import db


class ScriptedChat(BaseChatModel):
    """Fake tool-calling chat model: returns scripted AIMessages and records its inputs."""
    script: list = []
    seen: list = []
    loop_tool: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append([m.content for m in messages])
        if self.loop_tool:
            msg = AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": "alto"}, "id": f"c{len(self.seen)}"}])
        else:
            msg = self.script.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "b.db")
    monkeypatch.setattr("agent.tools.retrieve", lambda *a, **k: [])
    llm = ScriptedChat(script=[], seen=[])
    monkeypatch.setattr(graph, "get_llm", lambda: llm)
    graph.reset_agent()
    yield llm
    graph.reset_agent()


def test_tool_called_and_trace_recorded(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": "Suzuki Alto price"}, "id": "c1"}]),
        AIMessage(content="The Suzuki Alto starts at Rs 2,994,861."),
    ]
    out = graph.run_agent("s1", "Alto price?")
    assert out["reply"] == "The Suzuki Alto starts at Rs 2,994,861."
    assert len(out["steps"]) == 1
    step = out["steps"][0]
    assert step["tool"] == "search_cars" and step["args"] == {"query": "Suzuki Alto price"}
    assert "2,994,861" in step["result"]


def test_memory_keeps_session_context(fake):
    fake.script = [AIMessage(content="Hello Ali!"), AIMessage(content="Your name is Ali.")]
    graph.run_agent("s2", "My name is Ali")
    graph.run_agent("s2", "What is my name?")
    second_call_inputs = " ".join(fake.seen[-1])
    assert "My name is Ali" in second_call_inputs


def test_sessions_are_isolated(fake):
    fake.script = [AIMessage(content="ok"), AIMessage(content="ok")]
    graph.run_agent("a", "secret word is mango")
    graph.run_agent("b", "hello")
    assert "mango" not in " ".join(fake.seen[-1])


def test_step_limit_escalates(fake):
    fake.loop_tool = True
    out = graph.run_agent("s3", "loop forever")
    assert out["reply"].startswith("Ticket #")
    assert out["steps"][-1]["tool"] == "escalate_to_human"
    assert sum(s["tool"] == "search_cars" for s in out["steps"]) <= graph.MAX_TOOL_STEPS
    assert len(db.list_tickets(config.DB_PATH)) == 1


def test_system_prompt_sent(fake):
    fake.script = [AIMessage(content="hi")]
    graph.run_agent("s4", "hi")
    assert "PakCar Assistant" in fake.seen[0][0]


def test_refusal_without_tools_retries_with_search(fake):
    fake.script = [
        AIMessage(content=config.REFUSAL),  # lazy first answer, no tool used
        AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": "hybrid suv under 1 crore"}, "id": "r1"}]),
        AIMessage(content="Try the Toyota Corolla Cross 1.8 HEV at Rs 9,299,000."),
    ]
    out = graph.run_agent("s5", "Best hybrid SUV under 1 crore?")
    assert [s["tool"] for s in out["steps"]] == ["search_cars"]
    assert out["reply"].startswith("Try the Toyota Corolla Cross")


def test_refusal_after_tool_use_is_kept(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": "tesla"}, "id": "t1"}]),
        AIMessage(content=config.REFUSAL),
    ]
    out = graph.run_agent("s6", "Tesla price?")
    assert out["reply"] == config.REFUSAL and len(out["steps"]) == 1


def test_junk_tool_args_use_user_message(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": {"type": "string"}}, "id": "j1"}]),
        AIMessage(content="The Honda City starts at Rs 4,737,000."),
    ]
    out = graph.run_agent("s7", "Honda City price?")
    assert "4,737,000" in out["steps"][0]["result"]
