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
    assert out["reply"].startswith("The Suzuki Alto starts at Rs 2,994,861.")
    assert "3,326,446" in out["reply"]  # all variants appended
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


def test_summarised_reply_gets_full_variant_list(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": "Toyota Yaris price"}, "id": "y1"}]),
        AIMessage(content="Toyota Yaris: Rs 4,649,000 to Rs 6,389,000."),
    ]
    out = graph.run_agent("s8", "Toyota Yaris price")
    for price in ["4,649,000", "4,809,000", "4,829,000", "5,719,000", "6,389,000"]:
        assert price in out["reply"], price
    assert out["reply"].startswith("Toyota Yaris: Rs 4,649,000 to Rs 6,389,000.")


def test_complete_reply_not_duplicated(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": "Honda Civic"}, "id": "c9"}]),
        AIMessage(content="Civic: Standard Rs 8,499,000, Oriel Rs 8,834,000, RS Rs 10,100,000."),
    ]
    out = graph.run_agent("s9", "Honda Civic price")
    assert out["reply"].count("8,499,000") == 1


def test_unsourced_part_price_gets_honest_reply(fake, monkeypatch):
    monkeypatch.setattr("agent.tools.retrieve", lambda q, **k: [
        {"text": "# Hyundai Tucson\nPrice Rs 12,202,000", "source": "s", "brand": "-", "model": "-", "score": 0.7}])
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "parts_and_service", "args": {"query": "Tucson timing belt price"}, "id": "p1"}]),
        AIMessage(content="The Hyundai Tucson costs Rs 12,202,000."),
    ]
    out = graph.run_agent("s10", "Tucson timing belt price")
    assert out["reply"].startswith(config.REFUSAL)
    assert "12,202,000" not in out["reply"] and "dealer" in out["reply"].lower()


def test_invented_numbers_are_replaced_with_tool_data(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": "Toyota Yaris"}, "id": "g1"}]),
        AIMessage(content="Yaris from Rs 4,649,000. EMI only Rs 2,707/month with Rs 4,649,00 down."),
    ]
    out = graph.run_agent("s11", "Toyota Yaris price")
    assert "2,707" not in out["reply"] and "4,649,00 " not in out["reply"]
    assert "5,719,000" in out["reply"]


def test_grounded_numbers_kept(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "emi_calculator", "args": {"price": "50 lakh"}, "id": "e1"}]),
        AIMessage(content="Your monthly instalment is Rs 92,729 on a loan of Rs 3,500,000."),
    ]
    out = graph.run_agent("s12", "EMI for 50 lakh car")
    assert out["reply"] == "Your monthly instalment is Rs 92,729 on a loan of Rs 3,500,000."


def test_partial_reply_appends_only_missing_variants(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {"query": "Toyota Yaris"}, "id": "m1"}]),
        AIMessage(content="Yaris: GLi MT Rs 4,649,000, GLi CVT Rs 4,809,000, ATIV MT Rs 4,829,000."),
    ]
    out = graph.run_agent("s13", "Toyota Yaris price")
    assert out["reply"].count("4,649,000") == 1
    assert "5,719,000" in out["reply"] and "6,389,000" in out["reply"]


def test_compare_table_shown_when_model_summarises(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "compare_models", "args": {"model_a": "corolla", "model_b": "civic"}, "id": "k1"}]),
        AIMessage(content="The Corolla is cheaper; the Civic is more luxurious."),
    ]
    out = graph.run_agent("s14", "Corolla vs Civic")
    assert "| Price |" in out["reply"] and "8,499,000" in out["reply"]


def test_bank_list_shown_when_model_summarises(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "bank_financing", "args": {"car": "Suzuki Alto", "years": 5}, "id": "b1"}]),
        AIMessage(content="Meezan and Alfalah both finance the Alto."),
    ]
    out = graph.run_agent("s15", "Alto bank loan 5 years")
    assert "HBL" in out["reply"] and "per month" in out["reply"]


def test_bank_reply_is_exact_tool_quote(fake):
    fake.script = [
        AIMessage(content="", tool_calls=[{"name": "bank_financing", "args": {"car": "Toyota Corolla", "years": 5}, "id": "b2"}]),
        AIMessage(content="- Bank AL Habib: Rs 103,117 to Rs 104,585 per month"),  # mislabelled numbers
    ]
    out = graph.run_agent("s16", "Corolla car loan 5 saal ke liye")
    assert "3 years" in out["reply"] and "Rs 3,000,000" in out["reply"]
    assert out["reply"].startswith("Financing Toyota Corolla")
