"""LangGraph ReAct agent over the six car tools, with per-session memory and a step limit."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

import config
from agent.tools import ALL_TOOLS, escalate_to_human
from finetune.build_dataset import SYSTEM_PROMPT

MAX_TOOL_STEPS = 6
AGENT_PROMPT = SYSTEM_PROMPT + (
    "\n\nRules:\n"
    "- Use the tools for ANY price, spec, comparison, EMI or booking. Never guess numbers.\n"
    "- Before calling book_test_drive or book_service, restate all details and wait for the user to say yes.\n"
    "- If a tool returns the refusal sentence, repeat it and offer alternatives.\n"
    "- For complaints, discounts or anything you cannot do, call escalate_to_human.\n"
    "- Keep answers short and clear; end price answers with the dealer-confirmation note."
)

_AGENT = None
_MEMORY = MemorySaver()


def get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_URL, temperature=0.3)


def reset_agent() -> None:
    global _AGENT, _MEMORY
    _AGENT, _MEMORY = None, MemorySaver()


def _agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = create_react_agent(get_llm(), ALL_TOOLS, prompt=AGENT_PROMPT, checkpointer=_MEMORY)
    return _AGENT


def _steps(messages) -> list[dict]:
    calls, steps = {}, []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                calls[tc["id"]] = tc
        elif isinstance(m, ToolMessage):
            tc = calls.get(m.tool_call_id, {})
            steps.append({"tool": tc.get("name", m.name), "args": tc.get("args", {}), "result": str(m.content)})
    return steps


def run_agent(session_id: str, message: str) -> dict:
    """Run one user turn. Returns {"reply": str, "steps": [{"tool", "args", "result"}]}."""
    agent = _agent()
    cfg = {"configurable": {"thread_id": session_id}, "recursion_limit": 2 * MAX_TOOL_STEPS + 1}
    before = len(agent.get_state(cfg).values.get("messages", [])) if agent.get_state(cfg).values else 0
    try:
        state = agent.invoke({"messages": [HumanMessage(content=message)]}, cfg)
    except GraphRecursionError:
        new = agent.get_state(cfg).values.get("messages", [])[before:]
        steps = _steps(new)[:MAX_TOOL_STEPS]
        summary = f"Agent hit the {MAX_TOOL_STEPS}-step limit. User asked: {message[:300]}"
        reply = escalate_to_human.invoke({"summary": summary})
        steps.append({"tool": "escalate_to_human", "args": {"summary": summary}, "result": reply})
        return {"reply": reply, "steps": steps}
    new = state["messages"][before:]
    final = next((m for m in reversed(new) if isinstance(m, AIMessage) and not m.tool_calls), None)
    reply = (final.content if final else "").strip() or config.REFUSAL
    return {"reply": reply, "steps": _steps(new)}
