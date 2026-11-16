"""LangGraph ReAct agent over the six car tools, with per-session memory and a step limit."""
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

import config
from agent.tools import ALL_TOOLS, CURRENT_MESSAGE, escalate_to_human
from finetune.build_dataset import SYSTEM_PROMPT

MAX_TOOL_STEPS = 6
AGENT_PROMPT = SYSTEM_PROMPT + (
    "\n\nRules:\n"
    "- Use the tools for ANY price, spec, comparison, EMI or booking. Never guess numbers.\n"
    "- Before calling book_test_drive or book_service, restate all details and wait for the user to say yes.\n"
    "- If a tool returns the refusal sentence, repeat it and offer alternatives.\n"
    "- For complaints, discounts or anything you cannot do, call escalate_to_human.\n"
    "- For spare parts, part prices, service costs or maintenance, use parts_and_service.\n"
    "- When a tool lists several variants, show EVERY variant with its price; never collapse them into a range.\n"
    "- Keep answers clear; end price answers with the dealer-confirmation note."
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


RETRY_NUDGE = ("(System note: you answered without checking the database. Call search_cars "
               "(or another tool) for this question before answering.) ")


def run_agent(session_id: str, message: str) -> dict:
    """Run one user turn. Returns {"reply": str, "steps": [{"tool", "args", "result"}]}.

    If the model refuses without having used any tool, retry once with a nudge to search first,
    so a lazy refusal never hides data we actually have."""
    token = CURRENT_MESSAGE.set(message)
    try:
        return _run_with_retry(session_id, message)
    finally:
        CURRENT_MESSAGE.reset(token)


_PRICE = re.compile(r"Rs ([\d,]{7,})")
_AMOUNT = re.compile(r"Rs\.?\s?(\d[\d,]*)")


def _guard_numbers(out: dict) -> dict:
    """Any rupee amount in the reply must appear in a tool result; otherwise answer from tool data only."""
    if not out["steps"]:
        return out
    tool_text = " ".join(s["result"] for s in out["steps"])
    allowed = {n.replace(",", "") for n in re.findall(r"\d[\d,]*", tool_text)}
    invented = [a for a in _AMOUNT.findall(out["reply"]) if a.rstrip(",").replace(",", "") not in allowed]
    if not invented:
        return out
    lines = []
    for s in out["steps"]:
        for line in s["result"].splitlines():
            if line.strip() and not line.startswith("Note:"):
                lines.append(line.split(". Source:")[0].split(" (Source:")[0])
    return {**out, "reply": "Here is what I found:\n" + "\n".join(lines)}


def _ensure_complete(out: dict) -> dict:
    """Make sure the user sees what the tools found when the model summarised it away."""
    reply = out["reply"]
    for step in out["steps"]:
        if step["tool"] == "search_cars":
            lines = [l for l in step["result"].splitlines() if l.startswith("- ") and _PRICE.search(l)]
            missing = [l.split(". Source:")[0] for l in lines if _PRICE.search(l).group(1) not in reply]
            if missing and len(missing) == len(lines):
                reply += "\n\nAll matching variants:\n" + "\n".join(missing)
            elif missing:
                reply += "\n\nAlso matching:\n" + "\n".join(missing)
        elif step["tool"] == "compare_models" and step["result"].startswith("|"):
            table = [l for l in step["result"].splitlines() if l.startswith("|")]
            prices = _PRICE.findall(step["result"])
            if prices and not all(p in reply for p in prices):
                reply += "\n\n" + "\n".join(table)
    return {**out, "reply": reply}


UNSOURCED_PARTS_REPLY = (
    f"{config.REFUSAL} I don't have a sourced part price for that car. "
    "General advice: use genuine parts from an authorised dealer (copies sold far below market price are rarely "
    "genuine), and follow the service intervals in your owner's manual. I have sourced part prices for the "
    "Suzuki Cultus/Alto/Swift, Toyota Corolla and Honda; for this car, please ask an authorised dealer."
)


def _guard_parts(out: dict) -> dict:
    """If the parts tool said no part prices are sourced, never let the model improvise numbers."""
    if any(s["tool"] in ("parts_and_service", "search_cars") and "no verified part prices" in s["result"].lower()
           for s in out["steps"]):
        return {**out, "reply": UNSOURCED_PARTS_REPLY}
    return out


def _run_with_retry(session_id: str, message: str) -> dict:
    out = _guard_parts(_ensure_complete(_guard_numbers(_run_once(session_id, message))))
    if not out["steps"] and out["reply"].strip().startswith(config.REFUSAL):
        retry = _ensure_complete(_guard_numbers(_run_once(session_id, RETRY_NUDGE + message)))
        if retry["steps"]:
            return retry
    return out


def _run_once(session_id: str, message: str) -> dict:
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
