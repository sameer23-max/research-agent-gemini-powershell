"""
Free Research Agent
-------------------
A simple agent loop: the model decides when to search the web, read pages,
and save a report. Works with Google Gemini (free tier) or a local Ollama model.

Usage:
    set GEMINI_API_KEY=your_key_here        (Windows CMD)
    $env:GEMINI_API_KEY="your_key_here"     (PowerShell)
    python research_agent.py

To use a local model instead:
    set PROVIDER=ollama
    set MODEL=qwen2.5:3b
    python research_agent.py
"""

import json
import os
import re
from datetime import date

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import OpenAI

# ----------------------------------------------------------------------------
# 1. Model setup (both providers speak the OpenAI-compatible format)
# ----------------------------------------------------------------------------
PROVIDER = os.getenv("PROVIDER", "gemini").lower()

if PROVIDER == "ollama":
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL = os.getenv("MODEL", "qwen2.5:3b")
else:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set the GEMINI_API_KEY environment variable first "
                         "(free key: https://aistudio.google.com).")
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    # If this model name stops working, check Google AI Studio for a current free model.
    MODEL = os.getenv("MODEL", "gemini-2.5-flash")

MAX_STEPS = 12  # safety limit so the agent can never loop forever

SYSTEM_PROMPT = f"""You are a careful research assistant. Today's date is {date.today()}.
Process:
1. Use web_search to find sources. Try 2-3 different queries for broad topics.
2. Use read_page on the most promising results to get details (don't rely on snippets alone).
3. Write a clear, well-organized answer that cites sources with their URLs.
4. If sources disagree or information is missing, say so. Never invent facts or URLs.
5. Only call save_report if the user asks you to save the results.
"""

# ----------------------------------------------------------------------------
# 2. Tools (plain Python functions)
# ----------------------------------------------------------------------------
def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
    return "\n".join(lines)


def read_page(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (research-agent)"})
        resp.raise_for_status()
    except Exception as e:
        return f"Could not fetch page: {e}"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
    # Keep it short: small models and free tiers have limited context
    return text[:6000] if text else "Page had no readable text."


def save_report(filename: str, content: str) -> str:
    os.makedirs("reports", exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", filename)[:60] or "report"
    path = os.path.join("reports", f"{safe}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved report to {path}"


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "read_page": read_page,
    "save_report": save_report,
}

# ----------------------------------------------------------------------------
# 3. Tool descriptions the model sees (JSON schema)
# ----------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns titles, URLs and short snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Fetch a web page and return its main text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL starting with http(s)://"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "Save a markdown report to the reports/ folder. Only when the user asks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Short name, no extension"},
                    "content": {"type": "string", "description": "Full report in markdown"},
                },
                "required": ["filename", "content"],
            },
        },
    },
]

# ----------------------------------------------------------------------------
# 4. The agent loop
# ----------------------------------------------------------------------------
def run_agent(messages: list) -> str:
    """Runs until the model gives a final answer or MAX_STEPS is reached."""
    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # No tool calls means the model is done
        if not msg.tool_calls:
            return msg.content or "(no answer)"

        # Otherwise run each requested tool and feed the result back
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                func = TOOL_FUNCTIONS[name]
                print(f"  [tool] {name}({args})")
                result = func(**args)
            except KeyError:
                result = f"Unknown tool: {name}"
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": str(result)}
            )

    return "Stopped: reached the maximum number of steps without a final answer."


def main():
    print(f"Research agent ready (provider={PROVIDER}, model={MODEL})")
    print("Ask a research question. Type 'quit' to exit, 'new' to reset memory.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break
        if question.lower() == "new":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Memory cleared.\n")
            continue

        messages.append({"role": "user", "content": question})
        try:
            answer = run_agent(messages)
        except Exception as e:
            answer = f"Error talking to the model: {e}"
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
