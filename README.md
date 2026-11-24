# AI Agents: Course Code

Source code for the AI Agents section of the AI Engineer Bootcamp.

## Setup update (November 2026)

Use **Python 3.12** and install the pinned dependencies in a fresh virtual environment:

```bash
pip install -r requirements.txt
```

Dependencies are pinned to the versions used in the lectures, because newer LangChain releases moved
several imports used in this code.

The human-in-the-loop example (`simple_agent_hum_in_loop.py`) was also updated: newer versions of
`langgraph-checkpoint-sqlite` no longer accept `SqliteSaver.from_conn_string()` directly as a checkpointer.

## Model update: gpt-6-luna

OpenAI shuts down `gpt-4o-mini` on Jan 14, 2027. The code now uses `gpt-6-luna` with
`reasoning_effort="none"` (needed for `temperature` and tools). If your screen shows `gpt-4o-mini`
in a lecture, use the repo code. See `HOTFIX-NOTES.md` for every line that differs from the videos.
