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
