# Hotfix notes

## gpt-4o-mini → gpt-6-luna (November 2026)

OpenAI shuts down `gpt-4o`, `gpt-4o-mini` and `gpt-4.1(-mini)` on **Jan 14, 2027**. The official
replacement is `gpt-6-luna`. Every file in this repo used `gpt-4o-mini`, so every file changed.
Dependency pins are unchanged (`openai` 1.63.2 and `langchain-openai` 0.3.6 already accept `reasoning_effort`).

### Why `reasoning_effort="none"`

`gpt-6-luna` is a reasoning model. On Chat Completions it rejects `temperature` and function tools
unless reasoning is switched off with `reasoning_effort="none"`. With it, the code behaves like before.

### Video vs code now

| File | In the video | In the code now | Bootcamp S22 lecture |
|---|---|---|---|
| `simple_agent.py` | `llm_name = "gpt-4o-mini"`; `chat.completions.create(model=..., temperature=0.0, ...)` | `"gpt-6-luna"`; adds `reasoning_effort="none"` to both `create(...)` calls | Build an agent from scratch (Thought/Action/PAUSE loop) |
| `simple_agent.py` (prompt) | ReAct prompt ends at the Earth + Mars example | **One new last line:** "Now it's your turn. Begin every reply with Thought:, then an Action line and PAUSE, until you have every Observation you need." Without it, gpt-6-luna answers from memory and never runs an action | same |
| `simple_agent.py` (regex) | `action_re = re.compile(r"^Action: (\w+): (.*)$")` | `r"^Action: (\w+): (.*?)\s*$"`. gpt-6-luna sometimes adds trailing spaces after `Action: planet_mass: Earth`, which crashed `planet_mass` with `KeyError: 'Earth  '` | same |
| `simple_agent_lngraph.py` | `ChatOpenAI(api_key=openai_key, model=llm_name)` with `gpt-4o-mini` | `gpt-6-luna` + `reasoning_effort="none"` | LangGraph basics (StateGraph chatbot) |
| `simple_agent_lngraph_tools.py` | same | same | Tools + memory (Tavily, ToolNode, MemorySaver) |
| `simple_agent_hum_in_loop.py` | same | same | Human-in-the-loop (L280, `interrupt_before`) |
| `finance_agent.py` | same | same | Financial report agent (Streamlit, `with_structured_output`) |

Graph structure, prompts (except the one line above), tools, checkpointers and the Streamlit UI are unchanged.

### Still true from the videos (not a regression)

- `simple_agent_hum_in_loop.py` has no edge from `tools` back to `bot`, so after you approve, the run
  ends on the Tavily result. It did that with `gpt-4o-mini` too.
- On a very simple question ("What is the mass of Earth?") the from-scratch agent may still answer
  directly instead of calling `planet_mass`. Multi-planet questions use the actions.

### Live test (gpt-6-luna, pinned requirements, Python 3.12)

All 5 files pass `py_compile` and ran end to end against the live API: the ReAct loop ran
planet_mass/calculate to a final Answer, the LangGraph chatbot answered, the memory demo recalled
"Bond / 100 years", the HITL demo paused before `tools` and resumed, and the finance graph produced a report.
