# Northwind scheduling agent

A text scheduling agent for a fictional clinic, plus an evaluation loop that turns one failed run into a policy reinforcement and re-scores the same scenarios.

The clinic is an in-memory object. Gemini is the chat model. LangGraph routes the conversation. The calendar, not the model's wording, decides whether a booking happened. The eval loop stays a terminal command. A Streamlit page is only for watching a conversation and the tool log.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_free_key
```

The key stays in the environment. Do not commit it. The patients are fictional.

## Talk to the agent

```bash
python -m agent.cli
```

Or open the chat with the tool log beside it:

```bash
streamlit run app.py
```

The sidebar lists the four fictional patients. Anyone else is not in the clinic, so lookup should return not found instead of booking them.

## Run the eval loop

```bash
python -m eval.loop
```

That command scores every scenario, writes a reinforcement from whichever scenarios failed, and scores the same scenarios again. It repeats until they all pass, a round stops changing the scoreboard, or three rounds are done. It exits 0 only when every scenario that failed in the first round passes and no scenario that passed at the start fails later. Accepted reinforcements are written to `agent/reinforcements/`. Transcripts and scoreboards land in `runs/`.

`AGENT_MODEL=fake` runs the same loop with a deterministic stand-in, so the harness can be checked without the network. The submitted demo uses Gemini.

```bash
AGENT_MODEL=fake python -m eval.loop
```

Optional: `GEMINI_MODEL` overrides the default `gemini-2.5-flash-lite`.

## Tests

```bash
AGENT_MODEL=fake pytest
```

These cover the clinic tools and the offline improvement loop. They do not call Gemini.
