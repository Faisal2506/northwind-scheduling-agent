from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent.runtime import ROOT, build_graph, build_prompt, run_turn
from clinic.fixtures import make_session
from eval.patient import ScriptedPatient
from eval.score import ScenarioResult, score_session

SCENARIO_ORDER = [
    "happy_book",
    "missing_identity",
    "ambiguous_name",
    "no_slots",
    "slot_taken",
    "cancel_policy",
    "no_medical_advice",
    "social_engineer",
]


def load_scenarios() -> list[dict]:
    folder = ROOT / "eval" / "scenarios"
    by_id = {}
    for path in folder.glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        by_id[data["id"]] = data
    missing = [sid for sid in SCENARIO_ORDER if sid not in by_id]
    if missing:
        raise RuntimeError(f"missing scenarios: {missing}")
    return [by_id[sid] for sid in SCENARIO_ORDER]


def run_scenario(scenario: dict, prompt: str, graph) -> ScenarioResult:
    session = make_session(race_first_book=bool(scenario.get("race_first_book")))
    calendar_before = session.clinic.snapshot()
    patient = ScriptedPatient(scenario)
    user_text = patient.opening()
    for _ in range(int(scenario.get("max_turns", 8))):
        run_turn(session, user_text, prompt, graph)
        if session.terminal:
            break
        nxt = patient.respond(_last_assistant(session), session)
        if not nxt:
            if session.phase not in {"done", "escalated"}:
                session.phase = "done"
            session.terminal = True
            break
        user_text = nxt
    return score_session(scenario, session, calendar_before)


def run_suite(patches: list[dict], graph) -> list[ScenarioResult]:
    prompt = build_prompt(patches)
    return [run_scenario(scenario, prompt, graph) for scenario in load_scenarios()]


def make_graph():
    return build_graph()


def _last_assistant(session) -> str:
    for event in reversed(session.events):
        if event["type"] == "assistant":
            return event.get("text") or ""
    return ""


def result_to_dict(result: ScenarioResult) -> dict:
    return {
        "scenario_id": result.scenario_id,
        "passed": result.passed,
        "score": result.score,
        "failures": result.failures,
        "transcript": result.transcript,
        "tool_trace": result.tool_trace,
        "calendar_before": result.calendar_before,
        "calendar_after": result.calendar_after,
        "state": result.state,
    }


def write_results(folder: Path, results: list[ScenarioResult]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    payload = [result_to_dict(result) for result in results]
    (folder / "scoreboard.json").write_text(json.dumps(payload, indent=2))
    for result in results:
        (folder / f"{result.scenario_id}.json").write_text(
            json.dumps(result_to_dict(result), indent=2)
        )
