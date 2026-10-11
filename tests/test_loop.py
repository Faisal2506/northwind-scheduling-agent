import json
import os

os.environ["AGENT_MODEL"] = "fake"

from agent.runtime import apply_reinforcements, build_graph, load_base_prompt
from eval.loop import build_patch, gate
from eval.run_suite import run_suite
from eval.score import ScenarioResult


def test_seed_policy_is_stripped_by_the_reinforcement():
    base = load_base_prompt()
    assert "hide booking errors" in base
    patched = apply_reinforcements(
        base,
        [{"id": "never-confirm-without-tool-success", "strips_policy": "hide_booking_errors", "rule": "Do not lie."}],
    )
    assert "hide booking errors" not in patched
    assert "Do not lie." in patched


def test_loop_closes_without_regressions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # run_loop writes under the package root, not the cwd. Call the pieces directly.
    graph = build_graph()
    before = run_suite([], graph)
    by_id = {result.scenario_id: result for result in before}
    assert by_id["happy_book"].passed, by_id["happy_book"].failures
    assert by_id["missing_identity"].passed, by_id["missing_identity"].failures
    assert by_id["ambiguous_name"].passed, by_id["ambiguous_name"].failures
    assert by_id["no_slots"].passed, by_id["no_slots"].failures
    assert by_id["cancel_policy"].passed, by_id["cancel_policy"].failures
    assert by_id["no_medical_advice"].passed, by_id["no_medical_advice"].failures
    assert by_id["social_engineer"].passed, by_id["social_engineer"].failures
    assert not by_id["slot_taken"].passed

    patch = build_patch(before)
    after = run_suite([patch], graph)
    decision = gate(before, after)
    assert decision["accepted"], decision
    assert decision["after_mean"] > decision["before_mean"]
    assert json.loads(json.dumps(patch))["patch_type"] == "policy_rule"
    assert patch["from_scenario"] == "slot_taken"
    assert patch["strips_policy"] == "hide_booking_errors"


def _result(scenario_id: str, passed: bool) -> ScenarioResult:
    return ScenarioResult(scenario_id=scenario_id, passed=passed, score=1.0 if passed else 0.0)


def test_gate_follows_whichever_scenario_failed():
    before = [_result("happy_book", True), _result("no_slots", False)]
    after = [_result("happy_book", True), _result("no_slots", True)]
    decision = gate(before, after)
    assert decision["accepted"], decision
    patch = build_patch(
        [
            ScenarioResult(
                scenario_id="no_slots",
                passed=False,
                score=0.0,
                failures=["ended without a terminal state"],
                tool_trace=[{"tool": "list_availability", "result": {"ok": True, "slots": []}}],
                transcript=[{"role": "assistant", "text": "Let me keep looking."}],
            )
        ]
    )
    assert patch["from_scenario"] == "no_slots"
    assert "strips_policy" not in patch
    assert "no slots" in patch["rule"]
    assert "finish_when_no_slots" in patch["repairs"]
    assert "quote_confirmation" in patch["repairs"]


def test_missing_booking_repair_lists_slots_then_books():
    patch = build_patch(
        [
            ScenarioResult(
                scenario_id="missing_identity",
                passed=False,
                score=0.0,
                failures=["calendar has no new appointment for p_maya_chen"],
                tool_trace=[
                    {"tool": "lookup_patient", "result": {"ok": True, "patient": {"id": "p_maya_chen"}}},
                    {"tool": "note_confirmation", "result": {"ok": False, "error": "SLOT_NOT_OFFERED"}},
                ],
                transcript=[{"role": "assistant", "text": "There is a general appointment at 2026-10-16T15:00."}],
            )
        ]
    )
    assert "offer_open_slots" in patch["repairs"]
    graph = build_graph()
    results = {item.scenario_id: item for item in run_suite([patch], graph)}
    assert results["missing_identity"].passed, results["missing_identity"].failures
    assert results["happy_book"].passed, results["happy_book"].failures
    assert results["no_slots"].passed, results["no_slots"].failures
    assert results["slot_taken"].passed, results["slot_taken"].failures
    assert results["cancel_policy"].passed, results["cancel_policy"].failures
    assert results["no_medical_advice"].passed, results["no_medical_advice"].failures
    assert results["social_engineer"].passed, results["social_engineer"].failures
    assert results["ambiguous_name"].passed, results["ambiguous_name"].failures
