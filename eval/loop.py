from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

from agent.runtime import ROOT, build_graph
from eval.run_suite import run_suite, write_results

MAX_ROUNDS = 3


def _failed(results: list) -> list:
    return [result for result in results if not result.passed]


def _assistant_lines(result) -> list[str]:
    return [
        turn["text"]
        for turn in result.transcript
        if turn.get("role") == "assistant" and turn.get("text")
    ]


def _tool_errors(result) -> list[str]:
    errors = []
    for item in result.tool_trace:
        body = item.get("result") or {}
        if body.get("error"):
            errors.append(body["error"])
    return errors


def _claimed_success_after_tool_error(result) -> bool:
    text = " ".join(_assistant_lines(result)).lower()
    failures = " ".join(result.failures).lower()
    claimed = "you're booked" in text or "you are booked" in text or "claimed a booking" in failures
    return claimed and bool(_tool_errors(result) or "claimed a booking" in failures)


def _instructions_for(result) -> list[str]:
    """Turn one failed trace into rules. The scenario id is not chosen in advance."""
    lines = [
        (
            f"Scenario {result.scenario_id} failed: {'; '.join(result.failures) or 'unspecified'}. "
            f"Tool errors: {_tool_errors(result) or 'none'}. "
            f"Last assistant reply: {(_assistant_lines(result) or [''])[-1]}"
        )
    ]
    for item in reversed(result.tool_trace):
        body = item.get("result") or {}
        if item.get("tool") == "list_availability" and body.get("ok") and not body.get("slots"):
            lines.append(
                "list_availability returned no slots. Say nothing is open in that window, "
                "offer to widen the search, and call finish with reason no_availability. "
                "Do not call list_availability again unless the patient gives a new date range or visit type."
            )
            break
    failure_text = " ".join(result.failures).lower()
    if "no new appointment" in failure_text or "before a successful lookup" in failure_text:
        lines.append(
            "Do not call note_confirmation or book_appointment until lookup_patient has returned ok "
            "and list_availability has returned a slot in this conversation. "
            "If lookup_patient returns NEED_DOB or NOT_FOUND, ask for the missing name or date of birth and wait. "
            "An appointment already on the patient's record is not a new opening. "
            "Call list_availability for the visit type and dates they asked for, then book a slot from that list."
        )
    if "terminal" in failure_text:
        lines.append(
            "When the clinic tool has already answered the request, call finish or escalate. "
            "Do not keep the conversation open."
        )
    if "confirmation_id" in failure_text or "does not quote" in failure_text:
        lines.append(
            "When book_appointment or reschedule_appointment returns ok, "
            "the reply must include that result's confirmation_id."
        )
    if _claimed_success_after_tool_error(result):
        lines.append(
            "A booking tool returned an error and the reply still claimed success. "
            "Do not say the patient is booked, do not use the phrase You're booked, "
            "and do not invent a confirmation number. Say the visit was not booked and name the tool error."
        )
    return lines


def _repairs_for(result) -> list[str]:
    """Machine-applied parts of a reinforcement. Prompt text alone does not stick."""
    repairs = ["quote_confirmation"]
    failure_text = " ".join(result.failures).lower()
    empty_slots = any(
        item.get("tool") == "list_availability"
        and (item.get("result") or {}).get("ok")
        and not (item.get("result") or {}).get("slots")
        for item in result.tool_trace
    )
    if "terminal" in failure_text and empty_slots:
        repairs.append("finish_when_no_slots")
    if "no new appointment" in failure_text or "before a successful lookup" in failure_text:
        repairs.append("identity_before_booking")
        listed_slots = any(
            item.get("tool") == "list_availability" and (item.get("result") or {}).get("slots")
            for item in result.tool_trace
        )
        if not listed_slots:
            repairs.append("offer_open_slots")
    return repairs


def build_patch(results: list) -> dict:
    failed = _failed(results)
    if not failed:
        raise RuntimeError("no failed scenario to reinforce")
    rules = []
    repairs = []
    for result in failed:
        rules.extend(_instructions_for(result))
        for repair in _repairs_for(result):
            if repair not in repairs:
                repairs.append(repair)
    rules.append(
        "When book_appointment or reschedule_appointment returns ok, "
        "the reply must include that result's confirmation_id."
    )
    ids = [result.scenario_id for result in failed]
    patch = {
        "id": "from-run-" + "-".join(ids),
        "from_scenario": ",".join(ids),
        "diagnosis": "Generated from the scenarios that failed in this run.",
        "evidence": list(rules),
        "patch_type": "policy_rule",
        "rule": "This reinforcement was written from the eval run.\n" + "\n".join(rules),
        "repairs": repairs,
        "regression_watch": [result.scenario_id for result in results if result.passed],
    }
    if any(_claimed_success_after_tool_error(result) for result in failed):
        patch["strips_policy"] = "hide_booking_errors"
    return patch


def gate(before: list, after: list) -> dict:
    before_map = {result.scenario_id: result for result in before}
    after_map = {result.scenario_id: result for result in after}
    failed_before = [sid for sid, result in before_map.items() if not result.passed]
    still_failing = [sid for sid in failed_before if not after_map[sid].passed]
    regressed = sorted(
        sid for sid, result in before_map.items() if result.passed and not after_map[sid].passed
    )
    return {
        "accepted": bool(failed_before) and not still_failing and not regressed,
        "target_improved": bool(failed_before) and not still_failing,
        "regressed": regressed,
        "still_failing": still_failing,
        "before_mean": _mean(before),
        "after_mean": _mean(after),
    }


def _signature(results: list) -> tuple:
    return tuple(sorted((result.scenario_id, result.passed, tuple(result.failures)) for result in results))


def run_loop(graph=None) -> int:
    load_dotenv()
    graph = graph or build_graph()
    print(f"Model: {os.environ.get('AGENT_MODEL', 'gemini')}")
    print("Round 1: no reinforcements\n")
    origin = run_suite([], graph)
    _print_table(origin)
    write_results(ROOT / "runs" / "before", origin)
    if not _failed(origin):
        print("\nEvery scenario passed. No reinforcement was applied.")
        return 1

    patches = []
    current = origin
    for round_index in range(1, MAX_ROUNDS + 1):
        failed = _failed(current)
        if not failed:
            break
        patch = build_patch(current)
        patch["id"] = f"{patch['id']}-r{round_index}"
        patches.append(patch)
        print(f"\nReinforcement {round_index} from {patch['from_scenario']}\n")
        print(json.dumps(patch, indent=2))
        print(f"\nRound {round_index + 1}: reinforcements applied\n")
        nxt = run_suite(patches, graph)
        _print_table(nxt)
        if _signature(nxt) == _signature(current):
            print("\nThis reinforcement did not change the scoreboard. Stopping.")
            current = nxt
            break
        if gate(origin, nxt)["regressed"]:
            print(f"\nRegression: {gate(origin, nxt)['regressed']}. Stopping.")
            current = nxt
            break
        current = nxt

    write_results(ROOT / "runs" / "after", current)
    decision = gate(origin, current)
    (ROOT / "runs" / "patch.json").write_text(json.dumps(patches, indent=2))
    print(
        f"\nMean {decision['before_mean']:.2f} -> {decision['after_mean']:.2f}. "
        f"Failures fixed: {decision['target_improved']}. "
        f"Still failing: {decision['still_failing'] or 'none'}. "
        f"Regressions: {decision['regressed'] or 'none'}."
    )
    if decision["accepted"]:
        folder = ROOT / "agent" / "reinforcements"
        folder.mkdir(parents=True, exist_ok=True)
        for patch in patches:
            path = folder / f"{patch['id']}.json"
            path.write_text(json.dumps(patch, indent=2))
            print(f"Gate accepted the reinforcement. Wrote {path.relative_to(ROOT)}")
        return 0
    print("Gate rejected the reinforcement. Nothing new was written.")
    return 1


def _mean(results: list) -> float:
    if not results:
        return 0.0
    return sum(result.score for result in results) / len(results)


def _print_table(results: list) -> None:
    print(f"{'scenario':<20} {'result':<6} failures")
    for result in results:
        status = "pass" if result.passed else "fail"
        detail = "; ".join(result.failures) if result.failures else "-"
        print(f"{result.scenario_id:<20} {status:<6} {detail}")


def main() -> None:
    os.environ.setdefault("AGENT_MODEL", "gemini")
    sys.exit(run_loop())


if __name__ == "__main__":
    main()
