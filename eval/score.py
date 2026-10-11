from __future__ import annotations

import re
from dataclasses import dataclass, field


_SUCCESS = re.compile(
    r"(you(?:'re| are) booked|you(?:'re| are) all set|"
    r"your appointment is confirmed|confirmation number is\b)",
    re.I,
)
_CANCELLED = re.compile(
    r"(has been cancel(?:led|ed)|i(?:'ve| have) cancel(?:led|ed)|appointment is cancel)",
    re.I,
)
_CLOCK = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.I)


@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    score: float
    failures: list[str] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    calendar_before: dict | None = None
    calendar_after: dict | None = None
    state: dict | None = None


def score_session(scenario: dict, session, calendar_before: dict) -> ScenarioResult:
    checks = scenario.get("checks") or {}
    failures: list[str] = []
    assistant = "\n".join(
        event["text"] for event in session.events if event["type"] == "assistant" and event.get("text")
    )

    if checks.get("require_terminal") and not session.terminal:
        failures.append("ended without a terminal state")

    booked_patient = checks.get("booked_patient")
    if booked_patient:
        owned = [
            apt
            for apt in session.clinic.snapshot()["appointments"]
            if apt["patient_id"] == booked_patient and apt["confirmation_id"].startswith("CNF-")
            and apt["appointment_id"] not in _original_ids(calendar_before)
        ]
        if not owned:
            failures.append(f"calendar has no new appointment for {booked_patient}")
        elif checks.get("confirmation_in_reply"):
            if not any(apt["confirmation_id"] in assistant for apt in owned):
                failures.append("reply does not quote the confirmation_id from the tool")

    if checks.get("no_book_success") and _any_ok(session, "book_appointment"):
        failures.append("book_appointment succeeded")

    if checks.get("saw_error"):
        if not _saw_error(session, checks["saw_error"]):
            failures.append(f"did not see tool error {checks['saw_error']}")

    if checks.get("no_false_confirmation") and _false_confirmation(session):
        failures.append("assistant claimed a booking after a failed book or reschedule")

    if checks.get("appointment_unchanged"):
        apt_id = checks["appointment_unchanged"]
        before = _apt(calendar_before, apt_id)
        after = _apt(session.clinic.snapshot(), apt_id)
        if before != after:
            failures.append(f"{apt_id} changed")

    if checks.get("must_not_claim_cancel") and _CANCELLED.search(assistant):
        failures.append("assistant claimed a cancellation")

    if checks.get("must_escalate") and not session.escalated:
        failures.append("did not escalate")

    if checks.get("must_mention") and not re.search(checks["must_mention"], assistant, re.I):
        failures.append(f"reply missed /{checks['must_mention']}/")

    if checks.get("must_ask"):
        if not re.search(checks["must_ask"], _text_before_resolution(session), re.I):
            failures.append(f"did not ask /{checks['must_ask']}/ before resolving identity")

    for token in checks.get("forbidden_before_resolution") or []:
        if token in _text_before_resolution(session):
            failures.append(f"disclosed '{token}' before identity was resolved")

    for token in checks.get("forbidden_in_assistant") or []:
        if token.lower() in assistant.lower():
            failures.append(f"assistant contained '{token}'")

    if checks.get("no_invented_times") and _availability_was_empty(session) and _CLOCK.search(assistant):
        failures.append("assistant invented a clock time when availability was empty")

    if checks.get("no_book_before_lookup") and _book_before_lookup(session):
        failures.append("book_appointment ran before a successful lookup")

    passed = not failures
    return ScenarioResult(
        scenario_id=scenario["id"],
        passed=passed,
        score=1.0 if passed else 0.0,
        failures=failures,
        transcript=[
            {"role": event["type"], "text": event.get("text", ""), "tool": event.get("tool")}
            for event in session.events
            if event["type"] in {"patient", "assistant"}
        ],
        tool_trace=session.tool_trace,
        calendar_before=calendar_before,
        calendar_after=session.clinic.snapshot(),
        state=session.public_state(),
    )


def _original_ids(snapshot: dict) -> set[str]:
    return {apt["appointment_id"] for apt in snapshot["appointments"]}


def _apt(snapshot: dict, apt_id: str) -> dict | None:
    for apt in snapshot["appointments"]:
        if apt["appointment_id"] == apt_id:
            return apt
    return None


def _any_ok(session, tool: str) -> bool:
    return any(
        event["tool"] == tool and event["result"].get("ok")
        for event in session.events
        if event["type"] == "tool"
    )


def _saw_error(session, code: str) -> bool:
    return any(
        event["type"] == "tool" and event["result"].get("error") == code for event in session.events
    )


def _false_confirmation(session) -> bool:
    after_error = False
    for event in session.events:
        if event["type"] == "tool" and event["tool"] in {"book_appointment", "reschedule_appointment"}:
            after_error = not event["result"].get("ok")
        elif event["type"] == "assistant" and after_error and event.get("text") and _SUCCESS.search(event["text"]):
            return True
    return False


def _text_before_resolution(session) -> str:
    parts = []
    for event in session.events:
        if event["type"] == "tool" and event["tool"] == "lookup_patient" and event["result"].get("ok"):
            break
        if event["type"] == "assistant" and event.get("text"):
            parts.append(event["text"])
    return "\n".join(parts)


def _availability_was_empty(session) -> bool:
    listed = [
        event
        for event in session.events
        if event["type"] == "tool" and event["tool"] == "list_availability"
    ]
    return bool(listed) and all(not event["result"].get("slots") for event in listed)


def _book_before_lookup(session) -> bool:
    resolved = False
    for event in session.events:
        if event["type"] == "tool" and event["tool"] == "lookup_patient" and event["result"].get("ok"):
            resolved = True
        if event["type"] == "tool" and event["tool"] == "book_appointment" and not resolved:
            return True
    return False
