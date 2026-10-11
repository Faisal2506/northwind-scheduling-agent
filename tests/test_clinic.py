from agent.identity import parse_identity
from agent.runtime import build_graph, build_prompt, run_turn
from clinic.fixtures import make_session
from clinic.tools import (
    book_appointment,
    cancel_appointment,
    list_availability,
    lookup_patient,
    note_confirmation,
    reschedule_appointment,
)


def test_ambiguous_lookup_has_no_phi():
    session = make_session()
    result = lookup_patient(session, first_name="Maya", last_name="", dob="", patient_id="")
    assert result["error"] == "AMBIGUOUS"
    blob = str(result)
    assert "Chen" not in blob
    assert "Patel" not in blob
    assert "1990-04-12" not in blob


def test_book_requires_offer_identity_and_confirmation():
    session = make_session()
    lookup_patient(session, first_name="Maya", last_name="Chen", dob="1990-04-12", patient_id="")
    session.resolved_patient_id = "p_maya_chen"
    denied = book_appointment(session, patient_id="p_maya_chen", slot_id="slot_mon_10")
    assert denied["error"] == "SLOT_NOT_OFFERED"

    list_availability(session, visit_type="general", date_from="2026-10-13", date_to="2026-10-17")
    session.last_human_text = "not yet"
    unconfirmed = book_appointment(session, patient_id="p_maya_chen", slot_id="slot_mon_10")
    assert unconfirmed["error"] == "NOT_CONFIRMED"

    session.last_human_text = "Yes, I confirm. Please book slot_mon_10."
    note_confirmation(session, action="book", target_id="slot_mon_10")
    booked = book_appointment(session, patient_id="p_maya_chen", slot_id="slot_mon_10")
    assert booked["ok"]
    again = book_appointment(session, patient_id="p_maya_chen", slot_id="slot_mon_10")
    assert again["ok"] and again["idempotent"]
    assert again["confirmation_id"] == booked["confirmation_id"]


def test_race_does_not_create_an_appointment():
    session = make_session(race_first_book=True)
    lookup_patient(session, first_name="Owen", last_name="Brooks", dob="1978-11-20", patient_id="")
    session.resolved_patient_id = "p_owen"
    list_availability(session, visit_type="general", date_from="2026-10-13", date_to="2026-10-13")
    session.last_human_text = "Yes, I confirm. Please book it."
    note_confirmation(session, action="book", target_id="slot_mon_10")
    result = book_appointment(session, patient_id="p_owen", slot_id="slot_mon_10")
    assert result["error"] == "SLOT_TAKEN"
    assert all(apt["patient_id"] != "p_owen" or apt["slot_id"] != "slot_mon_10" for apt in session.clinic.snapshot()["appointments"])


def test_cancel_inside_window_does_not_drop_the_appointment():
    session = make_session()
    lookup_patient(session, first_name="Priya", last_name="Shah", dob="1992-07-07", patient_id="")
    session.resolved_patient_id = "p_priya"
    session.last_human_text = "Yes, I confirm. Please cancel it."
    note_confirmation(session, action="cancel", target_id="apt_priya")
    result = cancel_appointment(session, appointment_id="apt_priya")
    assert result["error"] == "OUTSIDE_POLICY"
    assert any(apt["appointment_id"] == "apt_priya" for apt in session.clinic.snapshot()["appointments"])


def test_reschedule_moves_owen():
    session = make_session()
    lookup_patient(session, first_name="Owen", last_name="Brooks", dob="1978-11-20", patient_id="")
    session.resolved_patient_id = "p_owen"
    list_availability(session, visit_type="general", date_from="2026-10-13", date_to="2026-10-17")
    session.last_human_text = "Yes, I confirm. Please book slot_mon_14."
    note_confirmation(session, action="reschedule", target_id="slot_mon_14")
    result = reschedule_appointment(session, appointment_id="apt_owen", slot_id="slot_mon_14")
    assert result["ok"]
    moved = next(apt for apt in session.clinic.snapshot()["appointments"] if apt["appointment_id"] == "apt_owen")
    assert moved["slot_id"] == "slot_mon_14"


def test_bare_name_is_looked_up_instead_of_asking_again(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "fake")
    assert parse_identity("faisal Shah")["last_name"] == "Shah"
    assert parse_identity("book appointment") is None
    assert parse_identity("general") is None
    parsed = parse_identity("Maya Patel, 1985-01-03")
    assert parsed["last_name"] == "Patel"
    assert parsed["dob"] == "1985-01-03"
    session = make_session()
    reply = run_turn(session, "faisal Shah", build_prompt([]), build_graph())
    assert "Nobody in this clinic" in reply
    assert session.tool_trace[-1]["tool"] == "lookup_patient"
    assert session.tool_trace[-1]["result"]["error"] == "NOT_FOUND"
    again = run_turn(session, "Maya Chen", build_prompt([]), build_graph())
    assert "date of birth" in again.lower()
    booked_path = run_turn(session, "1990-04-12", build_prompt([]), build_graph())
    assert "full name" not in booked_path.lower()


def test_plain_text_drops_html():
    from agent.runtime import _plain

    raw = "</td></tr><tr><td>I have one follow-up appointment available.</td></tr>"
    assert _plain(raw) == "I have one follow-up appointment available."


def test_reschedule_keeps_the_goal_and_lists_a_real_slot(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "fake")
    session = make_session()
    graph = build_graph()
    prompt = build_prompt([])
    ask = run_turn(session, "reschedule", prompt, graph)
    assert "date of birth" in ask.lower()
    listed = run_turn(session, "Owen Brooks, 1978-11-20", prompt, graph)
    assert "slot_tue_11" in listed
    assert "2026-10-20T10:00" in listed
    assert "book a new visit" not in listed.lower()
    assert "appointment id" not in listed.lower()
    moved = run_turn(session, "yes", prompt, graph)
    assert "2026-10-14T11:00" in moved
    assert session.clinic.appointments["apt_owen"].slot_id == "slot_tue_11"


def test_cancel_does_not_ask_for_an_id_the_clinic_already_has(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "fake")
    session = make_session()
    graph = build_graph()
    prompt = build_prompt([])
    run_turn(session, "cancel", prompt, graph)
    found = run_turn(session, "Owen Brooks, 1978-11-20", prompt, graph)
    assert "say yes" in found.lower()
    assert "do not need the appointment id" in found.lower()
    missing = run_turn(session, "i dont have appointment id", prompt, graph)
    assert "say yes" in missing.lower()
    done = run_turn(session, "yes", prompt, graph)
    assert "cancelled" in done.lower()
    assert session.clinic.appointments["apt_owen"].status == "cancelled"


def test_hio_is_not_a_patient_lookup():
    assert parse_identity("hio") is None
    assert parse_identity("monday") is None


def test_monday_asks_which_time_then_yes_books(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "fake")
    session = make_session()
    lookup_patient(session, first_name="Owen", last_name="Brooks", dob="1978-11-20", patient_id="")
    session.resolved_patient_id = "p_owen"
    session.intent = "book"
    list_availability(session, visit_type="general", date_from="2026-10-13", date_to="2026-10-17")
    graph = build_graph()
    prompt = build_prompt([])
    monday = run_turn(session, "monday", prompt, graph)
    assert "2026-10-13" in monday
    assert "slot_mon_10" in monday
    assert "no new message" not in monday.lower()
    assert session.clinic.slots["slot_mon_10"].status == "open"
    narrowed = run_turn(session, "monday slot", prompt, graph)
    assert "slot_tue_09" in narrowed
    picked = run_turn(session, "10:00", prompt, graph)
    assert "slot_mon_10" in picked
    assert "Say yes" in picked
    booked = run_turn(session, "yes", prompt, graph)
    assert "CNF-" in booked
    assert session.clinic.slots["slot_mon_10"].status == "booked"
    assert "no new message" not in booked.lower()


def test_reinforcement_asks_for_identity_and_copies_the_confirmation(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "fake")
    from agent.runtime import _quote_confirmation

    session = make_session()
    prompt = build_prompt(
        [{"id": "from-run", "rule": "Ask before booking.", "repairs": ["identity_before_booking", "quote_confirmation"]}]
    )
    asked = run_turn(session, "I need a general visit between 2026-10-13 and 2026-10-17.", prompt, build_graph())
    assert "full name" in asked.lower()
    assert session.tool_trace == []
    spoken = _quote_confirmation(
        "You are booked with Dr. Lee on 2026-10-13 at 10:00 AM.",
        [{"tool": "book_appointment", "result": {"ok": True, "confirmation_id": "CNF-101"}}],
    )
    assert "CNF-101" in spoken


def test_failed_lookup_is_not_followed_by_a_slot_list():
    from agent.runtime import fallback_for_tools

    text = fallback_for_tools(
        [
            {
                "tool": "lookup_patient",
                "args": {"first_name": "Maya", "last_name": "Chen", "dob": "2002-06-25"},
                "result": {"ok": False, "error": "NOT_FOUND"},
            },
            {
                "tool": "list_availability",
                "result": {
                    "ok": True,
                    "slots": [{"slot_id": "slot_mon_10", "provider_name": "Dr. Lee", "start": "2026-10-13T10:00"}],
                },
            },
        ],
        patient_already_known=False,
        user_text="I need a general visit between 2026-10-13 and 2026-10-17.",
    )
    assert "full name" in text.lower()
    assert "slot_mon_10" not in text


def test_empty_model_text_uses_the_latest_tool_not_an_old_lookup():
    from agent.runtime import fallback_for_tools

    slots = fallback_for_tools(
        [
            {
                "tool": "lookup_patient",
                "result": {
                    "ok": True,
                    "patient": {
                        "first_name": "Maya",
                        "last_name": "Chen",
                        "dob": "1990-04-12",
                        "appointments": [],
                    },
                },
            },
            {
                "tool": "list_availability",
                "result": {
                    "ok": True,
                    "slots": [
                        {
                            "slot_id": "slot_tue_11",
                            "provider_name": "Dr. Okonkwo",
                            "start": "2026-10-14T11:00",
                            "visit_type": "follow_up",
                        }
                    ],
                },
            },
        ],
        patient_already_known=True,
    )
    assert "Dr. Okonkwo" in slots
    assert "2026-10-14T11:00" in slots
    assert "Maya Chen" not in slots


def test_availability_has_no_patient_names():
    session = make_session()
    result = list_availability(session, visit_type="general", date_from="2026-10-13", date_to="2026-10-17")
    blob = str(result)
    assert "Maya" not in blob
    assert "Priya" not in blob
    assert result["slots"]
