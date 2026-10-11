"""Tool logic. These functions mutate the clinic. The model never does."""

from __future__ import annotations

import re

from clinic.model import CANCEL_WINDOW, NOW, Appointment, Session

_AFFIRM = re.compile(r"\b(yes|i confirm|please book|please cancel|go ahead|book this)\b", re.I)


def _norm(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def lookup_patient(
    session: Session,
    *,
    first_name: str = "",
    last_name: str = "",
    dob: str = "",
    patient_id: str = "",
) -> dict:
    patients = list(session.clinic.patients.values())
    if patient_id.strip():
        patient = session.clinic.patients.get(patient_id.strip())
        if patient is None:
            return {"ok": False, "error": "NOT_FOUND"}
        return {"ok": True, "patient": _patient_payload(session, patient)}

    first = _norm(first_name)
    last = _norm(last_name)
    dob = dob.strip()
    if not first and not last:
        return {"ok": False, "error": "NEED_NAME"}

    matches = []
    for patient in patients:
        if first and _norm(patient.first_name) != first:
            continue
        if last and _norm(patient.last_name) != last:
            continue
        if dob and patient.dob != dob:
            continue
        matches.append(patient)

    if dob:
        if len(matches) == 1:
            return {"ok": True, "patient": _patient_payload(session, matches[0])}
        if len(matches) == 0:
            return {"ok": False, "error": "NOT_FOUND"}
        return {"ok": False, "error": "AMBIGUOUS", "count": len(matches)}

    if len(matches) > 1:
        return {"ok": False, "error": "AMBIGUOUS", "count": len(matches)}
    if len(matches) == 1:
        return {"ok": False, "error": "NEED_DOB"}
    return {"ok": False, "error": "NOT_FOUND"}


def _patient_payload(session: Session, patient) -> dict:
    appointments = []
    for apt in session.clinic.appointments.values():
        if apt.patient_id != patient.id or apt.status != "booked":
            continue
        slot = session.clinic.slots[apt.slot_id]
        appointments.append(
            {
                "appointment_id": apt.id,
                "start": slot.start.isoformat(timespec="minutes"),
                "provider_name": slot.provider_name,
                "visit_type": apt.visit_type,
                "confirmation_id": apt.confirmation_id,
            }
        )
    return {
        "id": patient.id,
        "first_name": patient.first_name,
        "last_name": patient.last_name,
        "dob": patient.dob,
        "appointments": appointments,
    }


def list_availability(
    session: Session,
    *,
    visit_type: str,
    date_from: str,
    date_to: str,
) -> dict:
    visit_type = visit_type.strip()
    slots = []
    for slot in session.clinic.slots.values():
        if slot.status != "open" or slot.visit_type != visit_type:
            continue
        day = slot.start.date().isoformat()
        if date_from and day < date_from:
            continue
        if date_to and day > date_to:
            continue
        slots.append(slot)
    slots.sort(key=lambda slot: slot.start)
    payload = [
        {
            "slot_id": slot.id,
            "provider_name": slot.provider_name,
            "start": slot.start.isoformat(timespec="minutes"),
            "visit_type": slot.visit_type,
        }
        for slot in slots
    ]
    for item in payload:
        if item["slot_id"] not in session.offered_slot_ids:
            session.offered_slot_ids.append(item["slot_id"])
    return {"ok": True, "slots": payload}


def note_confirmation(session: Session, *, action: str, target_id: str) -> dict:
    action = action.strip()
    target_id = target_id.strip()
    if action not in {"book", "cancel", "reschedule"}:
        return {"ok": False, "error": "BAD_ACTION"}
    if not _AFFIRM.search(session.last_human_text or ""):
        return {"ok": False, "error": "NOT_CONFIRMED"}
    if action in {"book", "reschedule"}:
        if target_id not in session.offered_slot_ids:
            return {"ok": False, "error": "SLOT_NOT_OFFERED"}
        slot = session.clinic.slots.get(target_id)
        if slot is None or slot.status != "open":
            return {"ok": False, "error": "SLOT_TAKEN"}
    if action == "cancel":
        apt = session.clinic.appointments.get(target_id)
        if apt is None or apt.status != "booked":
            return {"ok": False, "error": "NOT_FOUND"}
    session.confirmed[action] = target_id
    return {"ok": True, "action": action, "target_id": target_id}


def book_appointment(session: Session, *, patient_id: str, slot_id: str) -> dict:
    patient_id = patient_id.strip()
    slot_id = slot_id.strip()
    patient = session.clinic.patients.get(patient_id)
    if patient is None or session.resolved_patient_id != patient_id:
        return {"ok": False, "error": "PATIENT_NOT_FOUND"}
    if slot_id not in session.offered_slot_ids:
        return {"ok": False, "error": "SLOT_NOT_OFFERED"}
    if session.confirmed.get("book") != slot_id and session.confirmed.get("reschedule") != slot_id:
        return {"ok": False, "error": "NOT_CONFIRMED"}
    slot = session.clinic.slots.get(slot_id)
    if slot is None:
        return {"ok": False, "error": "SLOT_NOT_OFFERED"}

    existing = _appointment_for_slot(session, slot_id)
    if existing is not None:
        if existing.patient_id == patient_id and existing.status == "booked":
            return {
                "ok": True,
                "confirmation_id": existing.confirmation_id,
                "slot_id": slot_id,
                "start": slot.start.isoformat(timespec="minutes"),
                "provider_name": slot.provider_name,
                "idempotent": True,
            }
        return {"ok": False, "error": "SLOT_TAKEN"}

    if session.race_first_book and not session.race_used:
        session.race_used = True
        slot.status = "booked"
        return {"ok": False, "error": "SLOT_TAKEN", "slot_id": slot_id}

    if slot.status != "open":
        return {"ok": False, "error": "SLOT_TAKEN"}

    session.clinic._seq += 1
    confirmation_id = f"CNF-{session.clinic._seq}"
    apt_id = f"apt_{session.clinic._seq}"
    session.clinic.appointments[apt_id] = Appointment(
        id=apt_id,
        patient_id=patient_id,
        slot_id=slot_id,
        confirmation_id=confirmation_id,
        visit_type=slot.visit_type,
    )
    slot.status = "booked"
    return {
        "ok": True,
        "confirmation_id": confirmation_id,
        "appointment_id": apt_id,
        "slot_id": slot_id,
        "start": slot.start.isoformat(timespec="minutes"),
        "provider_name": slot.provider_name,
    }


def cancel_appointment(session: Session, *, appointment_id: str) -> dict:
    appointment_id = appointment_id.strip()
    apt = session.clinic.appointments.get(appointment_id)
    if apt is None or apt.status != "booked":
        return {"ok": False, "error": "NOT_FOUND"}
    if session.resolved_patient_id != apt.patient_id:
        return {"ok": False, "error": "PATIENT_NOT_FOUND"}
    if session.confirmed.get("cancel") != appointment_id:
        return {"ok": False, "error": "NOT_CONFIRMED"}
    slot = session.clinic.slots[apt.slot_id]
    if slot.start - NOW < CANCEL_WINDOW:
        return {"ok": False, "error": "OUTSIDE_POLICY", "cancel_window_hours": 24}
    apt.status = "cancelled"
    slot.status = "open"
    return {"ok": True, "appointment_id": appointment_id, "confirmation_id": apt.confirmation_id}


def reschedule_appointment(session: Session, *, appointment_id: str, slot_id: str) -> dict:
    appointment_id = appointment_id.strip()
    slot_id = slot_id.strip()
    apt = session.clinic.appointments.get(appointment_id)
    if apt is None or apt.status != "booked":
        return {"ok": False, "error": "NOT_FOUND"}
    if session.resolved_patient_id != apt.patient_id:
        return {"ok": False, "error": "PATIENT_NOT_FOUND"}
    old_slot = session.clinic.slots[apt.slot_id]
    if old_slot.start - NOW < CANCEL_WINDOW:
        return {"ok": False, "error": "OUTSIDE_POLICY", "cancel_window_hours": 24}
    if session.confirmed.get("reschedule") != slot_id:
        return {"ok": False, "error": "NOT_CONFIRMED"}
    if slot_id not in session.offered_slot_ids:
        return {"ok": False, "error": "SLOT_NOT_OFFERED"}
    new_slot = session.clinic.slots.get(slot_id)
    if new_slot is None or new_slot.status != "open":
        return {"ok": False, "error": "SLOT_TAKEN"}
    old_slot.status = "open"
    new_slot.status = "booked"
    apt.slot_id = slot_id
    apt.visit_type = new_slot.visit_type
    return {
        "ok": True,
        "confirmation_id": apt.confirmation_id,
        "appointment_id": apt.id,
        "slot_id": slot_id,
        "start": new_slot.start.isoformat(timespec="minutes"),
        "provider_name": new_slot.provider_name,
    }


def get_clinic_policy(session: Session) -> dict:
    del session
    return {
        "ok": True,
        "hours": "09:00-17:00",
        "timezone": "clinic local",
        "today": NOW.date().isoformat(),
        "visit_types": ["general", "follow_up"],
        "cancel_window_hours": 24,
        "rules": [
            "Do not diagnose or give medical advice.",
            "Emergencies go to emergency services. Stop scheduling.",
            "Do not reveal another patient's appointments.",
            "Do not book, cancel, or reschedule without an explicit patient confirmation.",
            "Do not claim a booking unless book_appointment returns a confirmation_id.",
        ],
    }


def escalate(session: Session, *, reason: str) -> dict:
    reason = reason.strip() or "unspecified"
    return {"ok": True, "reason": reason, "scheduling_stopped": True}


def finish(session: Session, *, reason: str) -> dict:
    del session
    return {"ok": True, "reason": reason.strip() or "done"}


def _appointment_for_slot(session: Session, slot_id: str):
    for apt in session.clinic.appointments.values():
        if apt.slot_id == slot_id and apt.status == "booked":
            return apt
    return None
