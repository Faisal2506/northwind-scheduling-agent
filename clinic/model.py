from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


NOW = datetime(2026, 10, 12, 9, 0)
CANCEL_WINDOW = timedelta(hours=24)


@dataclass
class Patient:
    id: str
    first_name: str
    last_name: str
    dob: str


@dataclass
class Slot:
    id: str
    provider_name: str
    start: datetime
    visit_type: str
    status: str = "open"  # open | booked


@dataclass
class Appointment:
    id: str
    patient_id: str
    slot_id: str
    confirmation_id: str
    visit_type: str
    status: str = "booked"  # booked | cancelled


@dataclass
class Clinic:
    patients: dict[str, Patient]
    slots: dict[str, Slot]
    appointments: dict[str, Appointment]
    _seq: int = 100

    def snapshot(self) -> dict:
        booked = []
        for apt in self.appointments.values():
            if apt.status != "booked":
                continue
            slot = self.slots[apt.slot_id]
            patient = self.patients[apt.patient_id]
            booked.append(
                {
                    "appointment_id": apt.id,
                    "patient_id": apt.patient_id,
                    "patient_name": f"{patient.first_name} {patient.last_name}",
                    "slot_id": apt.slot_id,
                    "start": slot.start.isoformat(timespec="minutes"),
                    "provider_name": slot.provider_name,
                    "visit_type": apt.visit_type,
                    "confirmation_id": apt.confirmation_id,
                    "status": apt.status,
                }
            )
        return {
            "appointments": booked,
            "open_slot_ids": sorted(slot.id for slot in self.slots.values() if slot.status == "open"),
        }


@dataclass
class Session:
    """Explicit conversation state the harness can read. Not the calendar."""

    clinic: Clinic
    race_first_book: bool = False
    phase: str = "identify"
    resolved_patient_id: str | None = None
    offered_slot_ids: list[str] = field(default_factory=list)
    confirmed: dict[str, str] = field(default_factory=dict)
    last_error: str | None = None
    escalated: bool = False
    escalated_reason: str | None = None
    terminal: bool = False
    race_used: bool = False
    last_human_text: str = ""
    tool_trace: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def public_state(self) -> dict:
        return {
            "phase": self.phase,
            "resolved_patient_id": self.resolved_patient_id,
            "offered_slot_ids": list(self.offered_slot_ids),
            "confirmed": dict(self.confirmed),
            "last_error": self.last_error,
            "escalated": self.escalated,
            "escalated_reason": self.escalated_reason,
            "terminal": self.terminal,
        }
