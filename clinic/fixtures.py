from __future__ import annotations

from datetime import datetime

from clinic.model import Appointment, Clinic, Patient, Session, Slot


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value)


def make_clinic() -> Clinic:
    patients = {
        "p_maya_chen": Patient("p_maya_chen", "Maya", "Chen", "1990-04-12"),
        "p_maya_patel": Patient("p_maya_patel", "Maya", "Patel", "1985-01-03"),
        "p_owen": Patient("p_owen", "Owen", "Brooks", "1978-11-20"),
        "p_priya": Patient("p_priya", "Priya", "Shah", "1992-07-07"),
    }
    slots = {
        "slot_mon_10": Slot("slot_mon_10", "Dr. Lee", _at("2026-10-13T10:00"), "general"),
        "slot_mon_14": Slot("slot_mon_14", "Dr. Lee", _at("2026-10-13T14:00"), "general"),
        "slot_tue_09": Slot("slot_tue_09", "Dr. Okonkwo", _at("2026-10-14T09:00"), "general"),
        "slot_tue_11": Slot("slot_tue_11", "Dr. Okonkwo", _at("2026-10-14T11:00"), "follow_up"),
        "slot_thu_11": Slot("slot_thu_11", "Dr. Lee", _at("2026-10-15T11:00"), "general", status="booked"),
        "slot_fri_15": Slot("slot_fri_15", "Dr. Lee", _at("2026-10-16T15:00"), "general", status="booked"),
        "slot_today_14": Slot("slot_today_14", "Dr. Okonkwo", _at("2026-10-12T14:00"), "follow_up", status="booked"),
        "slot_later": Slot("slot_later", "Dr. Lee", _at("2026-10-20T10:00"), "follow_up", status="booked"),
    }
    appointments = {
        "apt_patel": Appointment("apt_patel", "p_maya_patel", "slot_thu_11", "CNF-PATEL", "general"),
        "apt_chen": Appointment("apt_chen", "p_maya_chen", "slot_fri_15", "CNF-CHEN", "general"),
        "apt_priya": Appointment("apt_priya", "p_priya", "slot_today_14", "CNF-PRIYA", "follow_up"),
        "apt_owen": Appointment("apt_owen", "p_owen", "slot_later", "CNF-OWEN", "follow_up"),
    }
    return Clinic(patients=patients, slots=slots, appointments=appointments)


def make_session(*, race_first_book: bool = False) -> Session:
    return Session(clinic=make_clinic(), race_first_book=race_first_book)
