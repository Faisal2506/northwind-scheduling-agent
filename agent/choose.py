"""Match a short patient reply to slots that were already offered."""

from __future__ import annotations

import re

_WEEKDAY = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}
_ORDINAL = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "last": -1,
}


def open_offered(session) -> list:
    slots = []
    for slot_id in session.offered_slot_ids:
        slot = session.clinic.slots.get(slot_id)
        if slot is not None and slot.status == "open":
            slots.append(slot)
    return slots


def describe(slot) -> str:
    start = slot.start.isoformat(timespec="minutes")
    return f"{start} with {slot.provider_name} ({slot.id})"


def match_offered(session, text: str) -> list | None:
    """Return matching open slots, or None when the message is not a slot choice."""
    slots = open_offered(session)
    if not slots:
        return None
    raw = " ".join(text.casefold().split())
    by_id = [slot for slot in slots if slot.id.casefold() in raw]
    if by_id:
        return by_id

    exact = re.findall(r"\d{4}-\d{2}-\d{2}t\d{2}:\d{2}", raw)
    if exact:
        return [
            slot
            for slot in slots
            if slot.start.isoformat(timespec="minutes").casefold() in exact
        ]

    days = [day for word, day in _WEEKDAY.items() if re.search(rf"\b{word}\b", raw)]
    scrubbed = re.sub(r"\d{4}-\d{2}-\d{2}(?:t\d{2}:\d{2})?", " ", raw)
    times = _times(scrubbed)
    month_days = [int(number) for number in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)\b", raw)]
    doctor_names = {slot.provider_name.split()[-1].casefold() for slot in slots}
    doctors = [name for name in doctor_names if re.search(rf"\b{re.escape(name)}\b", raw)]
    ordinals = [name for name in _ORDINAL if re.search(rf"\b{name}\b", raw)]
    if not days and not times and not doctors and not ordinals and not month_days:
        return None

    if ordinals and not days and not times and not doctors and not month_days:
        index = _ORDINAL[ordinals[0]]
        if index == -1:
            return [slots[-1]]
        if 0 <= index < len(slots):
            return [slots[index]]
        return []

    chosen = slots
    if days:
        chosen = [slot for slot in chosen if slot.start.weekday() in days]
    if times:
        chosen = [slot for slot in chosen if (slot.start.hour, slot.start.minute) in times]
    if month_days:
        chosen = [slot for slot in chosen if slot.start.day in month_days]
    if doctors:
        chosen = [slot for slot in chosen if slot.provider_name.split()[-1].casefold() in doctors]
    return chosen


def choice_reply(matches: list, offered: list | None = None) -> str:
    if not matches:
        listed = "; ".join(describe(slot) for slot in offered or [])
        if listed:
            return (
                "None of the open times fall on that day. "
                f"These are open: {listed}. Which time should I book? Say yes after you pick one."
            )
        return "That does not match an open time I offered. Pick a day and time from the list."
    if len(matches) == 1:
        return f"I can book {describe(matches[0])}. Say yes to confirm."
    listed = "; ".join(describe(slot) for slot in matches)
    return f"That matches more than one time: {listed}. Which time should I book? Say yes after you pick one."


def _times(text: str) -> list[tuple[int, int]]:
    found = []
    for match in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text, re.I):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = (match.group(3) or "").lower()
        if not ampm and match.group(2) is None and hour < 8:
            continue
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            continue
        found.append((hour, minute))
    return found
