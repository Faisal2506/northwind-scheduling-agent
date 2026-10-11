"""Pull a name and date of birth out of a normal sentence."""

from __future__ import annotations

import re

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_SKIP = re.compile(
    r"chest pain|cannot breathe|can't breathe|trouble breathing|assistant|skip verification",
    re.I,
)
_GREETING = {
    "hi",
    "hii",
    "hio",
    "hello",
    "hey",
    "helo",
    "yo",
    "thanks",
    "thank",
    "ok",
    "okay",
    "yes",
    "no",
}
_NOT_A_NAME = {
    "book",
    "appointment",
    "appointments",
    "general",
    "follow",
    "followup",
    "visit",
    "help",
    "please",
    "cancel",
    "reschedule",
    "weekend",
    "today",
    "tomorrow",
    "next",
    "week",
    "slot",
    "time",
    "date",
    "range",
    "check",
    "checkup",
    "new",
    "schedule",
    "scheduling",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
}


def parse_identity(text: str) -> dict | None:
    raw = " ".join(text.strip().split())
    if not raw or _SKIP.search(raw):
        return None
    if raw.casefold() in _GREETING:
        return None

    dob = _dob(raw)
    named = re.fullmatch(
        r"([A-Za-z]+(?:\s+[A-Za-z]+)?)\s*,?\s*(\d{4}-\d{2}-\d{2})",
        raw,
    )
    if named and not _blocked_name(named.group(1)):
        first, _, last = named.group(1).partition(" ")
        return {
            "first_name": first[:1].upper() + first[1:],
            "last_name": last,
            "dob": named.group(2),
        }
    name = _explicit_name(raw) or _bare_name(raw)
    if not name:
        return None
    parts = [part.strip(".,") for part in name.split() if part.strip(".,")]
    if not parts or parts[0].casefold() in _GREETING or parts[0].casefold() == "dr":
        return None
    first = parts[0]
    last = parts[1] if len(parts) > 1 else ""
    return {"first_name": first[:1].upper() + first[1:], "last_name": last, "dob": dob}


def _explicit_name(text: str) -> str:
    match = re.search(r"(?:i'm|i am|my name is)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", text, re.I)
    return match.group(1) if match else ""


def _blocked_name(name: str) -> bool:
    words = name.casefold().split()
    return any(word in _NOT_A_NAME or word in _GREETING for word in words)


def _bare_name(text: str) -> str:
    if not re.fullmatch(r"[A-Za-z]+(?:\s+[A-Za-z]+){0,2}", text):
        return ""
    if _blocked_name(text):
        return ""
    words = text.split()
    if len(words) == 1 and words[0].islower():
        return ""
    return text


def date_in(text: str) -> str:
    found = _dob(text)
    if found:
        return found
    bare = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", bare):
        return bare
    return ""


def _dob(text: str) -> str:
    labeled = re.search(
        r"(?:born|date of birth(?: is)?|dob(?: is)?)\s*(\d{4}-\d{2}-\d{2})",
        text,
        re.I,
    )
    if labeled:
        return labeled.group(1)
    written = re.search(
        r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b",
        text,
    )
    if not written:
        return ""
    month = _MONTHS.get(written.group(2).casefold())
    if not month:
        return ""
    return f"{int(written.group(3)):04d}-{month:02d}-{int(written.group(1)):02d}"


def reply_for_lookup(result: dict) -> str | None:
    error = result.get("error")
    if error == "NOT_FOUND":
        return (
            "Nobody in this clinic matches that name. "
            "Try Maya Chen (1990-04-12), Maya Patel (1985-01-03), "
            "Owen Brooks (1978-11-20), or Priya Shah (1992-07-07)."
        )
    if error == "AMBIGUOUS":
        return "More than one patient has that name. What is the last name and date of birth?"
    if error == "NEED_DOB":
        return "What is your date of birth?"
    return None
