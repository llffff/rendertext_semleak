"""Classify OCR recognition of a requested word or phrase."""

import difflib
import re


def woi_status(expected: str, recognized: list[str]) -> str:
    ignored = set("a an the of to in on at with and or for into over under above below left right next near".split())

    def words(text):
        normalized = re.sub(r"[^a-z0-9]+", " ", str(text).casefold())
        return [word for word in normalized.split() if word not in ignored]

    target = words(expected)
    observed = words(" ".join(recognized))
    if not target:
        return "missing"
    if all(word in observed for word in target):
        return "normal"
    ratio = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()
    token = max((ratio(a, b) for a in target for b in observed), default=0)
    phrases = (" ".join(observed[start:start + size])
               for size in range(max(1, len(target) - 1), len(target) + 2)
               for start in range(len(observed) - size + 1))
    phrase = max((ratio(" ".join(target), candidate) for candidate in phrases), default=0)
    return "spelling_error" if max(token, phrase) >= 0.7 else "missing"
