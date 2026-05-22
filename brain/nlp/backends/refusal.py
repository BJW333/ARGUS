"""
Refusal detection — regex-based heuristic for spotting policy refusals
from API models (Claude, GPT, etc.).

Used by RouterBackend to decide if the primary's response is a refusal
that should trigger fallback to the local model.

The check only scans the head of the response (first ~300-500 chars)
because refusals always front-load. Scanning the whole response causes
false positives on long answers that mention refusal patterns in passing
(e.g., "I cannot stress enough how important...").
"""
import re

REFUSAL_PATTERNS = [
    r"i (?:can'?t|cannot|am not able to|won'?t) (?:help|assist|provide|do|generate|create|write|engage|comply)",
    r"i'?m (?:not able|unable|sorry,? but i)",
    r"i (?:apologize|must decline|do not feel comfortable)",
    r"against (?:my|our|the) (?:guidelines|policies|principles|values|rules)",
    r"violates (?:openai|anthropic|usage|content|our) polic",
    r"as an ai (?:language model|assistant),? i (?:can'?t|cannot)",
    r"goes against my (?:programming|guidelines|values)",
    r"not (?:appropriate|something i can help)",
    r"i'?m not (?:going to|able to) (?:help|assist|provide)",
    r"this (?:request|content) (?:violates|is against)",
    r"i must (?:refuse|decline|emphasize that i can'?t)",
]
_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def looks_like_refusal(text: str, head_chars: int = 500) -> bool:
    """
    Check if the start of `text` matches a refusal pattern.

    Args:
        text: Response text to check.
        head_chars: How many leading chars to scan.

    Returns:
        True if any refusal pattern matches in the head.
    """
    if not text:
        return False
    return bool(_RE.search(text.strip()[:head_chars]))
