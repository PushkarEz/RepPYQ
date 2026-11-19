import os
import re
from typing import List, Optional
import httpx
import numpy as np

from app.models.schemas import ExtractedQuestion
from app.services.clustering import encode_texts


COMPARE_RE = re.compile(
    r"^(?:compare(?:\s+and\s+contrast)?|distinguish\s+between|differentiate\s+between|"
    r"difference\s+between|contrast)\s+(.+)$",
    re.IGNORECASE,
)

COMMAND_PREFIX_RE = re.compile(
    r"^(?:"
    r"explain\s+in\s+detail|explain|discuss\s+in\s+detail|discuss|"
    r"describe|define|state\s+and\s+explain|state|draw\s+and\s+explain|"
    r"draw\s+neat\s+diagram\s+(?:of|for)|draw|write\s+a?\s*short\s+notes?\s+on|"
    r"write\s+note\s+on|what\s+is\s+meant\s+by|what\s+is|what\s+are|"
    r"enlist|list\s+and\s+explain|list|derive\s+an\s+expression\s+for|derive|"
    r"prove\s+that|show\s+that|enumerate|illustrate|with\s+neat\s+diagram|"
    r"calculate\s+the|calculate|determine\s+the|determine|find\s+the|find"
    r")\s+",
    re.IGNORECASE,
)

TRAILING_FLUFF_RE = re.compile(
    r"\b(?:with\s+(?:a\s+)?neat\s+(?:labelled\s+)?diagram|with\s+suitable\s+example|"
    r"on\s+the\s+basis\s+of\s+features.*|in\s+detail|in\s+brief|briefly|giving\s+examples|with\s+diagram)\b.*$",
    re.IGNORECASE,
)

DANGLING_TAIL_WORDS = {
    "a", "an", "the", "of", "and", "or", "with", "in", "on", "for", "to", "which", "that", "is", "are", "its"
}


def heuristic_topic_title(questions: List[ExtractedQuestion]) -> str:
    shortest = min(questions, key=lambda q: len(q.exact_text)).exact_text
    text = shortest.strip().rstrip(".?:;")

    # Keep comparison intent clear ("Compare FPGA and CPLD" -> "FPGA vs CPLD Comparison")
    cmp_m = COMPARE_RE.match(text)
    if cmp_m:
        body = TRAILING_FLUFF_RE.sub("", cmp_m.group(1)).strip(" ,.;:-")
        parts = re.split(r"\s+(?:and|with|vs\.?|versus)\s+", body, maxsplit=1, flags=re.I)
        if len(parts) == 2 and parts[0] and parts[1]:
            a = parts[0].strip(" ,.;:-")
            b = parts[1].strip(" ,.;:-")
            if len(a) <= 28 and len(b) <= 28:
                return f"{a} vs {b} Comparison"
        return f"Compare {body[:52]}"

    is_numerical = bool(re.search(r"\b(?:calculate|determine|find|mhz|ghz|kw|mw|db|vswr|ohm)\b", shortest, re.I))

    for _ in range(3):
        updated = COMMAND_PREFIX_RE.sub("", text).strip()
        if updated == text:
            break
        text = updated

    text = TRAILING_FLUFF_RE.sub("", text).strip(" ,.;:-")
    text = re.sub(r"^(?:the|an|a|of|about|on)\s+", "", text, flags=re.IGNORECASE)

    clause = re.split(r"\s+(?:which|where|if\s+its|whose)\s+", text, maxsplit=1, flags=re.I)[0]
    text = clause.strip(" ,.;:-")

    if len(text) > 72:
        text = text[:72].rsplit(" ", 1)[0].rstrip(",;:")

    words = text.split()
    while len(words) > 2 and words[-1].lower().rstrip(",;:") in DANGLING_TAIL_WORDS:
        words.pop()
    text = " ".join(words).rstrip(",;:-")

    if is_numerical and "numerical" not in text.lower() and len(text) < 56:
        text = f"{text} (Numerical)"

    if not text:
        text = shortest[:60]

    return text[0].upper() + text[1:] if text else "Untitled Topic"


def match_syllabus_unit(
    questions: List[ExtractedQuestion],
    syllabus_units: List[str],
    unit_embeddings: Optional[np.ndarray],
) -> Optional[str]:
    if not syllabus_units or unit_embeddings is None or len(unit_embeddings) == 0:
        return None
    combined_text = " ".join(q.exact_text for q in questions[:3])
    q_emb = encode_texts([combined_text])
    if unit_embeddings.shape[1] != q_emb.shape[1]:
        return None
    sims = (q_emb @ unit_embeddings.T)[0]
    best_idx = int(np.argmax(sims))
    if float(sims[best_idx]) >= 0.25:
        return syllabus_units[best_idx].split(":")[0].strip()[:45]
    return None


def generate_cluster_title(questions: List[ExtractedQuestion]) -> str:
    return heuristic_topic_title(questions)
