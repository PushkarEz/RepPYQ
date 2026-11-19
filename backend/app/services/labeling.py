import os
import re
from typing import List, Optional
import httpx
import numpy as np

from app.models.schemas import ExtractedQuestion
from app.services.clustering import encode_texts


COMMAND_PREFIX_RE = re.compile(
    r"^(?:"
    r"explain\s+in\s+detail|explain|discuss\s+in\s+detail|discuss|"
    r"describe|define|state\s+and\s+explain|state|draw\s+and\s+explain|"
    r"draw\s+neat\s+diagram\s+(?:of|for)|draw|write\s+a?\s*short\s+notes?\s+on|"
    r"write\s+note\s+on|compare\s+and\s+contrast|compare|differentiate\s+between|"
    r"distinguish\s+between|what\s+is\s+meant\s+by|what\s+is|what\s+are|"
    r"enlist|list\s+and\s+explain|list|derive\s+an\s+expression\s+for|derive|"
    r"prove\s+that|show\s+that|enumerate|illustrate|with\s+neat\s+diagram|"
    r"calculate\s+the|calculate|determine\s+the|determine|find\s+the|find"
    r")\s+",
    re.IGNORECASE,
)

TRAILING_FLUFF_RE = re.compile(
    r"\b(?:with\s+(?:a\s+)?neat\s+(?:labelled\s+)?diagram|with\s+suitable\s+example|"
    r"in\s+detail|briefly|giving\s+examples|with\s+diagram)\b.*$",
    re.IGNORECASE,
)

DANGLING_TAIL_WORDS = {
    "a", "an", "the", "of", "and", "or", "with", "in", "on", "for", "to", "which", "that", "is", "are"
}


def heuristic_topic_title(questions: List[ExtractedQuestion]) -> str:
    """Produces a crisp, plain-English topic title without dangling tail words."""
    shortest = min(questions, key=lambda q: len(q.exact_text)).exact_text
    is_numerical = bool(re.search(r"\b(?:calculate|determine|find|mhz|ghz|kw|mw|db|vswr|ohm)\b", shortest, re.I))
    text = shortest.strip().rstrip(".?:;")

    for _ in range(3):
        updated = COMMAND_PREFIX_RE.sub("", text).strip()
        if updated == text:
            break
        text = updated

    text = TRAILING_FLUFF_RE.sub("", text).strip(" ,.;:-")
    text = re.sub(r"^(?:the|an|a|of|about|on)\s+", "", text, flags=re.IGNORECASE)

    # Cut cleanly at clause boundaries before max length
    clause = re.split(r"\s+(?:which|where|if\s+its|whose)\s+", text, maxsplit=1, flags=re.I)[0]
    text = clause.strip(" ,.;:-")

    if len(text) > 58:
        text = text[:58].rsplit(" ", 1)[0].rstrip(",;:")

    words = text.split()
    while len(words) > 2 and words[-1].lower().rstrip(",;:") in DANGLING_TAIL_WORDS:
        words.pop()
    text = " ".join(words).rstrip(",;:-")

    if is_numerical and "numerical" not in text.lower() and len(text) < 46:
        text = f"{text} (Numerical)"

    if not text:
        text = shortest[:52]

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
    sims = (q_emb @ unit_embeddings.T)[0]
    best_idx = int(np.argmax(sims))
    if float(sims[best_idx]) >= 0.25:
        return syllabus_units[best_idx].split(":")[0].strip()[:45]
    return None


def generate_cluster_title(questions: List[ExtractedQuestion]) -> str:
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    ollama_url = os.getenv("OLLAMA_URL", "").strip()

    if not groq_key and not ollama_url:
        return heuristic_topic_title(questions)

    sample_qs = "\n".join(f"- {q.exact_text}" for q in questions[:4])
    prompt = (
        "Give a crisp, plain-English 3 to 7 word exam topic title for these university "
        "exam questions. No filler words, no quotes, no colon, no AI phrasing.\n"
        f"Questions:\n{sample_qs}\n\nTopic Title:"
    )

    if groq_key:
        try:
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 25,
                },
                timeout=8.0,
            )
            if resp.status_code == 200:
                title = resp.json()["choices"][0]["message"]["content"].strip().strip('"\'.')
                if 3 <= len(title) <= 75:
                    return title
        except Exception:
            pass

    if ollama_url:
        try:
            model = os.getenv("OLLAMA_MODEL", "llama3.2")
            resp = httpx.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=8.0,
            )
            if resp.status_code == 200:
                title = resp.json().get("response", "").strip().strip('"\'.')
                if 3 <= len(title) <= 75:
                    return title
        except Exception:
            pass

    return heuristic_topic_title(questions)
