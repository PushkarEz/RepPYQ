import json
import os
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import httpx

from app.models.schemas import ExtractedQuestion
from app.services.clustering import chronological_paper_key
from app.services.labeling import heuristic_topic_title


def _call_groq_group_unit(
    subject_name: str,
    unit_name: str,
    unit_questions: List[ExtractedQuestion],
    api_key: str,
) -> Optional[List[Tuple[str, List[int]]]]:
    """
    Sends ONLY numeric indices + question text for ONE unit to Groq.
    Returns list of (topic_title, [local_indices]) or None on failure.
    Never modifies or returns question text.
    """
    numbered_lines = "\n".join(
        f"{idx}: {q.exact_text}" for idx, q in enumerate(unit_questions)
    )

    prompt = (
        f"You are organizing SPPU university exam questions for '{subject_name}' ({unit_name}) "
        "into clean, practical revision topics.\n"
        "Rules:\n"
        "1. Group questions that test the SAME core syllabus concept together "
        "(e.g., 'CMOS Inverter, VTC & Noise Margin' or 'Dynamic & Static Power Dissipation'), "
        "even if one asks for a derivation, diagram, numerical, or short note on that concept.\n"
        "2. Keep genuinely distinct syllabus concepts separate.\n"
        "3. Aim for concise, clean study buckets (typically 5 to 12 topics per unit when concepts repeat).\n"
        "4. Use ONLY the exact integer IDs given below. Every ID from 0 to "
        f"{len(unit_questions) - 1} must appear exactly once.\n"
        "5. Topic titles must be short, plain English (3 to 7 words, no quotes, no colons).\n"
        'Return ONLY valid JSON: {"topics": [{"title": "...", "ids": [0, 3]}]}\n\n'
        f"Questions:\n{numbered_lines}"
    )

    models_to_try = [
        os.getenv("GROQ_GROUP_MODEL", "llama-3.3-70b-versatile"),
        "llama-3.1-8b-instant",
    ]

    for model in models_to_try:
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "max_tokens": 2000,
                },
                timeout=25.0,
            )
            if resp.status_code != 200:
                continue

            raw_content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw_content)
            topics = parsed.get("topics")
            if not isinstance(topics, list):
                continue

            cleaned: List[Tuple[str, List[int]]] = []
            for t in topics:
                if not isinstance(t, dict):
                    continue
                title = str(t.get("title", "")).strip()
                ids = t.get("ids", [])
                if isinstance(ids, list):
                    valid_ids = [
                        int(i)
                        for i in ids
                        if isinstance(i, (int, str)) and str(i).isdigit()
                    ]
                    if valid_ids:
                        cleaned.append((title, valid_ids))

            if cleaned:
                return cleaned
        except Exception:
            continue

    return None


def group_questions_with_ai(
    subject_name: str,
    questions: List[ExtractedQuestion],
) -> Optional[List[Tuple[str, str, List[ExtractedQuestion], float]]]:
    """
    Runs ID-only AI Concept Grouping unit by unit.
    Returns list of (cluster_id, topic_title, verbatim_questions, confidence)
    or None if GROQ_API_KEY is not configured / fails completely.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or not questions:
        return None

    by_unit: Dict[str, List[ExtractedQuestion]] = defaultdict(list)
    for q in questions:
        by_unit[q.unit_number or "General"].append(q)

    all_clusters: List[Tuple[str, str, List[ExtractedQuestion], float]] = []
    any_ai_success = False

    for unit_name, unit_qs in by_unit.items():
        if len(unit_qs) == 1:
            q_only = unit_qs[0]
            all_clusters.append(
                (str(uuid.uuid4()), heuristic_topic_title([q_only]), [q_only], 1.0)
            )
            continue

        grouped = _call_groq_group_unit(subject_name, unit_name, unit_qs, api_key)
        if not grouped:
            # Skip AI for this unit; caller falls back if needed
            return None

        any_ai_success = True
        seen_local_ids = set()

        for title, local_ids in grouped:
            member_qs: List[ExtractedQuestion] = []
            for lid in local_ids:
                if 0 <= lid < len(unit_qs) and lid not in seen_local_ids:
                    seen_local_ids.add(lid)
                    member_qs.append(unit_qs[lid])

            if not member_qs:
                continue

            member_qs.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))
            clean_title = title.strip('"\'.') if title else heuristic_topic_title(member_qs)
            all_clusters.append((str(uuid.uuid4()), clean_title, member_qs, 0.92))

        # Safety net: preserve any question ID missed by the LLM without losing anything
        for idx, q in enumerate(unit_qs):
            if idx not in seen_local_ids:
                all_clusters.append(
                    (str(uuid.uuid4()), heuristic_topic_title([q]), [q], 1.0)
                )

    if not any_ai_success:
        return None

    all_clusters.sort(
        key=lambda item: (
            len({q.paper_id for q in item[2]}),
            len(item[2]),
            item[3],
        ),
        reverse=True,
    )
    return all_clusters
