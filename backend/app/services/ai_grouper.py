import json
import os
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import httpx

from app.models.schemas import ExtractedQuestion
from app.services.clustering import (
    chronological_paper_key,
    cluster_questions,
    encode_texts,
)
from app.services.labeling import heuristic_topic_title


def _call_groq_group_unit(
    subject_name: str,
    unit_name: str,
    unit_questions: List[ExtractedQuestion],
    api_key: str,
) -> Optional[List[Tuple[str, List[int]]]]:
    numbered_lines = "\n".join(
        f"{idx}: {q.exact_text}" for idx, q in enumerate(unit_questions)
    )

    prompt = (
        f"You are organizing SPPU university exam questions for '{subject_name}' ({unit_name}) "
        "into clean, accurate revision groups.\n"
        "Strict Grouping Rules:\n"
        "1. Treat exam command synonyms as IDENTICAL intent: "
        "'Compare A and B' == 'Distinguish between A and B' == 'Difference between A and B' -> keep all of those together!\n"
        "2. Similarly, 'Draw block diagram of X' == 'Draw and explain architecture of X' -> group together.\n"
        "3. 'Explain in brief PLD design flow' == 'Draw and explain PLD design flow' -> group together.\n"
        "4. Do NOT merge distinct topics (e.g. keep 'FPGA vs CPLD Comparison' separate from 'FPGA Architecture/CLB', and separate from 'CPLD Architecture/Macrocell').\n"
        "5. Use ONLY the exact integer IDs given below. Include EVERY ID from 0 to "
        f"{len(unit_questions) - 1} exactly once.\n"
        "6. Give each group a crisp 3 to 6 word plain-English title (e.g. 'FPGA vs CPLD Comparison', 'PLD Design Flow').\n"
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
                    "max_tokens": 3000,
                },
                timeout=30.0,
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
    fallback_threshold: float = 0.56,
) -> Optional[List[Tuple[str, str, List[ExtractedQuestion], float]]]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or not questions:
        return None

    by_unit: Dict[str, List[ExtractedQuestion]] = defaultdict(list)
    for q in questions:
        by_unit[q.unit_number or "General"].append(q)

    all_clusters: List[Tuple[str, str, List[ExtractedQuestion], float]] = []
    ai_units_succeeded = 0

    for unit_name, unit_qs in by_unit.items():
        if len(unit_qs) == 1:
            q_only = unit_qs[0]
            all_clusters.append(
                (str(uuid.uuid4()), heuristic_topic_title([q_only]), [q_only], 1.0)
            )
            continue

        grouped = _call_groq_group_unit(subject_name, unit_name, unit_qs, api_key)
        if not grouped:
            # Fallback ONLY for this unit instead of throwing away other successful units
            unit_embs = encode_texts([q.exact_text for q in unit_qs])
            for cid, m_qs, avg_sim in cluster_questions(unit_qs, unit_embs, fallback_threshold):
                all_clusters.append((cid, heuristic_topic_title(m_qs), m_qs, avg_sim))
            continue

        ai_units_succeeded += 1
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

        for idx, q in enumerate(unit_qs):
            if idx not in seen_local_ids:
                all_clusters.append(
                    (str(uuid.uuid4()), heuristic_topic_title([q]), [q], 1.0)
                )

    if ai_units_succeeded == 0:
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
