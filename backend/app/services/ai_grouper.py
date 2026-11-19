import json
import os
import re
import time
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


def _extract_json_object(raw: str) -> Optional[dict]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def _call_groq_group_unit(
    subject_name: str,
    unit_name: str,
    unit_questions: List[ExtractedQuestion],
    api_key: str,
) -> Tuple[Optional[List[Tuple[str, List[int]]]], Optional[str]]:
    numbered_lines = "\n".join(
        f"{idx}: {q.exact_text}" for idx, q in enumerate(unit_questions)
    )

    prompt = (
        f"You are organizing SPPU university exam questions for '{subject_name}' ({unit_name}) "
        "into clean, accurate revision groups.\n"
        "Strict Rules:\n"
        "1. Treat synonyms as the SAME topic: 'Compare A and B' == 'Distinguish between A and B' == 'Difference between A and B'.\n"
        "2. Fix obvious exam paper typos mentally when grouping (e.g., 'CPCD' = 'CPLD').\n"
        "3. If a question asks 'Draw architecture of X and compare CPLD and FPGA', put it in the 'CPLD vs FPGA Comparison' group so comparison frequency is accurate!\n"
        "4. Group 'Draw block diagram / architecture / CLB / Macrocell of X' together for the same device X.\n"
        "5. Keep genuinely distinct topics separate (e.g. FPGA Architecture != CPLD Architecture != CPLD vs FPGA Comparison).\n"
        f"6. Use ONLY the integer IDs 0 to {len(unit_questions) - 1}. Include EVERY ID exactly once.\n"
        "7. Give each group a crisp 3 to 6 word plain-English title (e.g., 'CPLD vs FPGA Comparison').\n"
        'Return ONLY valid JSON: {"topics": [{"title": "...", "ids": [0, 3]}]}\n\n'
        f"Questions:\n{numbered_lines}"
    )

    models_to_try = [
        os.getenv("GROQ_GROUP_MODEL", "llama-3.3-70b-versatile").strip(),
        "llama-3.1-8b-instant",
    ]

    last_err = None
    for model in models_to_try:
        for attempt in range(2):
            try:
                resp = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                        "max_tokens": 2500,
                    },
                    timeout=30.0,
                )
                if resp.status_code == 429:
                    last_err = f"Groq rate limit (429) on {model}"
                    time.sleep(1.5)
                    continue
                if resp.status_code != 200:
                    last_err = f"Groq HTTP {resp.status_code} on {model}: {resp.text[:120]}"
                    break

                raw_content = resp.json()["choices"][0]["message"]["content"]
                parsed = _extract_json_object(raw_content)
                if not parsed:
                    last_err = f"Groq returned non-JSON on {model}"
                    break

                topics = parsed.get("topics")
                if not isinstance(topics, list):
                    last_err = f"Groq JSON missing 'topics' list on {model}"
                    break

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
                    return cleaned, None
            except Exception as exc:
                last_err = f"Groq request error on {model}: {exc}"
                break

    return None, last_err


def group_questions_with_ai(
    subject_name: str,
    questions: List[ExtractedQuestion],
    fallback_threshold: float = 0.56,
) -> Tuple[Optional[List[Tuple[str, str, List[ExtractedQuestion], float]]], Optional[str]]:
    """
    Returns (clusters_or_None, diagnostic_error_or_None).
    Never modifies original question text.
    """
    raw_key = os.getenv("GROQ_API_KEY", "")
    api_key = raw_key.strip().strip('"').strip("'")
    if not api_key:
        return None, "GROQ_API_KEY is not set in environment."
    if not questions:
        return None, None

    by_unit: Dict[str, List[ExtractedQuestion]] = defaultdict(list)
    for q in questions:
        by_unit[q.unit_number or "General"].append(q)

    all_clusters: List[Tuple[str, str, List[ExtractedQuestion], float]] = []
    ai_units_succeeded = 0
    errors: List[str] = []

    for unit_idx, (unit_name, unit_qs) in enumerate(sorted(by_unit.items())):
        if len(unit_qs) == 1:
            q_only = unit_qs[0]
            all_clusters.append(
                (str(uuid.uuid4()), heuristic_topic_title([q_only]), [q_only], 1.0)
            )
            continue

        if unit_idx > 0:
            time.sleep(0.35)

        grouped, err = _call_groq_group_unit(subject_name, unit_name, unit_qs, api_key)
        if not grouped:
            if err:
                errors.append(f"{unit_name}: {err}")
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
            all_clusters.append((str(uuid.uuid4()), clean_title, member_qs, 0.94))

        for idx, q in enumerate(unit_qs):
            if idx not in seen_local_ids:
                all_clusters.append(
                    (str(uuid.uuid4()), heuristic_topic_title([q]), [q], 1.0)
                )

    if ai_units_succeeded == 0:
        return None, "; ".join(errors) if errors else "Groq grouping returned no groups."

    all_clusters.sort(
        key=lambda item: (
            len({q.paper_id for q in item[2]}),
            len(item[2]),
            item[3],
        ),
        reverse=True,
    )
    warn_msg = "; ".join(errors) if errors else None
    return all_clusters, warn_msg
