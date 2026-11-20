import json
import os
import re
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import httpx

from _lib.schemas import ExtractedQuestion
from _lib.clustering import chronological_paper_key, cluster_questions
from _lib.labeling import heuristic_topic_title


def _extract_json_object(raw: str) -> Optional[dict]:
    cleaned = raw.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S | re.I).strip()
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


def _discover_groq_models(api_key: str) -> List[str]:
    env_model = os.getenv("GROQ_GROUP_MODEL", "").strip()
    try:
        resp = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=6.0,
        )
        if resp.status_code == 200:
            items = resp.json().get("data", [])
            ids = [
                m["id"]
                for m in items
                if isinstance(m, dict)
                and m.get("id")
                and not any(
                    bad in m["id"].lower()
                    for bad in ("whisper", "tts", "guard", "safeguard", "embed", "audio", "orpheus", "vision")
                )
            ]

            def rank(mid: str) -> tuple:
                low = mid.lower()
                if "llama-4" in low or "70b" in low or "llama-3.3" in low:
                    return (0, mid)
                if "gpt-oss" in low or "kimi" in low or "gemma" in low or "8b" in low:
                    return (1, mid)
                if "qwen" in low or "deepseek" in low:
                    return (2, mid)
                return (3, mid)

            ids.sort(key=rank)
            if env_model and env_model in ids:
                ids.remove(env_model)
                ids.insert(0, env_model)
            if ids:
                return ids[:3]
    except Exception:
        pass
    return [m for m in [env_model, "llama-3.3-70b-versatile", "openai/gpt-oss-20b"] if m]


def _call_groq_group_unit(
    subject_name: str,
    unit_name: str,
    unit_questions: List[ExtractedQuestion],
    api_key: str,
    models_to_try: List[str],
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
        "3. If a question asks 'Draw architecture of X and compare CPLD and FPGA', put it in the 'CPLD vs FPGA Comparison' group.\n"
        "4. Group 'Draw block diagram / architecture / CLB / Macrocell of X' together for the same device X.\n"
        "5. Keep genuinely distinct topics separate (e.g. FPGA Architecture != CPLD Architecture != CPLD vs FPGA Comparison).\n"
        f"6. Use ONLY the integer IDs 0 to {len(unit_questions) - 1}. Include EVERY ID exactly once.\n"
        "7. Give each group a crisp 3 to 6 word plain-English title.\n"
        'Return ONLY valid JSON: {"topics": [{"title": "...", "ids": [0, 3]}]}\n\n'
        f"Questions:\n{numbered_lines}"
    )

    last_err = None
    for model in models_to_try:
        use_json_mode = True
        for _ in range(2):
            try:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 3000,
                }
                if use_json_mode:
                    body["response_format"] = {"type": "json_object"}

                resp = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                    timeout=18.0,
                )
                if resp.status_code == 429:
                    last_err = f"Groq rate limit (429) on {model}"
                    time.sleep(1.0)
                    continue
                if resp.status_code == 400 and use_json_mode:
                    use_json_mode = False
                    continue
                if resp.status_code != 200:
                    last_err = f"Groq HTTP {resp.status_code} on {model}"
                    break

                raw_content = resp.json()["choices"][0]["message"]["content"]
                parsed = _extract_json_object(raw_content)
                if not parsed:
                    last_err = f"Groq non-JSON on {model}"
                    break

                topics = parsed.get("topics")
                if not isinstance(topics, list):
                    last_err = f"Groq missing topics on {model}"
                    break

                cleaned: List[Tuple[str, List[int]]] = []
                for t in topics:
                    if not isinstance(t, dict):
                        continue
                    title = str(t.get("title", "")).strip()
                    ids = t.get("ids", [])
                    if isinstance(ids, list):
                        valid_ids = [
                            int(i) for i in ids if isinstance(i, (int, str)) and str(i).isdigit()
                        ]
                        if valid_ids:
                            cleaned.append((title, valid_ids))

                if cleaned:
                    return cleaned, None
            except Exception as exc:
                last_err = f"Groq error on {model}: {exc}"
                break

    return None, last_err


def group_questions_with_ai(
    subject_name: str,
    questions: List[ExtractedQuestion],
    fallback_threshold: float = 0.56,
) -> Tuple[Optional[List[Tuple[str, str, List[ExtractedQuestion], float]]], Optional[str]]:
    raw_key = os.getenv("GROQ_API_KEY", "")
    api_key = raw_key.strip().strip('"').strip("'")
    if not api_key:
        return None, "GROQ_API_KEY is not set in Vercel Environment Variables."
    if not questions:
        return None, None

    models_to_try = _discover_groq_models(api_key)

    by_unit: Dict[str, List[ExtractedQuestion]] = defaultdict(list)
    for q in questions:
        by_unit[q.unit_number or "General"].append(q)

    def process_unit(item):
        unit_name, unit_qs = item
        if len(unit_qs) == 1:
            q_only = unit_qs[0]
            return [(str(uuid.uuid4()), heuristic_topic_title([q_only]), [q_only], 1.0)], True, None

        grouped, err = _call_groq_group_unit(subject_name, unit_name, unit_qs, api_key, models_to_try)
        if not grouped:
            fallback_res = [
                (cid, heuristic_topic_title(m_qs), m_qs, avg_sim)
                for cid, m_qs, avg_sim in cluster_questions(unit_qs, fallback_threshold)
            ]
            return fallback_res, False, f"{unit_name}: {err}" if err else None

        unit_clusters = []
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
            unit_clusters.append((str(uuid.uuid4()), clean_title, member_qs, 0.94))

        for idx, q in enumerate(unit_qs):
            if idx not in seen_local_ids:
                unit_clusters.append((str(uuid.uuid4()), heuristic_topic_title([q]), [q], 1.0))

        return unit_clusters, True, None

    all_clusters: List[Tuple[str, str, List[ExtractedQuestion], float]] = []
    ai_units_succeeded = 0
    errors: List[str] = []

    sorted_items = sorted(by_unit.items())
    # Run max 2 units concurrently to stay fast for Vercel while avoiding Groq 429 burst
    with ThreadPoolExecutor(max_workers=2) as pool:
        for unit_res, used_ai, err in pool.map(process_unit, sorted_items):
            all_clusters.extend(unit_res)
            if used_ai:
                ai_units_succeeded += 1
            if err:
                errors.append(err)

    if ai_units_succeeded == 0:
        return None, "; ".join(errors) if errors else "Groq grouping returned no groups."

    all_clusters.sort(
        key=lambda item: (len({q.paper_id for q in item[2]}), len(item[2]), item[3]),
        reverse=True,
    )
    return all_clusters, "; ".join(errors) if errors else None
