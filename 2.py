from pathlib import Path

FILES = {
    "backend/app/services/clustering.py": r'''
import math
import os
import re
import uuid
from typing import Dict, List, Set, Tuple
import httpx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models.schemas import ExtractedQuestion, PriorityTier

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None

_MODEL_INSTANCE = None

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "in", "of", "and", "or", "for", "to",
    "with", "explain", "discuss", "describe", "define", "state", "draw", "write",
    "note", "notes", "short", "detail", "details", "brief", "neat", "diagram",
    "suitable", "example", "examples", "types", "different", "need", "enlist",
    "various", "its", "how", "why", "role", "steps", "involved", "fundamental",
    "components", "characteristics", "following", "respect", "points", "give",
    "list", "typical", "between", "basis",
}

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalize_for_matching(text: str) -> str:
    """
    Normalizes exam command fluff & synonyms ONLY for similarity calculation.
    NEVER touches q.exact_text (which stays 100% verbatim).
    """
    s = text.lower().strip()

    # Map comparison synonyms to one canonical intent token
    s = re.sub(
        r"\b(?:compare\s+and\s+contrast|compare|distinguish\s+between|distinguish|"
        r"differentiate\s+between|differentiate|difference\s+between|differences\s+between|"
        r"contrast)\b",
        " _cmp_ ",
        s,
    )
    # Map architecture / diagram synonyms
    s = re.sub(
        r"\b(?:block\s+diagram|internal\s+architecture|architecture|constructional\s+diagram|"
        r"construction\s+and\s+working|construction)\b",
        " _arch_ ",
        s,
    )

    # Strip boilerplate command phrases that hide the real concept
    filler_patterns = [
        r"\bon\s+the\s+basis\s+of\s+features[,\s]+specifications\s+and\s+applications\b",
        r"\bwith\s+respect\s+to\s+the\s+following\s+points\b",
        r"\bwith\s+respect\s+to\b",
        r"\bwith\s+(?:a\s+)?(?:neat|suitable|labelled)?\s*diagram\b",
        r"\bwrite\s+(?:a\s+)?short\s+notes?\s+on\b",
        r"\b(?:draw\s+and\s+explain|list\s+and\s+explain|state\s+and\s+explain|identify\s+and\s+elaborate)\b",
        r"\b(?:explain|discuss|describe|define|draw|write|list|enlist|give|state|what\s+are|what\s+is)\b",
        r"\b(?:in\s+detail|in\s+brief|briefly|various|the\s+following|any\s+two\s+of\s+it)\b",
    ]
    for pat in filler_patterns:
        s = re.sub(pat, " ", s)

    s = re.sub(r"[^a-z0-9_\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or text.lower()


def chronological_paper_key(paper_name: str, question_number: str = "") -> tuple:
    low = paper_name.lower()
    year_m = re.search(r"\b(20\d{2})\b", low)
    year = int(year_m.group(1)) if year_m else 9999

    months_found = [
        MONTH_MAP[m[:3]]
        for m in re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", low)
    ]
    month = min(months_found) if months_found else 13

    q_m = re.match(r"Q\.(\d+)(?:\(([a-z]+)\))?", question_number, re.IGNORECASE)
    q_num = int(q_m.group(1)) if q_m else 99
    q_sub = (q_m.group(2) or "").lower() if q_m else ""

    return (year, month, low, q_num, q_sub)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def _encode_via_hf_api(texts: List[str], token: str) -> np.ndarray | None:
    endpoints = [
        "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction",
        "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-small-en-v1.5",
    ]
    headers = {"Authorization": f"Bearer {token}"}
    for url in endpoints:
        try:
            resp = httpx.post(
                url,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = np.asarray(resp.json(), dtype=np.float32)
                if data.ndim == 2 and data.shape[0] == len(texts):
                    return _l2_normalize(data)
        except Exception:
            continue
    return None


def _encode_tfidf_hybrid(texts: List[str]) -> np.ndarray:
    """High-precision TF-IDF word + char_wb hybrid for 512MB cloud tier."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
    )
    w_mat = word_vec.fit_transform(texts).toarray().astype(np.float32)
    c_mat = char_vec.fit_transform(texts).toarray().astype(np.float32)
    combined = np.hstack([w_mat * 1.4, c_mat * 0.6])
    return _l2_normalize(combined)


def get_embedding_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None and SentenceTransformer is not None:
        _MODEL_INSTANCE = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL_INSTANCE


def encode_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    normed = [normalize_for_matching(t) for t in texts]

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        hf_embs = _encode_via_hf_api(normed, hf_token)
        if hf_embs is not None:
            return hf_embs

    if SentenceTransformer is not None:
        model = get_embedding_model()
        embeddings = model.encode(
            normed, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    return _encode_tfidf_hybrid(normed)


def _tech_tokens(text: str) -> Set[str]:
    normed = normalize_for_matching(text)
    tokens = set(re.findall(r"[a-z0-9_]{2,}", normed))
    # Simple plural stem trim ('tools' -> 'tool', 'applications' -> 'application')
    stemmed = {t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens}
    return stemmed - STOPWORDS


def compute_priority_tier(unique_papers: int, total_papers: int) -> PriorityTier:
    if total_papers >= 6:
        if unique_papers >= 5:
            return PriorityTier.HIGH
        if unique_papers >= 3:
            return PriorityTier.MEDIUM
        return PriorityTier.LOW
    if total_papers == 1:
        return PriorityTier.LOW

    high_cutoff = max(2, math.ceil(total_papers * 0.65))
    med_cutoff = max(2, math.ceil(total_papers * 0.4))
    if unique_papers >= high_cutoff:
        return PriorityTier.HIGH
    if unique_papers >= med_cutoff and total_papers >= 3:
        return PriorityTier.MEDIUM
    return PriorityTier.LOW


def cluster_questions(
    questions: List[ExtractedQuestion],
    embeddings: np.ndarray,
    threshold: float,
) -> List[Tuple[str, List[ExtractedQuestion], float]]:
    n = len(questions)
    if n == 0:
        return []
    if n == 1:
        return [(str(uuid.uuid4()), questions, 1.0)]

    if faiss is not None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        sims, idxs = index.search(embeddings, n)
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        row_indices = np.arange(n)[:, None]
        sim_matrix[row_indices, idxs] = sims
    else:
        sim_matrix = embeddings @ embeddings.T

    sim_matrix = np.clip((sim_matrix + sim_matrix.T) / 2.0, -1.0, 1.0)

    token_sets = [_tech_tokens(q.exact_text) for q in questions]
    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

    for i in range(n):
        for j in range(i + 1, n):
            ui, uj = questions[i].unit_number, questions[j].unit_number
            if ui and uj and ui != uj:
                dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                continue

            ti, tj = token_sets[i], token_sets[j]
            union = ti | tj
            jacc = (len(ti & tj) / len(union)) if union else 0.0
            hybrid_sim = 0.72 * float(sim_matrix[i, j]) + 0.28 * jacc
            dist = float(np.clip(1.0 - hybrid_sim, 0.0, 2.0))
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - threshold,
    )
    labels = clustering.fit_predict(dist_matrix)

    grouped: Dict[int, List[int]] = {}
    for q_idx, lbl in enumerate(labels):
        grouped.setdefault(int(lbl), []).append(q_idx)

    results: List[Tuple[str, List[ExtractedQuestion], float]] = []
    for member_indices in grouped.values():
        member_qs = [questions[i] for i in member_indices]
        member_qs.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))

        if len(member_indices) > 1:
            sub = sim_matrix[np.ix_(member_indices, member_indices)]
            triu = sub[np.triu_indices(len(member_indices), k=1)]
            avg_sim = float(np.mean(triu))
        else:
            avg_sim = 1.0

        results.append((str(uuid.uuid4()), member_qs, round(avg_sim, 3)))

    results.sort(
        key=lambda item: (
            len({q.paper_id for q in item[1]}),
            len(item[1]),
            item[2],
        ),
        reverse=True,
    )
    return results
''',

    "backend/app/services/ai_grouper.py": r'''
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
''',

    "backend/app/services/question_parser.py": r'''
import re
import uuid
from typing import List, Optional, Tuple
from app.models.schemas import ExtractedQuestion


MAIN_Q_RE = re.compile(
    r"^\s*(?:(?:Q(?:ue(?:stion)?)?\.?\s*(\d{1,2}))\s*[\.\)\]:\-]?|(\d{1,2})\s*[\)\]]\s*(?=\(?[a-h]\s*[\)\.\]]))\s*(.*)$",
    re.IGNORECASE,
)

SUB_Q_RE = re.compile(
    r"^\s*(?:\(([a-h])\)|\[([a-h])\]|([a-h])\s*[\)\.\]])\s+(.*)$",
    re.IGNORECASE,
)

OR_LINE_RE = re.compile(r"^\s*(?:OR|O\.?R\.?|O\s+R)\s*$", re.IGNORECASE)

MARKS_END_RE = re.compile(
    r"(?:\[\s*(\d{1,2})\s*(?:M|Marks?)?\s*\]|\(\s*(\d{1,2})\s*(?:M|Marks?)?\s*\)|[-–]\s*(\d{1,2})\s*Marks?)(?:\s+[A-Z]{1,2})?\s*$",
    re.IGNORECASE,
)

# Catches marks inline before sub-items or mid-question: "Draw block diagram of CPLD. [9] Explain..."
MARKS_INLINE_RE = re.compile(
    r"\s*\[\s*(\d{1,2})\s*(?:M|Marks?)?\s*\]\s*(?=(?:i|ii|1)\s*[\)\.]|[A-Z])",
)

INSTRUCTION_PREFIXES = (
    "attempt any", "answer any", "solve any", "all questions",
    "neat diagrams", "figures to the right", "assume suitable",
    "use of", "draw neat diagram", "note:", "n.b.", "section ", "part ",
    "instructions to",
)


def infer_sppu_unit(question_number: str, exam_type: str) -> Optional[str]:
    m = re.match(r"Q\.(\d{1,2})", question_number, re.IGNORECASE)
    if not m:
        return None
    q_num = int(m.group(1))
    if exam_type == "Insem":
        if q_num in (1, 2):
            return "Unit 1"
        if q_num in (3, 4):
            return "Unit 2"
        return None
    if q_num in (1, 2):
        return "Unit 3"
    if q_num in (3, 4):
        return "Unit 4"
    if q_num in (5, 6):
        return "Unit 5"
    if q_num in (7, 8):
        return "Unit 6"
    return None


def _extract_marks(text: str) -> Tuple[str, Optional[int]]:
    s = text.strip()
    m = MARKS_END_RE.search(s)
    if m:
        raw_num = m.group(1) or m.group(2) or m.group(3)
        val = int(raw_num)
        if val <= 30:
            cleaned = s[: m.start()].strip()
            # Also strip any second inline [9] if marks appeared twice
            cleaned = MARKS_INLINE_RE.sub(" ", cleaned)
            return cleaned, val

    m2 = MARKS_INLINE_RE.search(s)
    if m2:
        val = int(m2.group(1))
        if val <= 30:
            cleaned = (s[: m2.start()] + " " + s[m2.end():]).strip()
            return cleaned, val

    return s, None


def _clean_verbatim_whitespace(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+:", ":", text)
    text = re.sub(r":{2,}", ":", text)
    return text


def _looks_like_header_or_metadata(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    lower = s.lower()
    if any(lower.startswith(p) for p in INSTRUCTION_PREFIXES):
        return True
    header_keywords = [
        "semester", "insem", "endsem", "university", "faculty of",
        "pattern", "course code", "subject :", "subject:", "branch :",
        "credit", "savitribai phule", "sppu", "examination", "time :",
        "max. marks", "total no. of", "seat no",
    ]
    if len(s) < 100 and any(k in lower for k in header_keywords):
        return True
    return False


def parse_questions(
    cleaned_text: str, paper_id: str, paper_name: str, exam_type: str = "Endsem"
) -> List[ExtractedQuestion]:
    lines = [ln.strip() for ln in cleaned_text.splitlines()]
    questions: List[ExtractedQuestion] = []

    current_main_q: Optional[str] = None
    current_q_label: Optional[str] = None
    current_buffer: List[str] = []
    started_questions = False

    def flush_current():
        nonlocal current_q_label, current_buffer
        if not current_q_label or not current_buffer:
            current_buffer = []
            return
        raw_joined = " ".join(current_buffer).strip()
        text_no_marks, marks = _extract_marks(raw_joined)
        verbatim = _clean_verbatim_whitespace(text_no_marks)
        lower_v = verbatim.lower()
        if len(verbatim) >= 12 and not any(lower_v.startswith(p) for p in INSTRUCTION_PREFIXES):
            questions.append(
                ExtractedQuestion(
                    id=str(uuid.uuid4()),
                    paper_id=paper_id,
                    paper_name=paper_name,
                    question_number=current_q_label,
                    unit_number=infer_sppu_unit(current_q_label, exam_type),
                    marks=marks,
                    exact_text=verbatim,
                )
            )
        current_buffer = []

    for line in lines:
        if not line:
            continue

        if OR_LINE_RE.match(line):
            flush_current()
            current_q_label = None
            continue

        main_match = MAIN_Q_RE.match(line)
        if main_match:
            started_questions = True
            flush_current()
            q_num = main_match.group(1) or main_match.group(2)
            current_main_q = f"Q.{q_num}"
            rest = (main_match.group(3) or "").strip()

            if rest:
                sub_in_rest = SUB_Q_RE.match(rest)
                if sub_in_rest:
                    sub_letter = (
                        sub_in_rest.group(1)
                        or sub_in_rest.group(2)
                        or sub_in_rest.group(3)
                    ).lower()
                    current_q_label = f"{current_main_q}({sub_letter})"
                    current_buffer.append(sub_in_rest.group(4).strip())
                else:
                    current_q_label = current_main_q
                    current_buffer.append(rest)
            else:
                current_q_label = current_main_q
            continue

        if not started_questions and _looks_like_header_or_metadata(line):
            continue

        sub_match = SUB_Q_RE.match(line)
        if sub_match and (current_main_q or started_questions):
            flush_current()
            sub_letter = (
                sub_match.group(1) or sub_match.group(2) or sub_match.group(3)
            ).lower()
            prefix = current_main_q or "Q"
            current_q_label = f"{prefix}({sub_letter})"
            current_buffer.append(sub_match.group(4).strip())
            continue

        if current_q_label and not _looks_like_header_or_metadata(line):
            current_buffer.append(line)

    flush_current()

    if len(questions) == 0:
        questions = _fallback_numeric_parse(lines, paper_id, paper_name, exam_type)

    return questions


def _fallback_numeric_parse(
    lines: List[str], paper_id: str, paper_name: str, exam_type: str
) -> List[ExtractedQuestion]:
    num_re = re.compile(r"^\s*(\d{1,2})\s*[\.\)\]]\s+(.+)$")
    results: List[ExtractedQuestion] = []
    cur_num: Optional[str] = None
    buf: List[str] = []

    def flush():
        if cur_num and buf:
            joined = " ".join(buf)
            txt, marks = _extract_marks(joined)
            verbatim = _clean_verbatim_whitespace(txt)
            lower_v = verbatim.lower()
            if len(verbatim) >= 15 and not any(lower_v.startswith(p) for p in INSTRUCTION_PREFIXES):
                q_lbl = f"Q.{cur_num}"
                results.append(
                    ExtractedQuestion(
                        id=str(uuid.uuid4()),
                        paper_id=paper_id,
                        paper_name=paper_name,
                        question_number=q_lbl,
                        unit_number=infer_sppu_unit(q_lbl, exam_type),
                        marks=marks,
                        exact_text=verbatim,
                    )
                )

    for line in lines:
        if not line or _looks_like_header_or_metadata(line):
            continue
        m = num_re.match(line)
        if m:
            flush()
            cur_num = m.group(1)
            buf = [m.group(2).strip()]
        elif cur_num:
            buf.append(line)
    flush()
    return results
''',
}

# Tune default normalized threshold from 0.72 / 0.68 down to 0.56 for high-recall grouping
for target_file, old, new in [
    ("backend/app/main.py", "similarity_threshold: float = Form(0.68)", "similarity_threshold: float = Form(0.56)"),
    ("frontend/src/components/UploadPanel.tsx", "threshold: 0.72,", "threshold: 0.56,"),
]:
    p = Path(target_file)
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        if old in txt:
            p.write_text(txt.replace(old, new), encoding="utf-8")
            print(f"Updated threshold in {p}")

gi = Path(".gitignore")
if gi.exists():
    g = gi.read_text(encoding="utf-8")
    if "upgrade_smart_grouping.py" not in g:
        gi.write_text(g.rstrip() + "\nupgrade_smart_grouping.py\n", encoding="utf-8")

for rel, content in FILES.items():
    p = Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"Updated {p}")

print("\nSmart exam normalization + AI synonym grouping applied!")