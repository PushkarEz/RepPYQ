from pathlib import Path

FILES = {
    "backend/app/services/labeling.py": r'''
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
''',

    "backend/app/services/ai_grouper.py": r'''
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
''',

    "backend/app/main.py": r'''
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np

from app.models.schemas import (
    AnalysisResponse,
    ExtractedQuestion,
    MergeClustersRequest,
    PaperInfo,
    QuestionCluster,
    ReclusterRequest,
    SplitQuestionRequest,
    UpdateLabelsRequest,
)
from app.services.ai_grouper import group_questions_with_ai
from app.services.clustering import (
    chronological_paper_key,
    cluster_questions,
    compute_priority_tier,
    encode_texts,
)
from app.services.labeling import generate_cluster_title, match_syllabus_unit
from app.services.pdf_extractor import detect_paper_label, extract_pdf_text
from app.services.pdf_exporter import build_revision_pdf
from app.services.question_parser import parse_questions

load_dotenv()

app = FastAPI(title="RepPYQ API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATS_FILE = Path(os.getenv("STATS_FILE_PATH", "app_stats.json"))


def _read_views() -> int:
    try:
        if STATS_FILE.exists():
            data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
            return int(data.get("views", 0))
    except Exception:
        pass
    return 0


def _write_views(count: int) -> None:
    try:
        STATS_FILE.write_text(json.dumps({"views": count}), encoding="utf-8")
    except Exception:
        pass


@dataclass
class AnalysisSession:
    analysis_id: str
    subject_name: str
    papers: List[PaperInfo]
    questions: List[ExtractedQuestion]
    embeddings: np.ndarray
    syllabus_units: List[str]
    unit_embeddings: Optional[np.ndarray]
    similarity_threshold: float
    grouping_mode: str = "Semantic Similarity"
    clusters: List[QuestionCluster] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


SESSIONS: Dict[str, AnalysisSession] = {}


def _resolve_cluster_unit(
    q_list: List[ExtractedQuestion],
    syllabus_units: List[str],
    unit_embeddings: Optional[np.ndarray],
) -> Optional[str]:
    sppu_units = [q.unit_number for q in q_list if q.unit_number]
    base_unit = max(set(sppu_units), key=sppu_units.count) if sppu_units else None

    if syllabus_units and unit_embeddings is not None:
        matched = match_syllabus_unit(q_list, syllabus_units, unit_embeddings)
        if base_unit and matched:
            return f"{base_unit} · {matched}"
        return matched or base_unit
    return base_unit


def _build_cluster_objects(
    session: AnalysisSession, use_ai: bool = True
) -> List[QuestionCluster]:
    total_papers = len(session.papers)
    cluster_objs: List[QuestionCluster] = []

    if use_ai:
        ai_grouped, ai_err = group_questions_with_ai(
            session.subject_name, session.questions, session.similarity_threshold
        )
        if ai_grouped is not None:
            session.grouping_mode = "AI Concept Grouping"
            if ai_err:
                session.warnings.append(f"Partial AI fallback: {ai_err}")
            for cid, title, q_list, conf in ai_grouped:
                unique_papers = len({q.paper_id for q in q_list})
                unit_lbl = _resolve_cluster_unit(
                    q_list, session.syllabus_units, session.unit_embeddings
                )
                cluster_objs.append(
                    QuestionCluster(
                        cluster_id=cid,
                        topic_title=title,
                        unit_label=unit_lbl,
                        priority=compute_priority_tier(unique_papers, total_papers),
                        unique_paper_count=unique_papers,
                        total_papers=total_papers,
                        total_occurrences=len(q_list),
                        avg_similarity=conf,
                        questions=q_list,
                    )
                )
            return cluster_objs
        elif ai_err:
            session.warnings.append(f"AI note: {ai_err} (used fallback matching).")

    session.grouping_mode = "Semantic Similarity"
    raw_clusters = cluster_questions(
        session.questions, session.embeddings, session.similarity_threshold
    )
    for cid, q_list, avg_sim in raw_clusters:
        unique_papers = len({q.paper_id for q in q_list})
        title = generate_cluster_title(q_list)
        unit_lbl = _resolve_cluster_unit(
            q_list, session.syllabus_units, session.unit_embeddings
        )
        cluster_objs.append(
            QuestionCluster(
                cluster_id=cid,
                topic_title=title,
                unit_label=unit_lbl,
                priority=compute_priority_tier(unique_papers, total_papers),
                unique_paper_count=unique_papers,
                total_papers=total_papers,
                total_occurrences=len(q_list),
                avg_similarity=avg_sim,
                questions=q_list,
            )
        )
    return cluster_objs


def _to_response(session: AnalysisSession) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_id=session.analysis_id,
        subject_name=session.subject_name,
        similarity_threshold=session.similarity_threshold,
        grouping_mode=session.grouping_mode,
        total_papers=len(session.papers),
        total_questions=len(session.questions),
        papers=session.papers,
        clusters=session.clusters,
        warnings=session.warnings,
    )


@app.get("/")
def root():
    return {"app": "RepPYQ API", "status": "running"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/stats/visit")
def record_visit():
    views = _read_views() + 1
    _write_views(views)
    return {"views": views}


@app.get("/api/stats")
def get_stats():
    return {"views": _read_views()}


@app.post("/api/detect-labels")
async def detect_labels(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        raw = await f.read()
        extracted = extract_pdf_text(raw, f.filename or "")
        label = detect_paper_label(f.filename or "paper.pdf", extracted.raw_first_page)
        results.append({
            "filename": f.filename,
            "suggested_label": label,
            "exam_type": extracted.exam_type,
            "page_count": extracted.page_count,
            "used_ocr": extracted.used_ocr,
            "ocr_warning": extracted.ocr_warning,
        })
    return {"papers": results}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_pdfs(
    files: List[UploadFile] = File(...),
    paper_labels_json: str = Form("[]"),
    subject_name: str = Form("Subject Revision Notes"),
    similarity_threshold: float = Form(0.56),
    syllabus_text: str = Form(""),
):
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least 1 PDF.")

    try:
        provided_labels: List[str] = json.loads(paper_labels_json)
    except json.JSONDecodeError:
        provided_labels = []

    papers: List[PaperInfo] = []
    all_questions: List[ExtractedQuestion] = []
    warnings: List[str] = []

    for idx, f in enumerate(files):
        raw_bytes = await f.read()
        paper_id = str(uuid.uuid4())
        extracted = extract_pdf_text(raw_bytes, f.filename or "")
        exam_type = extracted.exam_type

        custom_label = (
            provided_labels[idx].strip()
            if idx < len(provided_labels) and provided_labels[idx].strip()
            else detect_paper_label(f.filename or "paper.pdf", extracted.raw_first_page)
        )

        qs = parse_questions(extracted.cleaned_text, paper_id, custom_label, exam_type)
        if extracted.ocr_warning:
            warnings.append(f"{custom_label}: {extracted.ocr_warning}")
        if len(qs) == 0:
            warnings.append(f"{custom_label}: No questions matched standard formats.")

        papers.append(
            PaperInfo(
                paper_id=paper_id,
                filename=f.filename or f"paper_{idx + 1}.pdf",
                paper_name=custom_label,
                exam_type=exam_type,
                page_count=extracted.page_count,
                used_ocr=extracted.used_ocr,
                ocr_warning=extracted.ocr_warning,
                extracted_count=len(qs),
            )
        )
        all_questions.extend(qs)

    if not all_questions:
        raise HTTPException(
            status_code=422,
            detail="Could not extract questions from the uploaded PDFs. Check if PDFs are scanned without OCR.",
        )

    papers.sort(key=lambda p: chronological_paper_key(p.paper_name))
    all_questions.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))

    embeddings = encode_texts([q.exact_text for q in all_questions])
    syllabus_units = [u.strip() for u in syllabus_text.splitlines() if u.strip()]
    unit_embeddings = encode_texts(syllabus_units) if syllabus_units else None

    analysis_id = str(uuid.uuid4())
    session = AnalysisSession(
        analysis_id=analysis_id,
        subject_name=subject_name.strip() or "Subject Revision Notes",
        papers=papers,
        questions=all_questions,
        embeddings=embeddings,
        syllabus_units=syllabus_units,
        unit_embeddings=unit_embeddings,
        similarity_threshold=similarity_threshold,
        warnings=warnings,
    )
    session.clusters = _build_cluster_objects(session, use_ai=True)
    SESSIONS[analysis_id] = session
    return _to_response(session)


@app.post("/api/analysis/{analysis_id}/recluster", response_model=AnalysisResponse)
def recluster(analysis_id: str, payload: ReclusterRequest):
    session = SESSIONS.get(analysis_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    session.similarity_threshold = payload.similarity_threshold
    session.clusters = _build_cluster_objects(session, use_ai=False)
    return _to_response(session)


@app.post("/api/analysis/{analysis_id}/merge", response_model=AnalysisResponse)
def merge_clusters(analysis_id: str, payload: MergeClustersRequest):
    session = SESSIONS.get(analysis_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    target_ids = set(payload.cluster_ids)
    to_merge = [c for c in session.clusters if c.cluster_id in target_ids]
    if len(to_merge) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 valid clusters to merge.")

    remaining = [c for c in session.clusters if c.cluster_id not in target_ids]
    merged_qs: List[ExtractedQuestion] = []
    for cl in to_merge:
        merged_qs.extend(cl.questions)
    merged_qs.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))

    unique_papers = len({q.paper_id for q in merged_qs})
    new_title = (
        payload.custom_title.strip()
        if payload.custom_title and payload.custom_title.strip()
        else to_merge[0].topic_title
    )

    merged_cluster = QuestionCluster(
        cluster_id=str(uuid.uuid4()),
        topic_title=new_title,
        unit_label=to_merge[0].unit_label,
        priority=compute_priority_tier(unique_papers, len(session.papers)),
        unique_paper_count=unique_papers,
        total_papers=len(session.papers),
        total_occurrences=len(merged_qs),
        avg_similarity=session.similarity_threshold,
        questions=merged_qs,
    )

    remaining.append(merged_cluster)
    remaining.sort(key=lambda c: (c.unique_paper_count, c.total_occurrences), reverse=True)
    session.clusters = remaining
    return _to_response(session)


@app.post("/api/analysis/{analysis_id}/split", response_model=AnalysisResponse)
def split_question(analysis_id: str, payload: SplitQuestionRequest):
    session = SESSIONS.get(analysis_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    target_cluster = next(
        (c for c in session.clusters if c.cluster_id == payload.cluster_id), None
    )
    if not target_cluster:
        raise HTTPException(status_code=404, detail="Cluster not found.")
    if len(target_cluster.questions) <= 1:
        raise HTTPException(status_code=400, detail="Cannot split a single-question cluster.")

    split_q = next((q for q in target_cluster.questions if q.id == payload.question_id), None)
    if not split_q:
        raise HTTPException(status_code=404, detail="Question not found in cluster.")

    target_cluster.questions = [q for q in target_cluster.questions if q.id != payload.question_id]
    uniq_orig = len({q.paper_id for q in target_cluster.questions})
    target_cluster.unique_paper_count = uniq_orig
    target_cluster.total_occurrences = len(target_cluster.questions)
    target_cluster.priority = compute_priority_tier(uniq_orig, len(session.papers))

    new_cluster = QuestionCluster(
        cluster_id=str(uuid.uuid4()),
        topic_title=generate_cluster_title([split_q]),
        unit_label=_resolve_cluster_unit(
            [split_q], session.syllabus_units, session.unit_embeddings
        ),
        priority=compute_priority_tier(1, len(session.papers)),
        unique_paper_count=1,
        total_papers=len(session.papers),
        total_occurrences=1,
        avg_similarity=1.0,
        questions=[split_q],
    )

    session.clusters.append(new_cluster)
    session.clusters.sort(key=lambda c: (c.unique_paper_count, c.total_occurrences), reverse=True)
    return _to_response(session)


@app.patch("/api/analysis/{analysis_id}/labels", response_model=AnalysisResponse)
def update_labels(analysis_id: str, payload: UpdateLabelsRequest):
    session = SESSIONS.get(analysis_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    if payload.subject_name and payload.subject_name.strip():
        session.subject_name = payload.subject_name.strip()

    for p in session.papers:
        if p.paper_id in payload.paper_labels:
            p.paper_name = payload.paper_labels[p.paper_id].strip() or p.paper_name

    name_map = {p.paper_id: p.paper_name for p in session.papers}
    for q in session.questions:
        if q.paper_id in name_map:
            q.paper_name = name_map[q.paper_id]
    for cl in session.clusters:
        for q in cl.questions:
            if q.paper_id in name_map:
                q.paper_name = name_map[q.paper_id]
        cl.questions.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))

    session.papers.sort(key=lambda p: chronological_paper_key(p.paper_name))
    return _to_response(session)


@app.get("/api/analysis/{analysis_id}/export-pdf")
def export_pdf(analysis_id: str):
    session = SESSIONS.get(analysis_id)
    if not session:
        raise HTTPException(status_code=404, detail="Analysis session not found.")

    pdf_bytes = build_revision_pdf(_to_response(session))
    safe_name = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_" for c in session.subject_name
    ).strip() or "RepPYQ_Revision_Notes"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_RepPYQ_Notes.pdf"'},
    )
''',

    "backend/app/services/pdf_exporter.py": r'''
import io
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.models.schemas import AnalysisResponse, PriorityTier, QuestionCluster
from app.services.clustering import chronological_paper_key


class CleanBookCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(num_pages)
            super().showPage()
        super().save()

    def _draw_footer(self, page_count: int):
        self.saveState()
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(18 * mm, 10.5 * mm, "RepPYQ by Pushkar Shinde · pushkarez.vercel.app")
        self.drawRightString(
            A4[0] - 18 * mm, 10.5 * mm, f"Page {self._pageNumber} of {page_count}"
        )
        self.restoreState()


def _unit_sort_key(unit: str) -> tuple:
    if unit.startswith("Unit "):
        try:
            num = int(unit.split()[1])
            return (0, num, unit)
        except Exception:
            pass
    return (1, 99, unit)


def build_revision_pdf(analysis: AnalysisResponse) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=21 * mm,
        title=f"{analysis.subject_name} - RepPYQ Revision Notes",
    )

    styles = getSampleStyleSheet()
    ink = colors.HexColor("#0F172A")
    body = colors.HexColor("#1E293B")
    slate = colors.HexColor("#475569")
    muted = colors.HexColor("#64748B")
    border_soft = colors.HexColor("#E2E8F0")
    card_bg = colors.HexColor("#F8FAFC")
    accent = colors.HexColor("#1E3A8A")
    high_col = colors.HexColor("#BE123C")
    med_col = colors.HexColor("#0369A1")

    title_style = ParagraphStyle(
        "TitleClean", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=22, leading=27, textColor=ink, spaceAfter=5,
    )
    meta_style = ParagraphStyle(
        "MetaClean", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=14.5, textColor=slate, spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "H2Clean", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=12.5, leading=16, textColor=ink, spaceBefore=10, spaceAfter=7,
    )
    unit_title_style = ParagraphStyle(
        "UnitTitle", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=16, leading=20, textColor=ink,
    )
    unit_sub_style = ParagraphStyle(
        "UnitSub", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9, leading=12, textColor=muted, alignment=2,
    )
    topic_style = ParagraphStyle(
        "TopicClean", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=11.2, leading=15, textColor=ink, spaceAfter=2,
    )
    rep_high_style = ParagraphStyle(
        "RepHigh", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=11, textColor=high_col, spaceAfter=2,
    )
    rep_med_style = ParagraphStyle(
        "RepMed", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=11, textColor=med_col, spaceAfter=2,
    )
    rep_low_style = ParagraphStyle(
        "RepLow", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=11, textColor=muted, spaceAfter=2,
    )
    cov_style = ParagraphStyle(
        "CovStrip", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7.8, leading=11, textColor=muted, spaceAfter=6,
    )
    q_meta_style = ParagraphStyle(
        "QMeta", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.3, leading=11, textColor=slate, spaceAfter=2.5,
    )
    q_text_style = ParagraphStyle(
        "QBody", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=body,
    )
    once_hdr_style = ParagraphStyle(
        "OnceHdr", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=10, leading=14, textColor=slate, spaceBefore=8, spaceAfter=6,
    )
    once_item_style = ParagraphStyle(
        "OnceItem", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.8, leading=12.5, textColor=slate, leftIndent=6, spaceAfter=5,
    )

    story = []

    sorted_papers = sorted(analysis.papers, key=lambda p: chronological_paper_key(p.paper_name))
    story.append(Paragraph(escape(analysis.subject_name), title_style))
    paper_list = "  ·  ".join(f"{p.paper_name} ({p.exam_type})" for p in sorted_papers)
    story.append(
        Paragraph(
            f"<b>Papers Analyzed ({analysis.total_papers}):</b> {escape(paper_list)}<br/>"
            f"<b>Exact Questions:</b> {analysis.total_questions}   ·   "
            f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y')}",
            meta_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=ink, spaceAfter=14))

    unit_groups: Dict[str, List[QuestionCluster]] = defaultdict(list)
    for cl in analysis.clusters:
        u = (cl.unit_label or "General / Unmapped").split("·")[0].strip()
        unit_groups[u].append(cl)

    sorted_units = sorted(unit_groups.keys(), key=_unit_sort_key)

    story.append(Paragraph("Unit-Wise Study Plan", h2_style))
    overview_rows = [["Unit", "Core Repeated Topics (2+)", "Asked Once (Optional)"]]
    for u in sorted_units:
        cls = unit_groups[u]
        rep = sum(1 for c in cls if c.unique_paper_count >= 2)
        once = len(cls) - rep
        overview_rows.append([u, str(rep), str(once)])

    ov_table = Table(overview_rows, colWidths=[64 * mm, 58 * mm, 52 * mm])
    ov_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ink),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, card_bg]),
            ("GRID", (0, 0), (-1, -1), 0.5, border_soft),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )
    story.append(ov_table)
    story.append(Spacer(1, 12))

    repeated_all = [c for c in analysis.clusters if c.unique_paper_count >= 2]
    if repeated_all:
        story.append(Paragraph("Top Repeated Concepts (Revise First)", h2_style))
        chk_rows = [["#", "Unit", "Topic", "Frequency"]]
        for i, cl in enumerate(repeated_all[:18], start=1):
            u_short = (cl.unit_label or "-").split("·")[0].strip()
            chk_rows.append([
                str(i),
                u_short,
                Paragraph(escape(cl.topic_title), ParagraphStyle("ChkT", fontSize=8.8, leading=12)),
                f"{cl.unique_paper_count} / {cl.total_papers} papers",
            ])
        chk_table = Table(chk_rows, colWidths=[10 * mm, 24 * mm, 108 * mm, 32 * mm])
        chk_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, border_soft),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(chk_table)

    for unit_name in sorted_units:
        story.append(PageBreak())

        unit_clusters = sorted(
            unit_groups[unit_name],
            key=lambda c: (c.unique_paper_count, c.total_occurrences),
            reverse=True,
        )
        rep_clusters = [c for c in unit_clusters if c.unique_paper_count >= 2]
        once_clusters = [c for c in unit_clusters if c.unique_paper_count < 2]

        main_clusters = rep_clusters if rep_clusters else once_clusters
        bonus_clusters = once_clusters if rep_clusters else []

        hdr_table = Table(
            [[
                Paragraph(escape(unit_name.upper()), unit_title_style),
                Paragraph(
                    f"{len(main_clusters)} Core Topics  ·  {len(bonus_clusters)} Asked Once",
                    unit_sub_style,
                ),
            ]],
            colWidths=[96 * mm, 78 * mm],
        )
        hdr_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 1.8, accent),
            ])
        )
        story.append(hdr_table)
        story.append(Spacer(1, 14))

        for idx, cl in enumerate(main_clusters, start=1):
            if cl.priority == PriorityTier.HIGH:
                b_style = rep_high_style
                tag = "MUST PREPARE"
                bar_col = high_col
            elif cl.priority == PriorityTier.MEDIUM:
                b_style = rep_med_style
                tag = "IMPORTANT"
                bar_col = med_col
            else:
                b_style = rep_low_style
                tag = "REPEATED" if cl.unique_paper_count >= 2 else "ASKED ONCE"
                bar_col = colors.HexColor("#94A3B8")

            topic_p = Paragraph(f"{idx}. {escape(cl.topic_title)}", topic_style)
            badge_p = Paragraph(
                f"{tag}   ·   Appeared in {cl.unique_paper_count} of {cl.total_papers} papers",
                b_style,
            )

            present_ids = {q.paper_id for q in cl.questions}
            cov_parts = [
                f"<b><font color='#0F172A'>[+] {escape(p.paper_name)}</font></b>"
                if p.paper_id in present_ids
                else f"<font color='#94A3B8'>[-] {escape(p.paper_name)}</font>"
                for p in sorted_papers
            ]
            cov_p = Paragraph("Coverage: " + "  ·  ".join(cov_parts), cov_style)

            ordered_qs = sorted(
                cl.questions,
                key=lambda q: chronological_paper_key(q.paper_name, q.question_number),
            )

            q_cards: List[Table] = []
            for q in ordered_qs:
                marks_str = f"   ·   {q.marks} Marks" if q.marks is not None else ""
                meta_p = Paragraph(
                    f"{escape(q.paper_name)}   ·   {escape(q.question_number)}{marks_str}",
                    q_meta_style,
                )
                body_p = Paragraph(escape(q.exact_text), q_text_style)

                card = Table([[meta_p], [body_p]], colWidths=[174 * mm])
                card.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                        ("LINEBEFORE", (0, 0), (0, -1), 2.2, bar_col),
                        ("BOX", (0, 0), (-1, -1), 0.4, border_soft),
                        ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, 0), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                        ("TOPPADDING", (0, 1), (-1, 1), 1),
                        ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
                    ])
                )
                q_cards.append(card)

            first_group = [topic_p, badge_p, cov_p]
            if q_cards:
                first_group.append(q_cards[0])
            story.append(KeepTogether(first_group))

            for extra_card in q_cards[1:]:
                story.append(Spacer(1, 5))
                story.append(extra_card)

            story.append(Spacer(1, 14))

        if bonus_clusters:
            story.append(HRFlowable(width="100%", thickness=0.6, color=border_soft, spaceAfter=6))
            story.append(
                Paragraph(
                    f"Asked Only Once in {escape(unit_name)} (Optional — Revise After Core Topics)",
                    once_hdr_style,
                )
            )
            for cl in bonus_clusters:
                for q in cl.questions:
                    marks_str = f" [{q.marks}M]" if q.marks is not None else ""
                    story.append(
                        Paragraph(
                            f"• <b>{escape(q.paper_name)} · {escape(q.question_number)}{marks_str}:</b> "
                            f"{escape(q.exact_text)}",
                            once_item_style,
                        )
                    )

    doc.build(story, canvasmaker=CleanBookCanvas)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
''',

    "frontend/src/components/ClusterCard.tsx": r'''
import React from "react";
import { Check, ChevronDown, ChevronRight, Minus, Unlink } from "lucide-react";
import { PaperInfo, QuestionCluster } from "../types";

interface ClusterCardProps {
  cluster: QuestionCluster;
  allPapers: PaperInfo[];
  expanded: boolean;
  selectedForMerge: boolean;
  onToggleExpand: (clusterId: string) => void;
  onToggleSelect: (clusterId: string) => void;
  onSplitQuestion: (clusterId: string, questionId: string) => void;
}

const PRIORITY_STYLES: Record<string, string> = {
  "High Priority":
    "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-500/10 dark:text-rose-300 dark:border-rose-500/25",
  "Medium Priority":
    "bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-500/10 dark:text-sky-300 dark:border-sky-500/25",
  "Low / Unique":
    "bg-slate-100 text-slate-600 border-slate-200 dark:bg-[#222631] dark:text-slate-400 dark:border-[#323746]",
};

export const ClusterCard: React.FC<ClusterCardProps> = ({
  cluster,
  allPapers,
  expanded,
  selectedForMerge,
  onToggleExpand,
  onToggleSelect,
  onSplitQuestion,
}) => {
  const presentPaperIds = new Set(cluster.questions.map((q) => q.paper_id));

  return (
    <div
      className={`bg-white dark:bg-[#181b22] border rounded-xl transition-all duration-200 ${
        selectedForMerge
          ? "border-indigo-500 ring-1 ring-indigo-500/50"
          : "border-slate-200 dark:border-[#272b36] hover:border-slate-300 dark:hover:border-[#3b4152]"
      }`}
    >
      <div className="flex items-start justify-between gap-2 sm:gap-4 p-3.5 sm:p-4">
        <div className="flex items-start gap-2.5 sm:gap-3 min-w-0 flex-1">
          <input
            type="checkbox"
            checked={selectedForMerge}
            onChange={() => onToggleSelect(cluster.cluster_id)}
            title="Select cluster to merge"
            className="mt-1 h-4 w-4 rounded border-slate-300 dark:border-slate-600 accent-indigo-500 cursor-pointer shrink-0"
          />
          <button
            type="button"
            onClick={() => onToggleExpand(cluster.cluster_id)}
            className="text-left min-w-0 flex-1 group"
          >
            <h3 className="text-sm sm:text-base font-semibold text-slate-900 dark:text-slate-100 leading-snug break-words group-hover:text-slate-600 dark:group-hover:text-white transition-colors">
              {cluster.topic_title}
            </h3>
            <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] sm:text-xs font-semibold bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-950">
                {cluster.unique_paper_count} / {cluster.total_papers} papers
              </span>
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] sm:text-xs font-medium border ${
                  PRIORITY_STYLES[cluster.priority]
                }`}
              >
                {cluster.priority}
              </span>
              {cluster.unit_label && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] sm:text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-200 dark:bg-indigo-500/10 dark:text-indigo-300 dark:border-indigo-500/25">
                  {cluster.unit_label}
                </span>
              )}
            </div>
          </button>
        </div>

        <button
          type="button"
          onClick={() => onToggleExpand(cluster.cluster_id)}
          className="shrink-0 inline-flex items-center gap-1 text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 px-2 py-1 rounded-md hover:bg-slate-100 dark:hover:bg-[#232732] transition-colors"
        >
          <span className="hidden sm:inline">
            {expanded ? "Hide" : `${cluster.questions.length} Qs`}
          </span>
          {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      </div>

      {expanded && (
        <div className="px-3.5 sm:px-4 pb-4 space-y-2.5 border-t border-slate-100 dark:border-[#272b36] pt-3">
          {/* Exam Paper Coverage Strip */}
          {allPapers.length > 1 && cluster.unique_paper_count >= 2 && (
            <div className="flex flex-wrap gap-1.5 pb-1">
              {allPapers.map((p) => {
                const asked = presentPaperIds.has(p.paper_id);
                return (
                  <span
                    key={p.paper_id}
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium border ${
                      asked
                        ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:border-emerald-500/25"
                        : "bg-slate-50 text-slate-400 border-slate-200 dark:bg-[#12151b] dark:text-slate-500 dark:border-[#232732]"
                    }`}
                  >
                    {asked ? <Check className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
                    {p.paper_name}
                  </span>
                );
              })}
            </div>
          )}

          {cluster.questions.map((q) => (
            <div
              key={q.id}
              className="group flex items-start justify-between gap-2.5 rounded-lg bg-slate-50 dark:bg-[#12151b] border border-slate-100 dark:border-[#232732] px-3 py-2 text-xs sm:text-sm"
            >
              <div className="text-slate-800 dark:text-slate-300 leading-relaxed break-words min-w-0">
                <span className="font-semibold text-slate-900 dark:text-slate-100">
                  {q.paper_name} · {q.question_number}
                  {q.marks !== null ? ` [${q.marks}M]` : ""}:
                </span>{" "}
                <span>&ldquo;{q.exact_text}&rdquo;</span>
              </div>
              {cluster.questions.length > 1 && (
                <button
                  type="button"
                  onClick={() => onSplitQuestion(cluster.cluster_id, q.id)}
                  title="Split this question into its own cluster"
                  className="sm:opacity-0 sm:group-hover:opacity-100 shrink-0 p-1 text-slate-400 hover:text-slate-900 dark:hover:text-white transition-all"
                >
                  <Unlink className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
''',
}

# Pass allPapers into ClusterCard inside Dashboard.tsx
dash_path = Path("frontend/src/components/Dashboard.tsx")
dash_txt = dash_path.read_text(encoding="utf-8")
if "allPapers={analysis.papers}" not in dash_txt:
    dash_txt = dash_txt.replace(
        "cluster={cluster}",
        "cluster={cluster}\n            allPapers={analysis.papers}",
    )
    dash_path.write_text(dash_txt, encoding="utf-8")
    print(f"Updated {dash_path}")

gi = Path(".gitignore")
if gi.exists():
    g = gi.read_text(encoding="utf-8")
    if "fix_all_100pct.py" not in g:
        gi.write_text(g.rstrip() + "\nfix_all_100pct.py\n", encoding="utf-8")

for rel, content in FILES.items():
    p = Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"Updated {p}")

print("\nAll fixes applied cleanly!")