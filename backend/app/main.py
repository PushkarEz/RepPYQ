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

app = FastAPI(title="RepPYQ API", version="2.0.0")

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
        ai_grouped = group_questions_with_ai(session.subject_name, session.questions)
        if ai_grouped is not None:
            session.grouping_mode = "AI Concept Grouping"
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
    similarity_threshold: float = Form(0.68),
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
