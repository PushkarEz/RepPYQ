import json
import os
import sys
import uuid
from pathlib import Path
from typing import List
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from _lib.ai_grouper import group_questions_with_ai
from _lib.clustering import (
    chronological_paper_key,
    cluster_questions,
    compute_priority_tier,
)
from _lib.labeling import heuristic_topic_title
from _lib.pdf_extractor import detect_paper_label, extract_pdf_text
from _lib.pdf_exporter import build_revision_pdf
from _lib.question_parser import parse_questions
from _lib.schemas import (
    AnalysisResponse,
    ExtractPaperResponse,
    ExtractedQuestion,
    GroupRequest,
    MergeStatelessRequest,
    PaperInfo,
    QuestionCluster,
    ReclusterStatelessRequest,
    SplitStatelessRequest,
)

app = FastAPI(title="RepPYQ Serverless API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOCAL_STATS = Path("/tmp/reppyq_views.json")


def _hit_views() -> int:
    try:
        r = httpx.get("https://api.counterapi.dev/v1/reppyq-pushkar/visits/up", timeout=2.5)
        if r.status_code == 200:
            count = int(r.json().get("count", 0))
            if count > 0:
                return count
    except Exception:
        pass
    try:
        cur = int(json.loads(LOCAL_STATS.read_text()).get("views", 0)) if LOCAL_STATS.exists() else 0
        cur += 1
        LOCAL_STATS.write_text(json.dumps({"views": cur}))
        return cur
    except Exception:
        return 1


@app.get("/api/health")
def health():
    return {"status": "ok", "runtime": "vercel-serverless"}


@app.post("/api/stats/visit")
def record_visit():
    return {"views": _hit_views()}


@app.post("/api/detect-label")
async def detect_single_label(file: UploadFile = File(...)):
    raw = await file.read()
    extracted = extract_pdf_text(raw, file.filename or "")
    label = detect_paper_label(file.filename or "paper.pdf", extracted.raw_first_page)
    return {
        "filename": file.filename,
        "suggested_label": label,
        "exam_type": extracted.exam_type,
        "page_count": extracted.page_count,
        "used_ocr": extracted.used_ocr,
        "ocr_warning": extracted.ocr_warning,
    }


@app.post("/api/extract-paper", response_model=ExtractPaperResponse)
async def extract_single_paper(
    file: UploadFile = File(...),
    paper_label: str = Form(""),
):
    raw_bytes = await file.read()
    paper_id = str(uuid.uuid4())
    extracted = extract_pdf_text(raw_bytes, file.filename or "")
    label = paper_label.strip() or detect_paper_label(
        file.filename or "paper.pdf", extracted.raw_first_page
    )
    qs = parse_questions(extracted.cleaned_text, paper_id, label, extracted.exam_type)

    warnings: List[str] = []
    if extracted.ocr_warning:
        warnings.append(f"{label}: {extracted.ocr_warning}")
    if len(qs) == 0:
        warnings.append(f"{label}: No selectable questions matched standard formats.")

    paper = PaperInfo(
        paper_id=paper_id,
        filename=file.filename or "paper.pdf",
        paper_name=label,
        exam_type=extracted.exam_type,
        page_count=extracted.page_count,
        used_ocr=extracted.used_ocr,
        ocr_warning=extracted.ocr_warning,
        extracted_count=len(qs),
    )
    return ExtractPaperResponse(paper=paper, questions=qs, warnings=warnings)


def _cluster_unit_label(q_list: List[ExtractedQuestion]) -> str | None:
    units = [q.unit_number for q in q_list if q.unit_number]
    return max(set(units), key=units.count) if units else None


@app.post("/api/group", response_model=AnalysisResponse)
def group_extracted_questions(payload: GroupRequest):
    if not payload.questions:
        raise HTTPException(
            status_code=422,
            detail="Could not extract questions from these PDFs. Make sure PDFs have selectable text (not image scans).",
        )

    papers = sorted(payload.papers, key=lambda p: chronological_paper_key(p.paper_name))
    questions = sorted(
        payload.questions,
        key=lambda q: chronological_paper_key(q.paper_name, q.question_number),
    )
    total_papers = len(papers)
    warnings = list(payload.warnings)
    cluster_objs: List[QuestionCluster] = []
    grouping_mode = "Semantic Similarity"

    ai_grouped, ai_err = group_questions_with_ai(
        payload.subject_name, questions, payload.similarity_threshold
    )
    if ai_grouped is not None:
        grouping_mode = "AI Concept Grouping"
        if ai_err:
            warnings.append(f"Partial AI fallback: {ai_err}")
        for cid, title, q_list, conf in ai_grouped:
            uniq = len({q.paper_id for q in q_list})
            cluster_objs.append(
                QuestionCluster(
                    cluster_id=cid,
                    topic_title=title,
                    unit_label=_cluster_unit_label(q_list),
                    priority=compute_priority_tier(uniq, total_papers),
                    unique_paper_count=uniq,
                    total_papers=total_papers,
                    total_occurrences=len(q_list),
                    avg_similarity=conf,
                    questions=q_list,
                )
            )
    else:
        if ai_err:
            warnings.append(f"AI note: {ai_err} (used fallback matching).")
        for cid, q_list, avg_sim in cluster_questions(questions, payload.similarity_threshold):
            uniq = len({q.paper_id for q in q_list})
            cluster_objs.append(
                QuestionCluster(
                    cluster_id=cid,
                    topic_title=heuristic_topic_title(q_list),
                    unit_label=_cluster_unit_label(q_list),
                    priority=compute_priority_tier(uniq, total_papers),
                    unique_paper_count=uniq,
                    total_papers=total_papers,
                    total_occurrences=len(q_list),
                    avg_similarity=avg_sim,
                    questions=q_list,
                )
            )

    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        subject_name=payload.subject_name.strip() or "SPPU Subject Revision Notes",
        similarity_threshold=payload.similarity_threshold,
        grouping_mode=grouping_mode,
        total_papers=total_papers,
        total_questions=len(questions),
        papers=papers,
        clusters=cluster_objs,
        warnings=warnings,
    )


@app.post("/api/recluster", response_model=AnalysisResponse)
def recluster_stateless(payload: ReclusterStatelessRequest):
    a = payload.analysis
    all_qs = [q for cl in a.clusters for q in cl.questions]
    total_papers = len(a.papers)
    cluster_objs: List[QuestionCluster] = []

    for cid, q_list, avg_sim in cluster_questions(all_qs, payload.similarity_threshold):
        uniq = len({q.paper_id for q in q_list})
        cluster_objs.append(
            QuestionCluster(
                cluster_id=cid,
                topic_title=heuristic_topic_title(q_list),
                unit_label=_cluster_unit_label(q_list),
                priority=compute_priority_tier(uniq, total_papers),
                unique_paper_count=uniq,
                total_papers=total_papers,
                total_occurrences=len(q_list),
                avg_similarity=avg_sim,
                questions=q_list,
            )
        )

    a.similarity_threshold = payload.similarity_threshold
    a.grouping_mode = "Semantic Similarity"
    a.clusters = cluster_objs
    return a


@app.post("/api/merge", response_model=AnalysisResponse)
def merge_stateless(payload: MergeStatelessRequest):
    a = payload.analysis
    target_ids = set(payload.cluster_ids)
    to_merge = [c for c in a.clusters if c.cluster_id in target_ids]
    if len(to_merge) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 clusters to merge.")

    remaining = [c for c in a.clusters if c.cluster_id not in target_ids]
    merged_qs = [q for cl in to_merge for q in cl.questions]
    merged_qs.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))
    uniq = len({q.paper_id for q in merged_qs})

    merged = QuestionCluster(
        cluster_id=str(uuid.uuid4()),
        topic_title=(payload.custom_title or "").strip() or to_merge[0].topic_title,
        unit_label=to_merge[0].unit_label,
        priority=compute_priority_tier(uniq, len(a.papers)),
        unique_paper_count=uniq,
        total_papers=len(a.papers),
        total_occurrences=len(merged_qs),
        avg_similarity=a.similarity_threshold,
        questions=merged_qs,
    )
    remaining.append(merged)
    remaining.sort(key=lambda c: (c.unique_paper_count, c.total_occurrences), reverse=True)
    a.clusters = remaining
    return a


@app.post("/api/split", response_model=AnalysisResponse)
def split_stateless(payload: SplitStatelessRequest):
    a = payload.analysis
    target = next((c for c in a.clusters if c.cluster_id == payload.cluster_id), None)
    if not target or len(target.questions) <= 1:
        raise HTTPException(status_code=400, detail="Cannot split this cluster.")

    split_q = next((q for q in target.questions if q.id == payload.question_id), None)
    if not split_q:
        raise HTTPException(status_code=404, detail="Question not found.")

    target.questions = [q for q in target.questions if q.id != payload.question_id]
    uniq_orig = len({q.paper_id for q in target.questions})
    target.unique_paper_count = uniq_orig
    target.total_occurrences = len(target.questions)
    target.priority = compute_priority_tier(uniq_orig, len(a.papers))

    new_cl = QuestionCluster(
        cluster_id=str(uuid.uuid4()),
        topic_title=heuristic_topic_title([split_q]),
        unit_label=_cluster_unit_label([split_q]),
        priority=compute_priority_tier(1, len(a.papers)),
        unique_paper_count=1,
        total_papers=len(a.papers),
        total_occurrences=1,
        avg_similarity=1.0,
        questions=[split_q],
    )
    a.clusters.append(new_cl)
    a.clusters.sort(key=lambda c: (c.unique_paper_count, c.total_occurrences), reverse=True)
    return a


@app.post("/api/export-pdf")
def export_pdf_stateless(analysis: AnalysisResponse):
    pdf_bytes = build_revision_pdf(analysis)
    safe_name = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_" for c in analysis.subject_name
    ).strip() or "RepPYQ_Revision_Notes"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_RepPYQ_Notes.pdf"'},
    )
