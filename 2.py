from pathlib import Path

FILES = {
    "backend/app/models/schemas.py": r'''
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PriorityTier(str, Enum):
    HIGH = "High Priority"
    MEDIUM = "Medium Priority"
    LOW = "Low / Unique"


class ExtractedQuestion(BaseModel):
    id: str
    paper_id: str
    paper_name: str
    question_number: str
    unit_number: Optional[str] = None
    marks: Optional[int] = None
    exact_text: str


class PaperInfo(BaseModel):
    paper_id: str
    filename: str
    paper_name: str
    exam_type: str = "Endsem"
    page_count: int
    used_ocr: bool = False
    ocr_warning: Optional[str] = None
    extracted_count: int = 0


class QuestionCluster(BaseModel):
    cluster_id: str
    topic_title: str
    unit_label: Optional[str] = None
    priority: PriorityTier
    unique_paper_count: int
    total_papers: int
    total_occurrences: int
    avg_similarity: float
    questions: List[ExtractedQuestion]


class AnalysisResponse(BaseModel):
    analysis_id: str
    subject_name: str
    similarity_threshold: float
    grouping_mode: str = "Semantic Similarity"
    total_papers: int
    total_questions: int
    papers: List[PaperInfo]
    clusters: List[QuestionCluster]
    warnings: List[str] = Field(default_factory=list)


class ReclusterRequest(BaseModel):
    similarity_threshold: float = Field(ge=0.45, le=0.95)


class MergeClustersRequest(BaseModel):
    cluster_ids: List[str] = Field(min_length=2)
    custom_title: Optional[str] = None


class SplitQuestionRequest(BaseModel):
    cluster_id: str
    question_id: str


class UpdateLabelsRequest(BaseModel):
    paper_labels: Dict[str, str]
    subject_name: Optional[str] = None
''',

    "backend/app/services/ai_grouper.py": r'''
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
''',

    "backend/app/services/clustering.py": r'''
import math
import os
import re
import uuid
from typing import Dict, List, Set, Tuple
import httpx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import HashingVectorizer

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
    "note", "short", "detail", "neat", "diagram", "suitable", "example", "examples",
    "compare", "differentiate", "between", "types", "different", "need", "enlist",
    "advantages", "disadvantages", "applications", "various", "its", "how", "why",
    "role", "steps", "involved", "fundamental", "components", "characteristics",
}

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


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


def _encode_lightweight_hybrid(texts: List[str]) -> np.ndarray:
    word_vec = HashingVectorizer(
        n_features=768, ngram_range=(1, 2), stop_words="english", alternate_sign=False
    )
    char_vec = HashingVectorizer(
        n_features=512, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False
    )
    w_mat = word_vec.transform(texts).toarray().astype(np.float32)
    c_mat = char_vec.transform(texts).toarray().astype(np.float32)
    combined = np.hstack([w_mat * 1.25, c_mat * 0.75])
    return _l2_normalize(combined)


def get_embedding_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None and SentenceTransformer is not None:
        _MODEL_INSTANCE = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL_INSTANCE


def encode_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        hf_embs = _encode_via_hf_api(texts, hf_token)
        if hf_embs is not None:
            return hf_embs

    if SentenceTransformer is not None:
        model = get_embedding_model()
        embeddings = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    return _encode_lightweight_hybrid(texts)


def _tech_tokens(text: str) -> Set[str]:
    tokens = set(re.findall(r"[a-z0-9]{2,}", text.lower()))
    return tokens - STOPWORDS


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
            # Keep distinct SPPU units separate, but allow same-concept questions in same paper to group
            ui, uj = questions[i].unit_number, questions[j].unit_number
            if ui and uj and ui != uj:
                dist_matrix[i, j] = dist_matrix[j, i] = 2.0
                continue

            ti, tj = token_sets[i], token_sets[j]
            union = ti | tj
            jacc = (len(ti & tj) / len(union)) if union else 0.0
            hybrid_sim = 0.84 * float(sim_matrix[i, j]) + 0.16 * jacc
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
        fontSize=8.5, leading=11, textColor=high_col, spaceAfter=6,
    )
    rep_med_style = ParagraphStyle(
        "RepMed", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=11, textColor=med_col, spaceAfter=6,
    )
    rep_low_style = ParagraphStyle(
        "RepLow", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=11, textColor=muted, spaceAfter=6,
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

    # --- PAGE 1: CLEAN STUDY OVERVIEW ---
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

    # --- UNIT BY UNIT CHAPTERS ---
    for unit_name in sorted_units:
        story.append(PageBreak())

        unit_clusters = sorted(
            unit_groups[unit_name],
            key=lambda c: (c.unique_paper_count, c.total_occurrences),
            reverse=True,
        )
        rep_clusters = [c for c in unit_clusters if c.unique_paper_count >= 2]
        once_clusters = [c for c in unit_clusters if c.unique_paper_count < 2]

        # If a unit only has single-paper clusters (e.g. 1 paper uploaded), show them as main
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

            first_group = [topic_p, badge_p]
            if q_cards:
                first_group.append(q_cards[0])
            story.append(KeepTogether(first_group))

            for extra_card in q_cards[1:]:
                story.append(Spacer(1, 5))
                story.append(extra_card)

            story.append(Spacer(1, 14))

        # Compact "Asked Only Once" Appendix at end of Unit
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
''',

    "frontend/src/types.ts": r'''
export type PriorityTier = "High Priority" | "Medium Priority" | "Low / Unique";

export interface ExtractedQuestion {
  id: string;
  paper_id: string;
  paper_name: string;
  question_number: string;
  unit_number: string | null;
  marks: number | null;
  exact_text: string;
}

export interface PaperInfo {
  paper_id: string;
  filename: string;
  paper_name: string;
  exam_type: string;
  page_count: number;
  used_ocr: boolean;
  ocr_warning: string | null;
  extracted_count: number;
}

export interface QuestionCluster {
  cluster_id: string;
  topic_title: string;
  unit_label: string | null;
  priority: PriorityTier;
  unique_paper_count: number;
  total_papers: number;
  total_occurrences: number;
  avg_similarity: number;
  questions: ExtractedQuestion[];
}

export interface AnalysisResponse {
  analysis_id: string;
  subject_name: string;
  similarity_threshold: number;
  grouping_mode?: string;
  total_papers: number;
  total_questions: number;
  papers: PaperInfo[];
  clusters: QuestionCluster[];
  warnings: string[];
}

export interface PendingPaper {
  file: File;
  label: string;
  pageCount?: number;
  usedOcr?: boolean;
  warning?: string | null;
}
''',

    "frontend/src/components/Dashboard.tsx": r'''
import React, { useEffect, useMemo, useState } from "react";
import {
  ChevronsDownUp,
  ChevronsUpDown,
  Download,
  GitMerge,
  RotateCcw,
  Search,
  AlertTriangle,
} from "lucide-react";
import { AnalysisResponse, PriorityTier } from "../types";
import { ClusterCard } from "./ClusterCard";
import { getExportPdfUrl } from "../api/client";

interface DashboardProps {
  analysis: AnalysisResponse;
  onRecluster: (threshold: number) => Promise<void>;
  onMerge: (clusterIds: string[], customTitle?: string) => Promise<void>;
  onSplit: (clusterId: string, questionId: string) => Promise<void>;
  onReset: () => void;
  busy: boolean;
}

type FilterTab = "REPEATED" | "ALL" | PriorityTier;

export const Dashboard: React.FC<DashboardProps> = ({
  analysis,
  onRecluster,
  onMerge,
  onSplit,
  onReset,
  busy,
}) => {
  const [threshold, setThreshold] = useState(analysis.similarity_threshold);
  const [priorityFilter, setPriorityFilter] = useState<FilterTab>("REPEATED");
  const [unitFilter, setUnitFilter] = useState<string>("ALL");
  const [search, setSearch] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const hasRepeated = analysis.clusters.some((c) => c.unique_paper_count >= 2);
    setPriorityFilter(hasRepeated ? "REPEATED" : "ALL");

    const initial: Record<string, boolean> = {};
    analysis.clusters.forEach((c) => {
      initial[c.cluster_id] = c.priority === "High Priority";
    });
    setExpandedIds(initial);
  }, [analysis.clusters]);

  const availableUnits = useMemo(() => {
    const units = new Set<string>();
    analysis.clusters.forEach((c) => {
      if (c.unit_label) {
        units.add(c.unit_label.split("·")[0].trim());
      }
    });
    return Array.from(units).sort();
  }, [analysis.clusters]);

  const filteredClusters = useMemo(() => {
    return analysis.clusters.filter((c) => {
      if (priorityFilter === "REPEATED" && c.unique_paper_count < 2) return false;
      if (
        priorityFilter !== "ALL" &&
        priorityFilter !== "REPEATED" &&
        c.priority !== priorityFilter
      ) {
        return false;
      }
      if (unitFilter !== "ALL") {
        const u = (c.unit_label || "").split("·")[0].trim();
        if (u !== unitFilter) return false;
      }
      if (search.trim()) {
        const needle = search.toLowerCase();
        const inTitle = c.topic_title.toLowerCase().includes(needle);
        const inQs = c.questions.some(
          (q) =>
            q.exact_text.toLowerCase().includes(needle) ||
            q.paper_name.toLowerCase().includes(needle)
        );
        return inTitle || inQs;
      }
      return true;
    });
  }, [analysis.clusters, priorityFilter, unitFilter, search]);

  const counts = useMemo(() => {
    const repeated = analysis.clusters.filter((c) => c.unique_paper_count >= 2).length;
    const high = analysis.clusters.filter((c) => c.priority === "High Priority").length;
    const med = analysis.clusters.filter((c) => c.priority === "Medium Priority").length;
    const low = analysis.clusters.filter((c) => c.priority === "Low / Unique").length;
    return { repeated, high, med, low };
  }, [analysis.clusters]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const setAllExpanded = (open: boolean) => {
    const next: Record<string, boolean> = {};
    filteredClusters.forEach((c) => {
      next[c.cluster_id] = open;
    });
    setExpandedIds((prev) => ({ ...prev, ...next }));
  };

  const handleMerge = async () => {
    if (selectedIds.length < 2) return;
    const customTitle = window.prompt(
      "Optional: Enter a merged topic title (leave blank to keep first title):"
    );
    await onMerge(selectedIds, customTitle || undefined);
    setSelectedIds([]);
  };

  const allVisibleExpanded =
    filteredClusters.length > 0 && filteredClusters.every((c) => expandedIds[c.cluster_id]);

  return (
    <div className="max-w-5xl mx-auto space-y-4 sm:space-y-5">
      {/* Header Summary */}
      <div className="bg-white dark:bg-[#181b22] border border-slate-200 dark:border-[#272b36] rounded-xl p-4 sm:p-5 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-lg sm:text-xl font-bold text-slate-900 dark:text-slate-100">
              {analysis.subject_name}
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-0.5">
              {analysis.total_papers} papers · {analysis.total_questions} exact questions ·{" "}
              {counts.repeated} repeated topics
              {analysis.grouping_mode ? ` · ${analysis.grouping_mode}` : ""}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-300 dark:border-[#323746] text-xs sm:text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-[#232732] transition-colors"
            >
              <RotateCcw className="w-4 h-4" /> New Upload
            </button>
            <a
              href={getExportPdfUrl(analysis.analysis_id)}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950 text-xs sm:text-sm font-medium hover:bg-slate-800 dark:hover:bg-white active:scale-[0.98] transition-all"
            >
              <Download className="w-4 h-4" /> Export Unit-Wise PDF
            </a>
          </div>
        </div>

        <div className="flex flex-wrap gap-1.5 pt-2 border-t border-slate-100 dark:border-[#272b36]">
          {analysis.papers.map((p) => (
            <span
              key={p.paper_id}
              className="text-[11px] sm:text-xs bg-slate-100 dark:bg-[#222631] text-slate-700 dark:text-slate-300 px-2.5 py-0.5 rounded-md font-medium"
            >
              {p.paper_name} ({p.exam_type}) · {p.extracted_count} Qs
            </span>
          ))}
        </div>
      </div>

      {analysis.warnings.length > 0 && (
        <div className="bg-rose-500/10 border border-rose-500/25 rounded-xl p-3.5 text-sm text-rose-700 dark:text-rose-300 space-y-1">
          <div className="font-semibold flex items-center gap-1.5 text-xs sm:text-sm">
            <AlertTriangle className="w-4 h-4" /> Extraction Notes
          </div>
          {analysis.warnings.map((w, i) => (
            <p key={i} className="text-xs">{w}</p>
          ))}
        </div>
      )}

      {/* Compact Sticky Filter Controls */}
      <div className="bg-white dark:bg-[#181b22] border border-slate-200 dark:border-[#272b36] rounded-xl p-4 space-y-3 sm:sticky sm:top-2 z-10 shadow-sm">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-5 items-center">
          <div>
            <div className="flex justify-between text-xs sm:text-sm mb-1">
              <span className="font-medium text-slate-700 dark:text-slate-300">
                Similarity Override Slider
              </span>
              <span className="font-mono text-xs bg-slate-100 dark:bg-[#222631] text-slate-700 dark:text-slate-300 px-2 py-0.5 rounded">
                {threshold.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <input
                type="range"
                min={0.5}
                max={0.9}
                step={0.02}
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <button
                type="button"
                disabled={busy || threshold === analysis.similarity_threshold}
                onClick={() => onRecluster(threshold)}
                className="shrink-0 px-3 py-1.5 rounded-lg bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950 text-xs font-medium hover:bg-slate-800 dark:hover:bg-white disabled:opacity-40 transition-colors"
              >
                Apply
              </button>
            </div>
          </div>

          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search topic, question, or year..."
              className="w-full pl-9 pr-3 py-2 text-xs sm:text-sm border border-slate-300 dark:border-[#323746] bg-white dark:bg-[#12151b] text-slate-900 dark:text-slate-100 placeholder:text-slate-400 rounded-lg focus:outline-none focus:border-slate-500"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100 dark:border-[#272b36]">
          <div className="flex flex-wrap gap-1.5">
            {(
              [
                ["REPEATED", `Repeated 2+ (${counts.repeated})`],
                ["ALL", `All (${analysis.clusters.length})`],
                ["High Priority", `High (${counts.high})`],
                ["Medium Priority", `Medium (${counts.med})`],
                ["Low / Unique", `Once (${counts.low})`],
              ] as const
            ).map(([val, label]) => (
              <button
                key={val}
                type="button"
                onClick={() => setPriorityFilter(val)}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                  priorityFilter === val
                    ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-950"
                    : "bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-[#222631] dark:text-slate-300 dark:hover:bg-[#2c3140]"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setAllExpanded(!allVisibleExpanded)}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border border-slate-200 dark:border-[#323746] text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-[#232732] transition-colors"
            >
              {allVisibleExpanded ? (
                <>
                  <ChevronsDownUp className="w-3.5 h-3.5" /> Collapse All
                </>
              ) : (
                <>
                  <ChevronsUpDown className="w-3.5 h-3.5" /> Expand All
                </>
              )}
            </button>

            {selectedIds.length >= 2 && (
              <button
                type="button"
                disabled={busy}
                onClick={handleMerge}
                className="inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-500 transition-colors"
              >
                <GitMerge className="w-3.5 h-3.5" /> Merge ({selectedIds.length})
              </button>
            )}
          </div>
        </div>

        {availableUnits.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-2 border-t border-slate-100 dark:border-[#272b36]">
            <button
              type="button"
              onClick={() => setUnitFilter("ALL")}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                unitFilter === "ALL"
                  ? "bg-indigo-600 text-white"
                  : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-500/10 dark:text-indigo-300 dark:hover:bg-indigo-500/20"
              }`}
            >
              All Units
            </button>
            {availableUnits.map((u) => (
              <button
                key={u}
                type="button"
                onClick={() => setUnitFilter(u)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  unitFilter === u
                    ? "bg-indigo-600 text-white"
                    : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-500/10 dark:text-indigo-300 dark:hover:bg-indigo-500/20"
                }`}
              >
                {u}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Compact Cluster List */}
      <div className="space-y-2.5">
        {filteredClusters.map((cluster) => (
          <ClusterCard
            key={cluster.cluster_id}
            cluster={cluster}
            expanded={!!expandedIds[cluster.cluster_id]}
            selectedForMerge={selectedIds.includes(cluster.cluster_id)}
            onToggleExpand={toggleExpand}
            onToggleSelect={toggleSelect}
            onSplitQuestion={onSplit}
          />
        ))}
        {filteredClusters.length === 0 && (
          <div className="text-center py-10 text-sm text-slate-500 dark:text-slate-400 bg-white dark:bg-[#181b22] border border-slate-200 dark:border-[#272b36] rounded-xl">
            No question clusters match the current filter.
          </div>
        )}
      </div>
    </div>
  );
};
''',
}

gi = Path(".gitignore")
if gi.exists():
    g = gi.read_text(encoding="utf-8")
    if "upgrade_ai_grouper.py" not in g:
        gi.write_text(g.rstrip() + "\nupgrade_ai_grouper.py\n", encoding="utf-8")

for rel, content in FILES.items():
    p = Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"Updated {p}")

print("\nID-only AI Concept Grouper + Repeated-First UI & PDF upgrade complete!")