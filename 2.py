from pathlib import Path

FILES = {
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

    story.append(Paragraph("Unit-Wise Question Summary", h2_style))
    overview_rows = [["Unit", "Total Topics", "Repeated (2+ Papers)", "Asked Once"]]
    for u in sorted_units:
        cls = unit_groups[u]
        rep = sum(1 for c in cls if c.unique_paper_count >= 2)
        once = len(cls) - rep
        overview_rows.append([u, str(len(cls)), str(rep), str(once)])

    ov_table = Table(overview_rows, colWidths=[56 * mm, 32 * mm, 48 * mm, 38 * mm])
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
        story.append(Paragraph("Top Repeated Topics (Must Revise First)", h2_style))
        chk_rows = [["#", "Unit", "Topic", "Frequency"]]
        for i, cl in enumerate(repeated_all[:15], start=1):
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
        total_u_qs = sum(len(c.questions) for c in unit_clusters)

        # Clean Editorial Unit Header
        hdr_table = Table(
            [[
                Paragraph(escape(unit_name.upper()), unit_title_style),
                Paragraph(
                    f"{len(unit_clusters)} Topics  ·  {total_u_qs} Questions",
                    unit_sub_style,
                ),
            ]],
            colWidths=[104 * mm, 70 * mm],
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

        for idx, cl in enumerate(unit_clusters, start=1):
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
                tag = "ASKED ONCE" if cl.unique_paper_count == 1 else "LOW FREQUENCY"
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

            # Keep topic header + first question card together so title is never orphaned
            first_group = [topic_p, badge_p]
            if q_cards:
                first_group.append(q_cards[0])
            story.append(KeepTogether(first_group))

            for extra_card in q_cards[1:]:
                story.append(Spacer(1, 5))
                story.append(extra_card)

            # Generous spacing between different topics
            story.append(Spacer(1, 15))

    doc.build(story, canvasmaker=CleanBookCanvas)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
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

# Also catches up to 2 stray diagram single-letters after marks like "[6] L"
MARKS_END_RE = re.compile(
    r"(?:\[\s*(\d{1,2})\s*(?:M|Marks?)?\s*\]|\(\s*(\d{1,2})\s*(?:M|Marks?)?\s*\)|[-–]\s*(\d{1,2})\s*Marks?)(?:\s+[A-Z]{1,2})?\s*$",
    re.IGNORECASE,
)

MARKS_BEFORE_ROMAN_RE = re.compile(
    r"\[\s*(\d{1,2})\s*(?:M|Marks?)?\s*\](?=\s+(?:i|ii|1)\s*[\)\.])",
    re.IGNORECASE,
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
            return s[: m.start()].strip(), val

    m2 = MARKS_BEFORE_ROMAN_RE.search(s)
    if m2:
        val = int(m2.group(1))
        if val <= 30:
            cleaned = (s[: m2.start()] + ":" + s[m2.end():]).strip()
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

    "backend/app/services/labeling.py": r'''
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
    "a", "an", "the", "of", "and", "or", "with", "in", "on", "for", "to", "which", "that", "is", "are", "its"
}


def heuristic_topic_title(questions: List[ExtractedQuestion]) -> str:
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
''',
}

gi = Path(".gitignore")
if gi.exists():
    g = gi.read_text(encoding="utf-8")
    if "upgrade_pdf_design.py" not in g:
        gi.write_text(g.rstrip() + "\nupgrade_pdf_design.py\n", encoding="utf-8")

for rel, content in FILES.items():
    p = Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"Updated {p}")

print("\nClean Question-Card PDF design + parser cleanup applied!")