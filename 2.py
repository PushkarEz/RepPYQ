from pathlib import Path

FILES = {
    "frontend/requirements.txt": r'''
fastapi>=0.115.6
pydantic>=2.10.4
pymupdf>=1.25.2
reportlab>=4.2.5
httpx>=0.28.1
python-multipart>=0.0.20
''',

    "frontend/vercel.json": r'''
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "/api/index" }
  ]
}
''',

    "frontend/vite.config.ts": r'''
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
''',

    "frontend/api/_lib/__init__.py": "",

    "frontend/api/_lib/schemas.py": r'''
from enum import Enum
from typing import List, Optional
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


class ExtractPaperResponse(BaseModel):
    paper: PaperInfo
    questions: List[ExtractedQuestion]
    warnings: List[str] = Field(default_factory=list)


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
    grouping_mode: str = "AI Concept Grouping"
    total_papers: int
    total_questions: int
    papers: List[PaperInfo]
    clusters: List[QuestionCluster]
    warnings: List[str] = Field(default_factory=list)


class GroupRequest(BaseModel):
    subject_name: str = "SPPU Subject Revision Notes"
    similarity_threshold: float = Field(default=0.56, ge=0.45, le=0.95)
    papers: List[PaperInfo]
    questions: List[ExtractedQuestion]
    warnings: List[str] = Field(default_factory=list)


class ReclusterStatelessRequest(BaseModel):
    analysis: AnalysisResponse
    similarity_threshold: float = Field(ge=0.45, le=0.95)


class MergeStatelessRequest(BaseModel):
    analysis: AnalysisResponse
    cluster_ids: List[str] = Field(min_length=2)
    custom_title: Optional[str] = None


class SplitStatelessRequest(BaseModel):
    analysis: AnalysisResponse
    cluster_id: str
    question_id: str
''',

    "frontend/api/_lib/pdf_extractor.py": r'''
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import fitz  # PyMuPDF


NOISE_LINE_PATTERNS = [
    r"^\s*P\.?\s*T\.?\s*O\.?\s*$",
    r"^\s*Total\s+No\.?\s+of\s+(Questions|Pages)\b.*$",
    r"^\s*\[?\s*Total\s+No\.?\s+of\s+Pages\s*[:\-].*$",
    r"^\s*Seat\s*No\.?\s*[:\-]?.*$",
    r"^\s*P[A-Z]?\s*\d{3,6}\s*$",
    r"^\s*\[\d{3,5}\]\s*-\s*\d+.*$",
    r"^\s*Time\s*[:\-]\s*[\d½\.\s]+Hours?.*$",
    r"^\s*\[?\s*Max\.?\s*Marks\s*[:\-]\s*\d+.*$",
    r"^\s*\(?\s*Total\s+Marks\s*[:\-]\s*\d+\s*\)?.*$",
    r"^\s*Instructions\s+to\s+(?:the\s+)?candidates\s*[:\-]?.*$",
    r"^\s*\d+\)\s*(?:All\s+questions|Solve\s+Q|Answer\s+Q|Neat\s+diagrams|Assume\s+suitable|Figures\s+to\s+the\s+right|Use\s+of\s+(?:logarithmic|electronic|non-programmable)).*$",
    r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$",
    r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$",
    r"^\s*(?:F\.?E\.?|S\.?E\.?|T\.?E\.?|B\.?E\.?|M\.?E\.?)\s*\(.*\)\s*$",
    r"^\s*\(\s*\d{4}\s+(?:Pattern|Course|Credit).*$",
    r"^\s*[\*•\-_=\uf000-\uf8ff\s]{2,}$",
    r"^\s*N\.?B\.?\s*[:\-].*$",
    r"^\s*[A-Z]{2,6}\d{4,8}\s*$",
    r"^.*\b\d{1,3}(?:\.\d{1,3}){3}\b.*static-\d+.*$",
]

COMPILED_NOISE = [re.compile(p, re.IGNORECASE) for p in NOISE_LINE_PATTERNS]


@dataclass
class ExtractedPDFResult:
    cleaned_text: str
    raw_first_page: str
    page_count: int
    exam_type: str
    used_ocr: bool
    ocr_warning: Optional[str]


def detect_paper_label(filename: str, first_page_text: str) -> str:
    month_year_re = re.compile(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-_,/]*?(20\d{2})\b",
        re.IGNORECASE,
    )
    for source in (first_page_text[:1500], filename):
        match = month_year_re.search(source)
        if match:
            month = match.group(1)[:3].title()
            year = match.group(2)
            low = source.lower()
            is_insem = any(k in low for k in ("insem", "in-sem", "in semester", "in-semester"))
            prefix = "Insem " if is_insem else ""
            return f"{prefix}{month} {year}"

    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem[:40] or "Unnamed Paper"


def detect_exam_type(raw_text: str, filename: str = "") -> str:
    header_low = (raw_text[:2500] + " " + filename).lower()
    if any(k in header_low for k in ("insem", "in-sem", "in semester", "in-semester")):
        return "Insem"
    if re.search(r"(?:max\.?|total)\s*marks\s*[:\-]?\s*30\b", header_low):
        return "Insem"
    if re.search(r"time\s*[:\-]?\s*1\s*(?:hr|hour)\b", header_low):
        return "Insem"
    if re.search(r"(?:max\.?|total)\s*marks\s*[:\-]?\s*70\b", header_low):
        return "Endsem"
    if re.search(r"\bQ\.?\s*[5-8]\s*[\.\)\]:\-]", raw_text, re.IGNORECASE):
        return "Endsem"

    has_q1_to_4 = bool(re.search(r"\bQ\.?\s*[1-4]\s*[\.\)\]:\-]", raw_text, re.IGNORECASE))
    has_q5_to_8 = bool(re.search(r"\bQ\.?\s*[5-8]\s*[\.\)\]:\-]", raw_text, re.IGNORECASE))
    if has_q1_to_4 and not has_q5_to_8:
        return "Insem"
    return "Endsem"


def _extract_horizontal_visual_lines(page: fitz.Page) -> str:
    data = page.get_text("dict")
    spans: List[Tuple[float, float, str]] = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1.0, 0.0))
            if abs(dx) < 0.92 or abs(dy) > 0.15:
                continue
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt:
                    continue
                txt = re.sub(r"[\uf000-\uf8ff]+", "", txt).strip()
                if not txt:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                y_mid = (y0 + y1) / 2.0
                spans.append((x0, y_mid, txt))

    if not spans:
        return ""

    spans.sort(key=lambda s: (s[1], s[0]))
    grouped_lines: List[List[Tuple[float, float, str]]] = []
    line_y: List[float] = []

    for sp in spans:
        if not grouped_lines or abs(sp[1] - line_y[-1]) > 4.5:
            grouped_lines.append([sp])
            line_y.append(sp[1])
        else:
            grouped_lines[-1].append(sp)
            n = len(grouped_lines[-1])
            line_y[-1] = ((line_y[-1] * (n - 1)) + sp[1]) / n

    out_lines: List[str] = []
    for line_spans in grouped_lines:
        line_spans.sort(key=lambda s: s[0])
        out_lines.append(" ".join(s[2] for s in line_spans))
    return "\n".join(out_lines)


def _strip_noise_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if any(pat.match(s) for pat in COMPILED_NOISE):
            continue
        if re.match(r"^(?:www\.|https?://|downloaded\s+from|join\s+telegram)", s, re.IGNORECASE):
            continue
        s = re.sub(r"^\[\d{3,5}\]-\d+\s+\d+\s*$", "", s).strip()
        if s:
            cleaned.append(s)
    return "\n".join(cleaned)


def extract_pdf_text(pdf_bytes: bytes, filename: str = "") -> ExtractedPDFResult:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    raw_pages: List[str] = []
    pages_text: List[str] = []
    raw_first_page = ""
    ocr_warning: Optional[str] = None

    for idx, page in enumerate(doc):
        text = _extract_horizontal_visual_lines(page)
        if len(text.strip()) < 30:
            text = page.get_text("text", sort=True) or ""

        if idx == 0:
            raw_first_page = text

        if len(text.strip()) < 40:
            ocr_warning = (
                f"Page {idx + 1} appears to be a scanned image (no selectable text layer)."
            )

        raw_pages.append(text)
        pages_text.append(_strip_noise_lines(text))

    doc.close()
    raw_combined = "\n".join(raw_pages)
    combined = "\n".join(pages_text)
    exam_type = detect_exam_type(raw_combined, filename)
    return ExtractedPDFResult(
        cleaned_text=combined,
        raw_first_page=raw_first_page,
        page_count=page_count,
        exam_type=exam_type,
        used_ocr=False,
        ocr_warning=ocr_warning,
    )
''',

    "frontend/api/_lib/question_parser.py": r'''
import re
import uuid
from typing import List, Optional, Tuple
from _lib.schemas import ExtractedQuestion


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

    "frontend/api/_lib/labeling.py": r'''
import re
from typing import List
from _lib.schemas import ExtractedQuestion


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
''',

    "frontend/api/_lib/clustering.py": r'''
import math
import re
import uuid
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
from _lib.schemas import ExtractedQuestion, PriorityTier

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
    s = text.lower().strip()
    s = re.sub(
        r"\b(?:compare\s+and\s+contrast|compare|distinguish\s+between|distinguish|"
        r"differentiate\s+between|differentiate|difference\s+between|differences\s+between|contrast)\b",
        " _cmp_ ",
        s,
    )
    s = re.sub(
        r"\b(?:block\s+diagram|internal\s+architecture|architecture|constructional\s+diagram|"
        r"construction\s+and\s+working|construction)\b",
        " _arch_ ",
        s,
    )
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


def _tech_tokens(text: str) -> Set[str]:
    normed = normalize_for_matching(text)
    tokens = set(re.findall(r"[a-z0-9_]{2,}", normed))
    stemmed = {t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens}
    return stemmed - STOPWORDS


def _doc_features(text: str) -> Counter:
    normed = normalize_for_matching(text)
    words = [w for w in normed.split() if w not in STOPWORDS and len(w) > 1]
    feats: Counter = Counter()
    for w in words:
        feats[f"w:{w}"] += 1.4
    for i in range(len(words) - 1):
        feats[f"b:{words[i]}_{words[i + 1]}"] += 1.4
    for w in words:
        padded = f" {w} "
        for n in (3, 4):
            if len(padded) >= n:
                for k in range(len(padded) - n + 1):
                    feats[f"c:{padded[k:k+n]}"] += 0.35
    return feats


def _build_tfidf_vectors(texts: List[str]) -> List[Dict[str, float]]:
    n = len(texts)
    raw_counts = [_doc_features(t) for t in texts]
    df: Counter = Counter()
    for c in raw_counts:
        for term in c.keys():
            df[term] += 1

    vectors: List[Dict[str, float]] = []
    for c in raw_counts:
        vec: Dict[str, float] = {}
        sq_sum = 0.0
        for term, tf in c.items():
            idf = math.log((1.0 + n) / (1.0 + df[term])) + 1.0
            w = (1.0 + math.log(tf)) * idf
            vec[term] = w
            sq_sum += w * w
        norm = math.sqrt(sq_sum) or 1.0
        for term in vec:
            vec[term] /= norm
        vectors.append(vec)
    return vectors


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[k] for k, v in a.items() if k in b)


def cluster_questions(
    questions: List[ExtractedQuestion],
    threshold: float,
) -> List[Tuple[str, List[ExtractedQuestion], float]]:
    """Pure-Python TF-IDF + Average-Linkage Clustering (zero numpy/sklearn)."""
    n = len(questions)
    if n == 0:
        return []
    if n == 1:
        return [(str(uuid.uuid4()), questions, 1.0)]

    vectors = _build_tfidf_vectors([q.exact_text for q in questions])
    token_sets = [_tech_tokens(q.exact_text) for q in questions]

    sim_matrix: List[List[float]] = [[1.0] * n for _ in range(n)]
    dist_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            ui, uj = questions[i].unit_number, questions[j].unit_number
            if ui and uj and ui != uj:
                sim_matrix[i][j] = sim_matrix[j][i] = 0.0
                dist_matrix[i][j] = dist_matrix[j][i] = 2.0
                continue

            cos_sim = _cosine_sparse(vectors[i], vectors[j])
            ti, tj = token_sets[i], token_sets[j]
            union = ti | tj
            jacc = (len(ti & tj) / len(union)) if union else 0.0
            hybrid_sim = max(0.0, min(1.0, 0.72 * cos_sim + 0.28 * jacc))
            sim_matrix[i][j] = sim_matrix[j][i] = cos_sim
            d = 1.0 - hybrid_sim
            dist_matrix[i][j] = dist_matrix[j][i] = d

    # Pure-Python Average-Linkage Agglomerative Clustering
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    cluster_dists: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            cluster_dists[(i, j)] = dist_matrix[i][j]

    max_dist = 1.0 - threshold
    while len(clusters) > 1:
        best_pair = None
        best_d = max_dist + 1e-9
        keys = sorted(clusters.keys())
        for idx_a in range(len(keys)):
            ci = keys[idx_a]
            for idx_b in range(idx_a + 1, len(keys)):
                cj = keys[idx_b]
                d = cluster_dists[(ci, cj)]
                if d <= best_d:
                    best_d = d
                    best_pair = (ci, cj)
        if best_pair is None or best_d > max_dist:
            break

        ci, cj = best_pair
        si, sj = len(clusters[ci]), len(clusters[cj])
        clusters[ci].extend(clusters[cj])
        del clusters[cj]

        for ck in list(clusters.keys()):
            if ck == ci:
                continue
            p_ik = (ci, ck) if ci < ck else (ck, ci)
            p_jk = (cj, ck) if cj < ck else (ck, cj)
            d_ik = cluster_dists.pop(p_ik)
            d_jk = cluster_dists.pop(p_jk)
            new_d = (d_ik * si + d_jk * sj) / (si + sj)
            cluster_dists[(ci, ck) if ci < ck else (ck, ci)] = new_d

        # Remove any stale pairs referencing cj
        for pair in [p for p in cluster_dists if cj in p]:
            cluster_dists.pop(pair, None)

    results: List[Tuple[str, List[ExtractedQuestion], float]] = []
    for member_indices in clusters.values():
        member_qs = [questions[i] for i in member_indices]
        member_qs.sort(key=lambda q: chronological_paper_key(q.paper_name, q.question_number))
        if len(member_indices) > 1:
            pair_sims = [
                sim_matrix[member_indices[a]][member_indices[b]]
                for a in range(len(member_indices))
                for b in range(a + 1, len(member_indices))
            ]
            avg_sim = sum(pair_sims) / len(pair_sims)
        else:
            avg_sim = 1.0
        results.append((str(uuid.uuid4()), member_qs, round(avg_sim, 3)))

    results.sort(
        key=lambda item: (len({q.paper_id for q in item[1]}), len(item[1]), item[2]),
        reverse=True,
    )
    return results
''',

    "frontend/api/_lib/ai_grouper.py": r'''
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
''',

    "frontend/api/_lib/pdf_exporter.py": r'''
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

from _lib.schemas import AnalysisResponse, PriorityTier, QuestionCluster
from _lib.clustering import chronological_paper_key


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
            times_word = "time" if cl.total_occurrences == 1 else "times"
            badge_p = Paragraph(
                f"{tag}   ·   Asked {cl.total_occurrences} {times_word} across {cl.unique_paper_count} of {cl.total_papers} papers",
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

    "frontend/api/index.py": r'''
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
''',

    "frontend/src/api/client.ts": r'''
import { AnalysisResponse, ExtractedQuestion, PaperInfo } from "../types";

const API_BASE = "/api";

export async function recordPageVisit(): Promise<number> {
  const res = await fetch(`${API_BASE}/stats/visit`, { method: "POST" });
  if (!res.ok) return 0;
  const data = await res.json();
  return data.views ?? 0;
}

export async function detectPaperLabels(files: File[]) {
  const papers = await Promise.all(
    files.map(async (file) => {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE}/detect-label`, { method: "POST", body: form });
      if (!res.ok) {
        return {
          filename: file.name,
          suggested_label: file.name.replace(/\.pdf$/i, ""),
          exam_type: "Endsem",
          page_count: 1,
          used_ocr: false,
          ocr_warning: null,
        };
      }
      return res.json();
    })
  );
  return { papers };
}

export async function analyzePdfs(params: {
  files: File[];
  labels: string[];
  subjectName: string;
  threshold: number;
  syllabusText: string;
}): Promise<AnalysisResponse> {
  // Extract 1 PDF per request in parallel so we never exceed Vercel's 4.5MB limit
  const extractedList = await Promise.all(
    params.files.map(async (file, i) => {
      const form = new FormData();
      form.append("file", file);
      form.append("paper_label", params.labels[i] || "");
      const res = await fetch(`${API_BASE}/extract-paper`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Failed to read ${file.name}`);
      }
      return res.json() as Promise<{
        paper: PaperInfo;
        questions: ExtractedQuestion[];
        warnings: string[];
      }>;
    })
  );

  const papers: PaperInfo[] = extractedList.map((e) => e.paper);
  const questions: ExtractedQuestion[] = extractedList.flatMap((e) => e.questions);
  const warnings: string[] = extractedList.flatMap((e) => e.warnings);

  const groupRes = await fetch(`${API_BASE}/group`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      subject_name: params.subjectName,
      similarity_threshold: params.threshold,
      papers,
      questions,
      warnings,
    }),
  });

  if (!groupRes.ok) {
    const err = await groupRes.json().catch(() => ({}));
    throw new Error(err.detail || "Grouping failed.");
  }
  return groupRes.json();
}

export async function reclusterAnalysis(
  analysis: AnalysisResponse,
  threshold: number
): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/recluster`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis, similarity_threshold: threshold }),
  });
  if (!res.ok) throw new Error("Failed to recluster.");
  return res.json();
}

export async function mergeClusters(
  analysis: AnalysisResponse,
  clusterIds: string[],
  customTitle?: string
): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis, cluster_ids: clusterIds, custom_title: customTitle }),
  });
  if (!res.ok) throw new Error("Failed to merge clusters.");
  return res.json();
}

export async function splitQuestion(
  analysis: AnalysisResponse,
  clusterId: string,
  questionId: string
): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis, cluster_id: clusterId, question_id: questionId }),
  });
  if (!res.ok) throw new Error("Failed to split question.");
  return res.json();
}

export async function downloadRevisionPdf(analysis: AnalysisResponse): Promise<void> {
  const res = await fetch(`${API_BASE}/export-pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(analysis),
  });
  if (!res.ok) throw new Error("Failed to generate PDF.");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const safeName =
    analysis.subject_name.replace(/[^a-zA-Z0-9 _-]/g, "_").trim() || "RepPYQ_Notes";
  a.href = url;
  a.download = `${safeName}_RepPYQ_Notes.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
''',

    "frontend/src/App.tsx": r'''
import React, { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { Clock, ExternalLink, Eye, FolderOpen, Moon, Sun } from "lucide-react";
import { AnalysisResponse } from "./types";
import {
  analyzePdfs,
  downloadRevisionPdf,
  mergeClusters,
  reclusterAnalysis,
  recordPageVisit,
  splitQuestion,
} from "./api/client";
import { UploadPanel } from "./components/UploadPanel";
import { Dashboard } from "./components/Dashboard";
import { RepPYQLogo } from "./components/RepPYQLogo";

const SPPU_PYQ_DRIVE_URL =
  "https://drive.google.com/drive/folders/0Bz9C0ysJZ7PnMGZKeWcybUpXWGM?resourcekey=0-S2yaWXvAG7ObM_GC8LRNTQ";

function formatISTTime(date: Date): string {
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  })
    .format(date)
    .toUpperCase();
}

export function App() {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [views, setViews] = useState<number | null>(null);
  const [istTime, setIstTime] = useState<string>(() => formatISTTime(new Date()));
  const visitedOnce = useRef(false);

  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem("reppyq-theme") === "dark";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    const timer = setInterval(() => {
      setIstTime(formatISTTime(new Date()));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (visitedOnce.current) return;
    visitedOnce.current = true;
    recordPageVisit()
      .then((count) => {
        if (count > 0) setViews(count);
      })
      .catch(() => {});
  }, []);

  const toggleTheme = (e: React.MouseEvent<HTMLButtonElement>) => {
    const nextDark = !dark;
    const apply = () => {
      document.documentElement.classList.toggle("dark", nextDark);
      try {
        localStorage.setItem("reppyq-theme", nextDark ? "dark" : "light");
      } catch {}
      setDark(nextDark);
    };

    const doc = document as Document & {
      startViewTransition?: (cb: () => void) => { ready: Promise<void> };
    };

    if (!doc.startViewTransition) {
      apply();
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX || rect.left + rect.width / 2;
    const y = e.clientY || rect.top + rect.height / 2;
    const endRadius = Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y)
    );

    const transition = doc.startViewTransition(() => {
      flushSync(apply);
    });

    transition.ready
      .then(() => {
        document.documentElement.animate(
          {
            clipPath: [
              `circle(0px at ${x}px ${y}px)`,
              `circle(${endRadius}px at ${x}px ${y}px)`,
            ],
          },
          {
            duration: 550,
            easing: "cubic-bezier(0.22, 1, 0.36, 1)",
            pseudoElement: "::view-transition-new(root)",
          }
        );
      })
      .catch(() => {});
  };

  const handleAnalyze = async (params: {
    files: File[];
    labels: string[];
    subjectName: string;
    threshold: number;
    syllabusText: string;
  }) => {
    setBusy(true);
    try {
      const res = await analyzePdfs(params);
      setAnalysis(res);
    } finally {
      setBusy(false);
    }
  };

  const handleRecluster = async (threshold: number) => {
    if (!analysis) return;
    setBusy(true);
    try {
      setAnalysis(await reclusterAnalysis(analysis, threshold));
    } finally {
      setBusy(false);
    }
  };

  const handleMerge = async (clusterIds: string[], customTitle?: string) => {
    if (!analysis) return;
    setBusy(true);
    try {
      setAnalysis(await mergeClusters(analysis, clusterIds, customTitle));
    } finally {
      setBusy(false);
    }
  };

  const handleSplit = async (clusterId: string, questionId: string) => {
    if (!analysis) return;
    setBusy(true);
    try {
      setAnalysis(await splitQuestion(analysis, clusterId, questionId));
    } finally {
      setBusy(false);
    }
  };

  const handleExportPdf = async () => {
    if (!analysis) return;
    setBusy(true);
    try {
      await downloadRevisionPdf(analysis);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-[#111318] text-slate-900 dark:text-slate-100">
      <header className="bg-white/95 dark:bg-[#151820]/95 backdrop-blur border-b border-slate-200 dark:border-[#272b36]">
        <div className="max-w-5xl mx-auto px-3 sm:px-4 py-3 flex flex-wrap items-center justify-between gap-2.5">
          <div className="flex items-center gap-2.5">
            <RepPYQLogo className="w-8 h-8 shrink-0" />
            <div>
              <h1 className="text-base font-bold text-slate-900 dark:text-slate-100 leading-none tracking-tight">
                RepPYQ
              </h1>
              <p className="text-[11px] sm:text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                SPPU 2019 Pattern · PYQ Analyzer
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <a
              href={SPPU_PYQ_DRIVE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white border border-slate-200 dark:border-[#2e3340] hover:border-slate-400 dark:hover:border-slate-500 hover:bg-slate-100 dark:hover:bg-[#1e222b] px-2.5 py-1.5 rounded-lg transition-all duration-150"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              <span>SPPU PYQ</span>
            </a>

            <a
              href="https://pushkarez.vercel.app/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 hover:text-slate-950 dark:hover:text-white bg-slate-100 dark:bg-[#1d212b] border border-slate-200 dark:border-[#2e3340] hover:border-slate-400 dark:hover:border-slate-500 px-2.5 sm:px-3 py-1.5 rounded-lg transition-all duration-150"
            >
              <span>Pushkar Shinde</span>
              <ExternalLink className="w-3 h-3" />
            </a>

            <button
              type="button"
              onClick={toggleTheme}
              aria-label="Toggle theme"
              title={dark ? "Switch to Light mode" : "Switch to Dark mode"}
              className="p-1.5 rounded-lg border border-slate-200 dark:border-[#2e3340] text-slate-600 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:border-slate-400 dark:hover:border-slate-500 hover:bg-slate-100 dark:hover:bg-[#1e222b] transition-all duration-150"
            >
              {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>

        <div className="border-t border-slate-100 dark:border-[#20242e] bg-slate-50/70 dark:bg-[#12151c]">
          <div className="max-w-5xl mx-auto px-3 sm:px-4 py-1.5 flex items-center justify-between text-[11px] sm:text-xs text-slate-500 dark:text-slate-400">
            <div className="inline-flex items-center gap-1.5 font-mono">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>IST · {istTime}</span>
            </div>

            <div className="inline-flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <Eye className="w-3.5 h-3.5 text-slate-400" />
              <span>{views !== null ? `${views.toLocaleString()} views` : "Live"}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 px-3 sm:px-4 py-5 sm:py-8">
        {!analysis ? (
          <UploadPanel onAnalyze={handleAnalyze} isLoading={busy} />
        ) : (
          <Dashboard
            analysis={analysis}
            onRecluster={handleRecluster}
            onMerge={handleMerge}
            onSplit={handleSplit}
            onExportPdf={handleExportPdf}
            onReset={() => setAnalysis(null)}
            busy={busy}
          />
        )}
      </main>

      <footer className="border-t border-slate-200 dark:border-[#272b36] bg-white dark:bg-[#151820] py-4 px-4 text-center text-xs text-slate-500 dark:text-slate-400">
        RepPYQ for SPPU Students · Built by{" "}
        <a
          href="https://pushkarez.vercel.app/"
          target="_blank"
          rel="noopener noreferrer"
          className="font-semibold text-slate-800 dark:text-slate-200 hover:underline"
        >
          Pushkar Shinde
        </a>
      </footer>
    </div>
  );
}

export default App;
''',
}

# Update Dashboard.tsx to use onExportPdf button instead of GET link
dash_path = Path("frontend/src/components/Dashboard.tsx")
dash_txt = dash_path.read_text(encoding="utf-8")
dash_txt = dash_txt.replace('import { getExportPdfUrl } from "../api/client";\n', "")
dash_txt = dash_txt.replace(
    "onSplit: (clusterId: string, questionId: string) => Promise<void>;",
    "onSplit: (clusterId: string, questionId: string) => Promise<void>;\n  onExportPdf: () => Promise<void>;",
)
dash_txt = dash_txt.replace(
    "  onSplit,\n  onReset,",
    "  onSplit,\n  onExportPdf,\n  onReset,",
)
dash_txt = dash_txt.replace(
    '<a\n              href={getExportPdfUrl(analysis.analysis_id)}\n              className="inline-flex items-center justify-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950 text-xs sm:text-sm font-medium hover:bg-slate-800 dark:hover:bg-white active:scale-[0.98] transition-all"\n            >\n              <Download className="w-4 h-4" /> Export Unit-Wise PDF\n            </a>',
    '<button\n              type="button"\n              disabled={busy}\n              onClick={onExportPdf}\n              className="inline-flex items-center justify-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950 text-xs sm:text-sm font-medium hover:bg-slate-800 dark:hover:bg-white active:scale-[0.98] disabled:opacity-50 transition-all"\n            >\n              <Download className="w-4 h-4" /> Export Unit-Wise PDF\n            </button>',
)
dash_path.write_text(dash_txt, encoding="utf-8")
print(f"Updated {dash_path}")

gi = Path(".gitignore")
if gi.exists():
    g = gi.read_text(encoding="utf-8")
    if "convert_to_vercel.py" not in g:
        gi.write_text(g.rstrip() + "\nconvert_to_vercel.py\n", encoding="utf-8")

for rel, content in FILES.items():
    p = Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"Created/Updated {p}")

print("\n100% Vercel Serverless conversion complete!")