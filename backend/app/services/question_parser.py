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
