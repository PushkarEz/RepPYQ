from pathlib import Path

# Safety check: make sure we are in D:\PDF PYQ\pyqradar
if not Path("frontend/package.json").exists():
    raise SystemExit("ERROR: Run this script inside D:\\PDF PYQ\\pyqradar !")

Path("frontend/.python-version").write_text("3.12\n", encoding="utf-8")

Path("frontend/requirements.txt").write_text(
    "fastapi>=0.115.6\n"
    "pydantic>=2.10.4\n"
    "pdfminer.six>=20231228\n"
    "reportlab>=4.2.5\n"
    "httpx>=0.28.1\n"
    "python-multipart>=0.0.20\n",
    encoding="utf-8",
)

PDF_EXTRACTOR = '''import io
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTAnno, LTChar, LTTextLine


NOISE_LINE_PATTERNS = [
    r"^\\s*P\\.?\\s*T\\.?\\s*O\\.?\\s*$",
    r"^\\s*Total\\s+No\\.?\\s+of\\s+(Questions|Pages)\\b.*$",
    r"^\\s*\\[?\\s*Total\\s+No\\.?\\s+of\\s+Pages\\s*[:\\-].*$",
    r"^\\s*Seat\\s*No\\.?\\s*[:\\-]?.*$",
    r"^\\s*P[A-Z]?\\s*\\d{3,6}\\s*$",
    r"^\\s*\\[\\d{3,5}\\]\\s*-\\s*\\d+.*$",
    r"^\\s*Time\\s*[:\\-]\\s*[\\d½\\.\\s]+Hours?.*$",
    r"^\\s*\\[?\\s*Max\\.?\\s*Marks\\s*[:\\-]\\s*\\d+.*$",
    r"^\\s*\\(?\\s*Total\\s+Marks\\s*[:\\-]\\s*\\d+\\s*\\)?.*$",
    r"^\\s*Instructions\\s+to\\s+(?:the\\s+)?candidates\\s*[:\\-]?.*$",
    r"^\\s*\\d+\\)\\s*(?:All\\s+questions|Solve\\s+Q|Answer\\s+Q|Neat\\s+diagrams|Assume\\s+suitable|Figures\\s+to\\s+the\\s+right|Use\\s+of\\s+(?:logarithmic|electronic|non-programmable)).*$",
    r"^\\s*[-–—]?\\s*\\d+\\s*[-–—]?\\s*$",
    r"^\\s*Page\\s+\\d+\\s*(?:of\\s+\\d+)?\\s*$",
    r"^\\s*(?:F\\.?E\\.?|S\\.?E\\.?|T\\.?E\\.?|B\\.?E\\.?|M\\.?E\\.?)\\s*\\(.*\\)\\s*$",
    r"^\\s*\\(\\s*\\d{4}\\s+(?:Pattern|Course|Credit).*$",
    r"^\\s*[\\*•\\-_=\\uf000-\\uf8ff\\s]{2,}$",
    r"^\\s*N\\.?B\\.?\\s*[:\\-].*$",
    r"^\\s*[A-Z]{2,6}\\d{4,8}\\s*$",
    r"^.*\\b\\d{1,3}(?:\\.\\d{1,3}){3}\\b.*static-\\d+.*$",
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
        r"\\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\\s\\-_,/]*?(20\\d{2})\\b",
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

    stem = re.sub(r"\\.pdf$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"[_\\-]+", " ", stem).strip()
    return stem[:40] or "Unnamed Paper"


def detect_exam_type(raw_text: str, filename: str = "") -> str:
    header_low = (raw_text[:2500] + " " + filename).lower()
    if any(k in header_low for k in ("insem", "in-sem", "in semester", "in-semester")):
        return "Insem"
    if re.search(r"(?:max\\.?|total)\\s*marks\\s*[:\\-]?\\s*30\\b", header_low):
        return "Insem"
    if re.search(r"time\\s*[:\\-]?\\s*1\\s*(?:hr|hour)\\b", header_low):
        return "Insem"
    if re.search(r"(?:max\\.?|total)\\s*marks\\s*[:\\-]?\\s*70\\b", header_low):
        return "Endsem"
    if re.search(r"\\bQ\\.?\\s*[5-8]\\s*[\\.\\)\\]:\\-]", raw_text, re.IGNORECASE):
        return "Endsem"

    has_q1_to_4 = bool(re.search(r"\\bQ\\.?\\s*[1-4]\\s*[\\.\\)\\]:\\-]", raw_text, re.IGNORECASE))
    has_q5_to_8 = bool(re.search(r"\\bQ\\.?\\s*[5-8]\\s*[\\.\\)\\]:\\-]", raw_text, re.IGNORECASE))
    if has_q1_to_4 and not has_q5_to_8:
        return "Insem"
    return "Endsem"


def _extract_page_horizontal_lines(page_layout) -> str:
    spans: List[Tuple[float, float, str]] = []

    def walk(obj):
        if isinstance(obj, LTTextLine):
            chars = []
            x0 = None
            y_sum = 0.0
            y_cnt = 0
            for child in obj:
                if isinstance(child, LTChar):
                    m = child.matrix
                    if abs(m[1]) > 0.15 or abs(m[2]) > 0.15 or m[0] <= 0:
                        continue
                    ch = re.sub(r"[\\uf000-\\uf8ff]+", "", child.get_text())
                    if not ch:
                        continue
                    if x0 is None:
                        x0 = child.x0
                    y_sum += (child.y0 + child.y1) / 2.0
                    y_cnt += 1
                    chars.append(ch)
                elif isinstance(child, LTAnno):
                    if chars and child.get_text() in (" ", "\\t"):
                        chars.append(" ")
            text = "".join(chars).strip()
            if text and x0 is not None and y_cnt > 0:
                spans.append((x0, y_sum / y_cnt, text))
            return

        if hasattr(obj, "__iter__"):
            for child in obj:
                walk(child)

    walk(page_layout)
    if not spans:
        return ""

    spans.sort(key=lambda s: (-s[1], s[0]))
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
    return "\\n".join(out_lines)


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
        if re.match(r"^(?:www\\.|https?://|downloaded\\s+from|join\\s+telegram)", s, re.IGNORECASE):
            continue
        s = re.sub(r"^\\[\\d{3,5}\\]-\\d+\\s+\\d+\\s*$", "", s).strip()
        if s:
            cleaned.append(s)
    return "\\n".join(cleaned)


def extract_pdf_text(pdf_bytes: bytes, filename: str = "") -> ExtractedPDFResult:
    laparams = LAParams(char_margin=2.0, line_margin=0.5, word_margin=0.1)
    page_count = 0
    raw_pages: List[str] = []
    pages_text: List[str] = []
    raw_first_page = ""
    ocr_warning: Optional[str] = None

    for idx, page_layout in enumerate(extract_pages(io.BytesIO(pdf_bytes), password="", laparams=laparams)):
        page_count += 1
        text = _extract_page_horizontal_lines(page_layout)
        if idx == 0:
            raw_first_page = text
        if len(text.strip()) < 40:
            ocr_warning = (
                f"Page {idx + 1} appears to be a scanned image (no selectable text layer)."
            )
        raw_pages.append(text)
        pages_text.append(_strip_noise_lines(text))

    raw_combined = "\\n".join(raw_pages)
    combined = "\\n".join(pages_text)
    exam_type = detect_exam_type(raw_combined, filename)
    return ExtractedPDFResult(
        cleaned_text=combined,
        raw_first_page=raw_first_page,
        page_count=page_count,
        exam_type=exam_type,
        used_ocr=False,
        ocr_warning=ocr_warning,
    )
'''

Path("frontend/api/_lib/pdf_extractor.py").write_text(PDF_EXTRACTOR, encoding="utf-8")

gi = Path(".gitignore")
if gi.exists():
    g = gi.read_text(encoding="utf-8")
    if "fix_now.py" not in g:
        gi.write_text(g.rstrip() + "\nfix_now.py\n", encoding="utf-8")

print("SUCCESS! frontend/requirements.txt is now:")
print(Path("frontend/requirements.txt").read_text(encoding="utf-8"))