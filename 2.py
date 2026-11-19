from pathlib import Path

FILES = {
    ".gitignore": r'''
.venv/
venv/
__pycache__/
*.pyc
.env
node_modules/
frontend/dist/
sample_pyqs/
app_stats.json
update_parser.py
upgrade_sppu_pro.py
upgrade_ui_chrono.py
upgrade_sppu_dark.py
upgrade_theme_polish.py
upgrade_reppyq.py
fix_upload_text.py
generate_test_pdfs.py
prep_deploy.py
''',

    "README.md": r'''
# RepPYQ — SPPU Question Paper Analyzer & Repeated PYQ Extractor

RepPYQ helps university students (built for **SPPU 2019 Pattern**) upload past exam PDFs, automatically extract exact verbatim questions, map them to **Unit 1–2 (Insem)** or **Unit 3–6 (Endsem)**, group rephrased repeated questions using semantic embeddings + FAISS, and export clean **Unit-Wise Revision PDFs**.

Built by **[Pushkar Shinde](https://pushkarez.vercel.app/)**

## Tech Stack
- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** FastAPI, PyMuPDF (`fitz`), Tesseract OCR fallback, `BAAI/bge-small-en-v1.5`, `faiss-cpu`, `scikit-learn`, `ReportLab`
''',

    "backend/Dockerfile": r'''
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/tmp/hf_cache
ENV STATS_FILE_PATH=/tmp/reppyq_stats.json

COPY . .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
''',

    "frontend/src/api/client.ts": r'''
import { AnalysisResponse } from "../types";

const API_BASE =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, "") ||
  "http://localhost:8000/api";

export async function recordPageVisit(): Promise<number> {
  const res = await fetch(`${API_BASE}/stats/visit`, { method: "POST" });
  if (!res.ok) return 0;
  const data = await res.json();
  return data.views ?? 0;
}

export async function detectPaperLabels(files: File[]) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API_BASE}/detect-labels`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Failed to scan uploaded PDFs.");
  return res.json() as Promise<{
    papers: Array<{
      filename: string;
      suggested_label: string;
      exam_type: string;
      page_count: number;
      used_ocr: boolean;
      ocr_warning: string | null;
    }>;
  }>;
}

export async function analyzePdfs(params: {
  files: File[];
  labels: string[];
  subjectName: string;
  threshold: number;
  syllabusText: string;
}): Promise<AnalysisResponse> {
  const form = new FormData();
  params.files.forEach((f) => form.append("files", f));
  form.append("paper_labels_json", JSON.stringify(params.labels));
  form.append("subject_name", params.subjectName);
  form.append("similarity_threshold", String(params.threshold));
  form.append("syllabus_text", params.syllabusText);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Analysis failed.");
  }
  return res.json();
}

export async function reclusterAnalysis(
  analysisId: string,
  threshold: number
): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/analysis/${encodeURIComponent(analysisId)}/recluster`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ similarity_threshold: threshold }),
  });
  if (!res.ok) throw new Error("Failed to recluster.");
  return res.json();
}

export async function mergeClusters(
  analysisId: string,
  clusterIds: string[],
  customTitle?: string
): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/analysis/${encodeURIComponent(analysisId)}/merge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cluster_ids: clusterIds, custom_title: customTitle }),
  });
  if (!res.ok) throw new Error("Failed to merge clusters.");
  return res.json();
}

export async function splitQuestion(
  analysisId: string,
  clusterId: string,
  questionId: string
): Promise<AnalysisResponse> {
  const res = await fetch(`${API_BASE}/analysis/${encodeURIComponent(analysisId)}/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cluster_id: clusterId, question_id: questionId }),
  });
  if (!res.ok) throw new Error("Failed to split question.");
  return res.json();
}

export function getExportPdfUrl(analysisId: string): string {
  return `${API_BASE}/analysis/${encodeURIComponent(analysisId)}/export-pdf`;
}
''',
}

# Update CORS and writable STATS_FILE path in backend/app/main.py
main_path = Path("backend/app/main.py")
main_txt = main_path.read_text(encoding="utf-8")
if "import os" not in main_txt:
    main_txt = "import os\n" + main_txt
main_txt = main_txt.replace(
    'STATS_FILE = Path("app_stats.json")',
    'STATS_FILE = Path(os.getenv("STATS_FILE_PATH", "app_stats.json"))',
)
main_txt = main_txt.replace(
    'allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],',
    'allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],\n    allow_origin_regex=r"https://.*\\.vercel\\.app",',
)
main_path.write_text(main_txt, encoding="utf-8")
print(f"Updated {main_path}")

for rel, content in FILES.items():
    p = Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"Created/Updated {p}")

print("\nReady for GitHub & Deployment!")