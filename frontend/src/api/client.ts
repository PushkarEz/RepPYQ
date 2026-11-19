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
