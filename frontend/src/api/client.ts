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
