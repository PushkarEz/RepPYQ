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
