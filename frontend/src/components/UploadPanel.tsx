import React, { useState } from "react";
import { AlertTriangle, FileText, Loader2, Upload, X } from "lucide-react";
import { PendingPaper } from "../types";
import { detectPaperLabels } from "../api/client";

interface UploadPanelProps {
  onAnalyze: (params: {
    files: File[];
    labels: string[];
    subjectName: string;
    threshold: number;
    syllabusText: string;
  }) => Promise<void>;
  isLoading: boolean;
}

export const UploadPanel: React.FC<UploadPanelProps> = ({ onAnalyze, isLoading }) => {
  const [papers, setPapers] = useState<PendingPaper[]>([]);
  const [subjectName, setSubjectName] = useState("");
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFilesSelected = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    const pdfs = Array.from(fileList).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfs.length === 0) {
      setError("Please select PDF files only.");
      return;
    }

    setError(null);
    setDetecting(true);
    try {
      const detected = await detectPaperLabels(pdfs);
      const mapped: PendingPaper[] = pdfs.map((file, i) => ({
        file,
        label: detected.papers[i]?.suggested_label || file.name.replace(/\.pdf$/i, ""),
        pageCount: detected.papers[i]?.page_count,
        usedOcr: detected.papers[i]?.used_ocr,
        warning: detected.papers[i]?.ocr_warning,
      }));
      setPapers((prev) => [...prev, ...mapped]);
    } catch {
      const mapped = pdfs.map((file) => ({
        file,
        label: file.name.replace(/\.pdf$/i, ""),
      }));
      setPapers((prev) => [...prev, ...mapped]);
    } finally {
      setDetecting(false);
    }
  };

  const updateLabel = (idx: number, val: string) => {
    setPapers((prev) => prev.map((p, i) => (i === idx ? { ...p, label: val } : p)));
  };

  const removePaper = (idx: number) => {
    setPapers((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (papers.length === 0) {
      setError("Upload at least one SPPU question paper PDF.");
      return;
    }
    setError(null);
    try {
      await onAnalyze({
        files: papers.map((p) => p.file),
        labels: papers.map((p) => p.label.trim() || p.file.name),
        subjectName: subjectName.trim() || "SPPU Subject Revision Notes",
        threshold: 0.56,
        syllabusText: "",
      });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to analyze PDFs.");
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-5">
      <div>
        <h2 className="text-xl sm:text-2xl font-semibold tracking-tight text-slate-900 dark:text-slate-100">
          Find Repeated Questions in SPPU Papers
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Drop your past SPPU exam PDFs (Insem or Endsem) from any branch. We sort every question into Unit 1&ndash;2 (Insem) or Unit 3&ndash;6 (Endsem) and show which ones repeat the most&mdash;so you know what to study first.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="bg-white dark:bg-[#181b22] border border-slate-200 dark:border-[#272b36] rounded-xl p-5 sm:p-6 space-y-5 shadow-sm">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
              Subject Name
            </label>
            <input
              type="text"
              value={subjectName}
              onChange={(e) => setSubjectName(e.target.value)}
              placeholder="e.g., Cloud Computing / Microwave Engineering / Embedded Systems"
              className="w-full rounded-lg border border-slate-300 dark:border-[#323746] bg-white dark:bg-[#12151b] px-3.5 py-2.5 text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:border-slate-500 dark:focus:border-slate-400 transition-colors"
              required
            />
          </div>

          <label className="group flex flex-col items-center justify-center border-2 border-dashed border-slate-300 dark:border-[#323746] rounded-xl p-7 sm:p-9 cursor-pointer hover:border-slate-500 dark:hover:border-slate-500 hover:bg-slate-50 dark:hover:bg-[#14171e] transition-all duration-200">
            <Upload className="w-7 h-7 text-slate-400 group-hover:text-slate-700 dark:group-hover:text-slate-200 group-hover:-translate-y-0.5 transition-all duration-200 mb-2" />
            <span className="text-sm font-medium text-slate-800 dark:text-slate-200 text-center">
              {detecting ? "Scanning PDF headers..." : "Click to upload SPPU Question Papers (PDF)"}
            </span>
            <span className="text-xs text-slate-500 dark:text-slate-400 mt-1 text-center">
              Select 3 to 10 papers in any order &mdash; auto-sorted chronologically
            </span>
            <input
              type="file"
              accept=".pdf,application/pdf"
              multiple
              className="hidden"
              onChange={(e) => handleFilesSelected(e.target.files)}
            />
          </label>

          {papers.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Uploaded Papers ({papers.length}) &mdash; Confirm Exam Labels
              </div>
              <div className="divide-y divide-slate-100 dark:divide-[#272b36] border border-slate-200 dark:border-[#272b36] rounded-lg overflow-hidden">
                {papers.map((p, idx) => (
                  <div
                    key={`${p.file.name}-${idx}`}
                    className="flex flex-wrap sm:flex-nowrap items-center gap-2.5 sm:gap-3 p-3 bg-white dark:bg-[#12151b]"
                  >
                    <FileText className="w-4 h-4 text-slate-400 shrink-0" />
                    <div className="min-w-0 flex-1">
                      <p className="text-xs text-slate-600 dark:text-slate-400 truncate">
                        {p.file.name}
                      </p>
                      {p.warning && (
                        <p className="text-xs text-rose-600 dark:text-rose-400 flex items-center gap-1 mt-0.5">
                          <AlertTriangle className="w-3 h-3 shrink-0" /> {p.warning}
                        </p>
                      )}
                    </div>
                    <input
                      type="text"
                      value={p.label}
                      onChange={(e) => updateLabel(idx, e.target.value)}
                      placeholder="e.g. May Jun 2024"
                      className="w-36 sm:w-40 rounded-md border border-slate-300 dark:border-[#323746] bg-white dark:bg-[#181b22] px-2.5 py-1 text-xs sm:text-sm font-medium text-slate-800 dark:text-slate-100 focus:outline-none focus:border-slate-500"
                    />
                    <button
                      type="button"
                      onClick={() => removePaper(idx)}
                      className="p-1 text-slate-400 hover:text-rose-500 transition-colors"
                      title="Remove paper"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-400 text-sm rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading || detecting || papers.length === 0}
          className="w-full flex items-center justify-center gap-2 bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950 font-medium py-3 px-5 rounded-xl hover:bg-slate-800 dark:hover:bg-white active:scale-[0.99] disabled:opacity-40 transition-all duration-200"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Extracting &amp; Grouping SPPU Questions...
            </>
          ) : (
            "Analyze Question Papers"
          )}
        </button>
      </form>
    </div>
  );
};
