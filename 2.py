from pathlib import Path

FILES = {
    "frontend/src/index.css": r'''
@tailwind base;
@tailwind components;
@tailwind utilities;

html {
  background-color: #f8fafc;
}

html.dark {
  background-color: #111318;
}

/* Circular Wave View Transition for Theme Toggle */
::view-transition-old(root),
::view-transition-new(root) {
  animation: none;
  mix-blend-mode: normal;
}

::view-transition-old(root) {
  z-index: 1;
}

::view-transition-new(root) {
  z-index: 9999;
}

/* Subtle circular wave ripple for PDF drag & drop */
@keyframes dropWave {
  0% {
    transform: scale(0.75);
    opacity: 0.55;
  }
  100% {
    transform: scale(1.75);
    opacity: 0;
  }
}

.animate-drop-wave {
  animation: dropWave 1.3s cubic-bezier(0.22, 1, 0.36, 1) infinite;
}
''',

    "frontend/src/components/UploadPanel.tsx": r'''
import React, { useRef, useState } from "react";
import { AlertTriangle, FileText, Loader2, Plus, X } from "lucide-react";
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

const FolderDropIcon: React.FC<{ active: boolean }> = ({ active }) => (
  <svg
    viewBox="0 0 64 56"
    fill="none"
    className={`w-14 h-12 transition-transform duration-300 ${
      active ? "scale-105 -translate-y-0.5" : "group-hover:-translate-y-0.5"
    }`}
    aria-hidden="true"
  >
    {/* Back folder tab */}
    <path
      d="M6 12C6 8.68629 8.68629 6 12 6H23.5C25.2 6 26.8 6.8 27.9 8.1L30.5 11.2H52C55.3137 11.2 58 13.8863 58 17.2V44C58 47.3137 55.3137 50 52 50H12C8.68629 50 6 47.3137 6 44V12Z"
      className="fill-slate-200 dark:fill-[#2b3140]"
    />
    {/* Peeking PDF document */}
    <path
      d="M17 14C17 11.7909 18.7909 10 21 10H39L47 18V36H17V14Z"
      className="fill-white dark:fill-[#1d212c] stroke-slate-300 dark:stroke-[#3b4254]"
      strokeWidth="1.2"
    />
    <path
      d="M39 10V16C39 17.1046 39.8954 18 41 18H47"
      className="fill-slate-100 dark:fill-[#282d3b]"
    />
    {/* Front folder flap */}
    <path
      d="M4 22C4 19.7909 5.79086 18 8 18H56C58.2091 18 60 19.7909 60 22L57.5 45C57.15 47.85 54.73 50 51.86 50H12.14C9.27 50 6.85 47.85 6.5 45L4 22Z"
      className="fill-slate-100/95 dark:fill-[#232836] stroke-slate-300/80 dark:stroke-[#383f50]"
      strokeWidth="1.2"
    />
  </svg>
);

export const UploadPanel: React.FC<UploadPanelProps> = ({ onAnalyze, isLoading }) => {
  const [papers, setPapers] = useState<PendingPaper[]>([]);
  const [subjectName, setSubjectName] = useState("");
  const [detecting, setDetecting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [justDropped, setJustDropped] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    setJustDropped(true);
    setTimeout(() => setJustDropped(false), 1200);
    handleFilesSelected(e.dataTransfer.files);
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

  const showWave = isDragging || justDropped || detecting;

  return (
    <div className="max-w-3xl mx-auto space-y-5">
      <div>
        <h2 className="text-xl sm:text-2xl font-semibold tracking-tight text-slate-900 dark:text-slate-100">
          Find Repeated Questions in SPPU Papers
        </h2>
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 mt-1 leading-relaxed">
          Drop your past SPPU exam PDFs (Insem or Endsem) from any branch. We sort every question into Unit 1&ndash;2 (Insem) or Unit 3&ndash;6 (Endsem) and show which ones repeat the most&mdash;so you know what to study first.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="bg-white dark:bg-[#181b22] border border-slate-200 dark:border-[#272b36] rounded-2xl p-4 sm:p-6 space-y-5 shadow-sm">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1.5">
              Subject Name
            </label>
            <input
              type="text"
              value={subjectName}
              onChange={(e) => setSubjectName(e.target.value)}
              placeholder="e.g., Cloud Computing / VLSI / Microwave Engineering"
              className="w-full rounded-xl border border-slate-300 dark:border-[#323746] bg-white dark:bg-[#12151b] px-3.5 py-2.5 text-sm text-slate-900 dark:text-slate-100 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-none focus:border-slate-500 dark:focus:border-slate-400 transition-colors"
              required
            />
          </div>

          {/* Drag & Drop PDF Box */}
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragEnter={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                fileInputRef.current?.click();
              }
            }}
            className={`group relative overflow-hidden flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-4 py-8 sm:py-11 cursor-pointer transition-all duration-200 ${
              isDragging
                ? "border-slate-700 dark:border-slate-300 bg-slate-100/90 dark:bg-[#1d222e] scale-[1.01]"
                : "border-slate-300 dark:border-[#323746] bg-slate-50/70 dark:bg-[#13161d] hover:border-slate-400 dark:hover:border-slate-500 hover:bg-slate-100/60 dark:hover:bg-[#161a23]"
            }`}
          >
            {/* Circular Wave Ripple */}
            <div className="relative flex items-center justify-center mb-3">
              {showWave && (
                <>
                  <span className="animate-drop-wave absolute w-16 h-16 rounded-full border border-slate-400/60 dark:border-slate-400/40 pointer-events-none" />
                  <span
                    className="animate-drop-wave absolute w-16 h-16 rounded-full border border-slate-400/40 dark:border-slate-400/25 pointer-events-none"
                    style={{ animationDelay: "0.45s" }}
                  />
                </>
              )}
              <FolderDropIcon active={isDragging} />
            </div>

            <p className="text-sm sm:text-base font-medium text-slate-800 dark:text-slate-100 text-center">
              {detecting
                ? "Scanning PDF headers..."
                : isDragging
                ? "Drop PDFs here"
                : "Upload PDF"}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 text-center max-w-xs">
              Drag &amp; drop 3 to 10 SPPU question papers here (auto-sorted chronologically)
            </p>

            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
              className="mt-4 inline-flex items-center gap-1.5 rounded-lg bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950 px-4 py-2 text-xs font-medium shadow-sm hover:bg-slate-800 dark:hover:bg-white active:scale-[0.97] transition-all duration-150"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Add files</span>
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,application/pdf"
              multiple
              className="hidden"
              onChange={(e) => {
                handleFilesSelected(e.target.files);
                e.target.value = "";
              }}
            />
          </div>

          {papers.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Uploaded Papers ({papers.length}) &mdash; Confirm Exam Labels
              </div>
              <div className="divide-y divide-slate-100 dark:divide-[#272b36] border border-slate-200 dark:border-[#272b36] rounded-xl overflow-hidden">
                {papers.map((p, idx) => (
                  <div
                    key={`${p.file.name}-${idx}`}
                    className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 p-3 bg-white dark:bg-[#12151b]"
                  >
                    <div className="flex items-center gap-2.5 min-w-0 flex-1">
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
                    </div>

                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={p.label}
                        onChange={(e) => updateLabel(idx, e.target.value)}
                        placeholder="e.g. May Jun 2024"
                        className="w-full sm:w-40 rounded-lg border border-slate-300 dark:border-[#323746] bg-white dark:bg-[#181b22] px-2.5 py-1.5 text-xs sm:text-sm font-medium text-slate-800 dark:text-slate-100 focus:outline-none focus:border-slate-500"
                      />
                      <button
                        type="button"
                        onClick={() => removePaper(idx)}
                        className="p-1.5 text-slate-400 hover:text-rose-500 transition-colors shrink-0"
                        title="Remove paper"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="bg-rose-500/10 border border-rose-500/30 text-rose-600 dark:text-rose-400 text-sm rounded-xl px-4 py-3">
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
''',

    "frontend/src/components/ClusterCard.tsx": r'''
import React from "react";
import { Check, ChevronDown, ChevronRight, Minus, Unlink } from "lucide-react";
import { PaperInfo, QuestionCluster } from "../types";

interface ClusterCardProps {
  cluster: QuestionCluster;
  allPapers: PaperInfo[];
  expanded: boolean;
  selectedForMerge: boolean;
  onToggleExpand: (clusterId: string) => void;
  onToggleSelect: (clusterId: string) => void;
  onSplitQuestion: (clusterId: string, questionId: string) => void;
}

const PRIORITY_STYLES: Record<string, string> = {
  "High Priority":
    "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-500/10 dark:text-rose-300 dark:border-rose-500/25",
  "Medium Priority":
    "bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-500/10 dark:text-sky-300 dark:border-sky-500/25",
  "Low / Unique":
    "bg-slate-100 text-slate-600 border-slate-200 dark:bg-[#222631] dark:text-slate-400 dark:border-[#323746]",
};

export const ClusterCard: React.FC<ClusterCardProps> = ({
  cluster,
  allPapers,
  expanded,
  selectedForMerge,
  onToggleExpand,
  onToggleSelect,
  onSplitQuestion,
}) => {
  const presentPaperIds = new Set(cluster.questions.map((q) => q.paper_id));
  const timesLabel = cluster.total_occurrences === 1 ? "time" : "times";

  return (
    <div
      className={`bg-white dark:bg-[#181b22] border rounded-xl transition-all duration-200 ${
        selectedForMerge
          ? "border-indigo-500 ring-1 ring-indigo-500/50"
          : "border-slate-200 dark:border-[#272b36] hover:border-slate-300 dark:hover:border-[#3b4152]"
      }`}
    >
      <div className="flex items-start justify-between gap-2 sm:gap-4 p-3.5 sm:p-4">
        <div className="flex items-start gap-2.5 sm:gap-3 min-w-0 flex-1">
          <input
            type="checkbox"
            checked={selectedForMerge}
            onChange={() => onToggleSelect(cluster.cluster_id)}
            title="Select cluster to merge"
            className="mt-1 h-4 w-4 rounded border-slate-300 dark:border-slate-600 accent-indigo-500 cursor-pointer shrink-0"
          />
          <button
            type="button"
            onClick={() => onToggleExpand(cluster.cluster_id)}
            className="text-left min-w-0 flex-1 group"
          >
            <h3 className="text-sm sm:text-base font-semibold text-slate-900 dark:text-slate-100 leading-snug break-words group-hover:text-slate-600 dark:group-hover:text-white transition-colors">
              {cluster.topic_title}
            </h3>
            <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] sm:text-xs font-semibold bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-950">
                {cluster.unique_paper_count} / {cluster.total_papers} papers &middot;{" "}
                {cluster.total_occurrences} {timesLabel}
              </span>
              <span
                className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] sm:text-xs font-medium border ${
                  PRIORITY_STYLES[cluster.priority]
                }`}
              >
                {cluster.priority}
              </span>
              {cluster.unit_label && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] sm:text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-200 dark:bg-indigo-500/10 dark:text-indigo-300 dark:border-indigo-500/25">
                  {cluster.unit_label}
                </span>
              )}
            </div>
          </button>
        </div>

        <button
          type="button"
          onClick={() => onToggleExpand(cluster.cluster_id)}
          className="shrink-0 inline-flex items-center gap-1 text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 px-2 py-1 rounded-md hover:bg-slate-100 dark:hover:bg-[#232732] transition-colors"
        >
          <span className="hidden sm:inline">
            {expanded ? "Hide" : `${cluster.questions.length} Qs`}
          </span>
          {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      </div>

      {expanded && (
        <div className="px-3.5 sm:px-4 pb-4 space-y-2.5 border-t border-slate-100 dark:border-[#272b36] pt-3">
          {allPapers.length > 1 && cluster.unique_paper_count >= 2 && (
            <div className="flex flex-wrap gap-1.5 pb-1">
              {allPapers.map((p) => {
                const asked = presentPaperIds.has(p.paper_id);
                return (
                  <span
                    key={p.paper_id}
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium border ${
                      asked
                        ? "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-300 dark:border-emerald-500/25"
                        : "bg-slate-50 text-slate-400 border-slate-200 dark:bg-[#12151b] dark:text-slate-500 dark:border-[#232732]"
                    }`}
                  >
                    {asked ? <Check className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
                    {p.paper_name}
                  </span>
                );
              })}
            </div>
          )}

          {cluster.questions.map((q) => (
            <div
              key={q.id}
              className="group flex items-start justify-between gap-2.5 rounded-lg bg-slate-50 dark:bg-[#12151b] border border-slate-100 dark:border-[#232732] px-3 py-2 text-xs sm:text-sm"
            >
              <div className="text-slate-800 dark:text-slate-300 leading-relaxed break-words min-w-0">
                <span className="font-semibold text-slate-900 dark:text-slate-100">
                  {q.paper_name} · {q.question_number}
                  {q.marks !== null ? ` [${q.marks}M]` : ""}:
                </span>{" "}
                <span>&ldquo;{q.exact_text}&rdquo;</span>
              </div>
              {cluster.questions.length > 1 && (
                <button
                  type="button"
                  onClick={() => onSplitQuestion(cluster.cluster_id, q.id)}
                  title="Split this question into its own cluster"
                  className="opacity-70 sm:opacity-0 sm:group-hover:opacity-100 shrink-0 p-1 text-slate-400 hover:text-slate-900 dark:hover:text-white transition-all"
                >
                  <Unlink className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
''',
}

# Update Dashboard sticky breakpoint to md:sticky + mobile full-width buttons
dash_path = Path("frontend/src/components/Dashboard.tsx")
dash_txt = dash_path.read_text(encoding="utf-8")
dash_txt = dash_txt.replace("sm:sticky sm:top-2", "md:sticky md:top-2")
dash_txt = dash_txt.replace(
    'className="flex flex-wrap items-center gap-2"',
    'className="grid grid-cols-2 sm:flex sm:flex-wrap items-center gap-2 w-full sm:w-auto"',
)
dash_txt = dash_txt.replace(
    'className="inline-flex items-center gap-1.5 px-3 py-2',
    'className="inline-flex items-center justify-center gap-1.5 px-3 py-2',
)
dash_txt = dash_txt.replace(
    'className="inline-flex items-center gap-1.5 px-3.5 py-2',
    'className="inline-flex items-center justify-center gap-1.5 px-3.5 py-2',
)
dash_path.write_text(dash_txt, encoding="utf-8")
print(f"Updated {dash_path}")

# Update PDF Exporter to show total times asked + unique papers clearly
pdf_path = Path("backend/app/services/pdf_exporter.py")
pdf_txt = pdf_path.read_text(encoding="utf-8")
pdf_txt = pdf_txt.replace(
    'f"{tag}   ·   Appeared in {cl.unique_paper_count} of {cl.total_papers} papers",',
    'f"{tag}   ·   Asked {cl.total_occurrences} time{\'s\' if cl.total_occurrences != 1 else \'\'} across {cl.unique_paper_count} of {cl.total_papers} papers",',
)
pdf_path.write_text(pdf_txt, encoding="utf-8")
print(f"Updated {pdf_path}")

gi = Path(".gitignore")
if gi.exists():
    g = gi.read_text(encoding="utf-8")
    if "upgrade_ux_final.py" not in g:
        gi.write_text(g.rstrip() + "\nupgrade_ux_final.py\n", encoding="utf-8")

for rel, content in FILES.items():
    p = Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"Updated {p}")

print("\nDrag-and-drop wave dropzone + exact paper/time counts + mobile polish applied!")