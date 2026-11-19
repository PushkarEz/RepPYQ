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
