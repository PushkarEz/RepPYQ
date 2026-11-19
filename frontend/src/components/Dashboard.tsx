import React, { useEffect, useMemo, useState } from "react";
import {
  ChevronsDownUp,
  ChevronsUpDown,
  Download,
  GitMerge,
  RotateCcw,
  Search,
  AlertTriangle,
} from "lucide-react";
import { AnalysisResponse, PriorityTier } from "../types";
import { ClusterCard } from "./ClusterCard";
import { getExportPdfUrl } from "../api/client";

interface DashboardProps {
  analysis: AnalysisResponse;
  onRecluster: (threshold: number) => Promise<void>;
  onMerge: (clusterIds: string[], customTitle?: string) => Promise<void>;
  onSplit: (clusterId: string, questionId: string) => Promise<void>;
  onReset: () => void;
  busy: boolean;
}

export const Dashboard: React.FC<DashboardProps> = ({
  analysis,
  onRecluster,
  onMerge,
  onSplit,
  onReset,
  busy,
}) => {
  const [threshold, setThreshold] = useState(analysis.similarity_threshold);
  const [priorityFilter, setPriorityFilter] = useState<"ALL" | PriorityTier>("ALL");
  const [unitFilter, setUnitFilter] = useState<string>("ALL");
  const [search, setSearch] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const initial: Record<string, boolean> = {};
    analysis.clusters.forEach((c) => {
      initial[c.cluster_id] = c.priority === "High Priority";
    });
    setExpandedIds(initial);
  }, [analysis.clusters]);

  const availableUnits = useMemo(() => {
    const units = new Set<string>();
    analysis.clusters.forEach((c) => {
      if (c.unit_label) {
        units.add(c.unit_label.split("·")[0].trim());
      }
    });
    return Array.from(units).sort();
  }, [analysis.clusters]);

  const filteredClusters = useMemo(() => {
    return analysis.clusters.filter((c) => {
      if (priorityFilter !== "ALL" && c.priority !== priorityFilter) return false;
      if (unitFilter !== "ALL") {
        const u = (c.unit_label || "").split("·")[0].trim();
        if (u !== unitFilter) return false;
      }
      if (search.trim()) {
        const needle = search.toLowerCase();
        const inTitle = c.topic_title.toLowerCase().includes(needle);
        const inQs = c.questions.some(
          (q) =>
            q.exact_text.toLowerCase().includes(needle) ||
            q.paper_name.toLowerCase().includes(needle)
        );
        return inTitle || inQs;
      }
      return true;
    });
  }, [analysis.clusters, priorityFilter, unitFilter, search]);

  const counts = useMemo(() => {
    const high = analysis.clusters.filter((c) => c.priority === "High Priority").length;
    const med = analysis.clusters.filter((c) => c.priority === "Medium Priority").length;
    const low = analysis.clusters.filter((c) => c.priority === "Low / Unique").length;
    return { high, med, low };
  }, [analysis.clusters]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const setAllExpanded = (open: boolean) => {
    const next: Record<string, boolean> = {};
    filteredClusters.forEach((c) => {
      next[c.cluster_id] = open;
    });
    setExpandedIds((prev) => ({ ...prev, ...next }));
  };

  const handleMerge = async () => {
    if (selectedIds.length < 2) return;
    const customTitle = window.prompt(
      "Optional: Enter a merged topic title (leave blank to keep first title):"
    );
    await onMerge(selectedIds, customTitle || undefined);
    setSelectedIds([]);
  };

  const allVisibleExpanded =
    filteredClusters.length > 0 && filteredClusters.every((c) => expandedIds[c.cluster_id]);

  return (
    <div className="max-w-5xl mx-auto space-y-4 sm:space-y-5">
      {/* Header Summary */}
      <div className="bg-white dark:bg-[#181b22] border border-slate-200 dark:border-[#272b36] rounded-xl p-4 sm:p-5 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-lg sm:text-xl font-bold text-slate-900 dark:text-slate-100">
              {analysis.subject_name}
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-400 mt-0.5">
              {analysis.total_papers} papers · {analysis.total_questions} exact questions ·{" "}
              {analysis.clusters.length} topics
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-300 dark:border-[#323746] text-xs sm:text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-[#232732] transition-colors"
            >
              <RotateCcw className="w-4 h-4" /> New Upload
            </button>
            <a
              href={getExportPdfUrl(analysis.analysis_id)}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950 text-xs sm:text-sm font-medium hover:bg-slate-800 dark:hover:bg-white active:scale-[0.98] transition-all"
            >
              <Download className="w-4 h-4" /> Export Unit-Wise PDF
            </a>
          </div>
        </div>

        {/* Chronological Paper Strip */}
        <div className="flex flex-wrap gap-1.5 pt-2 border-t border-slate-100 dark:border-[#272b36]">
          {analysis.papers.map((p) => (
            <span
              key={p.paper_id}
              className="text-[11px] sm:text-xs bg-slate-100 dark:bg-[#222631] text-slate-700 dark:text-slate-300 px-2.5 py-0.5 rounded-md font-medium"
            >
              {p.paper_name} · {p.extracted_count} Qs
            </span>
          ))}
        </div>
      </div>

      {analysis.warnings.length > 0 && (
        <div className="bg-rose-500/10 border border-rose-500/25 rounded-xl p-3.5 text-sm text-rose-700 dark:text-rose-300 space-y-1">
          <div className="font-semibold flex items-center gap-1.5 text-xs sm:text-sm">
            <AlertTriangle className="w-4 h-4" /> Extraction Notes
          </div>
          {analysis.warnings.map((w, i) => (
            <p key={i} className="text-xs">{w}</p>
          ))}
        </div>
      )}

      {/* Compact Sticky Filter Controls */}
      <div className="bg-white dark:bg-[#181b22] border border-slate-200 dark:border-[#272b36] rounded-xl p-4 space-y-3 sm:sticky sm:top-2 z-10 shadow-sm">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-5 items-center">
          <div>
            <div className="flex justify-between text-xs sm:text-sm mb-1">
              <span className="font-medium text-slate-700 dark:text-slate-300">
                Grouping Strictness
              </span>
              <span className="font-mono text-xs bg-slate-100 dark:bg-[#222631] text-slate-700 dark:text-slate-300 px-2 py-0.5 rounded">
                {threshold.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <input
                type="range"
                min={0.5}
                max={0.9}
                step={0.02}
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-indigo-500"
              />
              <button
                type="button"
                disabled={busy || threshold === analysis.similarity_threshold}
                onClick={() => onRecluster(threshold)}
                className="shrink-0 px-3 py-1.5 rounded-lg bg-slate-900 dark:bg-slate-100 text-white dark:text-slate-950 text-xs font-medium hover:bg-slate-800 dark:hover:bg-white disabled:opacity-40 transition-colors"
              >
                Apply
              </button>
            </div>
          </div>

          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search topic, question, or year..."
              className="w-full pl-9 pr-3 py-2 text-xs sm:text-sm border border-slate-300 dark:border-[#323746] bg-white dark:bg-[#12151b] text-slate-900 dark:text-slate-100 placeholder:text-slate-400 rounded-lg focus:outline-none focus:border-slate-500"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-slate-100 dark:border-[#272b36]">
          <div className="flex flex-wrap gap-1.5">
            {(
              [
                ["ALL", `All (${analysis.clusters.length})`],
                ["High Priority", `High (${counts.high})`],
                ["Medium Priority", `Medium (${counts.med})`],
                ["Low / Unique", `Once (${counts.low})`],
              ] as const
            ).map(([val, label]) => (
              <button
                key={val}
                type="button"
                onClick={() => setPriorityFilter(val)}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
                  priorityFilter === val
                    ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-950"
                    : "bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-[#222631] dark:text-slate-300 dark:hover:bg-[#2c3140]"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setAllExpanded(!allVisibleExpanded)}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border border-slate-200 dark:border-[#323746] text-xs font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-[#232732] transition-colors"
            >
              {allVisibleExpanded ? (
                <>
                  <ChevronsDownUp className="w-3.5 h-3.5" /> Collapse All
                </>
              ) : (
                <>
                  <ChevronsUpDown className="w-3.5 h-3.5" /> Expand All
                </>
              )}
            </button>

            {selectedIds.length >= 2 && (
              <button
                type="button"
                disabled={busy}
                onClick={handleMerge}
                className="inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-500 transition-colors"
              >
                <GitMerge className="w-3.5 h-3.5" /> Merge ({selectedIds.length})
              </button>
            )}
          </div>
        </div>

        {availableUnits.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-2 border-t border-slate-100 dark:border-[#272b36]">
            <button
              type="button"
              onClick={() => setUnitFilter("ALL")}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                unitFilter === "ALL"
                  ? "bg-indigo-600 text-white"
                  : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-500/10 dark:text-indigo-300 dark:hover:bg-indigo-500/20"
              }`}
            >
              All Units
            </button>
            {availableUnits.map((u) => (
              <button
                key={u}
                type="button"
                onClick={() => setUnitFilter(u)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  unitFilter === u
                    ? "bg-indigo-600 text-white"
                    : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-500/10 dark:text-indigo-300 dark:hover:bg-indigo-500/20"
                }`}
              >
                {u}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Compact Cluster List */}
      <div className="space-y-2.5">
        {filteredClusters.map((cluster) => (
          <ClusterCard
            key={cluster.cluster_id}
            cluster={cluster}
            expanded={!!expandedIds[cluster.cluster_id]}
            selectedForMerge={selectedIds.includes(cluster.cluster_id)}
            onToggleExpand={toggleExpand}
            onToggleSelect={toggleSelect}
            onSplitQuestion={onSplit}
          />
        ))}
        {filteredClusters.length === 0 && (
          <div className="text-center py-10 text-sm text-slate-500 dark:text-slate-400 bg-white dark:bg-[#181b22] border border-slate-200 dark:border-[#272b36] rounded-xl">
            No question clusters match the current filter.
          </div>
        )}
      </div>
    </div>
  );
};
