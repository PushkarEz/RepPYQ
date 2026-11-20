import React, { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { Clock, ExternalLink, Eye, FolderOpen, Moon, Sun } from "lucide-react";
import { AnalysisResponse } from "./types";
import {
  analyzePdfs,
  downloadRevisionPdf,
  mergeClusters,
  reclusterAnalysis,
  recordPageVisit,
  splitQuestion,
} from "./api/client";
import { UploadPanel } from "./components/UploadPanel";
import { Dashboard } from "./components/Dashboard";
import { RepPYQLogo } from "./components/RepPYQLogo";

const SPPU_PYQ_DRIVE_URL =
  "https://drive.google.com/drive/folders/0Bz9C0ysJZ7PnMGZKeWcybUpXWGM?resourcekey=0-S2yaWXvAG7ObM_GC8LRNTQ";

function formatISTTime(date: Date): string {
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  })
    .format(date)
    .toUpperCase();
}

export function App() {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [views, setViews] = useState<number | null>(null);
  const [istTime, setIstTime] = useState<string>(() => formatISTTime(new Date()));
  const visitedOnce = useRef(false);

  const [dark, setDark] = useState(() => {
    try {
      return localStorage.getItem("reppyq-theme") === "dark";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  useEffect(() => {
    const timer = setInterval(() => {
      setIstTime(formatISTTime(new Date()));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (visitedOnce.current) return;
    visitedOnce.current = true;
    recordPageVisit()
      .then((count) => {
        if (count > 0) setViews(count);
      })
      .catch(() => {});
  }, []);

  const toggleTheme = (e: React.MouseEvent<HTMLButtonElement>) => {
    const nextDark = !dark;
    const apply = () => {
      document.documentElement.classList.toggle("dark", nextDark);
      try {
        localStorage.setItem("reppyq-theme", nextDark ? "dark" : "light");
      } catch {}
      setDark(nextDark);
    };

    const doc = document as Document & {
      startViewTransition?: (cb: () => void) => { ready: Promise<void> };
    };

    if (!doc.startViewTransition) {
      apply();
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX || rect.left + rect.width / 2;
    const y = e.clientY || rect.top + rect.height / 2;
    const endRadius = Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y)
    );

    const transition = doc.startViewTransition(() => {
      flushSync(apply);
    });

    transition.ready
      .then(() => {
        document.documentElement.animate(
          {
            clipPath: [
              `circle(0px at ${x}px ${y}px)`,
              `circle(${endRadius}px at ${x}px ${y}px)`,
            ],
          },
          {
            duration: 550,
            easing: "cubic-bezier(0.22, 1, 0.36, 1)",
            pseudoElement: "::view-transition-new(root)",
          }
        );
      })
      .catch(() => {});
  };

  const handleAnalyze = async (params: {
    files: File[];
    labels: string[];
    subjectName: string;
    threshold: number;
    syllabusText: string;
  }) => {
    setBusy(true);
    try {
      const res = await analyzePdfs(params);
      setAnalysis(res);
    } finally {
      setBusy(false);
    }
  };

  const handleRecluster = async (threshold: number) => {
    if (!analysis) return;
    setBusy(true);
    try {
      setAnalysis(await reclusterAnalysis(analysis, threshold));
    } finally {
      setBusy(false);
    }
  };

  const handleMerge = async (clusterIds: string[], customTitle?: string) => {
    if (!analysis) return;
    setBusy(true);
    try {
      setAnalysis(await mergeClusters(analysis, clusterIds, customTitle));
    } finally {
      setBusy(false);
    }
  };

  const handleSplit = async (clusterId: string, questionId: string) => {
    if (!analysis) return;
    setBusy(true);
    try {
      setAnalysis(await splitQuestion(analysis, clusterId, questionId));
    } finally {
      setBusy(false);
    }
  };

  const handleExportPdf = async () => {
    if (!analysis) return;
    setBusy(true);
    try {
      await downloadRevisionPdf(analysis);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-[#111318] text-slate-900 dark:text-slate-100">
      <header className="bg-white/95 dark:bg-[#151820]/95 backdrop-blur border-b border-slate-200 dark:border-[#272b36]">
        <div className="max-w-5xl mx-auto px-3 sm:px-4 py-3 flex flex-wrap items-center justify-between gap-2.5">
          <div className="flex items-center gap-2.5">
            <RepPYQLogo className="w-8 h-8 shrink-0" />
            <div>
              <h1 className="text-base font-bold text-slate-900 dark:text-slate-100 leading-none tracking-tight">
                RepPYQ
              </h1>
              <p className="text-[11px] sm:text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                SPPU 2019 Pattern · PYQ Analyzer
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <a
              href={SPPU_PYQ_DRIVE_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white border border-slate-200 dark:border-[#2e3340] hover:border-slate-400 dark:hover:border-slate-500 hover:bg-slate-100 dark:hover:bg-[#1e222b] px-2.5 py-1.5 rounded-lg transition-all duration-150"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              <span>SPPU PYQ</span>
            </a>

            <a
              href="https://pushkarez.vercel.app/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 hover:text-slate-950 dark:hover:text-white bg-slate-100 dark:bg-[#1d212b] border border-slate-200 dark:border-[#2e3340] hover:border-slate-400 dark:hover:border-slate-500 px-2.5 sm:px-3 py-1.5 rounded-lg transition-all duration-150"
            >
              <span>Pushkar Shinde</span>
              <ExternalLink className="w-3 h-3" />
            </a>

            <button
              type="button"
              onClick={toggleTheme}
              aria-label="Toggle theme"
              title={dark ? "Switch to Light mode" : "Switch to Dark mode"}
              className="p-1.5 rounded-lg border border-slate-200 dark:border-[#2e3340] text-slate-600 dark:text-slate-300 hover:text-slate-950 dark:hover:text-white hover:border-slate-400 dark:hover:border-slate-500 hover:bg-slate-100 dark:hover:bg-[#1e222b] transition-all duration-150"
            >
              {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>

        <div className="border-t border-slate-100 dark:border-[#20242e] bg-slate-50/70 dark:bg-[#12151c]">
          <div className="max-w-5xl mx-auto px-3 sm:px-4 py-1.5 flex items-center justify-between text-[11px] sm:text-xs text-slate-500 dark:text-slate-400">
            <div className="inline-flex items-center gap-1.5 font-mono">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>IST · {istTime}</span>
            </div>

            <div className="inline-flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <Eye className="w-3.5 h-3.5 text-slate-400" />
              <span>{views !== null ? `${views.toLocaleString()} views` : "Live"}</span>
            </div>
          </div>
        </div>
      </header>

      <main className="flex-1 px-3 sm:px-4 py-5 sm:py-8">
        {!analysis ? (
          <UploadPanel onAnalyze={handleAnalyze} isLoading={busy} />
        ) : (
          <Dashboard
            analysis={analysis}
            onRecluster={handleRecluster}
            onMerge={handleMerge}
            onSplit={handleSplit}
            onExportPdf={handleExportPdf}
            onReset={() => setAnalysis(null)}
            busy={busy}
          />
        )}
      </main>

      <footer className="border-t border-slate-200 dark:border-[#272b36] bg-white dark:bg-[#151820] py-4 px-4 text-center text-xs text-slate-500 dark:text-slate-400">
        RepPYQ for SPPU Students · Built by{" "}
        <a
          href="https://pushkarez.vercel.app/"
          target="_blank"
          rel="noopener noreferrer"
          className="font-semibold text-slate-800 dark:text-slate-200 hover:underline"
        >
          Pushkar Shinde
        </a>
      </footer>
    </div>
  );
}

export default App;
