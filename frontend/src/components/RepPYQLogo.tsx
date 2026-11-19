import React from "react";

export const RepPYQLogo: React.FC<{ className?: string }> = ({ className = "w-8 h-8" }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 32 32"
    fill="none"
    className={className}
    aria-hidden="true"
  >
    <rect width="32" height="32" rx="8" className="fill-slate-900 dark:fill-slate-100" />
    <path
      d="M23.8 11.5A9 9 0 1 0 25 16"
      className="stroke-slate-50 dark:stroke-slate-900"
      strokeWidth="2.2"
      strokeLinecap="round"
    />
    <path
      d="M20.3 13A5.2 5.2 0 1 0 21.2 16"
      className="stroke-slate-400 dark:stroke-slate-500"
      strokeWidth="2"
      strokeLinecap="round"
    />
    <path
      d="M16 16L23.5 8.5"
      className="stroke-sky-400 dark:stroke-sky-600"
      strokeWidth="2.2"
      strokeLinecap="round"
    />
    <circle cx="16" cy="16" r="2.2" className="fill-slate-50 dark:fill-slate-900" />
    <circle cx="24" cy="8" r="1.8" className="fill-sky-400 dark:fill-sky-600" />
  </svg>
);
