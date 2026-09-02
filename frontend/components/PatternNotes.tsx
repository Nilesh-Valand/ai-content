import { DetectedPattern, Severity } from "@/lib/api";
import { AlertTriangle, AlertCircle, Info } from "lucide-react";

function severityTheme(sev: Severity) {
  switch (sev) {
    case "High":
      return {
        bar: "bg-danger-500",
        chip: "bg-danger-50 text-danger-600",
        Icon: AlertTriangle,
      };
    case "Medium":
      return {
        bar: "bg-warn-500",
        chip: "bg-warn-50 text-warn-600",
        Icon: AlertCircle,
      };
    default:
      return {
        bar: "bg-success-500",
        chip: "bg-success-50 text-success-600",
        Icon: Info,
      };
  }
}

export default function PatternNotes({ patterns }: { patterns: DetectedPattern[] }) {
  const sorted = [...patterns].sort((a, b) => b.score - a.score);

  return (
    <div className="fade-in rounded-2xl border border-border bg-surface shadow-card p-6 h-full">
      <p className="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-4">
        Detected Patterns
      </p>
      <div className="space-y-2.5">
        {sorted.map((p) => {
          const theme = severityTheme(p.severity);
          const Icon = theme.Icon;
          return (
            <div
              key={p.pattern}
              className="relative overflow-hidden rounded-xl border border-border bg-surface-muted pl-4 pr-3 py-3"
            >
              <span className={`absolute left-0 top-0 h-full w-1 ${theme.bar}`} />
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-ink">{p.pattern}</p>
                  {p.examples.length > 0 && (
                    <p className="text-xs text-ink-faint mt-1 truncate">
                      {p.examples.slice(0, 3).map((e) => `"${e}"`).join("  ·  ")}
                    </p>
                  )}
                </div>
                <span
                  className={`shrink-0 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold ${theme.chip}`}
                >
                  <Icon className="h-3 w-3" strokeWidth={2.5} />
                  {p.severity}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
