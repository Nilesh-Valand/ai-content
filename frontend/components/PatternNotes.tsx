import { DetectedPattern, Severity } from "@/lib/api";

function severityStyle(sev: Severity) {
  switch (sev) {
    case "High":
      return "border-pen text-pen-dim";
    case "Medium":
      return "border-ink-faint text-ink-muted";
    default:
      return "border-verdict text-verdict";
  }
}

export default function PatternNotes({ patterns }: { patterns: DetectedPattern[] }) {
  const sorted = [...patterns].sort((a, b) => b.score - a.score);

  return (
    <div className="fade-in">
      <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-muted mb-4">
        Editor&rsquo;s Notes
      </p>
      <div className="divide-y rule-hairline border-t border-b rule-hairline">
        {sorted.map((p) => (
          <div key={p.pattern} className="py-3 flex items-start gap-4">
            <span
              className={`font-mono text-[10px] uppercase tracking-wider border px-1.5 py-0.5 shrink-0 w-16 text-center ${severityStyle(
                p.severity
              )}`}
            >
              {p.severity}
            </span>
            <div className="min-w-0">
              <p className="font-body text-base">{p.pattern}</p>
              {p.examples.length > 0 && (
                <p className="font-mono text-xs text-ink-faint mt-1 truncate">
                  {p.examples.slice(0, 3).map((e) => `"${e}"`).join("  ·  ")}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
