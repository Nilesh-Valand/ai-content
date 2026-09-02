"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Sparkles, ScanSearch, GraduationCap, History } from "lucide-react";

const links = [
  { href: "/", label: "Analyzer", icon: ScanSearch },
  { href: "/history", label: "History", icon: History },
  { href: "/train", label: "Train Phrases", icon: GraduationCap },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-30 border-b border-border bg-surface/80 backdrop-blur-md">
      <div className="w-full px-4 sm:px-8 lg:px-12">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 shrink-0">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-gradient shadow-lift">
              <Sparkles className="h-[18px] w-[18px] text-white" strokeWidth={2.25} />
            </span>
            <span className="text-[15px] font-semibold tracking-tight text-ink">
              Content<span className="text-brand-600">IQ</span>
            </span>
          </Link>

          <nav className="flex items-center gap-1">
            {links.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors ${
                    active
                      ? "bg-brand-50 text-brand-700"
                      : "text-ink-muted hover:bg-surface-muted hover:text-ink"
                  }`}
                >
                  <Icon className="h-4 w-4" strokeWidth={2.25} />
                  {label}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
}
