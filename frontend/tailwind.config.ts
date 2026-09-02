import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#FAFAF6",
        "paper-dim": "#F2EFE6",
        ink: "#1B1B18",
        "ink-muted": "#6B6559",
        "ink-faint": "#A39C8C",
        pen: "#B23A2E",
        "pen-soft": "#F3DEDA",
        "pen-dim": "#8C2E24",
        verdict: "#3C6E58",
        "verdict-soft": "#DCE9E2",
        rule: "#DDD6C8",
      },
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        body: ["var(--font-body)", "Georgia", "serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
