import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#F5F6FB",
        surface: "#FFFFFF",
        "surface-muted": "#F9FAFC",
        border: "#E7E9F3",
        "border-strong": "#D6D9EA",
        ink: "#14162B",
        "ink-muted": "#6B7089",
        "ink-faint": "#9CA0B8",
        brand: {
          50: "#F1EEFE",
          100: "#E4DEFD",
          200: "#C9BEFB",
          300: "#AB9AF8",
          400: "#8D74F3",
          500: "#7452EE",
          600: "#6236E0",
          700: "#4F27BE",
          DEFAULT: "#7452EE",
        },
        accent: {
          pink: "#EC4899",
          cyan: "#06B6D4",
        },
        danger: {
          50: "#FEF2F2",
          400: "#F87171",
          500: "#EF4444",
          600: "#DC2626",
        },
        warn: {
          50: "#FFFBEB",
          400: "#FBBF24",
          500: "#F59E0B",
          600: "#D97706",
        },
        success: {
          50: "#ECFDF5",
          400: "#34D399",
          500: "#10B981",
          600: "#059669",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        soft: "0 1px 2px rgba(20, 22, 43, 0.04), 0 1px 3px rgba(20, 22, 43, 0.06)",
        card: "0 1px 2px rgba(20, 22, 43, 0.03), 0 8px 24px -8px rgba(20, 22, 43, 0.10)",
        lift: "0 12px 32px -12px rgba(116, 82, 238, 0.35)",
      },
      backgroundImage: {
        "brand-gradient": "linear-gradient(135deg, #7452EE 0%, #6236E0 55%, #4F27BE 100%)",
        "hero-gradient":
          "radial-gradient(1200px 480px at 15% -10%, rgba(116,82,238,0.16), transparent), radial-gradient(900px 420px at 100% 0%, rgba(236,72,153,0.10), transparent)",
      },
      borderRadius: {
        xl2: "1.25rem",
      },
    },
  },
  plugins: [],
};
export default config;
