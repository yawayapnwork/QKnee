import type { Config } from "tailwindcss";

/**
 * Design tokens per FRONTEND_REDESIGN.md Part 3.D-G. Deliberately narrow:
 * no gradient pair, no default glow/blur utility, one accent color, and
 * provenance/severity share one hue family (both answer "how much should
 * I trust this"). If a new color/shadow/animation is added here, it must
 * be traceable to a specific, named meaning in FRONTEND_REDESIGN.md --
 * not "looks advanced."
 */
const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          0: "#0b0f14",
          1: "#11161d",
          2: "#171d26",
          3: "#1f2733",
        },
        ink: {
          primary: "#e6e9ee",
          muted: "#8b95a3",
          faint: "#565f6c",
        },
        accent: {
          DEFAULT: "#2dd4bf",
          muted: "#0f766e",
        },
        status: {
          live: "#22c55e",
          demo: "#eab308",
          fallback: "#ef4444",
          proxy: "#64748b",
        },
        severity: {
          normal: "#22c55e",
          indeterminate: "#eab308",
          urgent: "#ef4444",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      // Spacing rhythm: Tailwind's default scale is already 4px-based
      // (p-2=8px, p-4=16px, p-6=24px, p-8=32px, p-12=48px) -- deliberately
      // NOT overridden here (remapping numeric keys like "4"/"6" would
      // silently change what every existing `p-4`/`gap-6` utility means
      // across the whole app). The discipline is a convention, enforced by
      // review, not a token: components use only {2, 4, 6, 8, 12} for
      // padding/gap, never arbitrary values like `px-5`/`py-3.5`.
      boxShadow: {
        // The one deliberate glow in the entire system -- reserved for a
        // MOCK/FALLBACK alert, never for decoration or a default button
        // state. No teal "glow" utility exists anymore.
        alert: "0 0 0 1px rgba(239, 68, 68, 0.4), 0 0 16px rgba(239, 68, 68, 0.18)",
      },
    },
  },
  plugins: [],
};

export default config;
