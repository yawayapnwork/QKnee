import type { Config } from "tailwindcss";

/**
 * Q-Knee design system — Tailwind theme, rebuilt from first principles.
 * Every color/radius/shadow utility here resolves to a CSS custom
 * property declared once in `app/globals.css`'s `:root` block — that
 * file is the actual source of truth (real "CSS variables/design
 * tokens," not just a Tailwind config object); this file only exposes
 * those variables as utility classes. See DESIGN_SYSTEM.md for the full
 * rationale of every token and the component hierarchy built on it.
 */
const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          0: "var(--color-surface-0)",
          1: "var(--color-surface-1)",
          2: "var(--color-surface-2)",
          3: "var(--color-surface-3)",
          4: "var(--color-surface-4)",
        },
        ink: {
          primary: "var(--color-ink-primary)",
          secondary: "var(--color-ink-secondary)",
          muted: "var(--color-ink-muted)",
          faint: "var(--color-ink-faint)",
        },
        accent: {
          DEFAULT: "var(--color-accent)",
          strong: "var(--color-accent-strong)",
          subtle: "var(--color-accent-subtle)",
        },
        info: "var(--color-info)",
        success: "var(--color-success)",
        warning: "var(--color-warning)",
        danger: "var(--color-danger)",
        neutral: "var(--color-neutral)",
        status: {
          live: "var(--color-status-live)",
          demo: "var(--color-status-demo)",
          fallback: "var(--color-status-fallback)",
          proxy: "var(--color-status-proxy)",
        },
        severity: {
          normal: "var(--color-severity-normal)",
          indeterminate: "var(--color-severity-indeterminate)",
          urgent: "var(--color-severity-urgent)",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains)", "JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      // Typography scale -- seven steps, deliberately smaller-than-default
      // at the base (13px) for the same reason a trading terminal or a
      // PACS viewer runs small, tight type: high information density
      // without the page feeling like a blog post. Section headers top
      // out at 20px; nothing in the product uses a "hero" size.
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],   // 11px -- micro labels, table headers
        xs: ["0.75rem", { lineHeight: "1.1rem" }],       // 12px -- captions, badges
        sm: ["0.8125rem", { lineHeight: "1.25rem" }],    // 13px -- body default
        base: ["0.875rem", { lineHeight: "1.4rem" }],    // 14px -- emphasized body
        md: ["1rem", { lineHeight: "1.5rem" }],          // 16px -- card titles
        lg: ["1.25rem", { lineHeight: "1.75rem" }],      // 20px -- section headers
        xl: ["1.75rem", { lineHeight: "2.1rem" }],       // 28px -- the ONE dominant number per screen (risk gauge)
      },
      // Spacing rhythm: Tailwind's default scale is already 4px-based
      // (p-2=8px, p-4=16px, p-6=24px, p-8=32px, p-12=48px) -- deliberately
      // NOT overridden here (remapping numeric keys like "4"/"6" would
      // silently change what every existing `p-4`/`gap-6` utility means
      // across the whole app). The discipline is a convention, enforced by
      // review: components use only {2, 4, 6, 8, 12} for padding/gap,
      // never arbitrary values like `px-5`/`py-3.5`.
      borderRadius: {
        xs: "var(--radius-xs)",
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
      },
      boxShadow: {
        1: "var(--elevation-1)",
        2: "var(--elevation-2)",
        3: "var(--elevation-3)",
        // The one deliberate glow in the entire system -- reserved for a
        // MOCK/FALLBACK status, never decoration, never a default button
        // or hover state.
        alert: "var(--elevation-alert)",
      },
      transitionDuration: {
        fast: "var(--duration-fast)",
        base: "var(--duration-base)",
      },
    },
  },
  plugins: [],
};

export default config;
