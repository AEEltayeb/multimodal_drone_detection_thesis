/**
 * Shared design tokens + tiny helpers for the hand-rolled SVG chart toolkit.
 *
 * Everything in `site/src/charts/` draws into an SVG `viewBox` and is sized with
 * `width:100%`, so the charts are resolution-independent and fluid from ~320px to
 * ~640px wide inside the sticky scrollytelling pane. Colours mirror the TALOS dark
 * theme already used across the site (see `index.css`): cyan-400 accent, emerald
 * "good", rose "bad", slate "muted".
 */

/** Semantic colour role for a bar / series. Matches the vocabulary in BarSet. */
export type Tone = 'accent' | 'good' | 'bad' | 'muted'

/** Raw hex per tone - used for SVG `fill`/`stroke` where Tailwind classes don't reach. */
export const toneHex: Record<Tone, string> = {
  accent: '#22d3ee', // cyan-400
  good: '#34d399', // emerald-400
  bad: '#f43f5e', // rose-500
  muted: '#64748b', // slate-500
}

/** A slightly lighter top-stop per tone, for the vertical fill gradient on bars. */
export const toneHexLight: Record<Tone, string> = {
  accent: '#67e8f9', // cyan-300
  good: '#6ee7b7', // emerald-300
  bad: '#fb7185', // rose-400
  muted: '#94a3b8', // slate-400
}

/** Slate palette pulled out so axes/gridlines/labels stay consistent across charts. */
export const palette = {
  text: '#e2e8f0', // slate-200
  textDim: '#94a3b8', // slate-400
  textFaint: '#64748b', // slate-500
  grid: '#1e293b', // slate-800
  gridFaint: '#0f172a', // slate-900
  baseline: '#334155', // slate-700
} as const

/** Cubic ease-out - the same curve Counter.tsx uses, so motion feels consistent. */
export const easeOutCubic = (t: number) => 1 - Math.pow(1 - t, 3)

/** Clamp helper (kept tiny; avoids pulling in a util module). */
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** Format a metric the way the thesis copy does: 3 decimals, never rounded away. */
export const fmt3 = (v: number) => v.toFixed(3)

/** Stable-ish id suffix so multiple chart instances don't collide on <defs> ids. */
let _uid = 0
export const nextUid = () => `c${(_uid += 1)}`
