import { useEffect, useMemo, useState } from 'react'
import { clamp, nextUid, palette, toneHex, toneHexLight } from './chartTheme'
import { useChartReveal } from './useChartReveal'
import { ChartTooltip, type TooltipState } from './ChartTooltip'

/* ===========================================================================
 * LdaSeparabilityChart - the supervised "Model MRI" panel.
 *
 * Drone (good / emerald) vs confuser (bad / rose) projected onto a SINGLE
 * Fisher discriminant axis (LD1) computed from the detector's OWN p3+p5
 * features. WITH supervision the two classes fall into two cleanly-separated
 * humps; a dashed decision boundary sits where LDA's own threshold lands.
 *
 * Data (real). The bars are a histogram of REAL LD1 values sampled (≤300/class)
 * from mri/results/v5_report_regen/features.npz - the canonical fused-RGB run.
 * They are fetched at runtime from /media/mri/lda_points.json. If that asset is
 * missing the chart degrades to two Gaussians explicitly parameterized by the
 * measured class means + the published separability, captioned as illustrative.
 *
 * Interaction mirrors the rest of the toolkit: reveals once on scroll-in (bars
 * grow from the axis on a shared eased clock; reduced-motion → final state),
 * hover/keyboard-focus a bin → tooltip with class + count + LD1 range, the
 * other class dims. Responsive via SVG viewBox + width:100%.
 * ========================================================================= */

/** Shape of /media/mri/lda_points.json (real per-detection LD1 projections). */
export interface LdaPointsData {
  /** axis label, e.g. "LD1" */
  axis?: string
  /** LDA decision threshold on the LD1 axis */
  boundary: number
  /** mean LD1 of the confuser class */
  mean_confuser: number
  /** mean LD1 of the drone class */
  mean_drone: number
  /** sampled LD1 values for real drones */
  drone: number[]
  /** sampled LD1 values for confusers */
  confuser: number[]
}

export interface LdaSeparabilityChartProps {
  /** Real sampled LD1 points. Omit to fetch from `src`. */
  data?: LdaPointsData
  /** Where to fetch the points JSON (default '/media/mri/lda_points.json'). */
  src?: string
  /** Headline separability to print on the chart (default 0.952, the RGB run). */
  separability?: number
  /** Optional second separability chip (e.g. IR 0.981). */
  separabilityAlt?: { label: string; value: number }
  /** number of histogram bins (default 34) */
  bins?: number
  title?: string
  caption?: string
  ariaLabel?: string
}

/* ----- fixed viewBox geometry (responsive via width:100% on the <svg>) ----- */
const VB_W = 520
const VB_H = 320
const PAD = { top: 30, right: 18, bottom: 46, left: 40 }
const PLOT_W = VB_W - PAD.left - PAD.right
const PLOT_H = VB_H - PAD.top - PAD.bottom
const BASE_Y = PAD.top + PLOT_H

type Bin = { x0: number; x1: number; drone: number; conf: number }

/** Box–Muller standard normal with a small seeded LCG, for the fallback only. */
function seededNormals(n: number, seed: number): number[] {
  let s = seed >>> 0
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return (s & 0xffffff) / 0x1000000
  }
  const out: number[] = []
  for (let i = 0; i < n; i += 1) {
    const u1 = Math.max(rand(), 1e-9)
    const u2 = rand()
    out.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2))
  }
  return out
}

/**
 * Build a stat-parameterized fallback: two Gaussians whose means are the
 * measured class means and whose spread/overlap reproduces the published
 * separability (higher separability → tighter, further-apart humps).
 */
function fallbackPoints(separability: number): LdaPointsData {
  const meanConf = -2.2
  const meanDrone = 1.6
  // overlap shrinks as separability → 1; sd chosen so the boundary error rate
  // ≈ (1 - separability). With means ±~1.9 from the midpoint this lands near
  // the real ~0.95 figure at sd ≈ 1.05.
  const sep = clamp(separability, 0.5, 0.999)
  const sd = (meanDrone - meanConf) / (2 * Math.max(0.6, normInvUpper(sep)))
  const n = 300
  const drone = seededNormals(n, 11).map((z) => meanDrone + z * sd)
  const conf = seededNormals(n, 97).map((z) => meanConf + z * sd)
  return {
    axis: 'LD1',
    boundary: (meanConf + meanDrone) / 2,
    mean_confuser: meanConf,
    mean_drone: meanDrone,
    drone,
    confuser: conf,
  }
}

/** Rough inverse-normal upper-tail (for the fallback sd only); good to ~2 dp. */
function normInvUpper(p: number): number {
  // map separability p∈(0.5,1) to a z via a compact rational approx of Φ⁻¹(p)
  const q = clamp(p, 0.5001, 0.9999)
  const t = Math.sqrt(-2 * Math.log(1 - q))
  return t - (2.30753 + 0.27061 * t) / (1 + 0.99229 * t + 0.04481 * t * t)
}

function histogram(pts: LdaPointsData, nBins: number): { bins: Bin[]; max: number; lo: number; hi: number } {
  const all = [...pts.drone, ...pts.confuser]
  if (all.length === 0) return { bins: [], max: 1, lo: 0, hi: 1 }
  let lo = Math.min(...all)
  let hi = Math.max(...all)
  if (hi - lo < 1e-6) hi = lo + 1
  const pad = (hi - lo) * 0.04
  lo -= pad
  hi += pad
  const w = (hi - lo) / nBins
  const bins: Bin[] = Array.from({ length: nBins }, (_, i) => ({
    x0: lo + i * w,
    x1: lo + (i + 1) * w,
    drone: 0,
    conf: 0,
  }))
  const put = (v: number, key: 'drone' | 'conf') => {
    const idx = clamp(Math.floor((v - lo) / w), 0, nBins - 1)
    bins[idx][key] += 1
  }
  pts.drone.forEach((v) => put(v, 'drone'))
  pts.confuser.forEach((v) => put(v, 'conf'))
  const max = Math.max(1, ...bins.map((b) => Math.max(b.drone, b.conf)))
  return { bins, max, lo, hi }
}

export function LdaSeparabilityChart({
  data,
  src = '/media/mri/lda_points.json',
  separability = 0.952,
  separabilityAlt,
  bins = 34,
  title,
  caption,
  ariaLabel,
}: LdaSeparabilityChartProps) {
  const { ref, progress, seen, reduced } = useChartReveal<HTMLDivElement>(1150)
  const [uid] = useState(nextUid)
  const [fetched, setFetched] = useState<LdaPointsData | null>(null)
  const [isFallback, setIsFallback] = useState(false)
  const [active, setActive] = useState<number | null>(null)

  // resolve the point source: prop > fetched asset > stat-parameterized fallback
  useEffect(() => {
    if (data) return
    let alive = true
    fetch(src)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j: LdaPointsData) => {
        if (alive) setFetched(j)
      })
      .catch(() => {
        if (alive) {
          setFetched(fallbackPoints(separability))
          setIsFallback(true)
        }
      })
    return () => {
      alive = false
    }
  }, [data, src, separability])

  const pts = data ?? fetched
  const hist = useMemo(() => (pts ? histogram(pts, bins) : null), [pts, bins])

  const span = hist ? hist.hi - hist.lo || 1 : 1
  const xOf = (v: number) => (hist ? PAD.left + ((v - hist.lo) / span) * PLOT_W : PAD.left)
  const yOf = (count: number) => (hist ? BASE_Y - (count / hist.max) * PLOT_H : BASE_Y)

  const boundaryX = pts ? xOf(pts.boundary) : PAD.left
  // boundary guide reveals after the bars are mostly grown
  const guideP = clamp((progress - 0.55) / 0.45, 0, 1)

  return (
    <div ref={ref} className="w-full">
      {title && <div className="mb-3 text-sm font-semibold text-slate-300">{title}</div>}

      {/* legend + separability chips */}
      <div
        className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1.5"
        style={{ opacity: seen ? 1 : 0, transition: 'opacity 500ms ease-out' }}
      >
        <span className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: toneHex.good }} />
          Drone
        </span>
        <span className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: toneHex.bad }} />
          Confuser
        </span>
        <span className="ml-auto flex items-center gap-2">
          <span className="rounded-full border border-cyan-400/40 bg-cyan-400/10 px-2.5 py-1 text-[11px] font-semibold tabular-nums text-cyan-200">
            LDA separability {separability.toFixed(3)}
          </span>
          {separabilityAlt && (
            <span className="rounded-full border border-slate-600/60 bg-slate-700/20 px-2.5 py-1 text-[11px] font-semibold tabular-nums text-slate-300">
              {separabilityAlt.label} {separabilityAlt.value.toFixed(3)}
            </span>
          )}
        </span>
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          width="100%"
          role="img"
          aria-label={
            ariaLabel ??
            `Drone versus confuser projected on one LDA axis; separability ${separability.toFixed(3)}.`
          }
          className="block select-none overflow-visible"
        >
          <defs>
            {(['good', 'bad'] as const).map((t) => (
              <linearGradient key={t} id={`${uid}-lda-${t}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={toneHexLight[t]} stopOpacity={0.95} />
                <stop offset="100%" stopColor={toneHex[t]} stopOpacity={0.55} />
              </linearGradient>
            ))}
          </defs>

          {/* baseline */}
          <line
            x1={PAD.left}
            x2={PAD.left + PLOT_W}
            y1={BASE_Y}
            y2={BASE_Y}
            stroke={palette.baseline}
            strokeWidth={1.25}
            style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
          />

          {/* histogram bins: drone + confuser overlaid (semi-transparent) */}
          {hist &&
            hist.bins.map((b, i) => {
              const bx0 = xOf(b.x0)
              const bx1 = xOf(b.x1)
              const w = Math.max(bx1 - bx0 - 1, 0.5)
              const isActive = active === i
              const dimAll = active != null && !isActive
              const drawBar = (count: number, tone: 'good' | 'bad', dim: boolean) => {
                if (count <= 0) return null
                const fullH = BASE_Y - yOf(count)
                const h = fullH * progress
                return (
                  <rect
                    x={bx0 + 0.5}
                    y={BASE_Y - h}
                    width={w}
                    height={Math.max(h, 0.001)}
                    fill={`url(#${uid}-lda-${tone})`}
                    stroke={toneHex[tone]}
                    strokeOpacity={0.35}
                    strokeWidth={0.5}
                    style={{
                      opacity: dim ? 0.22 : 0.82,
                      transition: 'opacity 180ms ease-out',
                    }}
                  />
                )
              }
              return (
                <g
                  key={i}
                  tabIndex={0}
                  role="img"
                  aria-label={`LD1 ${b.x0.toFixed(2)} to ${b.x1.toFixed(2)}: ${b.drone} drone, ${b.conf} confuser`}
                  onMouseEnter={() => setActive(i)}
                  onMouseLeave={() => setActive(null)}
                  onFocus={() => setActive(i)}
                  onBlur={() => setActive(null)}
                  style={{ cursor: 'default', outline: 'none' }}
                >
                  {/* full-height hit area for steadier hover */}
                  <rect x={bx0 + 0.5} y={PAD.top} width={w} height={PLOT_H} fill="transparent" />
                  {/* draw the shorter bar last so both stay visible */}
                  {b.drone >= b.conf ? (
                    <>
                      {drawBar(b.drone, 'good', dimAll)}
                      {drawBar(b.conf, 'bad', dimAll)}
                    </>
                  ) : (
                    <>
                      {drawBar(b.conf, 'bad', dimAll)}
                      {drawBar(b.drone, 'good', dimAll)}
                    </>
                  )}
                </g>
              )
            })}

          {/* decision boundary (LDA's own threshold on LD1) */}
          {pts && (
            <g style={{ opacity: guideP }}>
              <line
                x1={boundaryX}
                x2={boundaryX}
                y1={PAD.top - 6}
                y2={BASE_Y}
                stroke={toneHex.accent}
                strokeWidth={1.5}
                strokeDasharray="4 3"
              />
              <g transform={`translate(${clamp(boundaryX, PAD.left + 30, PAD.left + PLOT_W - 30)}, ${PAD.top - 14})`}>
                <rect x={-46} y={-11} width={92} height={20} rx={10} fill={toneHex.accent} fillOpacity={0.12} stroke={toneHex.accent} strokeOpacity={0.5} />
                <text x={0} y={0} textAnchor="middle" dominantBaseline="central" fontSize="10" fontWeight={600} fill={toneHexLight.accent}>
                  decision boundary
                </text>
              </g>
            </g>
          )}

          {/* axis label */}
          <text
            x={PAD.left + PLOT_W / 2}
            y={BASE_Y + 30}
            textAnchor="middle"
            fontSize="11"
            fontWeight={600}
            fill={palette.textDim}
            style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
          >
            {pts?.axis ?? 'LD1'} - Fisher discriminant of p3+p5 features
          </text>

          {/* class-mean ticks under the axis */}
          {pts &&
            (
              [
                { v: pts.mean_confuser, tone: 'bad' as const, label: 'confuser μ' },
                { v: pts.mean_drone, tone: 'good' as const, label: 'drone μ' },
              ]
            ).map((m) => (
              <g key={m.label} style={{ opacity: guideP }}>
                <line x1={xOf(m.v)} x2={xOf(m.v)} y1={BASE_Y} y2={BASE_Y + 6} stroke={toneHex[m.tone]} strokeWidth={1.5} />
              </g>
            ))}
        </svg>

        <ChartTooltip state={tooltipFor(active, hist, xOf, yOf)} />
      </div>

      {caption && <p className="mt-4 text-xs leading-relaxed text-slate-500">{caption}</p>}
      {isFallback && (
        <p className="mt-1 text-[11px] italic leading-relaxed text-slate-600">
          Distributions illustrative of the measured statistic (separability {separability.toFixed(3)}); per-detection coordinates
          unavailable at load time.
        </p>
      )}
      {reduced && <span className="sr-only">Animation reduced; chart shown in its final state.</span>}
    </div>
  )
}

function tooltipFor(
  active: number | null,
  hist: { bins: Bin[]; max: number; lo: number; hi: number } | null,
  xOf: (v: number) => number,
  yOf: (count: number) => number,
): TooltipState {
  if (active == null || !hist) return null
  const b = hist.bins[active]
  if (!b) return null
  const total = b.drone + b.conf
  const lead = b.drone >= b.conf ? 'drone' : 'confuser'
  const cx = (xOf(b.x0) + xOf(b.x1)) / 2
  const topCount = Math.max(b.drone, b.conf)
  return {
    xFrac: cx / VB_W,
    yFrac: yOf(topCount) / VB_H,
    series: lead === 'drone' ? 'Drone' : 'Confuser',
    tone: lead === 'drone' ? 'good' : 'bad',
    label: `LD1 ${b.x0.toFixed(2)} … ${b.x1.toFixed(2)}`,
    value: total === 0 ? '-' : `${b.drone} drone · ${b.conf} confuser`,
  }
}
