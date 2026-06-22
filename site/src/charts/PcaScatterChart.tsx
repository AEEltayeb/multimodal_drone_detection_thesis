import { useEffect, useMemo, useState } from 'react'
import { clamp, nextUid, palette, toneHex, toneHexLight } from './chartTheme'
import { useChartReveal } from './useChartReveal'
import { ChartTooltip, type TooltipState } from './ChartTooltip'

/* ===========================================================================
 * PcaScatterChart - the "it's difficult" counterpoint to the LDA panel.
 *
 * The SAME two classes (drone good/emerald, confuser bad/rose) under an
 * UNSUPERVISED 2-D PCA of the detector's p3+p5 features. Without labels the two
 * clouds sit on top of each other (measured silhouette only 0.067) - invisible
 * to any distance/clustering rule, which is precisely WHY a trained classifier,
 * not a threshold, is needed. The visual contrast with LdaSeparabilityChart is
 * the whole point: same data, supervised splits it, unsupervised does not.
 *
 * Data (real). Points are REAL [PC1, PC2] coordinates sampled (≤300/class) from
 * mri/results/v5_report_regen/features.npz (the canonical fused-RGB run),
 * fetched from /media/mri/pca_points.json. If the asset is missing the chart
 * degrades to two overlapping Gaussian blobs whose overlap is set to reproduce
 * the published silhouette, captioned as illustrative.
 *
 * Interaction: dots fade/scale in on a staggered scroll-reveal clock
 * (reduced-motion → final state); hover/keyboard-focus a class legend (or a dot)
 * highlights that class and dims the other; per-dot tooltip with class + PCs.
 * ========================================================================= */

/** Shape of /media/mri/pca_points.json (real per-detection PCA projections). */
export interface PcaPointsData {
  /** explained-variance fraction of PC1 (0..1) */
  pc1_var?: number
  /** explained-variance fraction of PC2 (0..1) */
  pc2_var?: number
  /** [PC1, PC2] per real drone detection */
  drone: Array<[number, number]>
  /** [PC1, PC2] per confuser detection */
  confuser: Array<[number, number]>
}

export interface PcaScatterChartProps {
  data?: PcaPointsData
  src?: string
  /** silhouette score to print (default 0.067 - the measured fused-RGB value). */
  silhouette?: number
  title?: string
  caption?: string
  ariaLabel?: string
}

const VB_W = 520
const VB_H = 340
const PAD = { top: 28, right: 18, bottom: 44, left: 44 }
const PLOT_W = VB_W - PAD.left - PAD.right
const PLOT_H = VB_H - PAD.top - PAD.bottom

type Pt = { x: number; y: number; cls: 0 | 1 }

function seededRand(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return (s & 0xffffff) / 0x1000000
  }
}
function gauss(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-9)
  const u2 = rand()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/** Two heavily-overlapping blobs; centroid offset set from the silhouette. */
function fallbackPoints(silhouette: number): PcaPointsData {
  const rand = seededRand(7)
  const n = 300
  // tiny class separation: low silhouette → nearly co-located centroids.
  const sep = clamp(silhouette, 0, 0.3) * 18 // ~1.2 units at s=0.067
  const blob = (cx: number, cy: number, count: number): Array<[number, number]> =>
    Array.from({ length: count }, () => [cx + gauss(rand) * 7, cy + gauss(rand) * 5] as [number, number])
  return {
    pc1_var: 0.488,
    pc2_var: 0.098,
    drone: blob(sep, 0, n),
    confuser: blob(-sep, 0, n),
  }
}

export function PcaScatterChart({
  data,
  src = '/media/mri/pca_points.json',
  silhouette = 0.067,
  title,
  caption,
  ariaLabel,
}: PcaScatterChartProps) {
  const { ref, progress, seen, reduced } = useChartReveal<HTMLDivElement>(1300)
  const [uid] = useState(nextUid)
  const [fetched, setFetched] = useState<PcaPointsData | null>(null)
  const [isFallback, setIsFallback] = useState(false)
  const [hover, setHover] = useState<number | null>(null)
  // focusing a legend entry highlights a whole class
  const [focusCls, setFocusCls] = useState<0 | 1 | null>(null)

  useEffect(() => {
    if (data) return
    let alive = true
    fetch(src)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((j: PcaPointsData) => {
        if (alive) setFetched(j)
      })
      .catch(() => {
        if (alive) {
          setFetched(fallbackPoints(silhouette))
          setIsFallback(true)
        }
      })
    return () => {
      alive = false
    }
  }, [data, src, silhouette])

  const pts = data ?? fetched

  // flatten to one list with class + bounds; interleave so the reveal stagger
  // and z-order mix the two classes rather than painting one fully on top.
  const { items, bounds } = useMemo(() => {
    if (!pts) return { items: [] as Pt[], bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 } }
    const d = pts.drone.map(([x, y]) => ({ x, y, cls: 1 as const }))
    const c = pts.confuser.map(([x, y]) => ({ x, y, cls: 0 as const }))
    const merged: Pt[] = []
    const n = Math.max(d.length, c.length)
    for (let i = 0; i < n; i += 1) {
      if (i < c.length) merged.push(c[i])
      if (i < d.length) merged.push(d[i])
    }
    const xs = merged.map((p) => p.x)
    const ys = merged.map((p) => p.y)
    const pad = 0.06
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const dx = (maxX - minX || 1) * pad
    const dy = (maxY - minY || 1) * pad
    return {
      items: merged,
      bounds: { minX: minX - dx, maxX: maxX + dx, minY: minY - dy, maxY: maxY + dy },
    }
  }, [pts])

  const spanX = bounds.maxX - bounds.minX || 1
  const spanY = bounds.maxY - bounds.minY || 1
  const sx = (x: number) => PAD.left + ((x - bounds.minX) / spanX) * PLOT_W
  // invert y so +PC2 is up
  const sy = (y: number) => PAD.top + (1 - (y - bounds.minY) / spanY) * PLOT_H

  const total = items.length || 1

  return (
    <div ref={ref} className="w-full">
      {title && <div className="mb-3 text-sm font-semibold text-slate-300">{title}</div>}

      {/* interactive legend (focusable → highlights a class) + silhouette chip */}
      <div
        className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1.5"
        style={{ opacity: seen ? 1 : 0, transition: 'opacity 500ms ease-out' }}
      >
        {(
          [
            { cls: 1 as const, tone: 'good' as const, label: 'Drone' },
            { cls: 0 as const, tone: 'bad' as const, label: 'Confuser' },
          ]
        ).map((l) => (
          <button
            key={l.label}
            type="button"
            onMouseEnter={() => setFocusCls(l.cls)}
            onMouseLeave={() => setFocusCls(null)}
            onFocus={() => setFocusCls(l.cls)}
            onBlur={() => setFocusCls(null)}
            className="flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-[11px] text-slate-400 outline-none transition-opacity"
            style={{ opacity: focusCls != null && focusCls !== l.cls ? 0.4 : 1 }}
            aria-label={`Highlight ${l.label} points`}
          >
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: toneHex[l.tone] }} />
            {l.label}
          </button>
        ))}
        <span className="ml-auto rounded-full border border-rose-400/40 bg-rose-400/10 px-2.5 py-1 text-[11px] font-semibold tabular-nums text-rose-200">
          silhouette {silhouette.toFixed(3)}
        </span>
      </div>

      <div className="relative">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          width="100%"
          role="img"
          aria-label={
            ariaLabel ??
            `Unsupervised PCA of drone and confuser features; the two classes overlap heavily, silhouette ${silhouette.toFixed(3)}.`
          }
          className="block select-none overflow-visible"
        >
          <defs>
            <clipPath id={`${uid}-pca-clip`}>
              <rect x={PAD.left} y={PAD.top} width={PLOT_W} height={PLOT_H} rx={6} />
            </clipPath>
          </defs>

          {/* plot frame */}
          <rect
            x={PAD.left}
            y={PAD.top}
            width={PLOT_W}
            height={PLOT_H}
            rx={6}
            fill={palette.gridFaint}
            stroke={palette.grid}
            strokeWidth={1}
            style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
          />
          {/* faint center crosshair */}
          <g style={{ opacity: seen ? 0.5 : 0, transition: 'opacity 700ms ease-out' }}>
            <line x1={sx((bounds.minX + bounds.maxX) / 2)} x2={sx((bounds.minX + bounds.maxX) / 2)} y1={PAD.top} y2={PAD.top + PLOT_H} stroke={palette.grid} strokeWidth={0.75} strokeDasharray="2 4" />
            <line x1={PAD.left} x2={PAD.left + PLOT_W} y1={sy((bounds.minY + bounds.maxY) / 2)} y2={sy((bounds.minY + bounds.maxY) / 2)} stroke={palette.grid} strokeWidth={0.75} strokeDasharray="2 4" />
          </g>

          {/* the two overlapping point clouds */}
          <g clipPath={`url(#${uid}-pca-clip)`}>
            {items.map((p, i) => {
              const tone = p.cls === 1 ? 'good' : 'bad'
              // staggered reveal: each dot crosses its own little threshold
              const local = clamp((progress - (i / total) * 0.5) / 0.5, 0, 1)
              if (local <= 0) return null
              const dimByClass = focusCls != null && focusCls !== p.cls
              const isHover = hover === i
              const baseOpacity = 0.55
              const op = (dimByClass ? 0.12 : isHover ? 0.95 : baseOpacity) * local
              const r = (isHover ? 4.5 : 2.6) * (0.6 + 0.4 * local)
              return (
                <circle
                  key={i}
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r={r}
                  fill={toneHex[tone]}
                  stroke={isHover ? toneHexLight[tone] : 'none'}
                  strokeWidth={isHover ? 1 : 0}
                  style={{ opacity: op, transition: 'opacity 160ms ease-out, r 120ms ease-out' }}
                  tabIndex={focusCls == null ? -1 : 0}
                  onMouseEnter={() => setHover(i)}
                  onMouseLeave={() => setHover((h) => (h === i ? null : h))}
                />
              )
            })}
          </g>

          {/* invisible hover targets on top (larger, steadier) - only when not staggering */}
          {progress >= 1 &&
            items.map((p, i) => (
              <circle
                key={`hit-${i}`}
                cx={sx(p.x)}
                cy={sy(p.y)}
                r={6}
                fill="transparent"
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover((h) => (h === i ? null : h))}
              />
            ))}

          {/* axis labels with explained variance */}
          <text
            x={PAD.left + PLOT_W / 2}
            y={VB_H - 12}
            textAnchor="middle"
            fontSize="11"
            fontWeight={600}
            fill={palette.textDim}
            style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
          >
            PC1{pts?.pc1_var != null ? ` (${(pts.pc1_var * 100).toFixed(1)}% var)` : ''}
          </text>
          <text
            x={14}
            y={PAD.top + PLOT_H / 2}
            textAnchor="middle"
            fontSize="11"
            fontWeight={600}
            fill={palette.textDim}
            transform={`rotate(-90, 14, ${PAD.top + PLOT_H / 2})`}
            style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
          >
            PC2{pts?.pc2_var != null ? ` (${(pts.pc2_var * 100).toFixed(1)}% var)` : ''}
          </text>
        </svg>

        <ChartTooltip state={tooltipFor(hover, items, sx, sy)} />
      </div>

      {caption && <p className="mt-4 text-xs leading-relaxed text-slate-500">{caption}</p>}
      {isFallback && (
        <p className="mt-1 text-[11px] italic leading-relaxed text-slate-600">
          Point clouds illustrative of the measured statistic (silhouette {silhouette.toFixed(3)}); per-detection coordinates
          unavailable at load time.
        </p>
      )}
      {reduced && <span className="sr-only">Animation reduced; chart shown in its final state.</span>}
    </div>
  )
}

function tooltipFor(
  hover: number | null,
  items: Pt[],
  sx: (x: number) => number,
  sy: (y: number) => number,
): TooltipState {
  if (hover == null) return null
  const p = items[hover]
  if (!p) return null
  return {
    xFrac: sx(p.x) / VB_W,
    yFrac: sy(p.y) / VB_H,
    series: p.cls === 1 ? 'Drone' : 'Confuser',
    tone: p.cls === 1 ? 'good' : 'bad',
    label: 'PCA projection',
    value: `PC1 ${p.x.toFixed(2)} · PC2 ${p.y.toFixed(2)}`,
  }
}
