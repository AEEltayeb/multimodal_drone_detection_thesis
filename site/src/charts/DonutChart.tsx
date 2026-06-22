import { useState } from 'react'
import type { Tone } from './chartTheme'
import { clamp, easeOutCubic, nextUid, palette, toneHex, toneHexLight } from './chartTheme'
import { useChartReveal } from './useChartReveal'
import { ChartTooltip, type TooltipState } from './ChartTooltip'

export type DonutSlice = {
  label: string
  value: number
  tone?: Tone
  /** override the printed/tooltip count (defaults to value.toLocaleString()) */
  display?: string
}

export type DonutChartProps = {
  slices: DonutSlice[]
  /** optional title shown above the ring */
  title?: string
  /** optional caption shown below the ring */
  caption?: string
  /** big label under the center total (e.g. "instances") */
  centerLabel?: string
  /** override the center total string (defaults to summed counts, localized) */
  centerValue?: string
  /** show the legend with per-slice swatch + count + % (default true) */
  legend?: boolean
  /** ring thickness as a fraction of the radius (default 0.42) */
  thickness?: number
  /** accessible description of the whole figure */
  ariaLabel?: string
}

/* ----- fixed viewBox geometry (responsive via width:100% on the <svg>) ----- */
const VB = 260
const CX = VB / 2
const CY = VB / 2
const R = 104
const GAP_DEG = 2 // small wedge gap so adjacent slices read as distinct

/** Polar → cartesian on the ring centerline, with 0° at 12 o'clock, clockwise. */
function polar(cx: number, cy: number, r: number, deg: number) {
  const rad = ((deg - 90) * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

/** SVG arc path for an annulus segment between start/end degrees at the given radii. */
function arcPath(startDeg: number, endDeg: number, rOuter: number, rInner: number) {
  // guard the degenerate "full circle" case (one arc can't draw 360°).
  const sweep = endDeg - startDeg
  const largeArc = sweep > 180 ? 1 : 0
  const o0 = polar(CX, CY, rOuter, startDeg)
  const o1 = polar(CX, CY, rOuter, endDeg)
  const i1 = polar(CX, CY, rInner, endDeg)
  const i0 = polar(CX, CY, rInner, startDeg)
  return [
    `M ${o0.x} ${o0.y}`,
    `A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${o1.x} ${o1.y}`,
    `L ${i1.x} ${i1.y}`,
    `A ${rInner} ${rInner} 0 ${largeArc} 0 ${i0.x} ${i0.y}`,
    'Z',
  ].join(' ')
}

/**
 * Hand-rolled interactive donut, drawn as SVG annulus segments.
 *
 * Same interaction grammar as the rest of the toolkit (GroupedBarChart /
 * ComparisonChart):
 * - reveals once on scroll-in - the ring sweeps clockwise from 12 o'clock on a
 *   shared eased clock (useChartReveal); reduced-motion jumps to the final ring.
 * - hover OR keyboard-focus a slice -> tooltip with label + exact count + share,
 *   the focused wedge nudges outward and glows, and the others dim.
 * - the center always shows the running total (counts up with the reveal) plus an
 *   optional label; each wedge is a focusable element with an aria-label.
 *
 * Designed for the dataset confuser-composition donuts, but generic over any set
 * of labeled counts.
 */
export function DonutChart({
  slices,
  title,
  caption,
  centerLabel,
  centerValue,
  legend = true,
  thickness = 0.42,
  ariaLabel,
}: DonutChartProps) {
  const { ref, progress, seen } = useChartReveal<HTMLDivElement>(1100)
  const [active, setActive] = useState<number | null>(null)
  const [uid] = useState(nextUid)

  const total = slices.reduce((s, d) => s + Math.max(0, d.value), 0)
  const rInner = R * (1 - clamp(thickness, 0.1, 0.9))

  // Lay each slice out on [0,360); the reveal sweeps a single global angle so the
  // whole ring fills clockwise rather than every wedge growing independently.
  const sweepDeg = progress * 360
  let cursor = 0
  const arcs = slices.map((d, i) => {
    const frac = total > 0 ? Math.max(0, d.value) / total : 0
    const start = cursor * 360
    const end = (cursor + frac) * 360
    cursor += frac
    // visible portion of this slice given the global reveal angle
    const visEnd = clamp(sweepDeg, start, end)
    const padG = frac > 0 ? GAP_DEG / 2 : 0
    return { d, i, start, end, frac, visStart: start + padG, visEnd: Math.max(start + padG, visEnd - padG) }
  })

  const totalShown = Math.round(total * easeOutCubic(clamp(progress, 0, 1)))
  const centerNum = centerValue ?? totalShown.toLocaleString()

  return (
    <div ref={ref} className="w-full">
      {title && <div className="mb-3 text-sm font-semibold text-slate-300">{title}</div>}

      <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-center sm:gap-6">
        <div className="relative shrink-0" style={{ width: '60%', maxWidth: 230, minWidth: 168 }}>
          <svg
            viewBox={`0 0 ${VB} ${VB}`}
            width="100%"
            role="img"
            aria-label={ariaLabel ?? title ?? 'Composition donut chart'}
            className="block select-none overflow-visible"
          >
            <defs>
              {(['accent', 'good', 'bad', 'muted'] as Tone[]).map((t) => (
                <linearGradient key={t} id={`${uid}-fill-${t}`} x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor={toneHexLight[t]} />
                  <stop offset="100%" stopColor={toneHex[t]} />
                </linearGradient>
              ))}
              <filter id={`${uid}-glow`} x="-60%" y="-60%" width="220%" height="220%">
                <feGaussianBlur stdDeviation="4" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* faint full-ring track so the shape reads before/while it fills */}
            <circle
              cx={CX}
              cy={CY}
              r={(R + rInner) / 2}
              fill="none"
              stroke={palette.grid}
              strokeWidth={R - rInner}
              style={{ opacity: seen ? 0.5 : 0, transition: 'opacity 500ms ease-out' }}
            />

            {arcs.map((a) => {
              if (a.frac <= 0 || a.visEnd <= a.visStart) return null
              const tone = a.d.tone ?? 'accent'
              const isActive = active === a.i
              const dimmed = active != null && !isActive
              // pop the active wedge outward a touch along its mid-angle
              const midDeg = (a.start + a.end) / 2
              const push = isActive ? 5 : 0
              const off = polar(0, 0, push, midDeg)
              const text = a.d.display ?? a.d.value.toLocaleString()
              const pct = total > 0 ? (a.d.value / total) * 100 : 0

              return (
                <g
                  key={a.d.label}
                  tabIndex={0}
                  role="img"
                  aria-label={`${a.d.label}: ${text} (${pct.toFixed(1)}%)`}
                  onMouseEnter={() => setActive(a.i)}
                  onMouseLeave={() => setActive(null)}
                  onFocus={() => setActive(a.i)}
                  onBlur={() => setActive(null)}
                  transform={`translate(${off.x} ${off.y})`}
                  style={{
                    cursor: 'default',
                    outline: 'none',
                    opacity: dimmed ? 0.35 : 1,
                    transition: 'opacity 180ms ease-out, transform 180ms ease-out',
                  }}
                >
                  <path
                    d={arcPath(a.visStart, a.visEnd, R, rInner)}
                    fill={`url(#${uid}-fill-${tone})`}
                    stroke={palette.gridFaint}
                    strokeWidth={1}
                    filter={isActive ? `url(#${uid}-glow)` : undefined}
                    style={{ transition: 'filter 180ms ease-out' }}
                  />
                </g>
              )
            })}

            {/* center readout: running total + optional label */}
            <text
              x={CX}
              y={centerLabel ? CY - 4 : CY}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="30"
              fontWeight={800}
              fill={active != null ? toneHexLight[slices[active]?.tone ?? 'accent'] : '#ffffff'}
              className="tabular-nums"
              style={{ opacity: progress, transition: 'fill 180ms ease-out' }}
            >
              {active != null ? (slices[active]?.display ?? slices[active]?.value.toLocaleString()) : centerNum}
            </text>
            {centerLabel && (
              <text
                x={CX}
                y={CY + 20}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize="11"
                fontWeight={500}
                fill={palette.textFaint}
                style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
              >
                {active != null ? (slices[active]?.label ?? centerLabel) : centerLabel}
              </text>
            )}
          </svg>

          <ChartTooltip state={tooltipFor(active, arcs, total)} />
        </div>

        {legend && (
          <ul
            className="flex-1 space-y-2"
            style={{ opacity: seen ? 1 : 0, transition: 'opacity 500ms ease-out' }}
          >
            {slices.map((d, i) => {
              const tone = d.tone ?? 'accent'
              const pct = total > 0 ? (d.value / total) * 100 : 0
              const isActive = active === i
              const dimmed = active != null && !isActive
              return (
                <li
                  key={d.label}
                  tabIndex={0}
                  aria-label={`${d.label}: ${d.display ?? d.value.toLocaleString()} (${pct.toFixed(1)}%)`}
                  onMouseEnter={() => setActive(i)}
                  onMouseLeave={() => setActive(null)}
                  onFocus={() => setActive(i)}
                  onBlur={() => setActive(null)}
                  className="flex items-center justify-between gap-3 rounded-md px-1.5 py-1 text-[12px] outline-none transition-opacity duration-200"
                  style={{ opacity: dimmed ? 0.4 : 1 }}
                >
                  <span className="flex min-w-0 items-center gap-2">
                    <span
                      className="inline-block h-2.5 w-2.5 shrink-0 rounded-[3px]"
                      style={{
                        background: toneHex[tone],
                        boxShadow: isActive ? `0 0 7px ${toneHex[tone]}` : 'none',
                      }}
                    />
                    <span className="truncate text-slate-300">{d.label}</span>
                  </span>
                  <span className="shrink-0 tabular-nums text-slate-500">
                    <span className="font-semibold text-slate-300">{d.display ?? d.value.toLocaleString()}</span>
                    <span className="ml-1.5 text-slate-500">{pct.toFixed(1)}%</span>
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </div>

      {caption && <p className="mt-4 text-xs leading-relaxed text-slate-500">{caption}</p>}
    </div>
  )
}

type ArcLayout = {
  d: DonutSlice
  i: number
  start: number
  end: number
  frac: number
}

/** Tooltip anchored just outside the ring at the active slice's mid-angle. */
function tooltipFor(active: number | null, arcs: ArcLayout[], total: number): TooltipState {
  if (active == null) return null
  const a = arcs.find((x) => x.i === active)
  if (!a) return null
  const midDeg = (a.start + a.end) / 2
  const p = polar(CX, CY, R - (R - R * 0.58) / 2, midDeg)
  const pct = total > 0 ? (a.d.value / total) * 100 : 0
  return {
    xFrac: p.x / VB,
    yFrac: p.y / VB,
    label: a.d.label,
    value: `${a.d.display ?? a.d.value.toLocaleString()}  ·  ${pct.toFixed(1)}%`,
    tone: a.d.tone ?? 'accent',
  }
}
