import { useState } from 'react'
import type { Tone } from './chartTheme'
import {
  clamp,
  fmt3,
  nextUid,
  palette,
  toneHex,
  toneHexLight,
} from './chartTheme'
import { useChartReveal } from './useChartReveal'
import { ChartTooltip, type TooltipState } from './ChartTooltip'

export type ChartBar = {
  label: string
  value: number
  tone?: Tone
  /** override the printed/tooltip string (defaults to value.toFixed(3)) */
  display?: string
}

export type ChartGroup = {
  label: string
  bars: ChartBar[]
}

export type GroupedBarChartProps = {
  groups: ChartGroup[]
  /** top of the value axis (default 1 - these are F1 scores) */
  max?: number
  /** axis floor (default 0) */
  min?: number
  /** optional title shown above the plot */
  title?: string
  /** optional caption shown below the plot */
  caption?: string
  /** draw the shared legend from the first group's series labels (default true) */
  legend?: boolean
  /** ring + glow the tallest bar in each group (default true) */
  highlightGroupMax?: boolean
  /** number of horizontal gridlines incl. top (default 4) */
  gridlines?: number
  /** y-axis tick label formatter */
  formatTick?: (v: number) => string
  /** accessible description of the whole figure */
  ariaLabel?: string
}

/* ----- fixed viewBox geometry (responsive via width:100% on the <svg>) ----- */
const VB_W = 520
const VB_H = 320
const PAD = { top: 20, right: 16, bottom: 52, left: 40 }
const PLOT_W = VB_W - PAD.left - PAD.right
const PLOT_H = VB_H - PAD.top - PAD.bottom
const BASE_Y = PAD.top + PLOT_H

/**
 * Grouped vertical bar chart, hand-rolled in SVG.
 *
 * Built for the "modality reversal" story: several groups, each with N aligned
 * series. The per-group tallest bar gets a ring + soft glow so the eye lands on
 * the winner; give the series you want to emphasise (e.g. "Routed") the `accent`
 * tone and it carries the cyan brand colour in every group.
 *
 * Interactivity
 * - reveals once on scroll-in: bars grow from the baseline, axis/labels fade in
 *   on a shared eased clock (useChartReveal); reduced-motion jumps to final state.
 * - hover OR keyboard-focus a bar -> tooltip with exact value + series + group,
 *   and the other bars dim so the focused one pops.
 * - every bar is a focusable element with an aria-label.
 */
export function GroupedBarChart({
  groups,
  max = 1,
  min = 0,
  title,
  caption,
  legend = true,
  highlightGroupMax = true,
  gridlines = 4,
  formatTick = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(2)),
  ariaLabel,
}: GroupedBarChartProps) {
  const { ref, progress, seen } = useChartReveal<HTMLDivElement>(1150)
  const [active, setActive] = useState<{ g: number; b: number } | null>(null)
  const [uid] = useState(nextUid)

  const span = max - min || 1
  const yOf = (v: number) => BASE_Y - (clamp(v, min, max) - min) / span * PLOT_H

  // legend comes from the longest series row so every series shows once.
  const legendSource =
    groups.reduce<ChartBar[]>((best, g) => (g.bars.length > best.length ? g.bars : best), [])

  // layout: each group occupies an equal slot; bars centered within it.
  const groupW = PLOT_W / groups.length
  const seriesCount = Math.max(1, ...groups.map((g) => g.bars.length))
  const groupInnerPad = groupW * 0.16
  const slotW = groupW - groupInnerPad * 2
  const barGap = seriesCount > 1 ? slotW * 0.1 : 0
  const barW = (slotW - barGap * (seriesCount - 1)) / seriesCount

  const ticks = Array.from({ length: gridlines + 1 }, (_, i) => min + (span * i) / gridlines)

  return (
    <div ref={ref} className="w-full">
      {title && <div className="mb-3 text-sm font-semibold text-slate-300">{title}</div>}

      {legend && (
        <div
          className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1.5"
          style={{ opacity: seen ? 1 : 0, transition: 'opacity 500ms ease-out' }}
        >
          {legendSource.map((b) => (
            <span key={b.label} className="flex items-center gap-1.5 text-[11px] text-slate-400">
              <span
                className="inline-block h-2.5 w-2.5 rounded-[3px]"
                style={{ background: toneHex[b.tone ?? 'accent'] }}
              />
              {b.label}
            </span>
          ))}
        </div>
      )}

      <div className="relative">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          width="100%"
          role="img"
          aria-label={ariaLabel ?? title ?? 'Grouped bar chart'}
          className="block select-none overflow-visible"
        >
          <defs>
            {(['accent', 'good', 'bad', 'muted'] as Tone[]).map((t) => (
              <linearGradient key={t} id={`${uid}-fill-${t}`} x1="0" y1="0" x2="0" y2="1">
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

          {/* gridlines + y tick labels */}
          {ticks.map((tv, i) => {
            const gy = yOf(tv)
            return (
              <g key={i} style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}>
                <line
                  x1={PAD.left}
                  x2={PAD.left + PLOT_W}
                  y1={gy}
                  y2={gy}
                  stroke={i === 0 ? palette.baseline : palette.grid}
                  strokeWidth={i === 0 ? 1.25 : 1}
                />
                <text
                  x={PAD.left - 8}
                  y={gy}
                  textAnchor="end"
                  dominantBaseline="central"
                  fontSize="10"
                  fill={palette.textFaint}
                  className="tabular-nums"
                >
                  {formatTick(tv)}
                </text>
              </g>
            )
          })}

          {/* bars */}
          {groups.map((g, gi) => {
            const gx = PAD.left + gi * groupW + groupInnerPad
            const maxVal = Math.max(...g.bars.map((b) => b.value))
            return (
              <g key={g.label}>
                {g.bars.map((b, bi) => {
                  const tone = b.tone ?? 'accent'
                  const x = gx + bi * (barW + barGap)
                  const fullTop = yOf(b.value)
                  const fullH = BASE_Y - fullTop
                  const h = fullH * progress
                  const y = BASE_Y - h
                  const isWinner = highlightGroupMax && b.value === maxVal
                  const isActive = active?.g === gi && active?.b === bi
                  const dimmed = active != null && !isActive
                  const text = b.display ?? fmt3(b.value)
                  const r = Math.min(barW / 2, 5)

                  return (
                    <g
                      key={b.label}
                      tabIndex={0}
                      role="img"
                      aria-label={`${g.label}, ${b.label}: ${text}`}
                      onMouseEnter={() => setActive({ g: gi, b: bi })}
                      onMouseLeave={() => setActive(null)}
                      onFocus={() => setActive({ g: gi, b: bi })}
                      onBlur={() => setActive(null)}
                      style={{
                        cursor: 'default',
                        outline: 'none',
                        opacity: dimmed ? 0.35 : 1,
                        transition: 'opacity 180ms ease-out',
                      }}
                    >
                      {/* hover hit-area spanning full column height (steadier than the bar alone) */}
                      <rect
                        x={x - barGap / 2}
                        y={PAD.top}
                        width={barW + barGap}
                        height={PLOT_H}
                        fill="transparent"
                      />
                      {/* winner ring (sits behind the bar) */}
                      {isWinner && (
                        <rect
                          x={x - 3}
                          y={y - 3}
                          width={barW + 6}
                          height={h + 3}
                          rx={r + 2}
                          fill="none"
                          stroke={toneHex[tone]}
                          strokeOpacity={0.45}
                          strokeWidth={1.25}
                          style={{ opacity: progress }}
                        />
                      )}
                      <rect
                        x={x}
                        y={y}
                        width={barW}
                        height={Math.max(h, 0.001)}
                        rx={r}
                        fill={`url(#${uid}-fill-${tone})`}
                        filter={isActive || isWinner ? `url(#${uid}-glow)` : undefined}
                        style={{
                          transition: 'filter 180ms ease-out',
                        }}
                      />
                      {/* value label above the bar, fades/rises in with the reveal */}
                      <text
                        x={x + barW / 2}
                        y={y - 7}
                        textAnchor="middle"
                        fontSize="11"
                        fontWeight={isWinner ? 700 : 500}
                        fill={isActive || isWinner ? toneHexLight[tone] : palette.textDim}
                        className="tabular-nums"
                        style={{
                          opacity: progress,
                          transition: 'fill 180ms ease-out',
                        }}
                      >
                        {text}
                      </text>
                    </g>
                  )
                })}

                {/* group label */}
                <text
                  x={gx + slotW / 2}
                  y={BASE_Y + 20}
                  textAnchor="middle"
                  fontSize="11.5"
                  fontWeight={600}
                  fill={palette.text}
                  style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
                >
                  {g.label}
                </text>
              </g>
            )
          })}
        </svg>

        <ChartTooltip state={tooltipFor(active, groups, yOf)} />
      </div>

      {caption && <p className="mt-4 text-xs leading-relaxed text-slate-500">{caption}</p>}
    </div>
  )
}

/** Build the tooltip state (anchored to the top-center of the active bar). */
function tooltipFor(
  active: { g: number; b: number } | null,
  groups: ChartGroup[],
  yOf: (v: number) => number,
): TooltipState {
  if (!active) return null
  const g = groups[active.g]
  const b = g?.bars[active.b]
  if (!g || !b) return null

  const groupW = PLOT_W / groups.length
  const seriesCount = Math.max(1, ...groups.map((gr) => gr.bars.length))
  const groupInnerPad = groupW * 0.16
  const slotW = groupW - groupInnerPad * 2
  const barGap = seriesCount > 1 ? slotW * 0.1 : 0
  const barW = (slotW - barGap * (seriesCount - 1)) / seriesCount
  const gx = PAD.left + active.g * groupW + groupInnerPad
  const cx = gx + active.b * (barW + barGap) + barW / 2

  return {
    xFrac: cx / VB_W,
    yFrac: yOf(b.value) / VB_H,
    label: b.label,
    series: g.label,
    value: b.display ?? fmt3(b.value),
    tone: b.tone ?? 'accent',
  }
}
