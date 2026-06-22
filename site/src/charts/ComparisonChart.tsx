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

export type ComparisonBar = {
  label: string
  value: number
  tone?: Tone
  display?: string
  /** optional small line under the bar label (e.g. "production weights") */
  sublabel?: string
}

export type ComparisonChartProps = {
  /** exactly two bars: [from, to]. The drop between them is the story. */
  from: ComparisonBar
  to: ComparisonBar
  max?: number
  min?: number
  title?: string
  caption?: string
  /** show the connecting drop arrow + delta chip between the two bar tops (default true) */
  showDelta?: boolean
  /** override the delta string (defaults to a signed (to-from).toFixed(3), e.g. "-0.655") */
  deltaDisplay?: string
  /** number of gridlines incl. baseline (default 4) */
  gridlines?: number
  formatTick?: (v: number) => string
  ariaLabel?: string
}

const VB_W = 520
const VB_H = 320
const PAD = { top: 24, right: 20, bottom: 56, left: 40 }
const PLOT_W = VB_W - PAD.left - PAD.right
const PLOT_H = VB_H - PAD.top - PAD.bottom
const BASE_Y = PAD.top + PLOT_H

/**
 * Two-bar "before / after" comparison built to dramatize a collapse - e.g. the
 * Svanström recall result, baseline R=0.961 (good) vs bird-suppression retrain
 * R=0.306 (bad). A dashed guide drops from the taller bar's top to the shorter
 * one, capped with an arrowhead and a signed delta chip ("−0.655").
 *
 * Same interaction model as GroupedBarChart: scroll-reveal grow, hover/focus
 * tooltips with dimming, keyboard-focusable bars, reduced-motion → final state.
 * The delta guide animates in only after the bars have essentially finished
 * growing, so the eye reads "tall, then it falls".
 */
export function ComparisonChart({
  from,
  to,
  max = 1,
  min = 0,
  title,
  caption,
  showDelta = true,
  deltaDisplay,
  gridlines = 4,
  formatTick = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(2)),
  ariaLabel,
}: ComparisonChartProps) {
  const { ref, progress, seen } = useChartReveal<HTMLDivElement>(1200)
  const [active, setActive] = useState<0 | 1 | null>(null)
  const [uid] = useState(nextUid)

  const bars: ComparisonBar[] = [from, to]
  const span = max - min || 1
  const yOf = (v: number) => BASE_Y - (clamp(v, min, max) - min) / span * PLOT_H

  // two centered columns
  const colW = PLOT_W / 2
  const barW = Math.min(colW * 0.5, 92)
  const xOf = (i: number) => PAD.left + colW * i + (colW - barW) / 2

  const ticks = Array.from({ length: gridlines + 1 }, (_, i) => min + (span * i) / gridlines)

  const delta = to.value - from.value
  const deltaText = deltaDisplay ?? `${delta >= 0 ? '+' : '−'}${Math.abs(delta).toFixed(3)}`
  // reveal the drop guide in the tail of the animation (last ~30%)
  const dropP = clamp((progress - 0.7) / 0.3, 0, 1)

  const fromTop = yOf(from.value)
  const toTop = yOf(to.value)
  const taller = from.value >= to.value ? 0 : 1
  const shorter = taller === 0 ? 1 : 0
  const guideX = xOf(shorter) + barW / 2
  const guideTopY = yOf(bars[taller].value)
  const guideEndY = yOf(bars[shorter].value)
  const midY = (guideTopY + guideEndY) / 2

  return (
    <div ref={ref} className="w-full">
      {title && <div className="mb-3 text-sm font-semibold text-slate-300">{title}</div>}

      <div className="relative">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          width="100%"
          role="img"
          aria-label={
            ariaLabel ?? `${from.label} ${fmt3(from.value)} versus ${to.label} ${fmt3(to.value)}`
          }
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
              <feGaussianBlur stdDeviation="4.5" result="b" />
              <feMerge>
                <feMergeNode in="b" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <marker
              id={`${uid}-arrow`}
              markerWidth="9"
              markerHeight="9"
              refX="4.5"
              refY="4.5"
              orient="auto"
            >
              <path d="M1,1 L8,4.5 L1,8 Z" fill={toneHex.bad} />
            </marker>
          </defs>

          {/* gridlines + y ticks */}
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
          {bars.map((b, i) => {
            const tone = b.tone ?? (i === 0 ? 'good' : 'bad')
            const x = xOf(i)
            const fullTop = yOf(b.value)
            const fullH = BASE_Y - fullTop
            const h = fullH * progress
            const y = BASE_Y - h
            const isActive = active === i
            const dimmed = active != null && !isActive
            const text = b.display ?? fmt3(b.value)
            const r = Math.min(barW / 2, 6)

            return (
              <g
                key={b.label}
                tabIndex={0}
                role="img"
                aria-label={`${b.label}${b.sublabel ? `, ${b.sublabel}` : ''}: ${text}`}
                onMouseEnter={() => setActive(i as 0 | 1)}
                onMouseLeave={() => setActive(null)}
                onFocus={() => setActive(i as 0 | 1)}
                onBlur={() => setActive(null)}
                style={{
                  cursor: 'default',
                  outline: 'none',
                  opacity: dimmed ? 0.4 : 1,
                  transition: 'opacity 180ms ease-out',
                }}
              >
                <rect
                  x={x - 10}
                  y={PAD.top}
                  width={barW + 20}
                  height={PLOT_H}
                  fill="transparent"
                />
                <rect
                  x={x}
                  y={y}
                  width={barW}
                  height={Math.max(h, 0.001)}
                  rx={r}
                  fill={`url(#${uid}-fill-${tone})`}
                  filter={isActive ? `url(#${uid}-glow)` : undefined}
                  style={{ transition: 'filter 180ms ease-out' }}
                />
                {/* big value label above the bar */}
                <text
                  x={x + barW / 2}
                  y={y - 9}
                  textAnchor="middle"
                  fontSize="15"
                  fontWeight={700}
                  fill={isActive ? toneHexLight[tone] : palette.text}
                  className="tabular-nums"
                  style={{ opacity: progress, transition: 'fill 180ms ease-out' }}
                >
                  {text}
                </text>
                {/* bar label / sublabel below the baseline */}
                <text
                  x={x + barW / 2}
                  y={BASE_Y + 20}
                  textAnchor="middle"
                  fontSize="11.5"
                  fontWeight={600}
                  fill={palette.text}
                  style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
                >
                  {b.label}
                </text>
                {b.sublabel && (
                  <text
                    x={x + barW / 2}
                    y={BASE_Y + 35}
                    textAnchor="middle"
                    fontSize="9.5"
                    fill={palette.textFaint}
                    style={{ opacity: seen ? 1 : 0, transition: 'opacity 700ms ease-out' }}
                  >
                    {b.sublabel}
                  </text>
                )}
              </g>
            )
          })}

          {/* drop guide + delta chip (only meaningful when one bar is taller) */}
          {showDelta && from.value !== to.value && (
            <g style={{ opacity: dropP }}>
              {/* horizontal lead-in across the top of the taller bar */}
              <line
                x1={xOf(taller) + barW / 2}
                x2={guideX}
                y1={guideTopY}
                y2={guideTopY}
                stroke={toneHex.bad}
                strokeWidth={1.25}
                strokeDasharray="3 3"
                strokeOpacity={0.7}
              />
              {/* vertical drop with arrowhead landing on the shorter bar */}
              <line
                x1={guideX}
                x2={guideX}
                y1={guideTopY}
                y2={guideEndY - 2}
                stroke={toneHex.bad}
                strokeWidth={1.5}
                strokeDasharray="3 3"
                markerEnd={`url(#${uid}-arrow)`}
              />
              {/* delta chip */}
              <g transform={`translate(${guideX + 12}, ${midY})`}>
                <rect
                  x={0}
                  y={-13}
                  width={64}
                  height={26}
                  rx={13}
                  fill={toneHex.bad}
                  fillOpacity={0.14}
                  stroke={toneHex.bad}
                  strokeOpacity={0.5}
                />
                <text
                  x={32}
                  y={0}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize="13"
                  fontWeight={700}
                  fill={toneHexLight.bad}
                  className="tabular-nums"
                >
                  {deltaText}
                </text>
              </g>
            </g>
          )}
        </svg>

        <ChartTooltip state={tooltipFor(active, bars, xOf, barW, yOf)} />
      </div>

      {caption && <p className="mt-4 text-xs leading-relaxed text-slate-500">{caption}</p>}
    </div>
  )
}

function tooltipFor(
  active: 0 | 1 | null,
  bars: ComparisonBar[],
  xOf: (i: number) => number,
  barW: number,
  yOf: (v: number) => number,
): TooltipState {
  if (active == null) return null
  const b = bars[active]
  if (!b) return null
  return {
    xFrac: (xOf(active) + barW / 2) / VB_W,
    yFrac: yOf(b.value) / VB_H,
    label: b.sublabel ?? b.label,
    series: b.label,
    value: b.display ?? fmt3(b.value),
    tone: b.tone ?? (active === 0 ? 'good' : 'bad'),
  }
}
