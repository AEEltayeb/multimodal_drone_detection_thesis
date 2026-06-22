import type { Tone } from './chartTheme'
import { toneHex } from './chartTheme'

export type TooltipState = {
  /** horizontal anchor as a 0..1 fraction of the chart's rendered width */
  xFrac: number
  /** vertical anchor as a 0..1 fraction of the chart's rendered height */
  yFrac: number
  label: string
  value: string
  series?: string
  tone?: Tone
} | null

/**
 * Floating tooltip rendered as an HTML overlay (crisp text, no SVG <text> scaling),
 * positioned via percentage `left/top` so it tracks the SVG regardless of the
 * responsive viewBox scale. It sits inside the chart's `position:relative` wrapper.
 *
 * Smoothness: opacity + translate are CSS-transitioned, so it fades/slides between
 * anchor points instead of snapping. `pointer-events:none` keeps it from stealing
 * hover/focus from the bars underneath.
 */
export function ChartTooltip({ state }: { state: TooltipState }) {
  const shown = state != null
  const xFrac = state?.xFrac ?? 0.5
  const yFrac = state?.yFrac ?? 0.5
  const dotColor = state?.tone ? toneHex[state.tone] : '#22d3ee'

  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none absolute z-10"
      style={{
        left: `${xFrac * 100}%`,
        top: `${yFrac * 100}%`,
        transform: `translate(-50%, calc(-100% - 10px)) translateY(${shown ? '0' : '4px'})`,
        opacity: shown ? 1 : 0,
        transition: 'opacity 140ms ease-out, transform 140ms ease-out',
      }}
    >
      <div className="relative whitespace-nowrap rounded-lg border border-slate-700/70 bg-slate-900/95 px-3 py-2 shadow-xl shadow-black/40 backdrop-blur-sm">
        {state?.series && (
          <div className="mb-0.5 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-slate-400">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: dotColor, boxShadow: `0 0 6px ${dotColor}` }}
            />
            {state.series}
          </div>
        )}
        <div className="text-[11px] leading-tight text-slate-400">{state?.label}</div>
        <div className="text-base font-semibold leading-tight tabular-nums text-white">
          {state?.value}
        </div>
        {/* little downward caret */}
        <span className="absolute left-1/2 top-full -translate-x-1/2 -translate-y-px border-x-[6px] border-t-[6px] border-x-transparent border-t-slate-700/70" />
      </div>
    </div>
  )
}
