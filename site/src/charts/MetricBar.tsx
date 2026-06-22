import { useState } from 'react'
import type { Tone } from './chartTheme'
import { clamp, fmt3, toneHex, toneHexLight } from './chartTheme'
import { useChartReveal } from './useChartReveal'

export type MetricBarItem = {
  label: string
  value: number
  tone?: Tone
  display?: string
}

export type MetricBarProps = {
  items: MetricBarItem[]
  max?: number
  min?: number
  title?: string
  caption?: string
  /** decimals for the count-up readout when no `display` is given (default 3) */
  decimals?: number
  ariaLabel?: string
}

/**
 * Refined horizontal stat bars - a polished cousin of the existing CSS BarSet,
 * but with an animated count-up readout, a soft track, rounded caps, an accent
 * glow + value highlight on hover/focus, and a focusable/labelled row for a11y.
 *
 * Each row's fill grows from 0 and the number counts up on the same eased reveal
 * clock; reduced-motion jumps straight to the final width + value. Generic and
 * props-driven so later acts can reuse it for any 0..max metric set.
 */
export function MetricBar({
  items,
  max = 1,
  min = 0,
  title,
  caption,
  decimals = 3,
  ariaLabel,
}: MetricBarProps) {
  const { ref, progress, seen } = useChartReveal<HTMLDivElement>(1100)
  const [active, setActive] = useState<number | null>(null)
  const span = max - min || 1

  return (
    <div
      ref={ref}
      className="w-full"
      role="group"
      aria-label={ariaLabel ?? title ?? 'Metric bars'}
    >
      {title && <div className="mb-4 text-sm font-semibold text-slate-300">{title}</div>}

      <div className="space-y-3.5">
        {items.map((it, i) => {
          const tone = it.tone ?? 'accent'
          const pct = clamp((it.value - min) / span, 0, 1) * 100
          const w = pct * progress
          const isActive = active === i
          const dimmed = active != null && !isActive
          const shown =
            it.display ?? (progress >= 1 ? it.value.toFixed(decimals) : (it.value * progress).toFixed(decimals))
          const color = toneHex[tone]

          return (
            <div
              key={it.label}
              tabIndex={0}
              aria-label={`${it.label}: ${it.display ?? fmt3(it.value)}`}
              onMouseEnter={() => setActive(i)}
              onMouseLeave={() => setActive(null)}
              onFocus={() => setActive(i)}
              onBlur={() => setActive(null)}
              className="rounded-md outline-none transition-opacity duration-200"
              style={{ opacity: dimmed ? 0.45 : 1 }}
            >
              <div className="mb-1 flex items-baseline justify-between text-xs">
                <span className="text-slate-400">{it.label}</span>
                <span
                  className="font-semibold tabular-nums transition-colors duration-200"
                  style={{ color: isActive ? toneHexLight[tone] : '#e2e8f0' }}
                >
                  {shown}
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${w}%`,
                    background: `linear-gradient(90deg, ${toneHex[tone]}, ${toneHexLight[tone]})`,
                    boxShadow: isActive ? `0 0 10px ${color}` : 'none',
                    transition: 'box-shadow 200ms ease-out',
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {caption && (
        <p
          className="mt-4 text-xs leading-relaxed text-slate-500"
          style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
        >
          {caption}
        </p>
      )}
    </div>
  )
}
