import { useRef } from 'react'
import { useInViewOnce } from '../engine/useInViewOnce'
import { useReducedMotion } from '../engine/useReducedMotion'

export type Bar = {
  label: string
  value: number
  display?: string
  tone?: 'accent' | 'good' | 'bad' | 'muted'
}

const toneClass: Record<NonNullable<Bar['tone']>, string> = {
  accent: 'bg-cyan-400',
  good: 'bg-emerald-400',
  bad: 'bg-rose-500',
  muted: 'bg-slate-500',
}

/** Horizontal bars that grow from 0 on first view. Values are scaled against `max`. */
export function BarSet({
  bars,
  max = 1,
  title,
  caption,
}: {
  bars: Bar[]
  max?: number
  title?: string
  caption?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const seen = useInViewOnce(ref)
  const reduced = useReducedMotion()
  const animate = seen || reduced

  return (
    <div ref={ref} className="w-full">
      {title && <div className="mb-4 text-sm font-semibold text-slate-300">{title}</div>}
      <div className="space-y-3.5">
        {bars.map((b) => {
          const pct = Math.max(0, Math.min(100, (b.value / max) * 100))
          return (
            <div key={b.label}>
              <div className="mb-1 flex items-baseline justify-between text-xs">
                <span className="text-slate-400">{b.label}</span>
                <span className="font-semibold tabular-nums text-slate-200">
                  {b.display ?? b.value.toFixed(3)}
                </span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full ${toneClass[b.tone ?? 'accent']} transition-[width] duration-[1100ms] ease-out`}
                  style={{ width: animate ? `${pct}%` : '0%' }}
                />
              </div>
            </div>
          )
        })}
      </div>
      {caption && <p className="mt-4 text-xs leading-relaxed text-slate-500">{caption}</p>}
    </div>
  )
}
