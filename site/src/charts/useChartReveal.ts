import { useEffect, useRef, useState } from 'react'
import { useInViewOnce } from '../engine/useInViewOnce'
import { useReducedMotion } from '../engine/useReducedMotion'
import { easeOutCubic } from './chartTheme'

/**
 * Drives a single 0..1 eased progress value the first time `ref`'s element scrolls
 * into view, via requestAnimationFrame. Bars/axes/value-labels interpolate against
 * this so the whole chart reveals on one shared clock.
 *
 * - Respects prefers-reduced-motion: jumps straight to `1` (final state), no rAF.
 * - `seen` is also returned so callers can gate things that shouldn't appear at all
 *   until in view (e.g. tooltips, count-up labels).
 *
 * Returns a ref to attach to the chart's root element plus the live progress.
 */
export function useChartReveal<T extends Element>(duration = 1100) {
  const ref = useRef<T>(null)
  const seen = useInViewOnce(ref)
  const reduced = useReducedMotion()
  const [p, setP] = useState(0)

  useEffect(() => {
    if (!seen) return
    if (reduced) {
      setP(1)
      return
    }
    let raf = 0
    let start = 0
    const tick = (t: number) => {
      if (!start) start = t
      const raw = Math.min(1, (t - start) / duration)
      setP(easeOutCubic(raw))
      if (raw < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [seen, reduced, duration])

  return { ref, progress: p, seen, reduced }
}
