import { useEffect, useRef, useState } from 'react'
import { useInViewOnce } from '../engine/useInViewOnce'
import { useReducedMotion } from '../engine/useReducedMotion'

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3)

/** Count-up number that animates from 0 to `to` the first time it scrolls into view. */
export function Counter({
  to,
  decimals = 0,
  suffix = '',
  prefix = '',
  duration = 1400,
  className = '',
}: {
  to: number
  decimals?: number
  suffix?: string
  prefix?: string
  duration?: number
  className?: string
}) {
  const ref = useRef<HTMLSpanElement>(null)
  const seen = useInViewOnce(ref)
  const reduced = useReducedMotion()
  const [val, setVal] = useState(0)

  useEffect(() => {
    if (!seen) return
    if (reduced) {
      setVal(to)
      return
    }
    let raf = 0
    let startT = 0
    const tick = (t: number) => {
      if (!startT) startT = t
      const p = Math.min(1, (t - startT) / duration)
      setVal(to * easeOut(p))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [seen, to, duration, reduced])

  return (
    <span ref={ref} className={className}>
      {prefix}
      {val.toFixed(decimals)}
      {suffix}
    </span>
  )
}
