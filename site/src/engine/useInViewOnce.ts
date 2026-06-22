import { useEffect, useState, type RefObject } from 'react'

/** Fires once when the element first scrolls into view; used to trigger enter animations. */
export function useInViewOnce<T extends Element>(
  ref: RefObject<T | null>,
  rootMargin = '0px 0px -15% 0px',
): boolean {
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || seen) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setSeen(true)
          io.disconnect()
        }
      },
      { rootMargin },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [ref, seen, rootMargin])
  return seen
}
