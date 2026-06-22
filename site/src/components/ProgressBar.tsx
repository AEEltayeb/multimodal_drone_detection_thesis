import { useEffect, useState } from 'react'

/** Thin scroll-progress bar pinned to the top of the page. */
export function ProgressBar() {
  const [pct, setPct] = useState(0)
  useEffect(() => {
    const onScroll = () => {
      const el = document.documentElement
      const max = el.scrollHeight - el.clientHeight
      setPct(max > 0 ? (el.scrollTop / max) * 100 : 0)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [])
  return (
    <div className="fixed inset-x-0 top-0 z-50 h-0.5">
      <div
        className="h-full bg-cyan-400 transition-[width] duration-150 ease-out"
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
