import { useEffect, useRef, useState, type ReactNode } from 'react'

export type ScrollyStep = { id: string; node: ReactNode }

/**
 * Two-column scrollytelling engine (the raihankalla reference pattern):
 * left = stepped narrative, right = ONE sticky visual that swaps as each step
 * crosses the viewport center. A single IntersectionObserver tracks the active step.
 * On mobile it collapses to one column and renders each step's visual inline.
 */
export function Scrolly({
  steps,
  renderVisual,
}: {
  steps: ScrollyStep[]
  renderVisual: (activeId: string, index: number) => ReactNode
}) {
  const [active, setActive] = useState(0)
  const refs = useRef<Array<HTMLDivElement | null>>([])

  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => {
        let bestIdx = -1
        let bestRatio = -1
        for (const e of entries) {
          if (e.isIntersecting && e.intersectionRatio > bestRatio) {
            bestRatio = e.intersectionRatio
            bestIdx = Number((e.target as HTMLElement).dataset.idx)
          }
        }
        if (bestIdx >= 0) setActive(bestIdx)
      },
      { rootMargin: '-45% 0px -45% 0px', threshold: [0, 0.25, 0.5, 0.75, 1] },
    )
    refs.current.forEach((el) => el && io.observe(el))
    return () => io.disconnect()
  }, [steps.length])

  const activeId = steps[active]?.id ?? steps[0].id

  return (
    <div className="scrolly">
      <div className="scrolly-steps">
        {steps.map((s, i) => (
          <div
            key={s.id}
            data-idx={i}
            ref={(el) => {
              refs.current[i] = el
            }}
            className="scrolly-step"
          >
            <div className="scrolly-visual-inline card p-5">{renderVisual(s.id, i)}</div>
            {s.node}
          </div>
        ))}
      </div>
      <div className="scrolly-sticky">
        <div className="scrolly-sticky-inner">
          <div className="w-full">{renderVisual(activeId, active)}</div>
        </div>
      </div>
    </div>
  )
}
