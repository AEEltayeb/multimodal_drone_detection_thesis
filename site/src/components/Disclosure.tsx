import { useState, type ReactNode } from 'react'

/**
 * A centered "Show …" button that smoothly drops its content down when opened
 * (grid-rows 0fr → 1fr, so it animates to the content's natural height).
 * Collapsed by default; accessible (aria-expanded, real <button>).
 */
export function Disclosure({
  label = 'Show examples',
  openLabel = 'Hide examples',
  defaultOpen = false,
  children,
  className = '',
}: {
  label?: string
  openLabel?: string
  defaultOpen?: boolean
  children: ReactNode
  className?: string
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className={`mx-auto max-w-6xl px-6 py-8 ${className}`}>
      <div className="flex justify-center">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="group inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900/60 px-5 py-2.5 text-sm font-medium text-slate-200 transition-colors hover:border-cyan-500/50 hover:text-white"
        >
          {open ? openLabel : label}
          <svg
            viewBox="0 0 16 16"
            className={`h-4 w-4 transition-transform duration-300 ${open ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            aria-hidden="true"
          >
            <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
      <div
        className={`grid transition-all duration-500 ease-out ${
          open ? 'mt-2 grid-rows-[1fr] opacity-100' : 'grid-rows-[0fr] opacity-0'
        }`}
      >
        <div className="overflow-hidden">{children}</div>
      </div>
    </div>
  )
}
