/** Slim fixed top bar with the TALOS brand mark. */
export function Header() {
  return (
    <header className="fixed inset-x-0 top-0 z-40 border-b border-slate-800/60 bg-slate-950/70 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
        <div className="flex items-center gap-2.5">
          <img src="/logo.png" alt="TALOS" className="h-6 w-auto" />
          <span className="text-sm font-medium text-slate-400">Drone Detection</span>
        </div>
        <span className="hidden text-xs text-slate-500 sm:block">
          MSc Thesis &middot; University of Salerno
        </span>
      </div>
    </header>
  )
}
