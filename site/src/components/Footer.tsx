/** Closing strip: signals more acts to come, links out to the source material. */
export function Footer() {
  return (
    <footer className="border-t border-slate-800/60 bg-slate-950">
      <div className="mx-auto max-w-6xl px-6 py-16">
        <p className="eyebrow mb-3">Real-Time Multimodal Drone Detection</p>
        <p className="max-w-2xl text-lg font-medium text-slate-200">
          An interactive companion to the MSc thesis.
        </p>
        <div className="mt-8 flex flex-wrap gap-x-6 gap-y-2 text-sm">
          <a
            className="text-cyan-400 hover:text-cyan-300"
            href="https://github.com/AEEltayeb/multimodal_drone_detection_thesis"
            target="_blank"
            rel="noreferrer"
          >
            Source code &amp; thesis repository &rarr;
          </a>
        </div>
        <p className="mt-10 text-xs text-slate-600">
          &copy; 2026 Ahmed Eltayeb &middot; Real-Time Multimodal Drone Detection
        </p>
      </div>
    </footer>
  )
}
