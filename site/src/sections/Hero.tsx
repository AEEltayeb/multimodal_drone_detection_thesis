/** Act 0.1 - title hero: the dual demand that frames the whole thesis. */
export function Hero() {
  return (
    <section className="relative flex min-h-screen items-center overflow-hidden">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950" />
      <div
        className="pointer-events-none absolute inset-0 opacity-20"
        style={{
          backgroundImage:
            'radial-gradient(circle at 72% 28%, rgba(34,211,238,0.18), transparent 42%)',
        }}
      />
      <div className="relative mx-auto max-w-4xl px-6 py-32 text-center">
        <p className="eyebrow mb-5">Interactive companion to the thesis</p>
        <h1 className="text-balance text-4xl font-extrabold leading-[1.05] tracking-tight text-white sm:text-6xl">
          Real-Time Multimodal
          <br />
          Drone Detection
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-lg text-cyan-300/90">
          Through Visual and Thermal Sensor Fusion
        </p>
        <p className="mx-auto mt-8 max-w-2xl text-pretty text-base leading-relaxed text-slate-300 sm:text-lg">
          Protecting a fixed site against consumer drones demands two things at once: catch every drone
          (<span className="text-white">high recall</span>), and almost never cry wolf (a{' '}
          <span className="text-white">low false-alarm rate</span>). This is the story of a system that
          does both - by moving the hard decisions <span className="accent">off the detector</span>.
        </p>
        <div className="mt-16 flex flex-col items-center gap-2 text-slate-500">
          <span className="text-[11px] uppercase tracking-[0.3em]">Scroll</span>
          <span className="h-9 w-px animate-pulse bg-gradient-to-b from-cyan-400/70 to-transparent" />
        </div>
      </div>
    </section>
  )
}
