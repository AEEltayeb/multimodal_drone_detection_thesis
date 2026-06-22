import { useRef, type ReactNode } from 'react'
import { useInViewOnce } from '../engine/useInViewOnce'

/* --------------------------------- data ------------------------------------ */

const ANSWERS = [
  {
    tag: 'RQ1',
    name: 'Capability',
    verdict: 'Yes - it raises performance, not just maintains it.',
    body: 'On Svanström the weak RGB channel goes 0.607 → 0.861 and the routed system reaches 0.946; out-of-distribution confuser fire drops 30.4% → 1.4%; the saturated Anti-UAV control is unharmed (0.973 → 0.984).',
  },
  {
    tag: 'RQ2',
    name: 'Attribution',
    verdict: 'Each stage is complementary, not redundant.',
    body: 'The filter does the false-positive work (835 → 39 confuser detections); the router buys back recall; composition order is a recall/precision dial. Both run on every frame for 1–4%.',
  },
  {
    tag: 'RQ3',
    name: 'Dual-modality',
    verdict: 'Yes - the ranking reverses, and routing pays on both sides.',
    body: 'IR leads Svanström (0.940 vs 0.607), RGB leads Anti-UAV (0.985 vs 0.961); the router matches the stronger modality on each without being told which it is.',
  },
]

const STACK = [
  { k: 'RGB detector', v: 'ft4 - bare detector P 0.84 / R 0.82 across 5 RGB benchmarks' },
  { k: 'IR detector', v: 'v3b - bare detector P 0.94 / R 0.97 across 3 thermal benchmarks' },
  { k: 'Trust router', v: 'robust8-nr - 8 free features, always routes (RGB / IR / both)' },
  { k: 'RGB confuser filter', v: 'mlp_v5_v4 - per-frame, P(drone) ≥ 0.25' },
  { k: 'IR confuser filter', v: 'mlp_aligned_thermalonly - thermal-native, ≥ 0.05' },
  { k: 'Temporal smoother', v: 'N-of-M sliding window above the cascade' },
  { k: 'Inference', v: 'imgsz 1280 (Svanström / CCTV; 960 deploy cam), 640 elsewhere; trust-aware scoring' },
]

/* ------------------------------- components -------------------------------- */

function Reveal({ children, delay = 0, className = '' }: { children: ReactNode; delay?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const seen = useInViewOnce(ref, '0px 0px -8% 0px')
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${seen ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  )
}

function AnswerCard({ a, i }: { a: (typeof ANSWERS)[number]; i: number }) {
  return (
    <Reveal delay={i * 110} className="h-full">
      <div className="card h-full p-6">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-500/15 text-sm text-emerald-300">
            ✓
          </span>
          <span className="text-xl font-extrabold text-cyan-400">{a.tag}</span>
          <span className="text-sm text-slate-400">{a.name}</span>
        </div>
        <p className="mt-3 text-[15px] font-semibold text-white">{a.verdict}</p>
        <p className="mt-2 text-[13px] leading-relaxed text-slate-400">{a.body}</p>
      </div>
    </Reveal>
  )
}

function DemoVideo() {
  return (
    <div className="card overflow-hidden">
      <video
        className="block w-full bg-slate-950"
        src="/media/video/demo_antiuav.mp4"
        poster="/media/video/demo_antiuav_poster.jpg"
        controls
        muted
        loop
        autoPlay
        playsInline
        preload="metadata"
      />
      <p className="border-t border-slate-800/70 px-5 py-3 text-[12px] leading-relaxed text-slate-500">
        The production cascade on a paired Anti-UAV sequence: ft4 and v3b detectors, the robust8-nr trust
        router, the mlp_v5_v4 RGB and thermal-native IR confuser filters, and the N-of-M temporal gate.
        Left is visible, right is thermal; every overlay (the trust decision, the MLP-V5 verifier verdicts,
        the confuser gate, the warning and alert counters) is produced by the engine.
      </p>
    </div>
  )
}

/* -------------------------------- section ---------------------------------- */

export function Conclusion() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <div className="mb-10">
        <p className="eyebrow mb-3">Act VII - The verdict</p>
        <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
          What holds - and what's honest about it
        </h2>
      </div>

      {/* the three questions, answered */}
      <p className="mb-5 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
        The three questions, answered
      </p>
      <div className="grid gap-5 md:grid-cols-3">
        {ANSWERS.map((a, i) => (
          <AnswerCard key={a.tag} a={a} i={i} />
        ))}
      </div>

      {/* see it run - the production cascade on real footage */}
      <div className="mt-16">
        <Reveal>
          <h3 className="text-2xl font-bold tracking-tight text-white">See it run</h3>
          <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-slate-400">
            The production cascade on real paired Anti-UAV footage: dual-modality detection, the per-frame
            trust decision, the confuser gate, and the temporal alert, all live.
          </p>
          <div className="mt-6">
            <DemoVideo />
          </div>
        </Reveal>
      </div>

      {/* what ships */}
      <div className="mt-16">
        <Reveal>
          <h3 className="text-2xl font-bold tracking-tight text-white">What ships</h3>
          <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-slate-400">
            The validated production stack: detectors tuned for recall, cheap learned stages for precision.
          </p>
          <div className="mt-6 card divide-y divide-slate-800/70">
            {STACK.map((s) => (
              <div key={s.k} className="flex flex-col gap-1 px-5 py-3 sm:flex-row sm:items-baseline sm:gap-4">
                <div className="w-44 shrink-0 text-sm font-semibold text-cyan-300">{s.k}</div>
                <div className="text-[13px] leading-snug text-slate-400">{s.v}</div>
              </div>
            ))}
          </div>
          <div className="mt-5 flex flex-wrap items-baseline gap-x-4 gap-y-1 rounded-lg border border-slate-700/60 bg-slate-900/50 px-5 py-4">
            <span className="text-xs font-semibold uppercase tracking-wider text-cyan-300/80">
              Full pipeline (routed)
            </span>
            <span className="text-2xl font-bold tabular-nums text-white">P 0.92 / R 0.93</span>
            <span className="text-[12px] text-slate-400">
              aggregate across the paired benchmarks (Svanström, Anti-UAV, DUT) - the bare detectors
              lifted by the filter and router
            </span>
          </div>
          <p className="mt-4 max-w-2xl text-[13px] leading-relaxed text-slate-500">
            One validated operating mode extends it: on recall-starved OOD RGB surfaces, drop the RGB floor
            to 0.05-0.10 and let the filter own precision (+10 pp F1 on SelCom, unchanged confuser safety).
          </p>
        </Reveal>
      </div>

    </section>
  )
}
