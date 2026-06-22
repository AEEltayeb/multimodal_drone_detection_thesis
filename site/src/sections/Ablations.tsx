import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Scrolly, type ScrollyStep } from '../engine/Scrolly'
import { useReducedMotion } from '../engine/useReducedMotion'
import { Disclosure } from '../components/Disclosure'
import { GroupedBarChart, MetricBar } from '../charts'

/* ----------------------------- tweened number ------------------------------ */

/** Smoothly animates the displayed value toward `value` whenever it changes. */
function useTween(value: number, reduced: boolean, duration = 750) {
  const [shown, setShown] = useState(value)
  const from = useRef(value)
  useEffect(() => {
    if (reduced || from.current === value) {
      setShown(value)
      from.current = value
      return
    }
    const start = from.current
    let raf = 0
    let t0 = 0
    const tick = (t: number) => {
      if (!t0) t0 = t
      const p = Math.min(1, (t - t0) / duration)
      const e = 1 - Math.pow(1 - p, 3)
      setShown(start + (value - start) * e)
      if (p < 1) raf = requestAnimationFrame(tick)
      else from.current = value
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value, reduced, duration])
  return shown
}

/* ------------------------------- scoreboard -------------------------------- */

const STAGE_PILLS = [
  { id: 'det', label: 'Detectors' },
  { id: 'filter', label: '+ Filter' },
  { id: 'router', label: '+ Router' },
  { id: 'temporal', label: '+ Temporal' },
]

type StageState = { rgb: number; ir: number; fire: number; routed: number | null; lit: string[] }

const STATE_BY_STEP: Record<string, StageState> = {
  // Svanström, the two detectors kept SEPARATE on purpose: the IR detector is
  // already strong (0.940); the proof is how far the pipeline lifts the weak RGB one.
  bare: { rgb: 0.607, ir: 0.94, fire: 0.304, routed: null, lit: ['det'] },
  filter: { rgb: 0.861, ir: 0.94, fire: 0.014, routed: null, lit: ['det', 'filter'] },
  router: { rgb: 0.861, ir: 0.94, fire: 0.014, routed: 0.946, lit: ['det', 'filter', 'router'] },
}

function Scoreboard({ rgb, ir, fire, routed, lit }: StageState) {
  const reduced = useReducedMotion()
  const trgb = useTween(rgb, reduced)
  const tfire = useTween(fire, reduced)
  const trouted = useTween(routed ?? 0.946, reduced)

  return (
    <div className="card mx-auto w-full max-w-md p-7">
      <p className="eyebrow mb-5">Svanström · keep the channels separate</p>

      <div className="grid grid-cols-2 gap-5">
        <div>
          <div className="text-4xl font-extrabold tabular-nums text-cyan-300">{trgb.toFixed(3)}</div>
          <div className="mt-1 text-xs uppercase tracking-wider text-slate-500">RGB detector F1</div>
        </div>
        <div>
          <div className="text-4xl font-extrabold tabular-nums text-emerald-300">{ir.toFixed(3)}</div>
          <div className="mt-1 text-xs uppercase tracking-wider text-slate-500">IR detector F1</div>
          <div className="mt-0.5 text-[10px] font-medium text-emerald-400/80">already strong</div>
        </div>
      </div>

      <div className="mt-6 flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3">
        <span className="text-xs text-slate-500">RGB confuser fire</span>
        <span className="text-2xl font-bold tabular-nums text-rose-300">{(tfire * 100).toFixed(1)}%</span>
      </div>

      <div
        className={`mt-3 flex items-center justify-between rounded-lg border px-4 py-3 transition-all duration-500 ${
          routed != null
            ? 'border-cyan-400/50 bg-cyan-500/10 opacity-100'
            : 'border-slate-800 bg-slate-900/20 opacity-40'
        }`}
      >
        <span className="text-xs text-slate-400">Routed system</span>
        <span className="text-2xl font-bold tabular-nums text-cyan-200">
          {routed != null ? trouted.toFixed(3) : '-'}
        </span>
      </div>

      <div className="mt-6 flex flex-wrap gap-2">
        {STAGE_PILLS.map((s) => {
          const on = lit.includes(s.id)
          return (
            <span
              key={s.id}
              className={`rounded-full border px-3 py-1 text-[11px] font-medium transition-colors duration-300 ${
                on
                  ? 'border-cyan-400/50 bg-cyan-500/10 text-cyan-200'
                  : 'border-slate-700 bg-slate-800/40 text-slate-500'
              }`}
            >
              {s.label}
            </span>
          )
        })}
      </div>
    </div>
  )
}

function CostRow({ big, small, tone }: { big: string; small: string; tone?: 'good' | 'accent' }) {
  const c = tone === 'good' ? 'text-emerald-300' : tone === 'accent' ? 'text-cyan-300' : 'text-white'
  return (
    <div className="flex items-baseline gap-3">
      <div className={`w-28 shrink-0 text-2xl font-bold tabular-nums ${c}`}>{big}</div>
      <div className="text-[13px] leading-snug text-slate-400">{small}</div>
    </div>
  )
}

function CostCard() {
  return (
    <div className="card mx-auto w-full max-w-md p-7">
      <p className="eyebrow mb-5">The cost side</p>
      <div className="space-y-4">
        <CostRow big="0.973→0.984" small="Anti-UAV F1 - no harm on the saturated control" tone="good" />
        <CostRow big="404×" small="trust router vs its predecessor (0.095 vs 38.3 ms / frame)" />
        <CostRow big="37–72×" small="confuser filter vs the patch CNN (1.3–2.1 vs 59–112 ms / detection)" />
        <CostRow big="1–4%" small="total pipeline overhead - so every stage runs on every frame" tone="accent" />
      </div>
    </div>
  )
}

/* -------------------------------- narrative -------------------------------- */

function Prose({ kicker, title, children }: { kicker: string; title: string; children: ReactNode }) {
  return (
    <div>
      <p className="eyebrow mb-2">{kicker}</p>
      <h3 className="text-2xl font-bold tracking-tight text-white sm:text-[28px]">{title}</h3>
      <div className="mt-4 space-y-4 text-[15px] leading-relaxed text-slate-300">{children}</div>
    </div>
  )
}

const steps: ScrollyStep[] = [
  {
    id: 'bare',
    node: (
      <Prose kicker="The ablation · 1" title="Don't combine the channels">
        <p>
          On Svanström the two detectors are not in the same place, so don&rsquo;t average them. The{' '}
          <span className="text-emerald-300">thermal detector is already strong</span> - F1{' '}
          <span className="text-white">0.940</span>.
        </p>
        <p>
          The <span className="text-cyan-300">visible detector</span> is the weak one:{' '}
          <span className="text-white">0.607</span>, and it fires on{' '}
          <span className="text-rose-300">30.4%</span> of confuser frames. This is the channel the
          pipeline has to prove itself on.
        </p>
      </Prose>
    ),
  },
  {
    id: 'filter',
    node: (
      <Prose kicker="The ablation · 2" title="The proof is the RGB lift">
        <p>
          Add the per-detection confuser filter to the RGB channel. Its F1 climbs{' '}
          <span className="accent">0.607 &rarr; 0.861</span> and its confuser fire collapses{' '}
          <span className="accent">30.4% &rarr; 1.4%</span>.
        </p>
        <p>
          That <span className="text-white">+25-point</span> lift, on the channel that needed it, is the
          proof - not a number the already-strong thermal detector ever needed help with.
        </p>
      </Prose>
    ),
  },
  {
    id: 'router',
    node: (
      <Prose kicker="The ablation · 3" title="The router fuses them">
        <p>
          Now let the trust router fuse the two, leaning on whichever it trusts per frame. The routed
          system reaches <span className="accent">0.946</span>.
        </p>
        <p>
          It matches the strong thermal detector <em>without being told which modality to believe</em>{' '}
          - and carries the lifted RGB channel along for the surfaces where visible is the one that
          works.
        </p>
      </Prose>
    ),
  },
  {
    id: 'cost',
    node: (
      <Prose kicker="The ablation · 4" title="…and it's nearly free">
        <p>
          On the saturated Anti-UAV control the pipeline does <span className="text-white">no harm</span>{' '}
          (0.973 &rarr; 0.984). The router is <span className="accent">404&times;</span> cheaper than its
          predecessor and the filter <span className="accent">37&ndash;72&times;</span>.
        </p>
        <p>
          So the whole cascade adds just <span className="accent">1&ndash;4%</span> over the two detector
          forward passes - every stage runs on every frame.
        </p>
      </Prose>
    ),
  },
]

function AblationVisual({ id }: { id: string }) {
  if (id === 'cost') return <CostCard />
  const s = STATE_BY_STEP[id] ?? STATE_BY_STEP.bare
  return <Scoreboard {...s} />
}

/* -------------------------------- section ---------------------------------- */

export function Ablations() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <div className="mb-6">
        <p className="eyebrow mb-3">Act V - The ablations</p>
        <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
          What does each stage actually buy?
        </h2>
        <p className="mt-4 max-w-2xl text-slate-400">
          Every cell replays the same stored detections, so each stage is isolated on identical frames.
          On Svanström the proof is specific: the thermal channel was already strong, so what matters is
          how far the pipeline lifts the weak visible one - RGB 0.607 &rarr; 0.861, routed 0.946.
        </p>
      </div>

      <Scrolly steps={steps} renderVisual={(id) => <AblationVisual id={id} />} />

      <Disclosure label="View the full per-surface ablation" openLabel="Hide per-surface ablation">
        <div className="mt-4 grid gap-8 lg:grid-cols-2">
          <div className="card p-7">
            <GroupedBarChart
              title="Drone F1 - bare detectors vs shipped pipeline"
              max={1}
              legend
              highlightGroupMax={false}
              groups={[
                {
                  label: 'Svanström',
                  bars: [
                    { label: 'bare', value: 0.742, display: '0.742', tone: 'muted' },
                    { label: 'shipped', value: 0.946, display: '0.946', tone: 'accent' },
                  ],
                },
                {
                  label: 'Anti-UAV',
                  bars: [
                    { label: 'bare', value: 0.973, display: '0.973', tone: 'muted' },
                    { label: 'shipped', value: 0.984, display: '0.984', tone: 'accent' },
                  ],
                },
                {
                  label: 'DUT (RGB)',
                  bars: [
                    { label: 'bare', value: 0.758, display: '0.758', tone: 'muted' },
                    { label: 'shipped', value: 0.835, display: '0.835', tone: 'accent' },
                  ],
                },
              ]}
              caption="Svanström 'bare' is the paired baseline; that lift is carried by the RGB channel (0.607 → 0.861) - the thermal channel was already at 0.940. Solo surfaces: IR test split 0.961 → 0.942 (a small filter recall cost), SelCom CCTV 0.591 → 0.612."
            />
          </div>
          <div className="card p-7">
            <MetricBar
              title="Confuser fire rate - bare vs shipped (lower is better)"
              max={0.35}
              decimals={1}
              items={[
                { label: 'RGB confusers · bare', value: 0.304, display: '30.4%', tone: 'bad' },
                { label: 'RGB confusers · shipped', value: 0.014, display: '1.4%', tone: 'good' },
                { label: 'IR confusers · bare', value: 0.294, display: '29.4%', tone: 'bad' },
                { label: 'IR confusers · shipped', value: 0.028, display: '2.8%', tone: 'good' },
              ]}
              caption="Per-frame fire rate on the out-of-distribution confuser surfaces. The reject-capable router can push RGB to 0.11%, the suppression the shipped stack trades for recall."
            />
          </div>
        </div>
      </Disclosure>
    </section>
  )
}
