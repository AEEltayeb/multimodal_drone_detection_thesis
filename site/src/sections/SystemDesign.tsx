import type { ReactNode } from 'react'
import { Scrolly, type ScrollyStep } from '../engine/Scrolly'

/* ------------------------------ pipeline diagram --------------------------- */

type Stage = 'detectors' | 'filter' | 'router' | 'temporal' | 'alert'

function StageChip({
  lit,
  title,
  sub,
  alert = false,
}: {
  lit: boolean
  title: string
  sub?: string
  alert?: boolean
}) {
  const cls = lit
    ? alert
      ? 'border-rose-400/60 bg-rose-500/10 text-rose-100 shadow-lg shadow-rose-500/30'
      : 'border-cyan-400/60 bg-cyan-500/10 text-white shadow-lg shadow-cyan-500/30'
    : 'border-slate-700/60 bg-slate-800/30 text-slate-300'
  return (
    <div className={`rounded-xl border px-4 py-3 text-center transition-all duration-300 ${cls}`}>
      <div className="text-sm font-semibold">{title}</div>
      {sub && (
        <div className={`mt-0.5 text-[11px] ${lit ? 'text-slate-300' : 'text-slate-500'}`}>{sub}</div>
      )}
    </div>
  )
}

const Connector = () => (
  <div className="flex justify-center py-1.5">
    <div className="h-4 w-px bg-slate-600" />
  </div>
)

/** The four-stage cascade as a vertical flow; the active stage lights up on scroll. */
export function PipelineDiagram({ active }: { active: Stage | 'all' | null }) {
  const lit = (s: Stage) => active === 'all' || active === s
  return (
    <div className="mx-auto w-full max-w-sm">
      <div className="grid grid-cols-2 gap-3">
        <StageChip lit={lit('detectors')} title="RGB detector" sub="tuned for recall" />
        <StageChip lit={lit('detectors')} title="IR detector" sub="thermal / grayscale" />
      </div>
      <Connector />
      <StageChip lit={lit('filter')} title="Confuser filter" sub="per-frame · feature reuse" />
      <Connector />
      <StageChip lit={lit('router')} title="Trust router" sub="8 features · always routes" />
      <Connector />
      <StageChip lit={lit('temporal')} title="Temporal smoother" sub="N-of-M window" />
      <Connector />
      <StageChip lit={lit('alert')} title="ALERT" alert />
    </div>
  )
}

/* ------------------------------- narrative --------------------------------- */

function Prose({ kicker, title, children }: { kicker: string; title: string; children: ReactNode }) {
  return (
    <div>
      <p className="eyebrow mb-2">{kicker}</p>
      <h3 className="text-2xl font-bold tracking-tight text-white sm:text-[28px]">{title}</h3>
      <div className="mt-4 space-y-4 text-[15px] leading-relaxed text-slate-300">{children}</div>
    </div>
  )
}

const STAGE_BY_STEP: Record<string, Stage | 'all' | null> = {
  principle: null,
  detectors: 'detectors',
  filter: 'filter',
  router: 'router',
  temporal: 'temporal',
  assembled: 'all',
}

const steps: ScrollyStep[] = [
  {
    id: 'principle',
    node: (
      <Prose kicker="The system · 1" title="One principle: asymmetric recoverability">
        <p>
          A confuser false positive raised by the detector can be vetoed downstream. A drone the detector
          <em> missed</em> cannot be recovered by any later stage. The two errors are not symmetric -
          so suppression is never the detector&rsquo;s job.
        </p>
        <p>
          Every discrimination the detectors cannot learn without costing recall is relocated into cheap,
          learned, downstream stages, and the detectors are left tuned purely for recall.
        </p>
      </Prose>
    ),
  },
  {
    id: 'detectors',
    node: (
      <Prose kicker="The system · 2" title="Two detectors, tuned for recall">
        <p>
          Two single-stage YOLO26n detectors run back to back - one on the visible stream, one on
          thermal. Each is trained with confuser negatives, but neither is made stricter to fight
          confusers; their confidence floors sit at normal optima. They propose; the downstream stages
          dispose.
        </p>
        <p>
          The thermal detector doubles as a <span className="accent">grayscale-RGB fallback</span> when no
          thermal camera is present - a finding we return to later.
        </p>
      </Prose>
    ),
  },
  {
    id: 'filter',
    node: (
      <Prose kicker="The system · 3" title="A confuser filter that reuses the detector's own features">
        <p>
          The per-detection confuser filter re-reads the detector&rsquo;s internal features - the{' '}
          <span className="text-white">p3</span> (fine spatial) and <span className="text-white">p5</span>{' '}
          (semantic) pyramid activations already computed for each box - through a small MLP, and
          vetoes a detection unless P(drone) clears a threshold.
        </p>
        <p>
          Because it reuses features rather than re-processing crops, it is{' '}
          <span className="accent">37&ndash;72&times; faster</span> than the CNN it replaced (1.3&ndash;2.1
          ms vs 59&ndash;112 ms per detection) - cheap enough to run on <em>every</em> frame.
        </p>
      </Prose>
    ),
  },
  {
    id: 'router',
    node: (
      <Prose kicker="The system · 4" title="A trust router that picks the modality per frame">
        <p>
          A per-frame trust router (XGBoost) decides which modality to believe - RGB, IR, or both. It
          reads <em>detection evidence only</em>: eight free features (per-modality confidence, box
          geometry, cross-modal agreement), selected by a <span className="accent">leakage statistic</span>{' '}
          that drops anything fingerprinting the scene rather than the drone.
        </p>
        <p>
          It always routes - never abstains - leaving false-positive rejection to the filter,
          and it is <span className="accent">404&times; cheaper</span> than the hand-engineered classifier
          it replaced.
        </p>
      </Prose>
    ),
  },
  {
    id: 'temporal',
    node: (
      <Prose kicker="The system · 5" title="A temporal smoother that turns decisions into alerts">
        <p>
          An <span className="text-white">N-of-M</span> sliding-window smoother fires an alert only when the
          per-frame decision persists across the window - absorbing the isolated firings
          characteristic of a bird or an aircraft transiting the field of view.
        </p>
      </Prose>
    ),
  },
  {
    id: 'assembled',
    node: (
      <Prose kicker="The system · 6" title="The whole cascade, for 1–4% extra">
        <p>
          Composed filter-then-route, the entire cascade adds just{' '}
          <span className="accent">1&ndash;4%</span> over the two detector forward passes. So every stage
          runs on every frame, and the alert-gating compromises of the predecessor design simply dissolve.
        </p>
        <p className="text-slate-400">
          Detectors for recall; cheap learned stages for precision. That division of labor is the system.
        </p>
      </Prose>
    ),
  },
]

export function SystemDesign() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <div className="mb-6">
        <p className="eyebrow mb-3">Act III - The system</p>
        <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
          Move the work downstream - one cheap stage at a time
        </h2>
      </div>
      <Scrolly steps={steps} renderVisual={(id) => <PipelineDiagram active={STAGE_BY_STEP[id] ?? null} />} />
    </section>
  )
}
