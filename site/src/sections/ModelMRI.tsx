import type { ReactNode } from 'react'
import { Scrolly, type ScrollyStep } from '../engine/Scrolly'
import { LdaSeparabilityChart, PcaScatterChart, ActivationOverlay } from '../charts'

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
    id: 'supervised',
    node: (
      <Prose kicker="Model MRI · 1" title="Is the signal even there?">
        <p>
          Before training the filter, the Model MRI asks a question with statistics, not a hunch: are
          drones and confusers already separable in the detector&rsquo;s own <span className="text-white">p3+p5</span>{' '}
          features?
        </p>
        <p>
          Project them onto a single supervised axis (LDA) and they fall into two clean humps -
          separability <span className="accent">0.952</span> in RGB, <span className="accent">0.981</span>{' '}
          in thermal. The signal is there, and it is linear.
        </p>
      </Prose>
    ),
  },
  {
    id: 'unsupervised',
    node: (
      <Prose kicker="Model MRI · 2" title="…but it isn't trivial">
        <p>
          It isn&rsquo;t trivial, though. Run <em>unsupervised</em> PCA on the same features and the two
          classes sit right on top of each other - silhouette just <span className="text-rose-300">0.067</span>.
        </p>
        <p>
          No distance rule or clustering would ever find the boundary. The signal is real but{' '}
          <span className="text-white">supervised</span> - which is exactly why the fix is a small
          trained classifier, not a threshold.
        </p>
      </Prose>
    ),
  },
  {
    id: 'verdict',
    node: (
      <Prose kicker="Model MRI · 3" title="An evidence-backed verdict">
        <p>
          So the verdict is backed by measurement: a classifier is worth building, and the MRI projects
          what it will buy - a <span className="accent">97.4%</span> false-positive cut at{' '}
          <span className="accent">98.9%</span> recall - before a single epoch runs.
        </p>
        <p>
          And it is a diagnostic, not a rubber stamp: pointed at the thermal detector, the same instrument
          returned the <em>opposite</em> call - already clean, no filter needed.{' '}
          <span className="text-white">See, don&rsquo;t guess.</span>
        </p>
      </Prose>
    ),
  },
]

/* ------------------------------ sticky visuals ----------------------------- */

function Stat({ v, l, tone }: { v: string; l: string; tone?: 'good' }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2.5">
      <div className={`text-lg font-bold tabular-nums ${tone === 'good' ? 'text-emerald-300' : 'text-white'}`}>
        {v}
      </div>
      <div className="mt-0.5 text-[11px] text-slate-500">{l}</div>
    </div>
  )
}

function VerdictCard() {
  return (
    <div className="card mx-auto w-full max-w-md p-7">
      <span className="inline-flex items-center gap-2 rounded-full border border-cyan-400/40 bg-cyan-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-cyan-200">
        <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" /> Verdict
      </span>
      <h4 className="mt-4 text-xl font-bold text-white">Classifier strongly recommended</h4>
      <p className="mt-2 text-[13px] leading-relaxed text-slate-400">
        A large false-positive cut at low recall cost - projected from the feature statistics,
        before a single epoch is trained.
      </p>
      <div className="mt-5 grid grid-cols-2 gap-3">
        <Stat v="97.4%" l="projected FP cut" tone="good" />
        <Stat v="98.9%" l="recall retained" tone="good" />
        <Stat v="0.952" l="LDA separability" />
        <Stat v="0.986" l="filter CV F1" />
      </div>
      <p className="mt-5 rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3 text-[12px] leading-relaxed text-slate-400">
        A diagnostic, not a rubber stamp: on the thermal detector the same instrument returned the
        opposite call - already clean (1.8% fire), <span className="text-slate-200">no filter needed</span>.
      </p>
    </div>
  )
}

function MriVisual({ id }: { id: string }) {
  switch (id) {
    case 'supervised':
      return (
        <div className="card mx-auto w-full max-w-lg p-6">
          <LdaSeparabilityChart
            title="Supervised (LDA): one axis splits them"
            separability={0.952}
            separabilityAlt={{ label: 'IR', value: 0.981 }}
            caption="Drone vs confuser on a single Fisher axis from the RGB detector's own p3+p5 features. The thermal detector separates even more cleanly (0.981)."
          />
        </div>
      )
    case 'unsupervised':
      return (
        <div className="card mx-auto w-full max-w-lg p-6">
          <PcaScatterChart
            title="Unsupervised (PCA): the same classes overlap"
            silhouette={0.067}
            caption="The identical features, without labels. Silhouette 0.067 - invisible to clustering, so a trained classifier is required, not a distance threshold."
          />
        </div>
      )
    case 'verdict':
      return <VerdictCard />
    default:
      return null
  }
}

/* -------------------------------- section ---------------------------------- */

export function ModelMRI() {
  return (
    <section className="border-t border-slate-900 bg-gradient-to-b from-slate-950 to-slate-900/30">
      <div className="mx-auto max-w-6xl px-6 py-20">
        <div className="mb-6">
          <p className="eyebrow mb-3">Act IV - The method &middot; Model MRI</p>
          <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
            See, don&rsquo;t guess
          </h2>
          <p className="mt-4 max-w-2xl text-slate-400">
            Before training the confuser filter, the Model MRI images the detector&rsquo;s feature space
            and decides - with classical statistics - whether a filter is needed and whether it
            will work. Measure first, then build.
          </p>
        </div>

        <Scrolly steps={steps} renderVisual={(id) => <MriVisual id={id} />} />

        <div className="mt-24">
          <ActivationOverlay />
        </div>
      </div>
    </section>
  )
}
