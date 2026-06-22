import type { ReactNode } from 'react'
import { Scrolly, type ScrollyStep } from '../engine/Scrolly'
import { Counter } from '../interactions/Counter'
import { GroupedBarChart, ComparisonChart } from '../charts'

/* ----------------------------- narrative steps ----------------------------- */

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
    id: 'good',
    node: (
      <Prose kicker="The problem · 1" title="The detectors are already good">
        <p>
          On ordinary footage - sky, buildings, traffic, vegetation - the false-alarm
          problem is already solved at the detection head. The production RGB detector reaches{' '}
          <span className="text-white">P&nbsp;=&nbsp;0.989, R&nbsp;=&nbsp;0.982</span> on the Anti-UAV
          benchmark, with just 41 false positives across 4,000 frames, and hallucinates on only{' '}
          <span className="text-white">2.8%</span> of its in-distribution test set.
        </p>
        <p>So the open problem is narrow. The question is what happens at the edges.</p>
      </Prose>
    ),
  },
  {
    id: 'confusers',
    node: (
      <Prose kicker="The problem · 2" title="Until the sky fills with look-alikes">
        <p>
          The failures concentrate in scenes with three aerial <em>confusers</em>: birds, fixed-wing
          aircraft, and helicopters. On a 2,633-image out-of-distribution confuser set, the same
          detector fires on <span className="text-rose-300">30.4%</span> of frames. On bird-only
          Svanstr&ouml;m footage, that rises to <span className="text-rose-300">94.4%</span>.
        </p>
        <p className="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3 text-[13px] text-slate-400">
          (Scope: this is the <span className="text-slate-200">hard</span> regime - small, distant
          birds against open sky in the Svanstr&ouml;m surveillance set, where at range a bird and a
          drone are nearly the same few-pixel silhouette. It is <em>not</em> &ldquo;all birds&rdquo;: on
          ordinary, close-range bird footage the detector is far less easily fooled. The cascade is built
          for exactly this narrow, stubborn slice.)
        </p>
        <p>
          Nearly one confuser frame in three becomes a false drone alert - and constant false
          alarms desensitize the very operator the system exists to serve.
        </p>
      </Prose>
    ),
  },
  {
    id: 'geometry',
    node: (
      <Prose kicker="The problem · 3" title="The cause is geometric">
        <p>
          At surveillance range, a drone, a bird, an airplane, and a helicopter all present as small,
          low-texture silhouettes against sky. The gap that separates them is too narrow for the
          detector to exploit <em>at its decision layer</em> without losing drones.
        </p>
        <p>
          But the separation does exist - deeper in the network, inside the detector&rsquo;s own
          intermediate features. Exploiting it <span className="accent">there</span>, rather than at the
          detection head, is the central idea of the thesis.
        </p>
      </Prose>
    ),
  },
  {
    id: 'retrain',
    node: (
      <Prose kicker="The problem · 4" title="And &ldquo;just retrain it&rdquo; backfires">
        <p>
          The obvious fix - retrain the detector with confuser images as hard negatives -
          gives class-specific, diminishing returns. It suppresses helicopters and airplanes, but barely
          touches birds. Push harder, and bird fire finally collapses - and Svanstr&ouml;m drone
          recall collapses with it, from <span className="text-white">0.961</span> to{' '}
          <span className="text-rose-300">0.306</span>.
        </p>
        <p>
          Worse, on its own in-domain test split those weights look <em>better</em> than the baseline.
          In-domain model selection would pick exactly the detector that fails in the field. And recall,
          once lost at the detector, cannot be recovered downstream.
        </p>
      </Prose>
    ),
  },
  {
    id: 'gaps',
    node: (
      <Prose kicker="The problem · 5" title="Two more gaps: resolution and modality">
        <p>
          Resolution: Svanstr&ouml;m&rsquo;s median drone is just <span className="text-white">29.8&nbsp;px</span>{' '}
          across, and recall collapses to <span className="text-white">0.63</span> for drones below 16&nbsp;px
          - small targets simply disappear.
        </p>
        <p>
          Modality: the visible channel degrades at night and under glare, exactly where thermal is
          strongest. And which modality wins <em>reverses</em> with the scene - thermal leads on
          Svanstr&ouml;m, visible leads on Anti-UAV. So the system is dual-modality from the start.
        </p>
      </Prose>
    ),
  },
  {
    id: 'move',
    node: (
      <Prose kicker="The problem · 6" title="So move the hard decisions downstream">
        <p>
          Any discrimination the detector cannot learn without costing recall must move <em>off</em> the
          detector. The thesis relocates confuser rejection, modality choice, and false-positive
          filtering into cheap downstream stages: a <span className="accent">trust router</span> that
          picks the reliable modality each frame, a per-detection{' '}
          <span className="accent">confuser filter</span> that re-reads the detector&rsquo;s own features,
          and a <span className="accent">temporal smoother</span> that turns per-frame decisions into
          alerts.
        </p>
        <p>
          The detectors themselves are left tuned purely for recall. That architectural move is the
          thesis.
        </p>
      </Prose>
    ),
  },
]

/* ------------------------------ sticky visuals ----------------------------- */

function Metric({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="text-3xl font-bold tabular-nums text-white">{children}</div>
      <div className="mt-1 text-xs uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  )
}

function Chip({ children, tone = 'plain' }: { children: ReactNode; tone?: 'plain' | 'accent' | 'alert' }) {
  const cls =
    tone === 'alert'
      ? 'justify-center border-rose-500/40 bg-rose-500/10 font-semibold text-rose-200'
      : tone === 'accent'
        ? 'border-cyan-500/30 bg-cyan-500/[0.06] text-slate-200'
        : 'border-slate-700/60 bg-slate-800/40 text-slate-200'
  return <div className={`flex items-center gap-2 rounded-lg border px-3.5 py-2.5 text-sm ${cls}`}>{children}</div>
}

const Arrow = () => <div className="text-center text-slate-600">&darr;</div>

function ProblemVisual({ id }: { id: string }) {
  switch (id) {
    case 'good':
      return (
        <div className="card mx-auto w-full max-w-md p-7">
          <p className="eyebrow mb-5">Anti-UAV &middot; in-distribution</p>
          <div className="grid grid-cols-2 gap-5">
            <Metric label="Precision">
              <Counter to={0.989} decimals={3} />
            </Metric>
            <Metric label="Recall">
              <Counter to={0.982} decimals={3} />
            </Metric>
          </div>
          <div className="mt-6 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4">
            <div className="text-3xl font-extrabold text-emerald-300">
              <Counter to={2.8} decimals={1} suffix="%" />
            </div>
            <div className="mt-1 text-xs text-slate-400">
              hallucination on the in-distribution test set
            </div>
          </div>
        </div>
      )
    case 'confusers':
      return (
        <div className="card mx-auto w-full max-w-md p-8 text-center">
          <p className="eyebrow mb-5">Out-of-distribution confusers</p>
          <div className="text-6xl font-extrabold text-rose-400">
            <Counter to={30.4} decimals={1} suffix="%" />
          </div>
          <div className="mt-2 text-sm text-slate-400">
            of confuser frames trigger a false drone alert
          </div>
          <div className="mt-6 border-t border-slate-800 pt-5">
            <div className="text-3xl font-bold text-rose-300">
              <Counter to={94.4} decimals={1} suffix="%" />
            </div>
            <div className="mt-1 text-xs text-slate-500">on bird-only Svanstr&ouml;m footage</div>
          </div>
        </div>
      )
    case 'geometry':
      return (
        <div className="card mx-auto w-full max-w-md p-7">
          <p className="eyebrow mb-5">At surveillance range</p>
          <div className="rounded-xl bg-gradient-to-b from-sky-950/50 to-slate-900 p-8">
            <div className="flex items-end justify-around">
              {['Drone', 'Bird', 'Airplane', 'Heli'].map((l) => (
                <div key={l} className="flex flex-col items-center gap-4">
                  <span className="block h-2 w-2 rounded-[1px] bg-slate-300" />
                  <span className="text-[11px] text-slate-500">{l}</span>
                </div>
              ))}
            </div>
          </div>
          <p className="mt-4 text-xs leading-relaxed text-slate-500">
            All four collapse to the same small silhouette. The separating signal is not at the decision
            layer - it is deeper, in the detector&rsquo;s own features.
          </p>
        </div>
      )
    case 'retrain':
      return (
        <div className="card mx-auto w-full max-w-md p-7">
          <ComparisonChart
            title="Svanström drone recall"
            max={1}
            from={{ label: 'Baseline', value: 0.961, display: '0.961', tone: 'good', sublabel: 'production detector' }}
            to={{ label: 'Retrained', value: 0.306, display: '0.306', tone: 'bad', sublabel: 'bird-suppression' }}
            caption="Suppressing birds collapses small-drone recall. Recall lost at the detector cannot be recovered by any later stage."
          />
        </div>
      )
    case 'gaps':
      return (
        <div className="card mx-auto w-full max-w-md p-7">
          <GroupedBarChart
            title="The winner flips - yet Routed matches it"
            max={1}
            highlightGroupMax
            groups={[
              {
                label: 'Svanström (F1)',
                bars: [
                  { label: 'RGB', value: 0.607, display: '0.607', tone: 'muted' },
                  { label: 'IR', value: 0.94, display: '0.940', tone: 'good' },
                  { label: 'Routed', value: 0.946, display: '0.946', tone: 'accent' },
                ],
              },
              {
                label: 'Anti-UAV (F1)',
                bars: [
                  { label: 'RGB', value: 0.985, display: '0.985', tone: 'good' },
                  { label: 'IR', value: 0.961, display: '0.961', tone: 'muted' },
                  { label: 'Routed', value: 0.984, display: '0.984', tone: 'accent' },
                ],
              },
            ]}
            caption="IR wins Svanström, RGB wins Anti-UAV - yet the routed system matches the winner both times, without being told which modality to trust."
          />
        </div>
      )
    case 'move':
      return (
        <div className="card mx-auto w-full max-w-md p-7">
          <p className="eyebrow mb-5">The architectural move</p>
          <div className="space-y-2">
            <Chip>
              RGB&nbsp;+&nbsp;IR detectors <span className="text-slate-500">- tuned for recall</span>
            </Chip>
            <Arrow />
            <Chip tone="accent">
              Confuser filter <span className="text-slate-500">- re-reads the detector&rsquo;s features</span>
            </Chip>
            <Arrow />
            <Chip tone="accent">
              Trust router <span className="text-slate-500">- reliable modality per frame</span>
            </Chip>
            <Arrow />
            <Chip tone="accent">
              Temporal smoother <span className="text-slate-500">- decisions &rarr; alerts</span>
            </Chip>
            <Arrow />
            <Chip tone="alert">ALERT</Chip>
          </div>
        </div>
      )
    default:
      return null
  }
}

/* -------------------------------- section ---------------------------------- */

export function ProblemStatement() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <div className="mb-6">
        <p className="eyebrow mb-3">Act I - The problem</p>
        <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
          A narrow but stubborn failure
        </h2>
      </div>
      <Scrolly steps={steps} renderVisual={(id) => <ProblemVisual id={id} />} />
    </section>
  )
}
