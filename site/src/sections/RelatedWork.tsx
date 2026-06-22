import { useRef } from 'react'
import { useInViewOnce } from '../engine/useInViewOnce'
import { BarSet } from '../interactions/BarSet'

const LINEAGES = [
  {
    t: 'Real-time detection',
    d: 'YOLO26n via Ultralytics - chosen for throughput on modest hardware and small-object strength. Two-stage and transformer detectors were too costly for a back-to-back RGB + IR cascade.',
  },
  {
    t: 'The confuser problem',
    d: 'Bird confusion is the perennial hard case (the Drone-vs-Bird challenge). Training-time hard negatives never drive it to zero without a recall cost.',
  },
  {
    t: 'Thermal infrared',
    d: 'Motor and ESC hot-spots separate drones from uniform-temperature birds; single-channel IR shares low-level statistics with grayscale RGB - the seed of the grayscale finding.',
  },
  {
    t: 'Multi-modal fusion',
    d: 'Decision-level fusion with a learned per-frame trust classifier - no paired-modality training data required, unlike intermediate fusion.',
  },
  {
    t: 'Cross-modal transfer',
    d: 'Modality hallucination and RGB-to-thermal translation inject cross-modal information at training time; here the transfer is emergent, from shared low-level structure alone.',
  },
  {
    t: 'Cascaded rejection',
    d: 'A high-recall first stage, then progressively discriminative rejection (Viola-Jones, Cascade R-CNN) - but here the stages are heterogeneous and refine the decision, not the box.',
  },
  {
    t: 'Data-centric AI',
    d: 'Route model-vs-ground-truth disagreements to a human adjudicator. The IR detector’s gains are driven almost entirely by dataset operations, not model changes.',
  },
  {
    t: 'Probing representations',
    d: 'Linear probes and feature-space statistics, used prescriptively: where the separation already exists, train the filter directly on those same detector features.',
  },
]

export function RelatedWork() {
  return (
    <section className="border-t border-slate-900 bg-gradient-to-b from-slate-950 to-slate-900/30">
      <div className="mx-auto max-w-6xl px-6 py-28">
        <p className="eyebrow mb-3">Act II - Where this sits</p>
        <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
          A different division of labor
        </h2>
        <p className="mt-4 max-w-2xl text-slate-400">
          Prior work uses either a single detector or a tightly-coupled multi-modal one, and treats
          confuser discrimination as a training-time problem. This thesis keeps the detectors strong on
          both precision and recall, and adds a separate discrimination stage - run per frame, with
          the modality choice made per frame. To our knowledge, no published system reports a
          separately-evaluated out-of-distribution confuser fire rate, because none separates a
          confuser-rejection stage from detection.
        </p>

        <h3 className="mt-16 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          Eight lineages it builds on
        </h3>
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {LINEAGES.map((l, i) => (
            <LineageCard key={l.t} l={l} i={i} />
          ))}
        </div>

        <div className="mt-20 grid gap-10 lg:grid-cols-2 lg:items-center">
          <div>
            <h3 className="text-2xl font-bold tracking-tight text-white">Competitive, not a strawman</h3>
            <p className="mt-4 text-[15px] leading-relaxed text-slate-300">
              On the public Svanstr&ouml;m benchmark, this work&rsquo;s per-modality detectors exceed the
              dataset paper&rsquo;s published averages. The comparison is offered as <em>context, not
              head-to-head</em>: part of the gap is detector generation (YOLO26n vs YOLOv2), inference
              resolution, and matcher leniency (IoP vs IoU) - stated as caveats, not hidden.
            </p>
            <ul className="mt-6 space-y-2.5 text-sm text-slate-400">
              <li className="flex gap-2">
                <span className="text-cyan-500">&bull;</span>
                <span>
                  <span className="text-slate-200">Scoring rule</span> - IoP@0.5 is more lenient than
                  IoU@0.5 for small objects.
                </span>
              </li>
              <li className="flex gap-2">
                <span className="text-cyan-500">&bull;</span>
                <span>
                  <span className="text-slate-200">Dataset split</span> - the paper&rsquo;s eval clip
                  list was never published, so the full set is used.
                </span>
              </li>
              <li className="flex gap-2">
                <span className="text-cyan-500">&bull;</span>
                <span>
                  <span className="text-slate-200">Task</span> - the Anti-UAV baselines are trackers,
                  not per-frame detectors.
                </span>
              </li>
            </ul>
          </div>
          <div className="card space-y-6 p-7">
            <BarSet
              title="Svanström · visible F1 - this work vs published"
              bars={[
                { label: 'This work (baseline RGB @1280)', value: 0.95, display: '0.950', tone: 'accent' },
                { label: 'Svanström 2021 (YOLOv2 @416, avg)', value: 0.785, display: '0.785', tone: 'muted' },
              ]}
            />
            <BarSet
              title="Svanström · thermal F1 - this work vs published"
              bars={[
                { label: 'This work (IR v3b @640)', value: 0.961, display: '0.961', tone: 'accent' },
                { label: 'Svanström 2021 (YOLOv2 @256, avg)', value: 0.76, display: '0.760', tone: 'muted' },
              ]}
              caption="Context, not a like-for-like result - read with the three caveats at left."
            />
          </div>
        </div>
      </div>
    </section>
  )
}

function LineageCard({ l, i }: { l: { t: string; d: string }; i: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const seen = useInViewOnce(ref, '0px 0px -8% 0px')
  return (
    <div
      ref={ref}
      className={`card p-5 transition-all duration-700 ease-out ${
        seen ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
      }`}
      style={{ transitionDelay: `${(i % 4) * 90}ms` }}
    >
      <h4 className="text-sm font-semibold text-cyan-300">{l.t}</h4>
      <p className="mt-2 text-[13px] leading-relaxed text-slate-400">{l.d}</p>
    </div>
  )
}
