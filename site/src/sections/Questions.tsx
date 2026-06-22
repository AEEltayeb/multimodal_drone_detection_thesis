import { useRef } from 'react'
import { useInViewOnce } from '../engine/useInViewOnce'

const RQS = [
  {
    tag: 'RQ1',
    name: 'Capability',
    body: 'Where retraining fails, can a layered pipeline maintain or improve drone detection while suppressing false positives - on both ordinary and confuser-rich scenes?',
  },
  {
    tag: 'RQ2',
    name: 'Attribution',
    body: 'What does each stage contribute? Measured by ablation - the trust router, the confuser filter, their composition order, and temporal aggregation - each isolated, with the runtime cost of each accounted for.',
  },
  {
    tag: 'RQ3',
    name: 'Dual-modality value',
    body: 'Does per-frame routing between two modalities beat either modality alone?',
  },
]

/** Act 0.2 - the three research questions, stated as the contract the work answers. */
export function Questions() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-28">
      <p className="eyebrow mb-3">The contract</p>
      <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
        Three research questions
      </h2>
      <p className="mt-4 max-w-2xl text-slate-400">
        Modern detectors are already strong. So the work is framed as three questions - each
        answered by ablation in the chapters that follow.
      </p>
      <div className="mt-12 grid gap-5 md:grid-cols-3">
        {RQS.map((q, i) => (
          <RQCard key={q.tag} q={q} i={i} />
        ))}
      </div>
    </section>
  )
}

function RQCard({ q, i }: { q: (typeof RQS)[number]; i: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const seen = useInViewOnce(ref, '0px 0px -10% 0px')
  return (
    <div
      ref={ref}
      className={`card p-6 transition-all duration-700 ease-out ${
        seen ? 'translate-y-0 opacity-100' : 'translate-y-6 opacity-0'
      }`}
      style={{ transitionDelay: `${i * 120}ms` }}
    >
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-extrabold text-cyan-400">{q.tag}</span>
        <span className="text-sm font-medium text-slate-400">{q.name}</span>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-slate-300">{q.body}</p>
    </div>
  )
}
