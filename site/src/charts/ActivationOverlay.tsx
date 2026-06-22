import { useRef, type ReactNode } from 'react'
import { useInViewOnce } from '../engine/useInViewOnce'

/* ===========================================================================
 * ActivationOverlay - the real "brain scan" panel of the Model MRI story.
 *
 * Two real, matched neuron-activation montages side by side: a DRONE the
 * detector fired on (kept) vs a CONFUSER it fired on (a false positive). Each
 * montage is a single image with three panes - Detection crop | P3 (stride 8,
 * fine spatial detail) | P5 (stride 32, semantic depth). Reading them together
 * is the punchline: on the drone the P5 semantic map locks onto one coherent
 * object; on the confuser the activation is diffuse and fragmented - the same
 * signal the LDA axis separates and the trained verifier reads.
 *
 * The images are REAL figures from the thesis (fig8_mri_act_*), copied into
 * /media/mri/. This component only lays them out, labels the three panes, and
 * carries the explanatory caption; it is fully props-driven with in-repo
 * defaults, matching the ExampleGallery idiom (card / Reveal / figure).
 * ========================================================================= */

export interface ActivationItem {
  /** Path under /public, e.g. '/media/mri/activation_drone.png'. */
  src: string
  /** Required alt text. */
  alt: string
  /** 'kept' (drone, emerald) or 'fired' (confuser false positive, rose). */
  verdict: 'kept' | 'fired'
  /** Short status pill, e.g. 'Drone - KEPT'. */
  badge: string
  /** Bold one-line headline under the montage. */
  title: string
  /** Supporting caption explaining what the activation shows. */
  caption: ReactNode
}

export interface ActivationOverlayData {
  eyebrow?: string
  heading?: string
  lead?: ReactNode
  drone: ActivationItem
  confuser: ActivationItem
  /** Pane legend shown once under the pair (Detection / P3 / P5). */
  panes?: { label: string; desc: string }[]
  /** Optional closing note tying the scan to the rest of the MRI story. */
  footnote?: ReactNode
}

export interface ActivationOverlayProps {
  data?: ActivationOverlayData
  className?: string
}

/* SOURCES (copied verbatim into site/public/media/mri/):
 *  activation_drone.png    ← docs/thesis_working_distilling_overleaf/figures/
 *      fig8_mri_act_drone.png    (RGB_images - DRONE, conf 0.72; three panes:
 *      Detection crop | P3 stride-8 | P5 stride-32, top-discriminative neuron)
 *  activation_confuser.png ← docs/thesis_working_distilling_overleaf/figures/
 *      fig8_mri_act_confuser.png (images_test - CONFUSER, conf 0.29; same panes)
 *  (alternates available: docs/analysis/images/v5_activation_{drone,confuser}_example.png)
 */
export const ACTIVATION_OVERLAY_DATA: ActivationOverlayData = {
  eyebrow: 'Model MRI · the brain scan',
  heading: 'What the detector sees, neuron by neuron',
  lead: (
    <>
      The confuser filter never looks at pixels - it reads the detector's <span className="text-white">own internal
      feature maps</span>. Here are two real ones: a drone the detector fired on, and a look-alike it also fired on.
      On the drone the deep <span className="accent">P5</span> map locks onto one coherent object; on the confuser the
      same neurons light up in a scattered, fragmented way. That difference is exactly the signal the MRI measures.
    </>
  ),
  drone: {
    src: '/media/mri/activation_drone.png',
    alt: 'Drone detection montage: left, a quadcopter against sky inside a green box (conf 0.72); middle, P3 stride-8 activation with detail concentrated on the airframe; right, P5 stride-32 semantic activation forming one tight blob locked onto the drone.',
    verdict: 'kept',
    badge: 'Drone - KEPT',
    title: 'A clean semantic lock',
    caption: (
      <>
        Detector confidence 0.72. The <span className="accent">P5</span> map (right) collapses to a single coherent
        region centred on the airframe - a confident "one object here." The verifier reads this and keeps it.
      </>
    ),
  },
  confuser: {
    src: '/media/mri/activation_confuser.png',
    alt: 'Confuser detection montage: left, a small dark bird-like blob against sky inside a green box (conf 0.29); middle, P3 stride-8 activation scattered across the frame; right, P5 stride-32 semantic activation fragmented with no single coherent object.',
    verdict: 'fired',
    badge: 'Confuser - FIRED',
    title: 'Diffuse, fragmented activation',
    caption: (
      <>
        Detector confidence 0.29 - it still fired. But the <span className="accent">P5</span> map (right) is
        smeared and broken across the frame, never settling on one object. Same neurons, no semantic lock - the
        cue the trained filter uses to veto it.
      </>
    ),
  },
  panes: [
    { label: 'Detection', desc: 'the crop the detector fired on (green box)' },
    { label: 'P3 · stride 8', desc: 'fine spatial detail - edges and texture' },
    { label: 'P5 · stride 32', desc: 'semantic depth - "is this one object?"' },
  ],
  footnote: (
    <>
      These maps are the raw material for the whole filter: flattened, they become the p3+p5 feature vector that the
      LDA axis splits at 0.952 separability and the distilled verifier scores per detection.
    </>
  ),
}

const VERDICT_TONE: Record<ActivationItem['verdict'], { ring: string; pill: string; dot: string }> = {
  kept: {
    ring: 'border-emerald-500/40 shadow-emerald-500/10',
    pill: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
    dot: 'bg-emerald-400',
  },
  fired: {
    ring: 'border-rose-500/40 shadow-rose-500/10',
    pill: 'border-rose-500/40 bg-rose-500/10 text-rose-200',
    dot: 'bg-rose-400',
  },
}

/** Fades + lifts children in the first time they enter view (matches ExampleGallery). */
function Reveal({
  children,
  delay = 0,
  className = '',
}: {
  children: ReactNode
  delay?: number
  className?: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const seen = useInViewOnce(ref, '0px 0px -8% 0px')
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${
        seen ? 'translate-y-0 opacity-100' : 'translate-y-4 opacity-0'
      } ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  )
}

function ActivationCard({ item, delay }: { item: ActivationItem; delay: number }) {
  const t = VERDICT_TONE[item.verdict]
  return (
    <Reveal delay={delay} className="h-full">
      <figure className="card flex h-full flex-col overflow-hidden shadow-lg">
        <div className="relative">
          <img
            src={item.src}
            alt={item.alt}
            loading="lazy"
            className="max-h-80 w-full bg-slate-950 object-contain"
          />
          <span
            className={`absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider backdrop-blur-sm ${t.pill}`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${t.dot}`} />
            {item.badge}
          </span>
        </div>
        <figcaption className="flex flex-1 flex-col p-5">
          <h4 className="text-base font-semibold text-white">{item.title}</h4>
          <p className="mt-2 text-[13px] leading-relaxed text-slate-400">{item.caption}</p>
        </figcaption>
      </figure>
    </Reveal>
  )
}

/**
 * Real neuron-activation overlays: a drone (kept) vs a confuser (fired) montage,
 * each Detection | P3 | P5, with a shared pane legend and explanatory copy.
 * Props-driven with in-repo defaults; pass `data` to swap content.
 */
export function ActivationOverlay({ data, className = '' }: ActivationOverlayProps) {
  const d = data ?? ACTIVATION_OVERLAY_DATA
  return (
    <div className={className}>
      {(d.eyebrow || d.heading || d.lead) && (
        <Reveal>
          {d.eyebrow && <p className="eyebrow mb-3">{d.eyebrow}</p>}
          {d.heading && (
            <h3 className="max-w-3xl text-2xl font-bold tracking-tight text-white sm:text-3xl">{d.heading}</h3>
          )}
          {d.lead && <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-slate-300">{d.lead}</p>}
        </Reveal>
      )}

      <div className="mt-8 grid items-stretch gap-5 md:grid-cols-2">
        <ActivationCard item={d.drone} delay={0} />
        <ActivationCard item={d.confuser} delay={110} />
      </div>

      {d.panes && d.panes.length > 0 && (
        <Reveal delay={120}>
          <div className="mt-6 grid gap-3 sm:grid-cols-3">
            {d.panes.map((p, i) => (
              <div key={p.label} className="card flex items-start gap-3 p-3.5">
                <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-cyan-400/40 bg-cyan-400/10 text-[11px] font-bold tabular-nums text-cyan-200">
                  {i + 1}
                </span>
                <div>
                  <div className="text-[13px] font-semibold text-white">{p.label}</div>
                  <div className="mt-0.5 text-[12px] leading-snug text-slate-400">{p.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </Reveal>
      )}

      {d.footnote && (
        <Reveal delay={160}>
          <p className="mt-6 max-w-3xl text-xs leading-relaxed text-slate-500">{d.footnote}</p>
        </Reveal>
      )}
    </div>
  )
}

export default ActivationOverlay
