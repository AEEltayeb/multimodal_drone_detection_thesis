import { useRef, type ReactNode } from 'react'
import { useInViewOnce } from '../engine/useInViewOnce'

/* ===========================================================================
 * ExampleGallery - real annotated image evidence for the scrollytelling page.
 *
 * Two self-contained blocks, both data-driven via props:
 *   (A) "It's the HARD birds, not all birds" - a labeled before/after pair.
 *       LEFT  : small/distant confusers the RGB detector FIRED on (a false
 *               positive against open sky, indistinguishable from a drone).
 *       RIGHT : ordinary, close-range birds - unmistakably birds; not fooled.
 *   (B) "The whole pipeline, on real frames" - a captioned gallery of frames
 *       the pipeline KEPT (drones, emerald) vs SUPPRESSED (confusers, rose).
 *
 * All imagery lives under /media/examples/ (Vite serves /public at the root).
 * The defaults below wire the gallery up out of the box; pass `contrast` /
 * `gallery` (or the whole `data` object) to swap or extend without editing JSX.
 * =========================================================================== */

/* ------------------------------- data model -------------------------------- */

/** One side of the (A) before/after contrast. */
export interface ContrastPanel {
  /** Path under /public, e.g. '/media/examples/A_left_hard_confusers_fired.png'. */
  src: string
  /** Required alt text for the image. */
  alt: string
  /** Short status pill, e.g. 'Detector FIRES' / 'No fire'. */
  badge: string
  /** Outcome tone - drives the border / pill / glow color. */
  tone: 'fire' | 'safe'
  /** Bold one-line headline under the frame. */
  title: string
  /** Supporting caption. */
  caption: string
}

/** The (A) "hard birds, not all birds" contrast pair. */
export interface ContrastData {
  eyebrow: string
  heading: string
  /** Optional lead-in paragraph above the pair. */
  lead?: ReactNode
  left: ContrastPanel
  right: ContrastPanel
}

/** One frame in the (B) detected-vs-suppressed gallery. */
export interface GalleryItem {
  src: string
  alt: string
  /** 'kept'  → drone correctly detected (emerald).
   *  'vetoed'→ confuser suppressed by the filter (rose). */
  verdict: 'kept' | 'vetoed'
  /** Short label, e.g. 'Drone - detected'. */
  label: string
  caption: string
  /** When true the tile spans both columns (good for wide montages). */
  wide?: boolean
}

/** The (B) full-pipeline gallery. */
export interface GalleryData {
  eyebrow: string
  heading: string
  lead?: ReactNode
  items: GalleryItem[]
}

export interface ExampleGalleryData {
  contrast: ContrastData
  gallery: GalleryData
}

export interface ExampleGalleryProps {
  /** Override the whole dataset. */
  data?: ExampleGalleryData
  /** Override just block (A). */
  contrast?: ContrastData
  /** Override just block (B). */
  gallery?: GalleryData
  /** Optional extra classes on the outer <section>. */
  className?: string
}

/* ------------------------- default (in-repo) content ----------------------- */
/* SOURCES (copied verbatim into site/public/media/examples/):
 *  A_left_hard_confusers_fired.png ← docs/thesis_working_distilling_overleaf/
 *      figures/fig_confuser_fp_examples.png  (FT4 false positives on the OOD
 *      confuser corpus; red detector boxes + per-box filter P(drone); the two
 *      bird cells are the small/distant "hard" birds at surveillance range).
 *  A_right_ordinary_birds.jpg ← datasets/confuser_videos/
 *      birds_birds_flying_overhead_various_sizes_short/images/test/frame_00010.jpg
 *  B_drone_detected_field.jpg ← training/demo_outputs/saved_alert_windows/
 *      gopro_006_20260201_124524_ea8aec79/frame_000444.jpg   (green box)
 *  B_drone_detected_sky.jpg ← training/demo_outputs/saved_alert_windows/
 *      GOPR5844_002_20260201_014951_fc322328/frame_000072.jpg (green box)
 *  B_verifier_veto_vs_keep.png ← docs/thesis_working_distilling_overleaf/
 *      figures/fig_confuser_panel.png  (bird P(drone)=0.00 VETO | drone 0.96 KEEP)
 *  B_pipeline_montage_dronevideo.png ← docs/analysis/full_pipeline_ablations/
 *      plots/drone_video_examples.png  (legend: GT / raw detector / after gate / dropped)
 */
export const EXAMPLE_GALLERY_DATA: ExampleGalleryData = {
  contrast: {
    eyebrow: 'The scope · in pictures',
    heading: "It's the hard birds - not all birds",
    lead: (
      <>
        The <span className="text-rose-300">94.4%</span> bird-fire rate is measured on{' '}
        <span className="text-white">small, distant birds against open sky</span>, where at
        surveillance range a bird and a drone collapse to the same few-pixel silhouette. On ordinary,
        close-range bird footage the detector is far harder to fool. Same detector, two regimes.
      </>
    ),
    left: {
      src: '/media/examples/A_left_hard_confusers_fired.png',
      alt: 'Six surveillance-range confusers (airplane, bird, helicopter) the RGB detector fired on - each marked with a red detection box and the confuser filter’s P(drone) score, all below 0.25.',
      badge: 'Detector FIRES',
      tone: 'fire',
      title: 'A few-pixel silhouette reads as a drone',
      caption:
        'Real false positives on the out-of-distribution confuser set. The detector fires (red box, conf up to 0.86) on tiny birds and aircraft it cannot separate from a drone at range - the filter’s P(drone) (≤ 0.077) is what later vetoes them.',
    },
    right: {
      src: '/media/examples/A_right_ordinary_birds.jpg',
      alt: 'Several large, close-range birds in flight against an overcast sky, clearly recognizable as birds by their wing shape and body.',
      badge: 'No fire',
      tone: 'safe',
      title: 'Up close, a bird is obviously a bird',
      caption:
        'Ordinary confuser footage: at this range wings, body and motion are unmistakable, and the detector is not fooled. The problem was never "birds" - it is the hard, small-and-distant slice.',
    },
  },
  gallery: {
    eyebrow: 'The pipeline · on real frames',
    heading: 'Drones kept, confusers vetoed',
    lead: (
      <>
        The same per-detection filter that produces the 94.4% scope figure runs on every frame:
        drones are <span className="text-emerald-300">detected</span>, look-alikes are{' '}
        <span className="text-rose-300">suppressed</span>. These are real frames from the system, not
        illustrations.
      </>
    ),
    items: [
      {
        src: '/media/examples/B_verifier_veto_vs_keep.png',
        alt: 'Side-by-side: left, a bird the detector fired on (conf 0.46) with verifier P(drone)=0.00 marked VETO; right, a drone (conf 0.85) with verifier P(drone)=0.96 marked KEEP.',
        verdict: 'kept',
        label: 'One filter, both calls',
        caption:
          'Bird: detector fires at conf 0.46, the verifier returns P(drone)=0.00 and VETOes it. Drone: conf 0.85, P(drone)=0.96, KEPT. The discrimination the detection head cannot make is made one stage later.',
        wide: true,
      },
      {
        src: '/media/examples/prod_B_drone_detected_field.jpg',
        alt: 'A quadcopter flying low over a wheat field, enclosed by a green production detection box labeled P(drone)=0.99.',
        verdict: 'kept',
        label: 'Drone - detected & kept',
        caption:
          'Real production output: the ft4 detector fires (conf 0.87) and the per-frame mlp_v5_v4 filter scores P(drone)=0.99 - kept.',
      },
      {
        src: '/media/examples/A_left_hard_confusers_fired.png',
        alt: 'Grid of six confusers (airplane, bird, helicopter) the detector fired on, each with a red box and the filter’s P(drone) below 0.25, all suppressed.',
        verdict: 'vetoed',
        label: 'Confusers - suppressed',
        caption:
          'Every detection here cleared the detector but failed the filter: P(drone) ranges 0.001–0.077, so each red box is dropped before it can raise an alert.',
        wide: true,
      },
    ],
  },
}

/* ------------------------------ tone tokens -------------------------------- */

const PANEL_TONE: Record<ContrastPanel['tone'], { ring: string; pill: string; dot: string }> = {
  fire: {
    ring: 'border-rose-500/40 shadow-rose-500/10',
    pill: 'border-rose-500/40 bg-rose-500/10 text-rose-200',
    dot: 'bg-rose-400',
  },
  safe: {
    ring: 'border-emerald-500/40 shadow-emerald-500/10',
    pill: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
    dot: 'bg-emerald-400',
  },
}

const VERDICT_TONE: Record<GalleryItem['verdict'], { ring: string; pill: string; label: string }> = {
  kept: {
    ring: 'border-emerald-500/30',
    pill: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
    label: 'KEPT',
  },
  vetoed: {
    ring: 'border-rose-500/30',
    pill: 'border-rose-500/40 bg-rose-500/10 text-rose-200',
    label: 'VETOED',
  },
}

/* --------------------------- reveal sub-component --------------------------- */

/** Wraps children and fades + lifts them in the first time they enter view. */
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

/* ------------------------------ (A) contrast ------------------------------- */

function ContrastCard({ panel, delay }: { panel: ContrastPanel; delay: number }) {
  const t = PANEL_TONE[panel.tone]
  return (
    <Reveal delay={delay} className="h-full">
      <figure className="card flex h-full flex-col overflow-hidden shadow-lg">
        <div className="relative">
          <img
            src={panel.src}
            alt={panel.alt}
            loading="lazy"
            className="aspect-video w-full bg-slate-950 object-contain"
          />
          <span
            className={`absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider backdrop-blur-sm ${t.pill}`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${t.dot}`} />
            {panel.badge}
          </span>
        </div>
        <figcaption className="flex flex-1 flex-col p-5">
          <h4 className="text-base font-semibold text-white">{panel.title}</h4>
          <p className="mt-2 text-[13px] leading-relaxed text-slate-400">{panel.caption}</p>
        </figcaption>
      </figure>
    </Reveal>
  )
}

function ContrastBlock({ data }: { data: ContrastData }) {
  return (
    <div>
      <Reveal>
        <p className="eyebrow mb-3">{data.eyebrow}</p>
        <h3 className="max-w-3xl text-2xl font-bold tracking-tight text-white sm:text-3xl">
          {data.heading}
        </h3>
        {data.lead && (
          <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-slate-300">{data.lead}</p>
        )}
      </Reveal>
      <div className="mt-8 grid items-stretch gap-5 md:grid-cols-2">
        <ContrastCard panel={data.left} delay={0} />
        <ContrastCard panel={data.right} delay={110} />
      </div>
    </div>
  )
}

/* ------------------------------- (B) gallery ------------------------------- */

function GalleryCard({ item, delay }: { item: GalleryItem; delay: number }) {
  const t = VERDICT_TONE[item.verdict]
  return (
    <Reveal delay={delay} className="h-full">
      <figure className="card flex h-full flex-col overflow-hidden">
        <div className="relative">
          <img
            src={item.src}
            alt={item.alt}
            loading="lazy"
            className="aspect-video w-full bg-slate-950 object-contain"
          />
          <span
            className={`absolute right-3 top-3 rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider backdrop-blur-sm ${t.pill}`}
          >
            {t.label}
          </span>
        </div>
        <figcaption className="flex flex-1 flex-col p-5">
          <h4 className="text-sm font-semibold text-white">{item.label}</h4>
          <p className="mt-1.5 text-[13px] leading-relaxed text-slate-400">{item.caption}</p>
        </figcaption>
      </figure>
    </Reveal>
  )
}

function GalleryBlock({ data }: { data: GalleryData }) {
  return (
    <div className="mt-24">
      <Reveal>
        <p className="eyebrow mb-3">{data.eyebrow}</p>
        <h3 className="max-w-3xl text-2xl font-bold tracking-tight text-white sm:text-3xl">
          {data.heading}
        </h3>
        {data.lead && (
          <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-slate-300">{data.lead}</p>
        )}
      </Reveal>
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {data.items.map((item, i) => (
          <GalleryCard key={item.src + i} item={item} delay={(i % 3) * 80} />
        ))}
      </div>
      <Reveal delay={120}>
        <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-slate-500">
          <span className="inline-flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-[3px] bg-emerald-400" /> kept - drone detected
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="h-2.5 w-2.5 rounded-[3px] bg-rose-400" /> vetoed - confuser suppressed by
            the filter
          </span>
        </div>
      </Reveal>
    </div>
  )
}

/* -------------------------------- section ---------------------------------- */

/**
 * Real annotated image evidence: (A) the "hard birds, not all birds" before/after
 * pair, and (B) the detected-vs-suppressed pipeline gallery. Self-contained with
 * in-repo defaults; override any block via props.
 */
export function ExampleGallery({ data, contrast, gallery, className = '' }: ExampleGalleryProps) {
  const resolved = data ?? EXAMPLE_GALLERY_DATA
  const contrastData = contrast ?? resolved.contrast
  const galleryData = gallery ?? resolved.gallery
  return (
    <section className={`mx-auto max-w-6xl px-6 py-20 ${className}`}>
      <ContrastBlock data={contrastData} />
      <GalleryBlock data={galleryData} />
    </section>
  )
}

export default ExampleGallery
