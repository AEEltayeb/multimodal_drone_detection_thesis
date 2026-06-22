import { useState, type ReactNode } from 'react'
import { Scrolly, type ScrollyStep } from '../engine/Scrolly'
import { MetricBar } from '../charts/MetricBar'
import { useChartReveal } from '../charts/useChartReveal'
import { ChartTooltip, type TooltipState } from '../charts/ChartTooltip'
import { palette, toneHex, toneHexLight, clamp, nextUid } from '../charts/chartTheme'

/* --------------------------- HITL trajectory chart ------------------------- */

type Rev = { v: string; f1: number; p: number; r: number; regress?: boolean; prod?: boolean }

const HITL: Rev[] = [
  { v: 'V2', f1: 0.503, p: 0.661, r: 0.406 },
  { v: 'V3', f1: 0.611, p: 0.648, r: 0.579 },
  { v: 'V4', f1: 0.765, p: 0.895, r: 0.669 },
  { v: 'V5', f1: 0.737, p: 0.768, r: 0.709, regress: true },
  { v: 'V6', f1: 0.931, p: 0.921, r: 0.941 },
  { v: 'Final', f1: 0.967, p: 0.955, r: 0.98 },
  { v: 'v3b', f1: 0.967, p: 0.957, r: 0.977, prod: true },
]

const VB_W = 520
const VB_H = 300
const PAD = { top: 26, right: 22, bottom: 40, left: 38 }
const PLOT_W = VB_W - PAD.left - PAD.right
const PLOT_H = VB_H - PAD.top - PAD.bottom
const Y_MIN = 0.4
const Y_MAX = 1.0

function HitlTrajectory() {
  const { ref, progress, seen } = useChartReveal<HTMLDivElement>(1500)
  const [uid] = useState(nextUid)
  const [active, setActive] = useState<number | null>(null)
  void uid

  const n = HITL.length
  const xOf = (i: number) => PAD.left + (i / (n - 1)) * PLOT_W
  const yOf = (f: number) => PAD.top + (1 - (clamp(f, Y_MIN, Y_MAX) - Y_MIN) / (Y_MAX - Y_MIN)) * PLOT_H
  const pts = HITL.map((d, i) => ({ x: xOf(i), y: yOf(d.f1), d, i }))
  const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ')
  let len = 0
  for (let i = 1; i < pts.length; i += 1) len += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)

  const ticks = [0.4, 0.6, 0.8, 1.0]

  return (
    <div ref={ref} className="card mx-auto w-full max-w-lg p-6">
      <div className="mb-3 text-sm font-semibold text-slate-300">
        IR detector F1 - six revisions, one fixed test split
      </div>
      <div className="relative">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          width="100%"
          role="img"
          aria-label="IR detector F1 across six HITL revisions, 0.503 to 0.967, with a dip at V5 where review was bypassed."
          className="block select-none overflow-visible"
        >
          {ticks.map((t) => (
            <g key={t} style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}>
              <line x1={PAD.left} x2={PAD.left + PLOT_W} y1={yOf(t)} y2={yOf(t)} stroke={palette.grid} strokeWidth={1} />
              <text x={PAD.left - 8} y={yOf(t)} textAnchor="end" dominantBaseline="central" fontSize="10" fill={palette.textFaint} className="tabular-nums">
                {t.toFixed(1)}
              </text>
            </g>
          ))}

          <path
            d={pathD}
            fill="none"
            stroke={toneHex.accent}
            strokeWidth={2.25}
            strokeLinejoin="round"
            strokeLinecap="round"
            strokeDasharray={len}
            strokeDashoffset={len * (1 - progress)}
          />

          {pts.map((p) => {
            const appear = clamp((progress - p.i / n) * 3, 0, 1)
            const isReg = p.d.regress
            const isProd = p.d.prod
            const col = isReg ? toneHex.bad : isProd ? toneHexLight.accent : toneHex.accent
            const isActive = active === p.i
            return (
              <g
                key={p.d.v}
                tabIndex={0}
                role="img"
                aria-label={`${p.d.v}: F1 ${p.d.f1.toFixed(3)}, precision ${p.d.p.toFixed(3)}, recall ${p.d.r.toFixed(3)}`}
                onMouseEnter={() => setActive(p.i)}
                onMouseLeave={() => setActive(null)}
                onFocus={() => setActive(p.i)}
                onBlur={() => setActive(null)}
                style={{ outline: 'none', opacity: appear }}
              >
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={isActive ? 6 : isReg || isProd ? 5 : 3.5}
                  fill={col}
                  stroke={palette.gridFaint}
                  strokeWidth={1.5}
                  style={{ transition: 'r 120ms ease-out' }}
                />
                <text
                  x={p.x}
                  y={VB_H - 16}
                  textAnchor="middle"
                  fontSize="10.5"
                  fontWeight={isProd ? 700 : 500}
                  fill={isReg ? toneHexLight.bad : isProd ? toneHexLight.accent : palette.textDim}
                >
                  {p.d.v}
                </text>
              </g>
            )
          })}

          {(() => {
            const v5 = pts.find((p) => p.d.regress)
            if (!v5) return null
            return (
              <text
                x={v5.x}
                y={v5.y - 14}
                textAnchor="middle"
                fontSize="9.5"
                fontWeight={600}
                fill={toneHexLight.bad}
                style={{ opacity: clamp((progress - 0.55) / 0.45, 0, 1) }}
              >
                review bypassed
              </text>
            )
          })()}
        </svg>
        <ChartTooltip state={tipFor(active, pts)} />
      </div>
      <p className="mt-4 text-xs leading-relaxed text-slate-500">
        0.503 → 0.967 across six revisions, driven almost entirely by dataset work. The V5 dip is the one
        batch ingested without review - precision 0.895 → 0.768.
      </p>
    </div>
  )
}

function tipFor(active: number | null, pts: Array<{ x: number; y: number; d: Rev }>): TooltipState {
  if (active == null) return null
  const p = pts[active]
  if (!p) return null
  return {
    xFrac: p.x / VB_W,
    yFrac: p.y / VB_H,
    series: p.d.v,
    tone: p.d.regress ? 'bad' : 'accent',
    label: `P ${p.d.p.toFixed(3)} · R ${p.d.r.toFixed(3)}`,
    value: `F1 ${p.d.f1.toFixed(3)}`,
  }
}

/* ------------------------------ other visuals ------------------------------ */

function MiniStat({ v, l }: { v: string; l: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2.5">
      <div className="text-base font-bold tabular-nums text-white">{v}</div>
      <div className="mt-0.5 text-[11px] leading-snug text-slate-500">{l}</div>
    </div>
  )
}

function CarveOutCard() {
  return (
    <div className="card mx-auto w-full max-w-md p-7">
      <MetricBar
        title="rgb_dataset test F1 - the carve-out, closed"
        max={1}
        decimals={3}
        items={[
          { label: 'earlier filter - over-vetoed small drones', value: 0.792, display: '0.792', tone: 'bad' },
          { label: 'production - size + bird re-mine', value: 0.916, display: '0.916', tone: 'accent' },
        ]}
      />
      <div className="mt-5 grid grid-cols-2 gap-3">
        <MiniStat v="91 → 30" l="held-out unseen-bird fires (of 230)" />
        <MiniStat v="48 → 6" l="thermal-native filter FP (CBAM)" />
      </div>
    </div>
  )
}

function GrayscaleCard() {
  return (
    <div className="card mx-auto w-full max-w-md p-7">
      <MetricBar
        title="Svanström F1 - the thermal model, three inputs"
        max={1}
        decimals={3}
        items={[
          { label: 'RGB detector (native)', value: 0.607, display: '0.607', tone: 'muted' },
          { label: 'IR on raw RGB', value: 0.187, display: '0.187', tone: 'bad' },
          { label: 'IR on grayscale RGB', value: 0.58, display: '0.580', tone: 'accent' },
        ]}
        caption="Raw RGB collapses the thermal model (0.187); a one-line grayscale conversion recovers it to 0.580 - within 2.7 pp of the dedicated RGB detector, with no retraining."
      />
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
    id: 'hitl',
    node: (
      <Prose kicker="The proof · 1" title="A data engine, not a model tweak">
        <p>
          The thermal detector had almost no public data, so it was built with its own dataset, in a
          loop: train, find where the model disagrees with the labels, review each disagreement by hand,
          feed the corrections back. Across six revisions its F1 went{' '}
          <span className="accent">0.503 &rarr; 0.967</span>.
        </p>
        <p>
          One revision proves the discipline by breaking it. In <span className="text-rose-300">V5</span>{' '}
          a large batch was ingested <em>without</em> review; it carried unlabeled drones, which trained
          as confuser-class negatives. Precision dropped <span className="text-white">0.895 &rarr; 0.768</span>.
          The model didn&rsquo;t get worse - the dataset state did, and the same review loop is what
          found and fixed it.
        </p>
      </Prose>
    ),
  },
  {
    id: 'carveout',
    node: (
      <Prose kicker="The proof · 2" title="The filter, diagnosed then closed">
        <p>
          The confuser filter once carried a carve-out: on the in-domain RGB test split it over-vetoed
          real drones, costing 11 points of F1. The Model MRI localized why - the falsely-vetoed
          drones weren&rsquo;t low-confidence, they were <span className="text-white">smaller</span>,
          sitting in a corner of feature space the filter&rsquo;s training drones never covered. A
          coverage gap, not a structural overlap.
        </p>
        <p>
          The diagnosis prescribed the fix: re-mine the training drones with a size&times;source balance,
          plus an in-domain bird split. Recall recovers <span className="accent">0.691 &rarr; 0.887</span>,
          F1 <span className="accent">0.792 &rarr; 0.916</span> - and on held-out <em>unseen</em>{' '}
          birds the production filter beats its predecessor (30 of 230 fires, vs 91).
        </p>
      </Prose>
    ),
  },
  {
    id: 'grayscale',
    node: (
      <Prose kicker="The proof · 3" title="An accident worth keeping">
        <p>
          The thermal detector was trained only on thermal. Feed it grayscale-converted visible-light
          video - a one-line conversion, no retraining - and it still detects drones:{' '}
          <span className="accent">F1 0.580</span>, within 2.7 points of the dedicated RGB detector
          (0.607) on the same frames.
        </p>
        <p>
          A control settles the mechanism: fed the raw three-channel RGB, the same model collapses to{' '}
          <span className="text-rose-300">0.187</span>. The single-channel conversion is the load-bearing
          step. We are not aware of this thermal-to-grayscale transfer being reported before - and
          it gives the system a usable RGB-only fallback when no thermal camera is present.
        </p>
      </Prose>
    ),
  },
]

function ProofVisual({ id }: { id: string }) {
  switch (id) {
    case 'hitl':
      return <HitlTrajectory />
    case 'carveout':
      return <CarveOutCard />
    case 'grayscale':
      return <GrayscaleCard />
    default:
      return null
  }
}

/* -------------------------------- section ---------------------------------- */

export function Proof() {
  return (
    <section className="border-t border-slate-900 bg-gradient-to-b from-slate-950 to-slate-900/30">
      <div className="mx-auto max-w-6xl px-6 py-20">
        <div className="mb-6">
          <p className="eyebrow mb-3">Act VI - The proof</p>
          <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Each component, on its own job
          </h2>
          <p className="mt-4 max-w-2xl text-slate-400">
            The ablation showed the stages add up. Here is the evidence behind three of them - the
            data engine that built the thermal detector, the filter that had to be diagnosed and fixed,
            and an accident that became a feature.
          </p>
        </div>
        <Scrolly steps={steps} renderVisual={(id) => <ProofVisual id={id} />} />
      </div>
    </section>
  )
}
