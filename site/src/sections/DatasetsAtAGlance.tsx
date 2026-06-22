import { useState, type ReactNode } from 'react'
import { DonutChart } from '../charts'
import { clamp, easeOutCubic, palette, toneHex, toneHexLight } from '../charts/chartTheme'
import { useChartReveal } from '../charts/useChartReveal'
import { ChartTooltip, type TooltipState } from '../charts/ChartTooltip'

/* ============================================================================
 * Datasets at a glance
 * ----------------------------------------------------------------------------
 * Every number here is verbatim from the thesis tables
 * (docs/thesis_working_distilling_overleaf/chapters/methodology.tex + appendices)
 * and the dataset YAMLs. Drone "size" is median sqrt(box area) in native pixels;
 * the ~8 px line is the YOLO26n imgsz=640 resolvable floor. For the all-drone
 * surfaces (Anti-UAV, DUT-Anti-UAV) we state plainly that there are NO confusers.
 * Confuser SIZE has no clean in-repo statistic, so we show composition only and
 * note that at range confuser silhouettes mirror drone size - we do not invent a
 * pixel figure.
 * ========================================================================== */

/** The resolvable floor: sqrt(area) below which an imgsz=640 detector can't see. */
const FLOOR_PX = 8

/** One median-drone-size datum, with how it was scored. */
type SizeDatum = {
  surface: string
  /** median sqrt(box area), native px */
  px: number
  /** modality the px figure is measured in */
  modality: 'RGB' | 'IR'
  tone: 'accent' | 'good' | 'bad' | 'muted'
  note: string
}

// Cross-surface median drone size (sqrt area, native px). The punchline: the two
// discriminating small-drone benchmarks sit just above the floor; Anti-UAV is an
// order of magnitude larger.
const SIZE_DATA: SizeDatum[] = [
  { surface: 'Svanström (IR)', px: 14.8, modality: 'IR', tone: 'muted', note: 'thermal box, even tighter' },
  { surface: 'Svanström (RGB)', px: 29.8, modality: 'RGB', tone: 'accent', note: 'just above the floor' },
  { surface: 'DUT-Anti-UAV', px: 40.9, modality: 'RGB', tone: 'good', note: 'median 40.9 px; boxes down to 25×9' },
  { surface: 'Anti-UAV', px: 93, modality: 'RGB', tone: 'bad', note: 'order-of-magnitude larger' },
]

/* -------------------------- comparable size strip -------------------------- */
/**
 * Hand-rolled "to-scale" size comparison. Each surface gets a square whose side
 * is the median sqrt(area) in px, drawn on a shared px→SVG scale so the squares
 * are directly comparable, with the ~8 px resolvable floor marked as a square +
 * baseline. Lands the Svanström/DUT-tiny vs Anti-UAV-large punchline literally
 * at scale. Same reveal/hover/focus grammar as the chart toolkit.
 */
function SizeScaleStrip() {
  const { ref, progress, seen } = useChartReveal<HTMLDivElement>(1150)
  const [active, setActive] = useState<number | null>(null)

  const VB_W = 520
  const VB_H = 190
  const PAD = { left: 16, right: 16, top: 24, bottom: 50 }
  const PLOT_W = VB_W - PAD.left - PAD.right
  const baseY = VB_H - PAD.bottom
  const maxPx = Math.max(...SIZE_DATA.map((d) => d.px)) // 93
  const col = PLOT_W / SIZE_DATA.length
  // cap the biggest square so it fits inside its OWN column (no overflow) and
  // leaves headroom above for the px label.
  const maxSide = Math.min(col * 0.66, baseY - PAD.top - 16)
  const scale = maxSide / maxPx

  const floorSide = FLOOR_PX * scale

  return (
    <div ref={ref} className="w-full">
      <div className="relative">
        <svg
          viewBox={`0 0 ${VB_W} ${VB_H}`}
          width="100%"
          role="img"
          aria-label="Median drone size to scale across evaluation surfaces, against the imgsz-640 resolvable floor"
          className="block select-none overflow-visible"
        >
          {/* baseline the squares sit on */}
          <line
            x1={PAD.left}
            x2={PAD.left + PLOT_W}
            y1={baseY}
            y2={baseY}
            stroke={palette.baseline}
            strokeWidth={1.25}
            style={{ opacity: seen ? 1 : 0, transition: 'opacity 500ms ease-out' }}
          />

          {/* resolvable-floor guide: a small square + dashed line across the plot */}
          <g style={{ opacity: seen ? 1 : 0, transition: 'opacity 700ms ease-out' }}>
            <line
              x1={PAD.left}
              x2={PAD.left + PLOT_W}
              y1={baseY - floorSide}
              y2={baseY - floorSide}
              stroke={toneHex.bad}
              strokeWidth={1}
              strokeDasharray="4 4"
              strokeOpacity={0.6}
            />
            <text
              x={PAD.left + PLOT_W}
              y={baseY - floorSide - 6}
              textAnchor="end"
              fontSize="10"
              fontWeight={600}
              fill={toneHexLight.bad}
            >
              ~8 px resolvable floor (imgsz 640)
            </text>
          </g>

          {SIZE_DATA.map((d, i) => {
            const side = d.px * scale * easeOutCubic(clamp(progress, 0, 1))
            const cx = PAD.left + col * i + col / 2
            const x = cx - side / 2
            const y = baseY - side
            const isActive = active === i
            const dimmed = active != null && !isActive
            return (
              <g
                key={d.surface}
                tabIndex={0}
                role="img"
                aria-label={`${d.surface}: median ${d.px} pixels`}
                onMouseEnter={() => setActive(i)}
                onMouseLeave={() => setActive(null)}
                onFocus={() => setActive(i)}
                onBlur={() => setActive(null)}
                style={{
                  cursor: 'default',
                  outline: 'none',
                  opacity: dimmed ? 0.4 : 1,
                  transition: 'opacity 180ms ease-out',
                }}
              >
                {/* hit area spanning the column */}
                <rect x={cx - col / 2} y={PAD.top} width={col} height={baseY - PAD.top} fill="transparent" />
                <rect
                  x={x}
                  y={y}
                  width={Math.max(side, 0.5)}
                  height={Math.max(side, 0.5)}
                  rx={2}
                  fill={toneHex[d.tone]}
                  fillOpacity={0.22}
                  stroke={toneHex[d.tone]}
                  strokeWidth={isActive ? 2 : 1.4}
                />
                {/* px value above the square */}
                <text
                  x={cx}
                  y={y - 7}
                  textAnchor="middle"
                  fontSize="12.5"
                  fontWeight={700}
                  fill={isActive ? toneHexLight[d.tone] : palette.text}
                  className="tabular-nums"
                  style={{ opacity: progress }}
                >
                  {d.px} px
                </text>
                {/* surface label under the baseline */}
                <text
                  x={cx}
                  y={baseY + 18}
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight={600}
                  fill={palette.text}
                  style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
                >
                  {d.surface}
                </text>
                <text
                  x={cx}
                  y={baseY + 33}
                  textAnchor="middle"
                  fontSize="9"
                  fill={palette.textFaint}
                  style={{ opacity: seen ? 1 : 0, transition: 'opacity 700ms ease-out' }}
                >
                  {d.modality}
                </text>
              </g>
            )
          })}
        </svg>

        <ChartTooltip state={sizeTooltip(active, SIZE_DATA, VB_W, VB_H, PAD.left, PLOT_W, baseY, scale)} />
      </div>
      <p className="mt-4 text-xs leading-relaxed text-slate-500">
        Squares are drawn to a shared scale: each side is the median&nbsp;
        <span className="text-slate-400">√area</span> in native pixels. Svanström&rsquo;s drones are the
        smallest (RGB 29.8&nbsp;px, IR 14.8&nbsp;px), nearest the ~8&nbsp;px imgsz-640 floor - where the
        Svanström baseline&rsquo;s recall is 0.684 at imgsz&nbsp;640 and recovers to 0.964 only when imgsz is
        doubled to 1280 (a 28&nbsp;pp swing). DUT-Anti-UAV sits higher at 40.9&nbsp;px; Anti-UAV at 93&nbsp;px
        is an order of magnitude larger and supra-floor at either resolution.
      </p>
    </div>
  )
}

function sizeTooltip(
  active: number | null,
  data: SizeDatum[],
  vbW: number,
  vbH: number,
  padLeft: number,
  plotW: number,
  baseY: number,
  scale: number,
): TooltipState {
  if (active == null) return null
  const d = data[active]
  if (!d) return null
  const col = plotW / data.length
  const cx = padLeft + col * active + col / 2
  const top = baseY - d.px * scale
  return {
    xFrac: cx / vbW,
    yFrac: top / vbH,
    label: `${d.surface} · ${d.modality}`,
    value: `median ${d.px} px - ${d.note}`,
    tone: d.tone,
  }
}

/* ------------------------------ small building blocks ----------------------- */

function Stat({ v, l, tone }: { v: string; l: string; tone?: 'good' | 'bad' | 'accent' }) {
  const c =
    tone === 'good'
      ? 'text-emerald-300'
      : tone === 'bad'
        ? 'text-rose-300'
        : tone === 'accent'
          ? 'text-cyan-300'
          : 'text-white'
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2.5">
      <div className={`text-lg font-bold tabular-nums ${c}`}>{v}</div>
      <div className="mt-0.5 text-[11px] leading-snug text-slate-500">{l}</div>
    </div>
  )
}

/** A small horizontal "size vs floor" gauge used inside a surface card. */
function FloorGauge({ px, modality, tone }: { px: number; modality: string; tone: 'accent' | 'good' | 'bad' | 'muted' }) {
  const { ref, progress, seen } = useChartReveal<HTMLDivElement>(1000)
  const VB_W = 320
  const VB_H = 58
  const padL = 6
  const padR = 6
  const plotW = VB_W - padL - padR
  // log-ish but readable: linear scale up to a generous ceiling so both tiny and
  // large surfaces fit; ceiling = 100 px (Anti-UAV ~93 fills the bar).
  const ceil = 100
  const xOf = (v: number) => padL + clamp(v / ceil, 0, 1) * plotW
  const floorX = xOf(FLOOR_PX)
  const barW = (xOf(px) - padL) * easeOutCubic(clamp(progress, 0, 1))
  return (
    <div ref={ref} className="w-full">
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} width="100%" role="img" aria-label={`Median drone size ${px} px versus 8 px floor`} className="block overflow-visible">
        {/* track */}
        <rect x={padL} y={26} width={plotW} height={8} rx={4} fill={palette.grid} />
        {/* fill up to median px */}
        <rect x={padL} y={26} width={Math.max(barW, 0.5)} height={8} rx={4} fill={toneHex[tone]} fillOpacity={0.85} />
        {/* floor marker */}
        <line x1={floorX} x2={floorX} y1={18} y2={42} stroke={toneHex.bad} strokeWidth={1.5} strokeDasharray="3 3" />
        <text x={floorX} y={14} textAnchor="middle" fontSize="9" fontWeight={600} fill={toneHexLight.bad} style={{ opacity: seen ? 1 : 0, transition: 'opacity 500ms ease-out' }}>
          8 px floor
        </text>
        {/* median px readout */}
        <text x={Math.min(xOf(px) + 6, VB_W - 4)} y={32} textAnchor={px / ceil > 0.8 ? 'end' : 'start'} dominantBaseline="central" fontSize="12" fontWeight={700} fill={toneHexLight[tone]} className="tabular-nums" style={{ opacity: progress }}>
          {px} px {modality}
        </text>
      </svg>
    </div>
  )
}

function NoConfusers({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3">
      <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-emerald-400/50 text-[10px] font-bold text-emerald-300">
        ∅
      </span>
      <p className="text-[12.5px] leading-relaxed text-slate-400">{children}</p>
    </div>
  )
}

/* --------------------------------- cards ----------------------------------- */

function SurfaceCard({
  kicker,
  title,
  blurb,
  tone,
  children,
}: {
  kicker: string
  title: string
  blurb: ReactNode
  tone: 'accent' | 'good' | 'bad' | 'muted'
  children: ReactNode
}) {
  return (
    <div className="card overflow-hidden p-6">
      <p className="eyebrow mb-2">{kicker}</p>
      <h4 className="text-xl font-bold tracking-tight text-white">{title}</h4>
      <p className="mt-2 text-[13px] leading-relaxed text-slate-400">{blurb}</p>
      <div className="mt-5 space-y-5">{children}</div>
    </div>
  )
}

/* -------------------------------- section ---------------------------------- */

export function DatasetsAtAGlance() {
  return (
    <section className="border-t border-slate-900 bg-gradient-to-b from-slate-950 to-slate-900/30">
      <div className="mx-auto max-w-6xl px-6 py-14">
        {/* header */}
        <div className="mb-10">
          <p className="eyebrow mb-3">The evidence base &middot; Datasets at a glance</p>
          <h2 className="max-w-3xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
            What each surface actually tests
          </h2>
          <p className="mt-4 max-w-2xl text-slate-400">
            Every headline number is earned on a specific surface, and the surfaces are not interchangeable.
            Two things decide what a benchmark can prove: how <span className="text-white">small</span> its
            drones are, and whether it contains <span className="text-white">confusers</span> - birds,
            airplanes, helicopters that a naïve detector mistakes for a target. Here is each surface on those
            two axes.
          </p>
        </div>

        {/* punchline: cross-surface drone size, to scale */}
        <div className="card p-6 sm:p-8">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="eyebrow mb-1">The punchline · drone size to scale</p>
              <h3 className="text-2xl font-bold tracking-tight text-white">
                Tiny targets vs a saturated control
              </h3>
            </div>
            <p className="max-w-xs text-[12.5px] leading-relaxed text-slate-500">
              The two discriminating benchmarks live at the resolution floor; the &ldquo;no-harm&rdquo;
              control does not. That gap is the whole reason small-drone results and saturated results read so
              differently.
            </p>
          </div>
          <div className="mx-auto max-w-2xl">
            <SizeScaleStrip />
          </div>
        </div>

        {/* per-surface eval cards */}
        <div className="mt-10">
          <h3 className="mb-1 text-lg font-semibold text-slate-200">Evaluation surfaces</h3>
          <p className="mb-6 max-w-2xl text-sm text-slate-500">
            Held-out benchmarks the pipeline is judged on - drone size, confuser composition, and the few
            stats that matter for each.
          </p>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* Svanström */}
            <SurfaceCard
              kicker="Surface · discriminating"
              title="Svanström (paired RGB + IR)"
              tone="accent"
              blurb={
                <>
                  The hard small-drone benchmark, and the only eval surface where confusers are{' '}
                  <span className="text-slate-200">labeled</span> - three classes alongside the drone. 28,710
                  paired frames across 279 sequences at native 640×512.
                </>
              }
            >
              <div className="grid grid-cols-2 gap-3">
                <Stat v="29.8 px" l="median √area, RGB (14.8 px IR)" tone="accent" />
                <Stat v="11,695" l="drone instances" />
                <Stat v="3 classes" l="labeled confusers" tone="good" />
                <Stat v="640×512" l="native resolution" />
              </div>
              <DonutChart
                title="Instance mix - drone vs labeled confusers"
                centerLabel="instances"
                ariaLabel="Svanström instance composition: drone, airplane, bird, helicopter"
                slices={[
                  { label: 'Drone', value: 11695, tone: 'accent' },
                  { label: 'Airplane', value: 6090, tone: 'muted' },
                  { label: 'Helicopter', value: 5627, tone: 'bad' },
                  { label: 'Bird', value: 5298, tone: 'good' },
                ]}
                caption="Drone is barely a third of the labeled instances - the confuser pressure here is real and in-distribution, which is what makes it discriminating."
              />
            </SurfaceCard>

            {/* Anti-UAV */}
            <SurfaceCard
              kicker="Surface · saturated control"
              title="Anti-UAV (RGBT)"
              tone="bad"
              blurb={
                <>
                  The no-harm control: large, well-resolved drones on ordinary footage. The pipeline has to
                  prove it does not <span className="text-slate-200">degrade</span> here while it lifts the
                  hard surfaces. 85,374 frames.
                </>
              }
            >
              <div className="grid grid-cols-2 gap-3">
                <Stat v="~93 px" l="median √area - order-of-mag larger" tone="bad" />
                <Stat v="85,374" l="frames" />
                <Stat v="All drone" l="single target class" />
                <Stat v="1920×1080" l="RGB (640×512 IR)" />
              </div>
              <NoConfusers>
                <span className="font-semibold text-slate-300">No confusers.</span> Anti-UAV is an all-drone
                surface - there are no bird / airplane / helicopter distractors to compose, so there is no
                confuser donut to show. It measures saturation and no-harm, not discrimination.
              </NoConfusers>
            </SurfaceCard>

            {/* DUT-Anti-UAV */}
            <SurfaceCard
              kicker="Surface · small-drone RGB"
              title="DUT-Anti-UAV (official test split)"
              tone="good"
              blurb={
                <>
                  A second small-drone RGB benchmark, scored on its official test split - boxes shrink to{' '}
                  <span className="text-slate-200">25×9 px</span>, well into the regime where imgsz starts to
                  matter. 2,200 frames / 2,245 GT drones.
                </>
              }
            >
              <div className="grid grid-cols-2 gap-3">
                <Stat v="25×9 px" l="smallest GT boxes" tone="good" />
                <Stat v="2,245" l="ground-truth drones" />
                <Stat v="2,200" l="frames (official test)" />
                <Stat v="All drone" l="single target class" />
              </div>
              <NoConfusers>
                <span className="font-semibold text-slate-300">No confusers.</span> Like Anti-UAV, DUT is
                all-drone - it stresses <span className="text-slate-200">resolution</span>, not confuser
                rejection. There is nothing to compose into a donut here.
              </NoConfusers>
            </SurfaceCard>

            {/* Confuser surfaces */}
            <SurfaceCard
              kicker="Surface · OOD confusers"
              title="Confuser benchmarks (RGB + IR)"
              tone="muted"
              blurb={
                <>
                  The out-of-distribution distractor sets used to measure false-alarm suppression: airplanes,
                  birds and helicopters with <span className="text-slate-200">no drone present</span>. Every
                  fire is a false alarm.
                </>
              }
            >
              <div className="grid grid-cols-2 gap-3">
                <Stat v="27,024" l="rgb_confusers_merged (2,633 test)" />
                <Stat v="5,938" l="IR_confusers instances" />
              </div>
              <DonutChart
                title="IR_confusers composition"
                centerLabel="instances"
                ariaLabel="IR confuser composition: airplane, bird, helicopter"
                slices={[
                  { label: 'Airplane', value: 4281, tone: 'accent' },
                  { label: 'Bird', value: 1200, tone: 'good' },
                  { label: 'Helicopter', value: 457, tone: 'bad' },
                ]}
                caption="Airplane-dominated. Confuser SIZE has no clean in-repo statistic, so this is composition only - at range a confuser silhouette mirrors drone size (small, ambiguous), which is exactly why a separate filter, not a size threshold, is needed."
              />
              <p className="rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-3 text-[12px] leading-relaxed text-slate-500">
                RGB test slice (2,633): Svanström airplane 1,034 / heli 311 / bird 409, plus 879 Roboflow +
                Kaggle distractors.
              </p>
            </SurfaceCard>
          </div>
        </div>

        {/* training corpora - the splits the detectors are fit on */}
        <div className="mt-10">
          <h3 className="mb-1 text-lg font-semibold text-slate-200">Training corpora (test splits)</h3>
          <p className="mb-6 max-w-2xl text-sm text-slate-500">
            The single-class detectors are fit on large multi-source corpora; held-out test splits below.
            Birds appear here only as background <span className="text-slate-300">negatives</span>, never as a
            labeled class - confuser rejection is delegated to the downstream filter, not the detector.
          </p>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* RGB corpus */}
            <div className="card p-6">
              <p className="eyebrow mb-2">RGB · rgb_dataset</p>
              <h4 className="text-lg font-bold text-white">17,209 test frames</h4>
              <p className="mt-1 text-[12.5px] text-slate-500">
                of 172,022 total (80/10/10) · 146,540 drone boxes · single class
              </p>
              <div className="mt-4">
                <PosNegBar pos={78.1} posLabel="78.1% positive (drone present)" negLabel="21.9% background" />
              </div>
              <DonutChart
                title="10 source datasets"
                legend={false}
                centerLabel="total frames"
                centerValue="172,022"
                ariaLabel="RGB training corpus composition by source dataset"
                slices={[
                  { label: 'anti', value: 59413, tone: 'accent' },
                  { label: 'wosdetc', value: 53259, tone: 'good' },
                  { label: 'anti-muav-roboflow', value: 14652, tone: 'bad' },
                  { label: 'AirBird', value: 10000, tone: 'muted' },
                  { label: 'FBD-SV', value: 9754, tone: 'accent' },
                  { label: 'mav', value: 9744, tone: 'good' },
                  { label: 'dut', value: 5200, tone: 'bad' },
                  { label: 'VIRAT', value: 3334, tone: 'muted' },
                  { label: 'UA-DETRAC', value: 3333, tone: 'accent' },
                  { label: 'BDD100K', value: 3333, tone: 'good' },
                ]}
                caption="anti + wosdetc dominate; AirBird / BDD100K / VIRAT / UA-DETRAC contribute hard negatives (birds, traffic, scenes) so the detector learns drone-vs-background, not drone-vs-nothing."
              />
            </div>

            {/* IR corpus */}
            <div className="card p-6">
              <p className="eyebrow mb-2">IR · ir_dset_final</p>
              <h4 className="text-lg font-bold text-white">9,612 test frames</h4>
              <p className="mt-1 text-[12.5px] text-slate-500">
                of 129,130 total (83.5/9.1/7.4) · 94,142 drone boxes · native 640×512
              </p>
              <div className="mt-4">
                <PosNegBar pos={72.5} posLabel="72.5% positive (drone present)" negLabel="27.5% negative" />
              </div>
              <DonutChart
                title="10 source datasets"
                legend={false}
                centerLabel="total frames"
                centerValue="129,130"
                ariaLabel="IR training corpus composition by source dataset"
                slices={[
                  { label: 'dv5', value: 53512, tone: 'accent' },
                  { label: 'cst', value: 22957, tone: 'good' },
                  { label: 'svan', value: 21637, tone: 'bad' },
                  { label: 'czoom', value: 9808, tone: 'muted' },
                  { label: 'sea', value: 8398, tone: 'accent' },
                  { label: 'flir', value: 6360, tone: 'good' },
                  { label: 'ovh2', value: 2898, tone: 'bad' },
                  { label: 'ovh1', value: 2866, tone: 'muted' },
                  { label: 'gemini', value: 392, tone: 'accent' },
                  { label: 'yt', value: 302, tone: 'good' },
                ]}
                caption="dv5-heavy, with svan thermal in the mix (its train/eval overlap is separately audited). Like the RGB corpus, a single drone class - thermal confuser rejection is the verifier's job."
              />
            </div>
          </div>
        </div>

        <p className="mt-10 max-w-3xl text-xs leading-relaxed text-slate-600">
          Sizes are median √area in native pixels; the ~8&nbsp;px line is the YOLO26n imgsz-640 resolvable
          floor. Confuser composition is shown where confusers exist; Anti-UAV and DUT-Anti-UAV are all-drone
          surfaces with none. Confuser <em>size</em> is not separately measured in-repo and is not invented -
          at range, confuser silhouettes mirror drone size.
        </p>
      </div>
    </section>
  )
}

/* ------------------------- positive/background split ----------------------- */
/** A single horizontal positive-vs-background split bar with reveal + readouts. */
function PosNegBar({ pos, posLabel, negLabel }: { pos: number; posLabel: string; negLabel: string }) {
  const { ref, progress, seen } = useChartReveal<HTMLDivElement>(1000)
  const w = pos * easeOutCubic(clamp(progress, 0, 1))
  return (
    <div ref={ref} className="w-full">
      <div className="mb-1.5 flex items-baseline justify-between text-[11px]">
        <span className="font-medium text-cyan-300">{posLabel}</span>
        <span className="text-slate-500">{negLabel}</span>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-800">
        <div
          className="h-full rounded-l-full"
          style={{
            width: `${w}%`,
            background: `linear-gradient(90deg, ${toneHex.accent}, ${toneHexLight.accent})`,
            transition: 'none',
          }}
        />
      </div>
      <div
        className="mt-1 text-[10px] text-slate-600"
        style={{ opacity: seen ? 1 : 0, transition: 'opacity 600ms ease-out' }}
      >
        background negatives keep precision honest - the detector must reject empty sky, not just find drones
      </div>
    </div>
  )
}
