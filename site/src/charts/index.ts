/**
 * Hand-rolled SVG chart toolkit for the TALOS scrollytelling site.
 *
 * No charting library - inline SVG + CSS transitions + requestAnimationFrame,
 * wired to the site's `useInViewOnce` / `useReducedMotion` hooks. Every chart
 * reveals once on scroll-in, exposes hover/focus tooltips, dims non-active
 * series, and is keyboard-focusable. Responsive via SVG viewBox + width:100%.
 */

export { GroupedBarChart } from './GroupedBarChart'
export type {
  GroupedBarChartProps,
  ChartGroup,
  ChartBar,
} from './GroupedBarChart'

export { ComparisonChart } from './ComparisonChart'
export type { ComparisonChartProps, ComparisonBar } from './ComparisonChart'

export { MetricBar } from './MetricBar'
export type { MetricBarProps, MetricBarItem } from './MetricBar'

export { LdaSeparabilityChart } from './LdaSeparabilityChart'
export type { LdaSeparabilityChartProps, LdaPointsData } from './LdaSeparabilityChart'

export { PcaScatterChart } from './PcaScatterChart'
export type { PcaScatterChartProps, PcaPointsData } from './PcaScatterChart'

export { ActivationOverlay } from './ActivationOverlay'
export type {
  ActivationOverlayProps,
  ActivationOverlayData,
  ActivationItem,
} from './ActivationOverlay'

export { DonutChart } from './DonutChart'
export type { DonutChartProps, DonutSlice } from './DonutChart'

export { ChartTooltip } from './ChartTooltip'
export type { TooltipState } from './ChartTooltip'

export { useChartReveal } from './useChartReveal'

export type { Tone } from './chartTheme'
export { toneHex, toneHexLight, palette } from './chartTheme'
