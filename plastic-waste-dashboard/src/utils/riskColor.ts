/**
 * Maps a risk score (0–100) to a color.
 * All thresholds are configurable — no magic numbers.
 */

interface RiskThreshold {
  max: number;
  color: string;
  label: string;
  bgClass: string;
  textClass: string;
  borderClass: string;
}

const RISK_THRESHOLDS: RiskThreshold[] = [
  {
    max: 30,
    color: '#22c55e',
    label: 'Low',
    bgClass: 'bg-green-500',
    textClass: 'text-green-400',
    borderClass: 'border-green-500',
  },
  {
    max: 60,
    color: '#eab308',
    label: 'Medium',
    bgClass: 'bg-yellow-500',
    textClass: 'text-yellow-400',
    borderClass: 'border-yellow-500',
  },
  {
    max: 80,
    color: '#f97316',
    label: 'High',
    bgClass: 'bg-orange-500',
    textClass: 'text-orange-400',
    borderClass: 'border-orange-500',
  },
  {
    max: 100,
    color: '#ef4444',
    label: 'Critical',
    bgClass: 'bg-red-500',
    textClass: 'text-red-400',
    borderClass: 'border-red-500',
  },
];

export function getRiskThreshold(score: number): RiskThreshold {
  const normalizedScore = Math.max(0, Math.min(100, score));
  return (
    RISK_THRESHOLDS.find((t) => normalizedScore <= t.max) ??
    RISK_THRESHOLDS[RISK_THRESHOLDS.length - 1]
  );
}

export function getRiskColor(score: number): string {
  return getRiskThreshold(score).color;
}

export function getRiskLabel(score: number): string {
  return getRiskThreshold(score).label;
}

export function getRiskColorWithOpacity(
  score: number,
  opacity: number = 0.4
): string {
  const hex = getRiskColor(score);
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}