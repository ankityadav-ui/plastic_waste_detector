import { format, formatDistanceToNow, parseISO, isValid } from 'date-fns';

/**
 * Safely format a timestamp string from backend.
 * Never crashes on unexpected formats.
 */
export function formatTimestamp(
  timestamp: string | undefined | null,
  pattern: string = 'MMM dd, HH:mm'
): string {
  if (!timestamp) return '—';
  try {
    const date = parseISO(timestamp);
    if (!isValid(date)) return timestamp;
    return format(date, pattern);
  } catch {
    return timestamp;
  }
}

export function formatRelativeTime(
  timestamp: string | undefined | null
): string {
  if (!timestamp) return '—';
  try {
    const date = parseISO(timestamp);
    if (!isValid(date)) return timestamp;
    return formatDistanceToNow(date, { addSuffix: true });
  } catch {
    return timestamp;
  }
}

export function formatNumber(value: number | undefined | null): string {
  if (value == null) return '—';
  return new Intl.NumberFormat().format(value);
}

export function formatPercentage(value: number | undefined | null): string {
  if (value == null) return '—';
  return `${value.toFixed(1)}%`;
}