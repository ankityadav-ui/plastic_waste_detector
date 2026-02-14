import { useEffect, useRef, useCallback } from 'react';

/**
 * Generic polling hook — calls a fetch function at a configurable interval.
 * Stops polling when component unmounts.
 */
export function usePolling(
  fetchFn: () => void,
  intervalMs?: number,
  enabled: boolean = true
) {
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const savedFetchFn = useRef(fetchFn);

  // Keep the fetch function reference fresh
  useEffect(() => {
    savedFetchFn.current = fetchFn;
  }, [fetchFn]);

  const startPolling = useCallback(() => {
    const interval =
      intervalMs ??
      Number(process.env.NEXT_PUBLIC_POLLING_INTERVAL) ??
      30000;

    intervalRef.current = setInterval(() => {
      savedFetchFn.current();
    }, interval);
  }, [intervalMs]);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (enabled) {
      startPolling();
    }

    return () => {
      stopPolling();
    };
  }, [enabled, startPolling, stopPolling]);

  return { startPolling, stopPolling };
}