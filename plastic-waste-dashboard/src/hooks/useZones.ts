import { useState, useCallback, useEffect } from 'react';
import { api } from '@/api/client';
import { Zone, RequestState } from '@/types';
import { usePolling } from './usePolling';

export function useZones(pollInterval?: number) {
  const [state, setState] = useState<RequestState<Zone[]>>({
    data: null,
    status: 'idle',
    error: null,
    lastUpdated: null,
  });

  const fetchZones = useCallback(async () => {
    // Only set loading on first fetch, not on polls
    setState((prev) => ({
      ...prev,
      status: prev.data ? prev.status : 'loading',
      error: null,
    }));

    try {
      const zones = await api.getZones();
      setState({
        data: Array.isArray(zones) ? zones : [],
        status: 'success',
        error: null,
        lastUpdated: Date.now(),
      });
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: err?.message || 'Failed to fetch zones',
      }));
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchZones();
  }, [fetchZones]);

  // Auto-poll
  usePolling(fetchZones, pollInterval);

  return {
    ...state,
    refetch: fetchZones,
  };
}