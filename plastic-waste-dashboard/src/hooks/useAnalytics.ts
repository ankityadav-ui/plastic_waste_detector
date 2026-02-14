import { useState, useCallback, useEffect } from 'react';
import { api } from '@/api/client';
import { AnalyticsData, RequestState } from '@/types';
import { usePolling } from './usePolling';

export function useAnalytics(pollInterval?: number) {
  const [state, setState] = useState<RequestState<AnalyticsData>>({
    data: null,
    status: 'idle',
    error: null,
    lastUpdated: null,
  });

  const fetchAnalytics = useCallback(async () => {
    setState((prev) => ({
      ...prev,
      status: prev.data ? prev.status : 'loading',
      error: null,
    }));

    try {
      const analytics = await api.getAnalytics();
      setState({
        data: analytics,
        status: 'success',
        error: null,
        lastUpdated: Date.now(),
      });
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: err?.message || 'Failed to fetch analytics',
      }));
    }
  }, []);

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  usePolling(fetchAnalytics, pollInterval);

  return {
    ...state,
    refetch: fetchAnalytics,
  };
}