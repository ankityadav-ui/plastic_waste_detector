import { useState, useCallback, useEffect } from 'react';
import { api } from '@/api/client';
import { PredictionData, RequestState } from '@/types';
import { usePolling } from './usePolling';

export function usePredictions(pollInterval?: number) {
  const [state, setState] = useState<RequestState<PredictionData>>({
    data: null,
    status: 'idle',
    error: null,
    lastUpdated: null,
  });

  const fetchPredictions = useCallback(async () => {
    setState((prev) => ({
      ...prev,
      status: prev.data ? prev.status : 'loading',
      error: null,
    }));

    try {
      const predictions = await api.getPredictions();
      setState({
        data: predictions,
        status: 'success',
        error: null,
        lastUpdated: Date.now(),
      });
    } catch (err: any) {
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: err?.message || 'Failed to fetch predictions',
      }));
    }
  }, []);

  useEffect(() => {
    fetchPredictions();
  }, [fetchPredictions]);

  usePolling(fetchPredictions, pollInterval);

  return {
    ...state,
    refetch: fetchPredictions,
  };
}