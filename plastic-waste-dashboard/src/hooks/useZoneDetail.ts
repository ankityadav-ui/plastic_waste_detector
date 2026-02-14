import { useState, useCallback } from 'react';
import { api } from '@/api/client';
import { ZoneDetail, RequestState } from '@/types';

export function useZoneDetail() {
  const [state, setState] = useState<RequestState<ZoneDetail>>({
    data: null,
    status: 'idle',
    error: null,
    lastUpdated: null,
  });

  const fetchZoneDetail = useCallback(async (zoneId: string) => {
    setState({
      data: null,
      status: 'loading',
      error: null,
      lastUpdated: null,
    });

    try {
      const detail = await api.getZoneById(zoneId);
      setState({
        data: detail,
        status: 'success',
        error: null,
        lastUpdated: Date.now(),
      });
    } catch (err: any) {
      setState({
        data: null,
        status: 'error',
        error: err?.message || 'Failed to fetch zone details',
        lastUpdated: null,
      });
    }
  }, []);

  const clearDetail = useCallback(() => {
    setState({
      data: null,
      status: 'idle',
      error: null,
      lastUpdated: null,
    });
  }, []);

  return {
    ...state,
    fetchZoneDetail,
    clearDetail,
  };
}