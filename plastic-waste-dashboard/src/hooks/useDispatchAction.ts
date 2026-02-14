import { useState, useCallback } from 'react';
import { api } from '@/api/client';
import {
  DispatchActionPayload,
  DispatchActionResponse,
  RequestState,
} from '@/types';

export function useDispatchAction() {
  const [state, setState] = useState<RequestState<DispatchActionResponse>>({
    data: null,
    status: 'idle',
    error: null,
    lastUpdated: null,
  });

  const dispatch = useCallback(async (payload: DispatchActionPayload) => {
    setState({
      data: null,
      status: 'loading',
      error: null,
      lastUpdated: null,
    });

    try {
      const result = await api.dispatchAction(payload);
      setState({
        data: result,
        status: 'success',
        error: null,
        lastUpdated: Date.now(),
      });
      return result;
    } catch (err: any) {
      setState({
        data: null,
        status: 'error',
        error: err?.message || 'Failed to dispatch action',
        lastUpdated: null,
      });
      throw err;
    }
  }, []);

  const reset = useCallback(() => {
    setState({
      data: null,
      status: 'idle',
      error: null,
      lastUpdated: null,
    });
  }, []);

  return {
    ...state,
    dispatch,
    reset,
  };
}