import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import {
  Zone,
  ZoneDetail,
  AnalyticsData,
  PredictionData,
  DispatchActionPayload,
  DispatchActionResponse,
} from '@/types';

// ============================================
// Centralized API Client
// Single source of truth for ALL backend calls
// ============================================

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

if (!BASE_URL) {
  console.warn(
    '[API Client] NEXT_PUBLIC_API_BASE_URL is not set. API calls will fail.'
  );
}

const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- Request Interceptor ---
apiClient.interceptors.request.use(
  (config) => {
    // Attach auth token if available in future
    const token =
      typeof window !== 'undefined'
        ? localStorage.getItem('auth_token')
        : null;
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    console.debug(
      `[API] ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`
    );
    return config;
  },
  (error) => Promise.reject(error)
);

// --- Response Interceptor ---
apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const status = error.response?.status;
    const url = error.config?.url;

    console.error(`[API Error] ${status} on ${url}:`, error.message);

    if (status === 401) {
      console.warn('[API] Unauthorized — token may be expired');
    }

    if (status === 503) {
      console.warn('[API] Service unavailable — backend may be down');
    }

    return Promise.reject(error);
  }
);

// --- Typed API Methods ---

export const api = {
  // Zones
  getZones: async (config?: AxiosRequestConfig): Promise<Zone[]> => {
    const response = await apiClient.get('/api/zones', config);
    return response.data?.data ?? response.data;
  },

  getZoneById: async (
    id: string,
    config?: AxiosRequestConfig
  ): Promise<ZoneDetail> => {
    const response = await apiClient.get(`/api/zones/${id}`, config);
    return response.data?.data ?? response.data;
  },

  // Analytics
  getAnalytics: async (
    config?: AxiosRequestConfig
  ): Promise<AnalyticsData> => {
    const response = await apiClient.get('/api/analytics', config);
    return response.data?.data ?? response.data;
  },

  // Predictions
  getPredictions: async (
    config?: AxiosRequestConfig
  ): Promise<PredictionData> => {
    const response = await apiClient.get('/api/predictions', config);
    return response.data?.data ?? response.data;
  },

  // Actions
  dispatchAction: async (
    payload: DispatchActionPayload,
    config?: AxiosRequestConfig
  ): Promise<DispatchActionResponse> => {
    const response = await apiClient.post('/api/actions', payload, config);
    return response.data?.data ?? response.data;
  },
};

export default apiClient;