// ============================================
// ALL types derived from API contract
// NO assumptions beyond documented endpoints
// ============================================

// --- Generic API Response Wrapper ---
export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

// --- Request State Machine ---
export type RequestStatus = 'idle' | 'loading' | 'success' | 'error';

export interface RequestState<T> {
  data: T | null;
  status: RequestStatus;
  error: string | null;
  lastUpdated: number | null;
}

// --- Zone Types ---
export interface Zone {
  id: string;
  name: string;
  riskScore: number;
  coordinates: Array<{
    lat: number;
    lng: number;
  }>;
  [key: string]: unknown; // Allow additional fields from backend
}

export interface ZoneDetail extends Zone {
  detectionCount?: number;
  lastDetection?: string;
  plasticTypes?: Array<{
    type: string;
    count: number;
  }>;
  recentActions?: Array<{
    id: string;
    type: string;
    timestamp: string;
    status: string;
  }>;
  [key: string]: unknown;
}

// --- Analytics Types ---
export interface AnalyticsData {
  plasticTypeDistribution: Array<{
    type: string;
    count: number;
    percentage?: number;
    [key: string]: unknown;
  }>;
  detectionOverTime: Array<{
    timestamp: string;
    count: number;
    [key: string]: unknown;
  }>;
  [key: string]: unknown;
}

// --- Prediction Types ---
export interface PredictionData {
  hotspots: Array<{
    zoneId: string;
    zoneName: string;
    predictedRisk: number;
    confidence?: number;
    [key: string]: unknown;
  }>;
  riskTrend: Array<{
    timestamp: string;
    riskLevel: number;
    [key: string]: unknown;
  }>;
  [key: string]: unknown;
}

// --- Action Types ---
export interface DispatchActionPayload {
  zoneId: string;
  actionType: string;
  priority?: string;
  notes?: string;
  [key: string]: unknown;
}

export interface DispatchActionResponse {
  id: string;
  status: string;
  message?: string;
  estimatedArrival?: string;
  [key: string]: unknown;
}