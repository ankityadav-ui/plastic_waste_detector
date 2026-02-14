'use client';

import React from 'react';
import { PredictionData, RequestStatus } from '@/types';
import { Card } from '@/components/common/Card';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorState } from '@/components/common/ErrorState';
import { EmptyState } from '@/components/common/EmptyState';
import { HotspotList } from './HotspotList';
import { RiskTrendChart } from './RiskTrendChart';
import { RefreshCw } from 'lucide-react';

interface PredictionPanelProps {
  data: PredictionData | null;
  status: RequestStatus;
  error: string | null;
  onRefresh: () => void;
  onZoneClick: (zoneId: string) => void;
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({
  data,
  status,
  error,
  onRefresh,
  onZoneClick,
}) => {
  const renderContent = () => {
    if (status === 'loading' && !data) {
      return <LoadingSpinner message="Loading predictions…" />;
    }

    if (status === 'error' && !data) {
      return (
        <ErrorState
          message={error || 'Failed to load predictions'}
          onRetry={onRefresh}
        />
      );
    }

    if (!data) {
      return <EmptyState message="No prediction data available" />;
    }

    return (
      <div className="space-y-6">
        {data.hotspots && data.hotspots.length > 0 && (
          <HotspotList hotspots={data.hotspots} onZoneClick={onZoneClick} />
        )}

        {data.riskTrend && data.riskTrend.length > 0 && (
          <RiskTrendChart data={data.riskTrend} />
        )}

        {(!data.hotspots || data.hotspots.length === 0) &&
          (!data.riskTrend || data.riskTrend.length === 0) && (
            <EmptyState message="No predictions available" />
          )}
      </div>
    );
  };

  return (
    <Card
      title="Predictions & Hotspots"
      subtitle="24–48 hour forecast"
      headerAction={
        <button
          onClick={onRefresh}
          className="p-1.5 rounded-md hover:bg-municipal-border transition-colors"
          title="Refresh predictions"
        >
          <RefreshCw
            className={`h-4 w-4 text-municipal-muted ${status === 'loading' ? 'animate-spin' : ''}`}
          />
        </button>
      }
    >
      {renderContent()}
    </Card>
  );
};