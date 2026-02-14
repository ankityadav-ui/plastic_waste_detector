'use client';

import React from 'react';
import { AnalyticsData, RequestStatus } from '@/types';
import { Card } from '@/components/common/Card';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorState } from '@/components/common/ErrorState';
import { EmptyState } from '@/components/common/EmptyState';
import { PlasticTypeChart } from './PlasticTypeChart';
import { DetectionTimelineChart } from './DetectionTimelineChart';
import { RefreshCw } from 'lucide-react';

interface AnalyticsPanelProps {
  data: AnalyticsData | null;
  status: RequestStatus;
  error: string | null;
  onRefresh: () => void;
}

export const AnalyticsPanel: React.FC<AnalyticsPanelProps> = ({
  data,
  status,
  error,
  onRefresh,
}) => {
  const renderContent = () => {
    if (status === 'loading' && !data) {
      return <LoadingSpinner message="Loading analytics…" />;
    }

    if (status === 'error' && !data) {
      return (
        <ErrorState
          message={error || 'Failed to load analytics'}
          onRetry={onRefresh}
        />
      );
    }

    if (!data) {
      return <EmptyState message="No analytics data available" />;
    }

    return (
      <div className="space-y-6">
        {data.plasticTypeDistribution &&
          data.plasticTypeDistribution.length > 0 && (
            <PlasticTypeChart data={data.plasticTypeDistribution} />
          )}

        {data.detectionOverTime && data.detectionOverTime.length > 0 && (
          <DetectionTimelineChart data={data.detectionOverTime} />
        )}

        {(!data.plasticTypeDistribution ||
          data.plasticTypeDistribution.length === 0) &&
          (!data.detectionOverTime ||
            data.detectionOverTime.length === 0) && (
            <EmptyState message="Analytics data is empty" />
          )}
      </div>
    );
  };

  return (
    <Card
      title="Detection Analytics"
      subtitle="Plastic waste analysis"
      headerAction={
        <button
          onClick={onRefresh}
          className="p-1.5 rounded-md hover:bg-municipal-border transition-colors"
          title="Refresh analytics"
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