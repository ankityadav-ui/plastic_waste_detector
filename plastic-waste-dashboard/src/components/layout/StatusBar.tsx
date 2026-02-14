'use client';

import React from 'react';
import { Activity } from 'lucide-react';
import { formatRelativeTime } from '@/utils/formatters';

interface StatusBarProps {
  zonesLastUpdated: number | null;
  analyticsLastUpdated: number | null;
  predictionsLastUpdated: number | null;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  zonesLastUpdated,
  analyticsLastUpdated,
  predictionsLastUpdated,
}) => {
  const formatUpdate = (ts: number | null): string => {
    if (!ts) return 'Never';
    return formatRelativeTime(new Date(ts).toISOString());
  };

  return (
    <div className="bg-municipal-bg border-t border-municipal-border px-6 py-2">
      <div className="flex items-center justify-between text-xs font-mono text-municipal-muted">
        <div className="flex items-center gap-2">
          <Activity className="h-3 w-3" />
          <span>System Active</span>
        </div>
        <div className="flex items-center gap-6">
          <span>Zones: {formatUpdate(zonesLastUpdated)}</span>
          <span>Analytics: {formatUpdate(analyticsLastUpdated)}</span>
          <span>Predictions: {formatUpdate(predictionsLastUpdated)}</span>
        </div>
      </div>
    </div>
  );
};