'use client';

import React from 'react';
import { Flame } from 'lucide-react';
import { getRiskThreshold } from '@/utils/riskColor';
import { formatPercentage } from '@/utils/formatters';

interface HotspotListProps {
  hotspots: Array<{
    zoneId: string;
    zoneName: string;
    predictedRisk: number;
    confidence?: number;
    [key: string]: unknown;
  }>;
  onZoneClick: (zoneId: string) => void;
}

export const HotspotList: React.FC<HotspotListProps> = ({
  hotspots,
  onZoneClick,
}) => {
  // Sort by predicted risk descending
  const sorted = [...hotspots].sort(
    (a, b) => (b.predictedRisk ?? 0) - (a.predictedRisk ?? 0)
  );

  return (
    <div>
      <h4 className="text-municipal-text text-xs font-semibold uppercase tracking-wider mb-3 flex items-center gap-2">
        <Flame className="h-3.5 w-3.5 text-municipal-orange" />
        Predicted Hotspots
      </h4>
      <div className="space-y-2">
        {sorted.map((hotspot, idx) => {
          const riskInfo = getRiskThreshold(hotspot.predictedRisk);
          return (
            <button
              key={hotspot.zoneId || idx}
              onClick={() => onZoneClick(hotspot.zoneId)}
              className="w-full flex items-center justify-between bg-municipal-bg rounded-lg p-3 hover:bg-municipal-border transition-colors text-left"
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-2 h-8 rounded-full"
                  style={{ backgroundColor: riskInfo.color }}
                />
                <div>
                  <p className="text-municipal-text text-sm font-medium">
                    {hotspot.zoneName || hotspot.zoneId}
                  </p>
                  {hotspot.confidence != null && (
                    <p className="text-municipal-muted text-xs font-mono">
                      Confidence: {formatPercentage(hotspot.confidence)}
                    </p>
                  )}
                </div>
              </div>
              <div className="text-right">
                <p
                  className="text-lg font-bold font-mono"
                  style={{ color: riskInfo.color }}
                >
                  {hotspot.predictedRisk}
                </p>
                <p
                  className="text-xs"
                  style={{ color: riskInfo.color }}
                >
                  {riskInfo.label}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};