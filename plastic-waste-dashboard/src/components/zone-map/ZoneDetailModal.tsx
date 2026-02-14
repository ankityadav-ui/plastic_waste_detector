'use client';

import React from 'react';
import { X, MapPin, Clock, BarChart3 } from 'lucide-react';
import { ZoneDetail, RequestStatus } from '@/types';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorState } from '@/components/common/ErrorState';
import { getRiskThreshold } from '@/utils/riskColor';
import { formatTimestamp, formatNumber } from '@/utils/formatters';

interface ZoneDetailModalProps {
  zone: ZoneDetail | null;
  status: RequestStatus;
  error: string | null;
  onClose: () => void;
  onDispatch: (zoneId: string) => void;
}

export const ZoneDetailModal: React.FC<ZoneDetailModalProps> = ({
  zone,
  status,
  error,
  onClose,
  onDispatch,
}) => {
  if (status === 'idle') return null;

  const riskInfo = zone ? getRiskThreshold(zone.riskScore) : null;

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4">
      <div className="bg-municipal-card border border-municipal-border rounded-xl w-full max-w-lg max-h-[80vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-municipal-border">
          <div className="flex items-center gap-2">
            <MapPin className="h-5 w-5 text-municipal-accent" />
            <h2 className="text-municipal-text font-bold text-base">
              Zone Details
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md hover:bg-municipal-border transition-colors"
          >
            <X className="h-5 w-5 text-municipal-muted" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5">
          {status === 'loading' && (
            <LoadingSpinner message="Loading zone details…" size="sm" />
          )}

          {status === 'error' && (
            <ErrorState message={error || 'Failed to load zone details'} />
          )}

          {status === 'success' && zone && (
            <div className="space-y-5">
              {/* Zone Name & Risk */}
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-municipal-text text-xl font-bold">
                    {zone.name || zone.id}
                  </h3>
                  <p className="text-municipal-muted text-sm font-mono">
                    ID: {zone.id}
                  </p>
                </div>
                <div
                  className={`px-4 py-2 rounded-lg border ${riskInfo?.borderClass} bg-opacity-20`}
                  style={{
                    backgroundColor: `${riskInfo?.color}20`,
                    borderColor: riskInfo?.color,
                  }}
                >
                  <p
                    className="text-2xl font-bold font-mono"
                    style={{ color: riskInfo?.color }}
                  >
                    {zone.riskScore}
                  </p>
                  <p
                    className="text-xs text-center"
                    style={{ color: riskInfo?.color }}
                  >
                    {riskInfo?.label}
                  </p>
                </div>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-3">
                {zone.detectionCount != null && (
                  <div className="bg-municipal-bg rounded-lg p-3">
                    <div className="flex items-center gap-2 text-municipal-muted text-xs mb-1">
                      <BarChart3 className="h-3 w-3" />
                      Detections
                    </div>
                    <p className="text-municipal-text font-bold text-lg font-mono">
                      {formatNumber(zone.detectionCount)}
                    </p>
                  </div>
                )}

                {zone.lastDetection && (
                  <div className="bg-municipal-bg rounded-lg p-3">
                    <div className="flex items-center gap-2 text-municipal-muted text-xs mb-1">
                      <Clock className="h-3 w-3" />
                      Last Detection
                    </div>
                    <p className="text-municipal-text font-bold text-sm font-mono">
                      {formatTimestamp(zone.lastDetection)}
                    </p>
                  </div>
                )}
              </div>

              {/* Plastic Types */}
              {zone.plasticTypes && zone.plasticTypes.length > 0 && (
                <div>
                  <h4 className="text-municipal-text text-sm font-semibold mb-2 uppercase tracking-wider">
                    Plastic Types Detected
                  </h4>
                  <div className="space-y-2">
                    {zone.plasticTypes.map((pt, idx) => {
                      const maxCount = Math.max(
                        ...zone.plasticTypes!.map((p) => p.count)
                      );
                      const widthPercent =
                        maxCount > 0 ? (pt.count / maxCount) * 100 : 0;

                      return (
                        <div key={pt.type || idx}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-municipal-text">
                              {pt.type}
                            </span>
                            <span className="text-municipal-muted font-mono">
                              {pt.count}
                            </span>
                          </div>
                          <div className="h-2 bg-municipal-bg rounded-full overflow-hidden">
                            <div
                              className="h-full bg-municipal-accent rounded-full transition-all duration-500"
                              style={{ width: `${widthPercent}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Recent Actions */}
              {zone.recentActions && zone.recentActions.length > 0 && (
                <div>
                  <h4 className="text-municipal-text text-sm font-semibold mb-2 uppercase tracking-wider">
                    Recent Actions
                  </h4>
                  <div className="space-y-2">
                    {zone.recentActions.map((action, idx) => (
                      <div
                        key={action.id || idx}
                        className="flex items-center justify-between bg-municipal-bg rounded-lg p-3 text-xs"
                      >
                        <div>
                          <span className="text-municipal-text font-medium">
                            {action.type}
                          </span>
                          <span className="text-municipal-muted ml-2">
                            {formatTimestamp(action.timestamp)}
                          </span>
                        </div>
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-mono ${
                            action.status === 'completed'
                              ? 'bg-green-500/20 text-green-400'
                              : action.status === 'pending'
                                ? 'bg-yellow-500/20 text-yellow-400'
                                : 'bg-gray-500/20 text-gray-400'
                          }`}
                        >
                          {action.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Dispatch Button */}
              <button
                onClick={() => onDispatch(zone.id)}
                className="w-full py-3 bg-municipal-accent hover:bg-blue-600 text-white font-semibold rounded-lg transition-colors text-sm"
              >
                Dispatch Cleanup Team to {zone.name || zone.id}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};