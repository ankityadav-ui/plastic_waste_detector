'use client';

import React, { useCallback, useState } from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { ZoneMap } from '@/components/zone-map/ZoneMap';
import { ZoneDetailModal } from '@/components/zone-map/ZoneDetailModal';
import { AnalyticsPanel } from '@/components/analytics/AnalyticsPanel';
import { PredictionPanel } from '@/components/predictions/PredictionPanel';
import { ActionPanel } from '@/components/actions/ActionPanel';

import { useZones } from '@/hooks/useZones';
import { useZoneDetail } from '@/hooks/useZoneDetail';
import { useAnalytics } from '@/hooks/useAnalytics';
import { usePredictions } from '@/hooks/usePredictions';
import { useDispatchAction } from '@/hooks/useDispatchAction';

export default function DashboardPage() {
  // ==========================================
  // ALL state derived from API hooks
  // ZERO hardcoded data
  // ==========================================

  const zones = useZones();
  const zoneDetail = useZoneDetail();
  const analytics = useAnalytics();
  const predictions = usePredictions();
  const dispatchAction = useDispatchAction();

  const [showZoneModal, setShowZoneModal] = useState(false);

  // --- Zone Click Handler ---
  const handleZoneClick = useCallback(
    (zoneId: string) => {
      zoneDetail.fetchZoneDetail(zoneId);
      setShowZoneModal(true);
    },
    [zoneDetail]
  );

  // --- Close Zone Modal ---
  const handleCloseZoneModal = useCallback(() => {
    setShowZoneModal(false);
    zoneDetail.clearDetail();
  }, [zoneDetail]);

  // --- Dispatch from Modal ---
  const handleDispatchFromModal = useCallback(
    (zoneId: string) => {
      setShowZoneModal(false);
      zoneDetail.clearDetail();
      // Scroll to action panel or pre-select zone
      const el = document.getElementById('action-panel');
      if (el) {
        el.scrollIntoView({ behavior: 'smooth' });
      }
    },
    [zoneDetail]
  );

  // --- Dispatch Action ---
  const handleDispatch = useCallback(
    async (
      zoneId: string,
      actionType: string,
      priority: string,
      notes: string
    ) => {
      try {
        await dispatchAction.dispatch({
          zoneId,
          actionType,
          priority,
          notes,
        });
        // Refresh zones after successful dispatch
        zones.refetch();
      } catch {
        // Error is captured in hook state
      }
    },
    [dispatchAction, zones]
  );

  return (
    <DashboardLayout
      zonesLastUpdated={zones.lastUpdated}
      analyticsLastUpdated={analytics.lastUpdated}
      predictionsLastUpdated={predictions.lastUpdated}
    >
      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 h-full">
        {/* Left Column: Map (takes 7 of 12 cols on XL) */}
        <div className="xl:col-span-7 space-y-4">
          <ZoneMap
            zones={zones.data}
            status={zones.status}
            error={zones.error}
            onZoneClick={handleZoneClick}
            onRefresh={zones.refetch}
          />

          {/* Analytics below map */}
          <AnalyticsPanel
            data={analytics.data}
            status={analytics.status}
            error={analytics.error}
            onRefresh={analytics.refetch}
          />
        </div>

        {/* Right Column: Predictions + Actions (5 of 12 cols on XL) */}
        <div className="xl:col-span-5 space-y-4">
          <PredictionPanel
            data={predictions.data}
            status={predictions.status}
            error={predictions.error}
            onRefresh={predictions.refetch}
            onZoneClick={handleZoneClick}
          />

          <div id="action-panel">
            <ActionPanel
              zones={zones.data}
              dispatchStatus={dispatchAction.status}
              dispatchError={dispatchAction.error}
              dispatchResult={dispatchAction.data}
              onDispatch={handleDispatch}
              onReset={dispatchAction.reset}
            />
          </div>
        </div>
      </div>

      {/* Zone Detail Modal */}
      {showZoneModal && (
        <ZoneDetailModal
          zone={zoneDetail.data}
          status={zoneDetail.status}
          error={zoneDetail.error}
          onClose={handleCloseZoneModal}
          onDispatch={handleDispatchFromModal}
        />
      )}
    </DashboardLayout>
  );
}