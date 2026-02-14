'use client';

import React from 'react';
import { Polygon, Tooltip } from 'react-leaflet';
import { Zone } from '@/types';
import { getRiskColor, getRiskLabel, getRiskColorWithOpacity } from '@/utils/riskColor';

interface ZonePolygonProps {
  zone: Zone;
  onClick: () => void;
}

export const ZonePolygon: React.FC<ZonePolygonProps> = ({ zone, onClick }) => {
  const positions = zone.coordinates.map((coord) => [
    coord.lat,
    coord.lng,
  ] as [number, number]);

  if (positions.length < 3) return null;

  const riskColor = getRiskColor(zone.riskScore);
  const fillColor = getRiskColorWithOpacity(zone.riskScore, 0.35);
  const riskLabel = getRiskLabel(zone.riskScore);

  return (
    <Polygon
      positions={positions}
      pathOptions={{
        color: riskColor,
        fillColor: fillColor,
        fillOpacity: 0.4,
        weight: 2,
      }}
      eventHandlers={{
        click: onClick,
      }}
    >
      <Tooltip
        direction="top"
        sticky
        className="!bg-municipal-card !border-municipal-border !text-municipal-text !rounded-lg !px-3 !py-2 !shadow-xl"
      >
        <div className="font-mono text-xs">
          <p className="font-bold">{zone.name || zone.id}</p>
          <p>
            Risk:{' '}
            <span style={{ color: riskColor }}>
              {zone.riskScore} ({riskLabel})
            </span>
          </p>
        </div>
      </Tooltip>
    </Polygon>
  );
};