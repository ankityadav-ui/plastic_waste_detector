'use client';

import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Zone, RequestStatus } from '@/types';
import { Card } from '@/components/common/Card';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorState } from '@/components/common/ErrorState';
import { EmptyState } from '@/components/common/EmptyState';
import { RefreshCw } from 'lucide-react';

// Leaflet must be loaded client-side only
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
);
const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
);

const ZonePolygonDynamic = dynamic(
  () =>
    import('./ZonePolygon').then((mod) => ({
      default: mod.ZonePolygon,
    })),
  { ssr: false }
);

interface ZoneMapProps {
  zones: Zone[] | null;
  status: RequestStatus;
  error: string | null;
  onZoneClick: (zoneId: string) => void;
  onRefresh: () => void;
}

export const ZoneMap: React.FC<ZoneMapProps> = ({
  zones,
  status,
  error,
  onZoneClick,
  onRefresh,
}) => {
  const center = useMemo(
    () => ({
      lat: Number(process.env.NEXT_PUBLIC_MAP_DEFAULT_LAT) || 28.6139,
      lng: Number(process.env.NEXT_PUBLIC_MAP_DEFAULT_LNG) || 77.209,
    }),
    []
  );

  const zoom = Number(process.env.NEXT_PUBLIC_MAP_DEFAULT_ZOOM) || 12;

  const renderContent = () => {
    if (status === 'loading' && !zones) {
      return <LoadingSpinner message="Loading zone data…" />;
    }

    if (status === 'error' && !zones) {
      return <ErrorState message={error || 'Unknown error'} onRetry={onRefresh} />;
    }

    if (!zones || zones.length === 0) {
      return <EmptyState message="No zones available" />;
    }

    return (
      <div className="h-[500px] w-full rounded-lg overflow-hidden relative">
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />
        <MapContainer
          center={[center.lat, center.lng]}
          zoom={zoom}
          className="h-full w-full"
          style={{ background: '#0a0f1a' }}
        >
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          {zones.map((zone) => {
            if (
              !zone.id ||
              !zone.coordinates ||
              !Array.isArray(zone.coordinates)
            ) {
              return null;
            }
            return (
              <ZonePolygonDynamic
                key={zone.id}
                zone={zone}
                onClick={() => onZoneClick(zone.id)}
              />
            );
          })}
        </MapContainer>
      </div>
    );
  };

  return (
    <Card
      title="Zone Map"
      subtitle="Real-time risk overview"
      noPadding
      headerAction={
        <button
          onClick={onRefresh}
          className="p-1.5 rounded-md hover:bg-municipal-border transition-colors"
          title="Refresh zones"
        >
          <RefreshCw
            className={`h-4 w-4 text-municipal-muted ${status === 'loading' ? 'animate-spin' : ''}`}
          />
        </button>
      }
    >
      <div className="p-2">{renderContent()}</div>
    </Card>
  );
};