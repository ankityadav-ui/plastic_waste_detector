'use client';

import React from 'react';
import { Header } from './Header';
import { StatusBar } from './StatusBar';

interface DashboardLayoutProps {
  children: React.ReactNode;
  zonesLastUpdated: number | null;
  analyticsLastUpdated: number | null;
  predictionsLastUpdated: number | null;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  zonesLastUpdated,
  analyticsLastUpdated,
  predictionsLastUpdated,
}) => {
  return (
    <div className="min-h-screen bg-municipal-bg flex flex-col">
      <Header />
      <main className="flex-1 p-4 overflow-auto">{children}</main>
      <StatusBar
        zonesLastUpdated={zonesLastUpdated}
        analyticsLastUpdated={analyticsLastUpdated}
        predictionsLastUpdated={predictionsLastUpdated}
      />
    </div>
  );
};