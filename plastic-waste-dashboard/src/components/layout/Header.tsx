'use client';

import React, { useState, useEffect } from 'react';
import { Shield, Wifi, WifiOff } from 'lucide-react';

export const Header: React.FC = () => {
  const [currentTime, setCurrentTime] = useState<string>('');
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(
        new Date().toLocaleTimeString('en-US', {
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        })
      );
    };

    updateTime();
    const timer = setInterval(updateTime, 1000);

    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      clearInterval(timer);
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <header className="bg-municipal-card border-b border-municipal-border px-6 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Shield className="h-8 w-8 text-municipal-accent" />
          <div>
            <h1 className="text-municipal-text font-bold text-lg tracking-wide">
              PLASTIC WASTE INTELLIGENCE
            </h1>
            <p className="text-municipal-muted text-xs font-mono">
              Municipal Smart-City Command Center
            </p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            {isOnline ? (
              <Wifi className="h-4 w-4 text-municipal-green" />
            ) : (
              <WifiOff className="h-4 w-4 text-municipal-red" />
            )}
            <span
              className={`text-xs font-mono ${isOnline ? 'text-municipal-green' : 'text-municipal-red'}`}
            >
              {isOnline ? 'CONNECTED' : 'OFFLINE'}
            </span>
          </div>

          <div className="text-municipal-text font-mono text-sm tabular-nums">
            {currentTime}
          </div>
        </div>
      </div>
    </header>
  );
};