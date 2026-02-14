import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  message,
  onRetry,
}) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4">
      <AlertTriangle className="h-12 w-12 text-municipal-red" />
      <div className="text-center">
        <p className="text-municipal-text font-medium mb-1">
          Data Unavailable
        </p>
        <p className="text-municipal-muted text-sm max-w-md">{message}</p>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="flex items-center gap-2 px-4 py-2 bg-municipal-card border border-municipal-border rounded-lg text-municipal-text hover:bg-municipal-border transition-colors text-sm"
        >
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      )}
    </div>
  );
};