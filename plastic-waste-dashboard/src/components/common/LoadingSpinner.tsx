import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

const sizeMap = {
  sm: 'h-6 w-6',
  md: 'h-10 w-10',
  lg: 'h-16 w-16',
};

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Loading data…',
  size = 'md',
}) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-4">
      <div
        className={`${sizeMap[size]} border-4 border-municipal-border border-t-municipal-accent rounded-full animate-spin`}
      />
      <p className="text-municipal-muted text-sm font-mono">{message}</p>
    </div>
  );
};