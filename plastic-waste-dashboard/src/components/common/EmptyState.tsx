import React from 'react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  message?: string;
  icon?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  message = 'No data available',
  icon,
}) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 gap-3 text-municipal-muted">
      {icon ?? <Inbox className="h-10 w-10" />}
      <p className="text-sm font-mono">{message}</p>
    </div>
  );
};