'use client';

import React from 'react';
import { Send, Loader2 } from 'lucide-react';

interface DispatchButtonProps {
  onClick: () => void;
  isLoading: boolean;
  disabled: boolean;
}

export const DispatchButton: React.FC<DispatchButtonProps> = ({
  onClick,
  isLoading,
  disabled,
}) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled || isLoading}
      className={`w-full flex items-center justify-center gap-2 py-3 rounded-lg font-semibold text-sm transition-all ${
        disabled || isLoading
          ? 'bg-municipal-border text-municipal-muted cursor-not-allowed'
          : 'bg-municipal-accent hover:bg-blue-600 text-white active:scale-[0.98]'
      }`}
    >
      {isLoading ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          Dispatching…
        </>
      ) : (
        <>
          <Send className="h-4 w-4" />
          Dispatch Cleanup Team
        </>
      )}
    </button>
  );
};