import React from 'react';

interface CardProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  headerAction?: React.ReactNode;
  className?: string;
  noPadding?: boolean;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  children,
  headerAction,
  className = '',
  noPadding = false,
}) => {
  return (
    <div
      className={`bg-municipal-card border border-municipal-border rounded-xl overflow-hidden ${className}`}
    >
      <div className="flex items-center justify-between px-5 py-4 border-b border-municipal-border">
        <div>
          <h3 className="text-municipal-text font-semibold text-sm uppercase tracking-wider">
            {title}
          </h3>
          {subtitle && (
            <p className="text-municipal-muted text-xs mt-0.5">{subtitle}</p>
          )}
        </div>
        {headerAction && <div>{headerAction}</div>}
      </div>
      <div className={noPadding ? '' : 'p-5'}>{children}</div>
    </div>
  );
};