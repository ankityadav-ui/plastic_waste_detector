'use client';

import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';

interface PlasticTypeChartProps {
  data: Array<{
    type: string;
    count: number;
    percentage?: number;
    [key: string]: unknown;
  }>;
}

// Color palette derived from municipal theme
const CHART_COLORS = [
  '#3b82f6',
  '#22c55e',
  '#eab308',
  '#ef4444',
  '#f97316',
  '#8b5cf6',
  '#06b6d4',
  '#ec4899',
  '#14b8a6',
  '#f43f5e',
];

export const PlasticTypeChart: React.FC<PlasticTypeChartProps> = ({ data }) => {
  return (
    <div>
      <h4 className="text-municipal-text text-xs font-semibold uppercase tracking-wider mb-3">
        Plastic Type Distribution
      </h4>
      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="type"
            cx="50%"
            cy="50%"
            outerRadius={90}
            innerRadius={45}
            paddingAngle={2}
            stroke="#111827"
            strokeWidth={2}
          >
            {data.map((entry, index) => (
              <Cell
                key={entry.type || index}
                fill={CHART_COLORS[index % CHART_COLORS.length]}
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: '#111827',
              border: '1px solid #1f2937',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '12px',
              fontFamily: 'monospace',
            }}
            formatter={(value: number, name: string) => [
              `${value} detections`,
              name,
            ]}
          />
          <Legend
            wrapperStyle={{
              color: '#6b7280',
              fontSize: '11px',
              fontFamily: 'monospace',
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};