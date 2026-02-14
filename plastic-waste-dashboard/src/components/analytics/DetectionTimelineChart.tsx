'use client';

import React, { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { formatTimestamp } from '@/utils/formatters';

interface DetectionTimelineChartProps {
  data: Array<{
    timestamp: string;
    count: number;
    [key: string]: unknown;
  }>;
}

export const DetectionTimelineChart: React.FC<DetectionTimelineChartProps> = ({
  data,
}) => {
  const chartData = useMemo(() => {
    return data.map((entry) => ({
      ...entry,
      formattedTime: formatTimestamp(entry.timestamp, 'HH:mm'),
    }));
  }, [data]);

  return (
    <div>
      <h4 className="text-municipal-text text-xs font-semibold uppercase tracking-wider mb-3">
        Detection Count Over Time
      </h4>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="detectionGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="formattedTime"
            stroke="#6b7280"
            tick={{ fontSize: 10, fontFamily: 'monospace' }}
            axisLine={{ stroke: '#1f2937' }}
          />
          <YAxis
            stroke="#6b7280"
            tick={{ fontSize: 10, fontFamily: 'monospace' }}
            axisLine={{ stroke: '#1f2937' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#111827',
              border: '1px solid #1f2937',
              borderRadius: '8px',
              color: '#e5e7eb',
              fontSize: '12px',
              fontFamily: 'monospace',
            }}
            formatter={(value: number) => [`${value} detections`, 'Count']}
          />
          <Area
            type="monotone"
            dataKey="count"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#detectionGradient)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};