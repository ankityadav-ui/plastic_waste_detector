'use client';

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { formatTimestamp } from '@/utils/formatters';
import { getRiskColor } from '@/utils/riskColor';

interface RiskTrendChartProps {
  data: Array<{
    timestamp: string;
    riskLevel: number;
    [key: string]: unknown;
  }>;
}

export const RiskTrendChart: React.FC<RiskTrendChartProps> = ({ data }) => {
  const chartData = useMemo(() => {
    return data.map((entry) => ({
      ...entry,
      formattedTime: formatTimestamp(entry.timestamp, 'MMM dd HH:mm'),
    }));
  }, [data]);

  // Dynamic dot coloring based on risk level
  const CustomDot = (props: any) => {
    const { cx, cy, payload } = props;
    if (!payload) return null;
    const color = getRiskColor(payload.riskLevel);
    return (
      <circle cx={cx} cy={cy} r={4} fill={color} stroke="#111827" strokeWidth={2} />
    );
  };

  return (
    <div>
      <h4 className="text-municipal-text text-xs font-semibold uppercase tracking-wider mb-3">
        Risk Trend (Next 24–48h)
      </h4>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="formattedTime"
            stroke="#6b7280"
            tick={{ fontSize: 10, fontFamily: 'monospace' }}
            axisLine={{ stroke: '#1f2937' }}
          />
          <YAxis
            domain={[0, 100]}
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
            formatter={(value: number) => [`Risk: ${value}`, 'Level']}
          />
          {/* Threshold reference lines */}
          <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="5 5" strokeOpacity={0.3} />
          <ReferenceLine y={60} stroke="#eab308" strokeDasharray="5 5" strokeOpacity={0.3} />
          <ReferenceLine y={80} stroke="#ef4444" strokeDasharray="5 5" strokeOpacity={0.3} />
          <Line
            type="monotone"
            dataKey="riskLevel"
            stroke="#f97316"
            strokeWidth={2}
            dot={<CustomDot />}
            activeDot={{ r: 6, stroke: '#f97316', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};