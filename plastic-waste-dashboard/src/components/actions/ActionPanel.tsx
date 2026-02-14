'use client';

import React, { useState } from 'react';
import { Zone, RequestStatus, DispatchActionResponse } from '@/types';
import { Card } from '@/components/common/Card';
import { DispatchButton } from './DispatchButton';
import { CheckCircle2, AlertCircle, Send } from 'lucide-react';
import { formatTimestamp } from '@/utils/formatters';

interface ActionPanelProps {
  zones: Zone[] | null;
  dispatchStatus: RequestStatus;
  dispatchError: string | null;
  dispatchResult: DispatchActionResponse | null;
  onDispatch: (zoneId: string, actionType: string, priority: string, notes: string) => void;
  onReset: () => void;
}

export const ActionPanel: React.FC<ActionPanelProps> = ({
  zones,
  dispatchStatus,
  dispatchError,
  dispatchResult,
  onDispatch,
  onReset,
}) => {
  const [selectedZoneId, setSelectedZoneId] = useState('');
  const [actionType, setActionType] = useState('cleanup');
  const [priority, setPriority] = useState('normal');
  const [notes, setNotes] = useState('');

  const handleSubmit = () => {
    if (!selectedZoneId) return;
    onDispatch(selectedZoneId, actionType, priority, notes);
  };

  return (
    <Card title="Action Center" subtitle="Dispatch & manage operations">
      <div className="space-y-4">
        {/* Success Banner */}
        {dispatchStatus === 'success' && dispatchResult && (
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle2 className="h-5 w-5 text-municipal-green" />
              <span className="text-municipal-green font-semibold text-sm">
                Action Dispatched Successfully
              </span>
            </div>
            <div className="text-xs font-mono text-municipal-muted space-y-1">
              {dispatchResult.id && <p>ID: {dispatchResult.id}</p>}
              {dispatchResult.status && <p>Status: {dispatchResult.status}</p>}
              {dispatchResult.message && <p>{dispatchResult.message}</p>}
              {dispatchResult.estimatedArrival && (
                <p>
                  ETA: {formatTimestamp(dispatchResult.estimatedArrival)}
                </p>
              )}
            </div>
            <button
              onClick={onReset}
              className="mt-3 text-xs text-municipal-accent hover:underline"
            >
              Dispatch another action
            </button>
          </div>
        )}

        {/* Error Banner */}
        {dispatchStatus === 'error' && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-1">
              <AlertCircle className="h-5 w-5 text-municipal-red" />
              <span className="text-municipal-red font-semibold text-sm">
                Dispatch Failed
              </span>
            </div>
            <p className="text-xs text-municipal-muted">
              {dispatchError || 'An unknown error occurred'}
            </p>
            <button
              onClick={onReset}
              className="mt-2 text-xs text-municipal-accent hover:underline"
            >
              Try again
            </button>
          </div>
        )}

        {/* Form */}
        {(dispatchStatus === 'idle' || dispatchStatus === 'loading') && (
          <>
            {/* Zone Selector */}
            <div>
              <label className="block text-municipal-muted text-xs font-mono mb-1.5 uppercase">
                Target Zone
              </label>
              <select
                value={selectedZoneId}
                onChange={(e) => setSelectedZoneId(e.target.value)}
                className="w-full bg-municipal-bg border border-municipal-border rounded-lg px-3 py-2.5 text-municipal-text text-sm font-mono focus:outline-none focus:ring-1 focus:ring-municipal-accent"
                disabled={dispatchStatus === 'loading'}
              >
                <option value="">Select a zone…</option>
                {zones?.map((zone) => (
                  <option key={zone.id} value={zone.id}>
                    {zone.name || zone.id} (Risk: {zone.riskScore})
                  </option>
                ))}
              </select>
            </div>

            {/* Action Type */}
            <div>
              <label className="block text-municipal-muted text-xs font-mono mb-1.5 uppercase">
                Action Type
              </label>
              <select
                value={actionType}
                onChange={(e) => setActionType(e.target.value)}
                className="w-full bg-municipal-bg border border-municipal-border rounded-lg px-3 py-2.5 text-municipal-text text-sm font-mono focus:outline-none focus:ring-1 focus:ring-municipal-accent"
                disabled={dispatchStatus === 'loading'}
              >
                <option value="cleanup">Cleanup Team</option>
                <option value="inspection">Inspection</option>
                <option value="monitoring">Enhanced Monitoring</option>
                <option value="enforcement">Enforcement</option>
              </select>
            </div>

            {/* Priority */}
            <div>
              <label className="block text-municipal-muted text-xs font-mono mb-1.5 uppercase">
                Priority
              </label>
              <div className="flex gap-2">
                {['low', 'normal', 'high', 'critical'].map((level) => (
                  <button
                    key={level}
                    onClick={() => setPriority(level)}
                    disabled={dispatchStatus === 'loading'}
                    className={`flex-1 py-2 rounded-lg text-xs font-mono uppercase transition-colors ${
                      priority === level
                        ? level === 'critical'
                          ? 'bg-red-500/20 border border-red-500 text-red-400'
                          : level === 'high'
                            ? 'bg-orange-500/20 border border-orange-500 text-orange-400'
                            : level === 'normal'
                              ? 'bg-blue-500/20 border border-blue-500 text-blue-400'
                              : 'bg-green-500/20 border border-green-500 text-green-400'
                        : 'bg-municipal-bg border border-municipal-border text-municipal-muted hover:bg-municipal-border'
                    }`}
                  >
                    {level}
                  </button>
                ))}
              </div>
            </div>

            {/* Notes */}
            <div>
              <label className="block text-municipal-muted text-xs font-mono mb-1.5 uppercase">
                Notes (Optional)
              </label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={2}
                className="w-full bg-municipal-bg border border-municipal-border rounded-lg px-3 py-2.5 text-municipal-text text-sm font-mono focus:outline-none focus:ring-1 focus:ring-municipal-accent resize-none"
                placeholder="Additional instructions…"
                disabled={dispatchStatus === 'loading'}
              />
            </div>

            {/* Submit */}
            <DispatchButton
              onClick={handleSubmit}
              isLoading={dispatchStatus === 'loading'}
              disabled={!selectedZoneId}
            />
          </>
        )}
      </div>
    </Card>
  );
};