import React from 'react';
import { X, ChevronRight, Zap } from 'lucide-react';
import { playTelemetryBeep } from '../utils/audioSynth';

export default function HotspotPanel({ hotspot, onClose }) {
  if (!hotspot) return null;

  return (
    <div className="absolute top-20 left-4 md:left-6 z-40 max-w-sm w-full animate-in fade-in slide-in-from-left-4 duration-300">
      <div className="glass-panel-glow p-5 rounded-2xl relative overflow-hidden">
        {/* Top telemetry line */}
        <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-3">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-[#00F5D4] animate-ping" />
            <span className="text-[10px] font-mono tracking-widest text-[#00F5D4] uppercase font-bold">
              {hotspot.category}
            </span>
          </div>
          <button
            onClick={() => {
              playTelemetryBeep(900);
              onClose();
            }}
            className="text-white/40 hover:text-white p-1 rounded-lg hover:bg-white/5 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Title */}
        <h3 className="text-xl font-bold tracking-tight text-white mb-2">
          {hotspot.title}
        </h3>

        {/* Spec Pill */}
        <div className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-[#E10600]/20 border border-[#E10600]/40 text-[#FFD166] font-mono text-xs font-semibold mb-3">
          <Zap size={13} className="text-[#FFD166]" />
          <span>{hotspot.spec}</span>
        </div>

        {/* Description */}
        <p className="text-xs text-white/70 leading-relaxed font-sans mb-4">
          {hotspot.description}
        </p>

        {/* Action Button */}
        <div className="pt-2 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-white/40">
          <span>COORDINATES: [{hotspot.position.join(', ')}]</span>
          <span className="text-[#00F5D4] flex items-center gap-0.5">
            INSPECTED <ChevronRight size={12} />
          </span>
        </div>
      </div>
    </div>
  );
}
