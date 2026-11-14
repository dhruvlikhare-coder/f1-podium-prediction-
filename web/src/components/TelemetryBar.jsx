import React from 'react';
import { Camera, Palette, Gauge } from 'lucide-react';
import { CAMERA_PRESETS, LIVERIES } from '../data/f1Data';
import { playTelemetryBeep } from '../utils/audioSynth';

export default function TelemetryBar({
  currentPreset,
  onSelectPreset,
  currentLivery,
  onSelectLivery
}) {
  return (
    <footer className="absolute bottom-0 left-0 right-0 z-30 p-4 md:p-6 pointer-events-none flex flex-col md:flex-row items-end md:items-center justify-between gap-3">
      {/* Telemetry Dashboard Ribbon */}
      <div className="glass-panel px-4 py-2.5 rounded-2xl flex items-center space-x-5 pointer-events-auto border border-white/10">
        <div className="flex items-center space-x-2">
          <Gauge size={16} className="text-[#E10600]" />
          <div>
            <div className="text-[9px] font-mono text-white/40 uppercase">TOP SPEED</div>
            <div className="text-xs font-mono font-bold text-white">348 KM/H</div>
          </div>
        </div>

        <div className="h-6 w-px bg-white/10" />

        <div>
          <div className="text-[9px] font-mono text-white/40 uppercase">GEAR</div>
          <div className="text-xs font-mono font-bold text-[#FFD166]">7TH</div>
        </div>

        <div className="h-6 w-px bg-white/10" />

        <div>
          <div className="text-[9px] font-mono text-white/40 uppercase">V10 RPM</div>
          <div className="text-xs font-mono font-bold text-[#00F5D4]">18,920</div>
        </div>

        <div className="h-6 w-px bg-white/10" />

        <div className="hidden sm:block">
          <div className="text-[9px] font-mono text-white/40 uppercase">DOWNFORCE</div>
          <div className="text-xs font-mono font-bold text-white">1,420 KG</div>
        </div>

        <div className="h-6 w-px bg-white/10 hidden sm:block" />

        <div className="hidden sm:block">
          <div className="text-[9px] font-mono text-white/40 uppercase">TYRE TEMP</div>
          <div className="text-xs font-mono font-bold text-[#FF8000]">104°C</div>
        </div>
      </div>

      {/* Camera Presets & Livery Selector */}
      <div className="flex items-center space-x-2 pointer-events-auto">
        {/* Livery Palette Picker */}
        <div className="glass-panel p-1.5 rounded-2xl flex items-center space-x-1 border border-white/10">
          <div className="px-2 text-[10px] font-mono text-white/40 flex items-center gap-1">
            <Palette size={12} />
          </div>
          {LIVERIES.map((liv) => {
            const isSelected = liv.id === currentLivery.id;
            return (
              <button
                key={liv.id}
                onClick={() => {
                  playTelemetryBeep(1000, 'sine', 0.05);
                  onSelectLivery(liv);
                }}
                className={`w-6 h-6 rounded-full border transition-all ${
                  isSelected
                    ? 'border-white scale-110 shadow-[0_0_10px_rgba(255,255,255,0.5)]'
                    : 'border-white/20 hover:scale-105'
                }`}
                style={{ backgroundColor: liv.hex }}
                title={liv.name}
              />
            );
          })}
        </div>

        {/* Camera Views Preset Selector */}
        <div className="glass-panel p-1 rounded-2xl flex items-center space-x-1 border border-white/10">
          <div className="px-2.5 text-[10px] font-mono text-white/40 flex items-center gap-1">
            <Camera size={12} />
          </div>
          {CAMERA_PRESETS.map((preset) => {
            const isSelected = currentPreset.id === preset.id;
            return (
              <button
                key={preset.id}
                onClick={() => {
                  playTelemetryBeep(850, 'sine', 0.04);
                  onSelectPreset(preset);
                }}
                className={`px-2.5 py-1.5 rounded-xl text-[11px] font-mono transition-all ${
                  isSelected
                    ? 'bg-[#E10600] text-white font-bold shadow-[0_0_12px_rgba(225,6,0,0.5)]'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                {preset.name}
              </button>
            );
          })}
        </div>
      </div>
    </footer>
  );
}
