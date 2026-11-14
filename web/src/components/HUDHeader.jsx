import React from 'react';
import { Volume2, VolumeX, Wind, Eye, RotateCw, Activity, Sparkles, SlidersHorizontal } from 'lucide-react';
import { playTelemetryBeep, revEngine } from '../utils/audioSynth';

export default function HUDHeader({
  isAudioEnabled,
  onToggleAudio,
  isWindTunnel,
  onToggleWindTunnel,
  isWireframe,
  onToggleWireframe,
  autoRotate,
  onToggleAutoRotate,
  onTogglePredictor,
  isPredictorOpen
}) {
  return (
    <header className="absolute top-0 left-0 right-0 z-40 p-4 md:p-6 flex items-center justify-between pointer-events-none">
      {/* Brand & Telemetry Badge */}
      <div className="flex items-center space-x-4 pointer-events-auto">
        <div className="flex items-center space-x-2.5 glass-panel px-3.5 py-2 rounded-xl border border-white/10">
          <span className="w-2.5 h-2.5 rounded-full bg-[#E10600] animate-ping" />
          <div className="flex flex-col">
            <span className="text-[10px] font-mono tracking-[0.25em] text-[#E10600] font-black uppercase">
              SCUDERIA LAB
            </span>
            <span className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
              FERRARI F2004 <span className="text-[10px] text-white/40 font-mono">#01</span>
            </span>
          </div>
        </div>

        {/* Live Engine Status */}
        <div className="hidden lg:flex items-center space-x-2 glass-panel px-3 py-2 rounded-xl text-xs font-mono text-white/60">
          <Activity size={14} className="text-[#00F5D4] animate-pulse" />
          <span>V10 TIPO 053 // 19,000 RPM</span>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center space-x-2 pointer-events-auto">
        {/* Rev Sound Button */}
        <button
          onClick={() => {
            playTelemetryBeep(1400);
            revEngine(18800);
          }}
          className="glass-panel px-3 py-2 rounded-xl text-xs font-mono font-medium flex items-center space-x-1.5 hover:border-[#E10600] hover:text-[#E10600] transition-colors group"
          title="Rev V10 Engine Sound"
        >
          <Sparkles size={14} className="text-[#E10600] group-hover:scale-125 transition-transform" />
          <span className="hidden sm:inline">REV V10</span>
        </button>

        {/* Audio Mute/Unmute */}
        <button
          onClick={onToggleAudio}
          className={`glass-panel p-2.5 rounded-xl text-xs font-mono transition-colors ${
            isAudioEnabled ? 'text-[#00F5D4] border-[#00F5D4]/40' : 'text-white/50 hover:text-white'
          }`}
          title="Toggle V10 Audio Ambience"
        >
          {isAudioEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
        </button>

        {/* Wind Tunnel Streamlines */}
        <button
          onClick={onToggleWindTunnel}
          className={`glass-panel p-2.5 rounded-xl text-xs font-mono transition-colors ${
            isWindTunnel ? 'text-[#00F5D4] border-[#00F5D4]/40 bg-[#00F5D4]/10' : 'text-white/50 hover:text-white'
          }`}
          title="Toggle Wind Tunnel Aero Particles"
        >
          <Wind size={16} />
        </button>

        {/* Wireframe Toggle */}
        <button
          onClick={onToggleWireframe}
          className={`glass-panel p-2.5 rounded-xl text-xs font-mono transition-colors ${
            isWireframe ? 'text-[#FFD166] border-[#FFD166]/40 bg-[#FFD166]/10' : 'text-white/50 hover:text-white'
          }`}
          title="Toggle Wireframe Chassis"
        >
          <Eye size={16} />
        </button>

        {/* Auto Rotate */}
        <button
          onClick={onToggleAutoRotate}
          className={`glass-panel p-2.5 rounded-xl text-xs font-mono transition-colors ${
            autoRotate ? 'text-[#00F5D4] border-[#00F5D4]/40' : 'text-white/50 hover:text-white'
          }`}
          title="Auto Rotate 3D Viewport"
        >
          <RotateCw size={16} className={autoRotate ? 'animate-spin-slow' : ''} />
        </button>

        {/* Open ML Predictor Drawer */}
        <button
          onClick={onTogglePredictor}
          className={`glass-panel px-4 py-2 rounded-xl text-xs font-mono font-bold flex items-center space-x-2 transition-all shadow-md ${
            isPredictorOpen
              ? 'bg-[#E10600] text-white border-transparent shadow-[0_0_20px_rgba(225,6,0,0.5)]'
              : 'hover:border-[#E10600] hover:text-[#E10600]'
          }`}
        >
          <SlidersHorizontal size={14} />
          <span>RACE PREDICTOR</span>
        </button>
      </div>
    </header>
  );
}
