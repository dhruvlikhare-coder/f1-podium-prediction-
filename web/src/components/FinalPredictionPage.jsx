import React, { useState, useEffect } from 'react';
import { RotateCcw, Trophy, Award, Sparkles, Cpu, CheckCircle2 } from 'lucide-react';
import confetti from 'canvas-confetti';
import { playTelemetryBeep, revEngine } from '../utils/audioSynth';

export default function FinalPredictionPage({ onRestart }) {
  const [activeTab, setActiveTab] = useState('DRIVERS');

  useEffect(() => {
    // Celebratory confetti upon podium reveal
    confetti({
      particleCount: 100,
      spread: 80,
      origin: { y: 0.55 },
      colors: ['#E10600', '#FF8000', '#0070F3', '#FFD166', '#FFFFFF']
    });
    revEngine(19000);
  }, []);

  return (
    <div className="relative w-full h-full min-h-screen bg-[#050608] text-white flex flex-col justify-between p-6 md:p-12 z-50 animate-in fade-in zoom-in-95 duration-700 select-none overflow-y-auto">
      {/* Top Header Navigation Tabs matching the layout */}
      <div className="w-full max-w-7xl mx-auto flex items-center justify-between border-b border-white/10 pb-4">
        {/* Left Tabs: DRIVERS / TEAMS */}
        <div className="flex items-center space-x-8 text-sm md:text-base font-bold tracking-wider font-mono">
          <button
            onClick={() => {
              playTelemetryBeep(900);
              setActiveTab('DRIVERS');
            }}
            className={`relative pb-3 uppercase transition-colors ${
              activeTab === 'DRIVERS' ? 'text-white' : 'text-white/40 hover:text-white/70'
            }`}
          >
            <span>DRIVERS</span>
            {activeTab === 'DRIVERS' && (
              <span className="absolute bottom-0 left-0 right-0 h-1 bg-[#E10600] rounded-full shadow-[0_0_10px_#E10600]" />
            )}
          </button>

          <button
            onClick={() => {
              playTelemetryBeep(1000);
              setActiveTab('TEAMS');
            }}
            className={`relative pb-3 uppercase transition-colors ${
              activeTab === 'TEAMS' ? 'text-white' : 'text-white/40 hover:text-white/70'
            }`}
          >
            <span>TEAMS</span>
            {activeTab === 'TEAMS' && (
              <span className="absolute bottom-0 left-0 right-0 h-1 bg-[#E10600] rounded-full shadow-[0_0_10px_#E10600]" />
            )}
          </button>
        </div>

        {/* Right: ML Model Tag & Re-Run Button */}
        <div className="flex items-center space-x-4">
          <div className="hidden sm:flex items-center space-x-2 text-[10px] font-mono text-[#00F5D4] bg-white/5 px-3 py-1.5 rounded-full border border-white/10">
            <Cpu size={12} className="text-[#00F5D4]" />
            <span>ADAPTIVESTACK ENSEMBLE // 2025</span>
          </div>

          <button
            onClick={() => {
              playTelemetryBeep(850);
              onRestart();
            }}
            className="flex items-center space-x-2 px-4 py-2 rounded-full glass-panel border border-white/10 text-xs font-mono text-white/70 hover:text-white hover:border-[#E10600] transition-all cursor-pointer"
          >
            <RotateCcw size={14} className="text-[#E10600]" />
            <span>RE-RUN PREDICTION</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="w-full max-w-7xl mx-auto my-auto py-8">
        {activeTab === 'DRIVERS' ? (
          <div>
            {/* The 3 Podium Cards With Real Model Predictions */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-end">
              {/* 2ND PLACE: Lando Norris (67.4%) */}
              <div className="relative rounded-3xl overflow-hidden bg-gradient-to-br from-[#FF8000]/25 via-[#4A2400]/40 to-[#120900] border border-[#FF8000]/40 p-6 md:p-8 h-[450px] flex flex-col justify-between shadow-[0_20px_50px_rgba(255,128,0,0.25)] group hover:scale-[1.02] transition-all">
                {/* Speed Halftone Texture */}
                <div className="absolute right-0 top-0 bottom-0 w-1/2 opacity-25 bg-[radial-gradient(#FF8000_1.5px,transparent_1.5px)] [background-size:12px_12px] pointer-events-none" />

                {/* Top Info */}
                <div>
                  <div className="text-3xl md:text-4xl font-display font-black text-[#FF8000] mb-2">
                    2<span className="text-xl font-bold text-white/80">ND</span>
                  </div>
                  <h3 className="text-2xl font-bold text-white tracking-tight">Lando Norris</h3>
                  <p className="text-xs font-mono text-[#FF8000]/90 uppercase font-semibold">McLaren F1 Team</p>
                </div>

                {/* Flag & Grid Tag */}
                <div className="my-2 flex items-center space-x-2">
                  <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-white/10 border border-white/20 text-sm" title="Great Britain">
                    🇬🇧
                  </span>
                  <span className="text-[10px] font-mono text-white/50 px-2 py-0.5 rounded-md bg-white/5 border border-white/10">
                    GRID P2
                  </span>
                </div>

                {/* Driver Photo Cutout */}
                <div className="absolute right-2 bottom-0 w-52 h-72 pointer-events-none flex items-end justify-center">
                  <img
                    src="/drivers/2026mclarenlannor01right.avif"
                    alt="Lando Norris"
                    className="max-h-full object-contain filter drop-shadow-[0_10px_20px_rgba(0,0,0,0.8)]"
                    onError={(e) => {
                      e.target.src = "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png";
                    }}
                  />
                </div>

                {/* Bottom Model Output Metrics */}
                <div className="relative z-10 pt-4 border-t border-white/10 space-y-1">
                  <div className="text-3xl md:text-4xl font-display font-black text-white">
                    67.4% <span className="text-xs font-mono text-[#FF8000] font-bold">PODIUM PROB</span>
                  </div>
                  <div className="text-[10px] font-mono text-white/50 flex space-x-3">
                    <span>Stacking: 73.9%</span>
                    <span>Soft Vote: 64.6%</span>
                  </div>
                </div>
              </div>

              {/* 1ST PLACE (CENTER, ELEVATED WINNER): Max Verstappen (79.9%) */}
              <div className="relative rounded-3xl overflow-hidden bg-gradient-to-br from-[#1C3D82] via-[#0E2454] to-[#06112C] border-2 border-[#FFD166]/60 p-6 md:p-8 h-[495px] flex flex-col justify-between shadow-[0_25px_60px_rgba(28,61,130,0.7)] ring-2 ring-[#FFD166]/30 group hover:scale-[1.03] transition-all z-10">
                {/* Speed Halftone Texture */}
                <div className="absolute right-0 top-0 bottom-0 w-1/2 opacity-35 bg-[radial-gradient(#3B82F6_1.5px,transparent_1.5px)] [background-size:12px_12px] pointer-events-none" />

                {/* Winner Gold Badge */}
                <div className="absolute top-4 right-4 flex items-center space-x-1.5 px-3.5 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-[#FFD166] text-[#FFD166] text-[11px] font-mono font-bold uppercase shadow-[0_0_15px_rgba(255,209,102,0.4)]">
                  <Trophy size={14} className="text-[#FFD166]" />
                  <span>PREDICTED WINNER</span>
                </div>

                {/* Top Info */}
                <div>
                  <div className="text-4xl md:text-5xl font-display font-black text-[#FFD166] mb-2">
                    1<span className="text-2xl font-bold text-white">ST</span>
                  </div>
                  <h3 className="text-2xl md:text-3xl font-bold text-white tracking-tight">Max Verstappen</h3>
                  <p className="text-xs font-mono text-blue-300 uppercase font-semibold">Red Bull Racing</p>
                </div>

                {/* Flag & Grid Tag */}
                <div className="my-2 flex items-center space-x-2">
                  <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-white/10 border border-white/20 text-sm" title="Netherlands">
                    🇳🇱
                  </span>
                  <span className="text-[10px] font-mono text-[#FFD166] px-2 py-0.5 rounded-md bg-[#FFD166]/10 border border-[#FFD166]/30 font-bold">
                    POLE POSITION (P1)
                  </span>
                </div>

                {/* Driver Photo Cutout */}
                <div className="absolute right-2 bottom-0 w-56 h-80 pointer-events-none flex items-end justify-center">
                  <img
                    src="/drivers/2026redbullracingmaxver01right.avif"
                    alt="Max Verstappen"
                    className="max-h-full object-contain filter drop-shadow-[0_10px_20px_rgba(0,0,0,0.8)]"
                    onError={(e) => {
                      e.target.src = "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png";
                    }}
                  />
                </div>

                {/* Bottom Model Output Metrics */}
                <div className="relative z-10 pt-4 border-t border-white/15 space-y-1">
                  <div className="text-4xl md:text-5xl font-display font-black text-white">
                    79.9% <span className="text-sm font-mono text-[#00F5D4] font-bold">PODIUM PROB</span>
                  </div>
                  <div className="text-[11px] font-mono text-white/60 flex space-x-3">
                    <span className="text-[#FFD166] font-semibold">Stacking (LR): 89.4%</span>
                    <span>Soft Vote: 79.7%</span>
                  </div>
                </div>
              </div>

              {/* 3RD PLACE: Charles Leclerc (64.3%) */}
              <div className="relative rounded-3xl overflow-hidden bg-gradient-to-br from-[#E10600]/30 via-[#660400]/40 to-[#180000] border border-[#E10600]/40 p-6 md:p-8 h-[450px] flex flex-col justify-between shadow-[0_20px_50px_rgba(225,6,0,0.3)] group hover:scale-[1.02] transition-all">
                {/* Speed Halftone Texture */}
                <div className="absolute right-0 top-0 bottom-0 w-1/2 opacity-25 bg-[radial-gradient(#E10600_1.5px,transparent_1.5px)] [background-size:12px_12px] pointer-events-none" />

                {/* Top Info */}
                <div>
                  <div className="text-3xl md:text-4xl font-display font-black text-[#E10600] mb-2">
                    3<span className="text-xl font-bold text-white/80">RD</span>
                  </div>
                  <h3 className="text-2xl font-bold text-white tracking-tight">Charles Leclerc</h3>
                  <p className="text-xs font-mono text-red-200/90 uppercase font-semibold">Scuderia Ferrari</p>
                </div>

                {/* Flag & Grid Tag */}
                <div className="my-2 flex items-center space-x-2">
                  <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-white/10 border border-white/20 text-sm" title="Monaco">
                    🇲🇨
                  </span>
                  <span className="text-[10px] font-mono text-white/50 px-2 py-0.5 rounded-md bg-white/5 border border-white/10">
                    GRID P3
                  </span>
                </div>

                {/* Driver Photo Cutout */}
                <div className="absolute right-2 bottom-0 w-52 h-72 pointer-events-none flex items-end justify-center">
                  <img
                    src="/drivers/2026ferrarichalec01right.avif"
                    alt="Charles Leclerc"
                    className="max-h-full object-contain filter drop-shadow-[0_10px_20px_rgba(0,0,0,0.8)]"
                    onError={(e) => {
                      e.target.src = "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png";
                    }}
                  />
                </div>

                {/* Bottom Model Output Metrics */}
                <div className="relative z-10 pt-4 border-t border-white/10 space-y-1">
                  <div className="text-3xl md:text-4xl font-display font-black text-white">
                    64.3% <span className="text-xs font-mono text-[#E10600] font-bold">PODIUM PROB</span>
                  </div>
                  <div className="text-[10px] font-mono text-white/50 flex space-x-3">
                    <span>Stacking: 66.7%</span>
                    <span>Soft Vote: 59.5%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Model Validation & Hit Rate Banner */}
            <div className="mt-8 flex flex-wrap items-center justify-between p-4 rounded-2xl glass-panel border border-white/10 text-xs font-mono text-white/60">
              <div className="flex items-center space-x-2 text-[#00F5D4]">
                <CheckCircle2 size={16} />
                <span className="font-bold">ADAPTIVESTACK ENSEMBLE PREDICTIONS CONVERGED</span>
              </div>
              <div className="flex items-center space-x-6">
                <span>HISTORICAL ROC-AUC: <strong className="text-white">0.948</strong></span>
                <span>LOG LOSS: <strong className="text-white">0.207</strong></span>
                <span>PODIUM HIT RATE: <strong className="text-white">75.0%</strong></span>
              </div>
            </div>
          </div>
        ) : (
          /* TEAMS CONSTRUCTOR TAB */
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="glass-panel p-8 rounded-3xl border-2 border-[#FFD166]/50 space-y-4 shadow-xl">
              <div className="text-4xl font-display font-black text-[#FFD166]">1ST</div>
              <h3 className="text-2xl font-bold text-white">Oracle Red Bull Racing</h3>
              <p className="text-xs font-mono text-white/50">Max Verstappen + Sergio Perez</p>
              <div className="text-3xl font-display font-bold text-white pt-4 border-t border-white/10">
                79.9% <span className="text-xs font-mono text-[#00F5D4]">AVG PODIUM INDEX</span>
              </div>
            </div>

            <div className="glass-panel p-8 rounded-3xl border border-[#FF8000]/40 space-y-4 shadow-xl">
              <div className="text-4xl font-display font-black text-[#FF8000]">2ND</div>
              <h3 className="text-2xl font-bold text-white">McLaren Formula 1 Team</h3>
              <p className="text-xs font-mono text-white/50">Lando Norris + Oscar Piastri</p>
              <div className="text-3xl font-display font-bold text-white pt-4 border-t border-white/10">
                67.4% <span className="text-xs font-mono text-[#00F5D4]">AVG PODIUM INDEX</span>
              </div>
            </div>

            <div className="glass-panel p-8 rounded-3xl border border-[#E10600]/40 space-y-4 shadow-xl">
              <div className="text-4xl font-display font-black text-[#E10600]">3RD</div>
              <h3 className="text-2xl font-bold text-white">Scuderia Ferrari</h3>
              <p className="text-xs font-mono text-white/50">Charles Leclerc + Carlos Sainz</p>
              <div className="text-3xl font-display font-bold text-white pt-4 border-t border-white/10">
                64.3% <span className="text-xs font-mono text-[#00F5D4]">AVG PODIUM INDEX</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Info */}
      <footer className="w-full max-w-7xl mx-auto flex items-center justify-between text-[11px] font-mono text-white/40 border-t border-white/10 pt-4">
        <span>F1 RACE WINNER & PODIUM PREDICTION SYSTEM</span>
        <span>EVALUATED ACROSS 2024-2025 SEASONS</span>
      </footer>
    </div>
  );
}
