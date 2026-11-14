import React, { useState } from 'react';
import { Wind, Gauge, Zap, Activity, ShieldCheck, Flame } from 'lucide-react';
import { playTelemetryBeep } from '../utils/audioSynth';

export default function AeroHUDOverlay() {
  const [activeTab, setActiveTab] = useState('AERO');

  const specs = {
    AERO: [
      { label: "FRONT AXLE DOWNFORCE", value: "530 KG", desc: "Multi-element carbon wing" },
      { label: "REAR DIFFUSER DOWNFORCE", value: "750 KG", desc: "Low pressure ground effect" },
      { label: "AERO BALANCE RATIO", value: "41.5% F / 58.5% R", desc: "Optimized for high-speed apex" },
      { label: "VENTURI FLOOR SUCTION", value: "-1,940 PA", desc: "Diffuser expansion tunnels" }
    ],
    POWERTRAIN: [
      { label: "ENGINE ARCHITECTURE", value: "90° 3.0L V10", desc: "Naturally aspirated Ferrari 053" },
      { label: "MAXIMUM HORSEPOWER", value: "920 BHP", desc: "At 19,000 RPM redline" },
      { label: "VALVETRAIN OPERATION", value: "Pneumatic", desc: "No mechanical valve springs" },
      { label: "POWER-TO-WEIGHT", value: "1,520 BHP / TON", desc: "Dry chassis 605 kg weight" }
    ],
    CHASSIS: [
      { label: "BRAKING FORCE", value: "5.5G DECEL", desc: "Brembo carbon-carbon discs" },
      { label: "0 – 200 KM/H", value: "4.6 SEC", desc: "Longitudinal 7-speed gearbox" },
      { label: "SURVIVAL MONOCOQUE", value: "Carbon-Nomex", desc: "Tailored to M. Schumacher" },
      { label: "TIRE TEMPERATURES", value: "104°C FRONT / 112°C REAR", desc: "Optimal Bridgestone window" }
    ]
  };

  return (
    <div className="w-full max-w-sm flex flex-col space-y-4 pointer-events-auto">
      {/* Top Engineering Badge */}
      <div className="glass-panel p-4 rounded-2xl border border-white/10 space-y-2 shadow-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-[#00F5D4] animate-ping" />
            <span className="text-[10px] font-mono tracking-widest text-[#00F5D4] uppercase font-bold">
              ENGINEERING TELEMETRY HUD
            </span>
          </div>
          <span className="text-[10px] font-mono text-white/40">MARANELLO WIND TUNNEL</span>
        </div>

        {/* Tab Switcher */}
        <div className="grid grid-cols-3 gap-1 bg-white/5 p-1 rounded-xl border border-white/5">
          {['AERO', 'POWERTRAIN', 'CHASSIS'].map((tab) => (
            <button
              key={tab}
              onClick={() => {
                playTelemetryBeep(1000);
                setActiveTab(tab);
              }}
              className={`py-1.5 rounded-lg text-[10px] font-mono font-bold tracking-wider transition-all ${
                activeTab === tab
                  ? 'bg-[#E10600] text-white shadow-[0_0_10px_rgba(225,6,0,0.5)]'
                  : 'text-white/50 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Dynamic Spec Data Cards */}
      <div className="glass-panel p-4 rounded-2xl border border-white/10 space-y-3 shadow-xl">
        {specs[activeTab].map((item, idx) => (
          <div
            key={idx}
            className="p-2.5 rounded-xl bg-white/[0.02] border border-white/5 hover:border-white/20 transition-all flex items-center justify-between"
          >
            <div>
              <div className="text-[9px] font-mono text-white/50 tracking-wider uppercase">
                {item.label}
              </div>
              <div className="text-[11px] font-sans text-white/60">
                {item.desc}
              </div>
            </div>
            <div className="text-right">
              <span className="text-sm font-mono font-bold text-[#E2C8A2] tracking-tight">
                {item.value}
              </span>
            </div>
          </div>
        ))}

        {/* Animated Frequency Telemetry Stream */}
        <div className="pt-2 border-t border-white/5 flex items-center justify-between text-[10px] font-mono text-white/40">
          <div className="flex items-center space-x-1">
            <Activity size={12} className="text-[#00F5D4] animate-pulse" />
            <span>DOWNFORCE LIVE STREAM</span>
          </div>
          <div className="flex items-center space-x-0.5">
            {[40, 70, 95, 60, 85, 100, 50, 75, 90, 65].map((h, i) => (
              <span
                key={i}
                className="w-1 bg-[#00F5D4] rounded-full inline-block animate-pulse"
                style={{
                  height: `${h * 0.16}px`,
                  animationDelay: `${i * 0.1}s`,
                  animationDuration: '1.2s'
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
