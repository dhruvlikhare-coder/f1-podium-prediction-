import React, { useState, useEffect } from 'react';
import { Cpu, Activity } from 'lucide-react';
import { playTelemetryBeep, revEngine, playGearShift } from '../utils/audioSynth';

export default function SkeletonLoadingOverlay({ onComplete }) {
  const [progress, setProgress] = useState(0);
  const [telemetryStep, setTelemetryStep] = useState(0);

  const steps = [
    "INITIALIZING ADAPTIVESTACK ENSEMBLE (LR + RF + XGBOOST)...",
    "INGESTING HISTORICAL SEASONS & TRACK TELEMETRY...",
    "SIMULATING 10,000 MONTE CARLO RACE LAPS...",
    "CALCULATING TIRE DEGRADATION & PIT STOP WINDOWS...",
    "COMPUTING APEX SPEEDS & OVERTAKE PROBABILITIES...",
    "FINE-TUNING ADAPTIVE NEURAL META-LAYER WEIGHTS...",
    "PREDICTION MATRIX CONVERGED // PODIUM RESOLVED."
  ];

  // 9-second realistic ML simulation progression
  useEffect(() => {
    const totalDuration = 9000;
    const intervalTime = 50;
    const increment = 100 / (totalDuration / intervalTime);

    playGearShift();
    revEngine(18500);

    const timer = setInterval(() => {
      setProgress((prev) => {
        const next = prev + increment;
        if (next >= 100) {
          clearInterval(timer);
          playTelemetryBeep(1500, 'sine', 0.2);
          setTimeout(() => {
            onComplete();
          }, 350);
          return 100;
        }

        const stepIdx = Math.min(
          Math.floor((next / 100) * steps.length),
          steps.length - 1
        );
        setTelemetryStep((currentStep) => {
          if (currentStep !== stepIdx) {
            playTelemetryBeep(900 + stepIdx * 80, 'sine', 0.04);
            if (stepIdx === 2 || stepIdx === 4) revEngine(19200);
          }
          return stepIdx;
        });

        return next;
      });
    }, intervalTime);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="fixed inset-0 z-40 bg-[#050608]/92 backdrop-blur-md flex flex-col items-center justify-between py-10 px-6 select-none animate-in fade-in duration-500 overflow-hidden">
      {/* Background High-Speed Aerodynamic Streaks moving right to left */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {[20, 35, 50, 65, 80].map((top, i) => (
          <div
            key={i}
            className="absolute h-px bg-gradient-to-l from-transparent via-[#00F5D4]/40 to-transparent w-96 animate-speed-streak"
            style={{
              top: `${top}%`,
              animationDelay: `${i * 0.18}s`,
              animationDuration: `${0.6 + i * 0.15}s`
            }}
          />
        ))}
      </div>

      {/* Top Telemetry Header */}
      <div className="flex items-center space-x-3 text-xs font-mono text-[#00F5D4] bg-white/[0.04] px-4 py-2 rounded-full border border-[#00F5D4]/30 shadow-[0_0_20px_rgba(0,245,212,0.25)] relative z-10">
        <Activity size={14} className="animate-pulse" />
        <span className="tracking-widest uppercase font-bold">
          AI SIMULATION IN PROGRESS // {progress.toFixed(0)}%
        </span>
      </div>

      {/* Center 2D Skeleton Car (Moves Left-Right to give illusion of high-speed velocity) */}
      <div className="relative w-full max-w-4xl h-64 md:h-80 flex items-center justify-center my-auto z-10">
        {/* Glow Aura behind car */}
        <div className="absolute w-[500px] h-36 bg-[#00F5D4]/12 blur-3xl rounded-full pointer-events-none animate-pulse" />

        {/* The Updated User-Uploaded Skeleton Image With Dynamic Left-Right Surging Motion */}
        <div className="relative w-full max-w-3xl flex items-center justify-center animate-car-racing">
          <img
            src="/images/car_skeleton.png?v=2"
            alt="F1 Wireframe Skeleton"
            className="w-full max-h-72 object-contain filter drop-shadow-[0_0_25px_rgba(0,245,212,0.85)] select-none pointer-events-none"
            style={{
              transform: 'rotate(-1.5deg) translateY(-4px)',
            }}
          />
        </div>
      </div>

      {/* 80% WIDTH LONG NEON BLUE LOADING BAR */}
      <div className="w-[80vw] max-w-5xl flex flex-col items-center space-y-3 pb-4 relative z-10">
        {/* Step Text & Percentage */}
        <div className="w-full flex items-center justify-between text-xs md:text-sm font-mono text-white/90">
          <div className="flex items-center space-x-2 text-[#00F5D4] truncate pr-4">
            <Cpu size={15} className="animate-spin-slow shrink-0" />
            <span className="truncate">{steps[telemetryStep]}</span>
          </div>
          <span className="font-bold text-white shrink-0 font-display text-base">
            {progress.toFixed(0)}%
          </span>
        </div>

        {/* The 80% Long Neon Blue Progress Bar */}
        <div className="w-full h-3.5 md:h-4 bg-white/10 rounded-full p-0.5 border border-white/20 shadow-[0_0_25px_rgba(0,112,243,0.35)] overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-[#0070F3] via-[#00DFD8] to-[#00F5D4] transition-all duration-75 relative shadow-[0_0_25px_#00F5D4]"
            style={{ width: `${progress}%` }}
          >
            {/* Trailing Light Pulse */}
            <div className="absolute right-0 top-0 bottom-0 w-8 bg-white/70 blur-sm rounded-full" />
          </div>
        </div>

        {/* Sub-telemetry metrics */}
        <div className="w-full flex justify-between text-[10px] font-mono text-white/40">
          <span>MODEL: ADAPTIVESTACK ENSEMBLE</span>
          <span>CALIBRATED WITH HISTORICAL PODIUM DATA</span>
        </div>
      </div>
    </div>
  );
}
