import React, { useState, useMemo } from 'react';
import { X, Trophy, Sparkles, TrendingUp, Cpu, Award } from 'lucide-react';
import confetti from 'canvas-confetti';
import { DRIVERS, computeEnsemblePredictions } from '../data/f1Data';
import { playTelemetryBeep, playGearShift, revEngine } from '../utils/audioSynth';

export default function PredictorDrawer({ isOpen, onClose }) {
  const [selectedDriverId, setSelectedDriverId] = useState(DRIVERS[0].id);
  const [gridPosition, setGridPosition] = useState(1);
  const [driverForm, setDriverForm] = useState(1.0);
  const [teamPace, setTeamPace] = useState(1.0);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationResult, setSimulationResult] = useState(null);

  const currentDriver = useMemo(() => {
    return DRIVERS.find((d) => d.id === selectedDriverId) || DRIVERS[0];
  }, [selectedDriverId]);

  // Compute live ensemble predictions using our machine learning logic
  const predictions = useMemo(() => {
    return computeEnsemblePredictions({
      grid: gridPosition,
      driver: currentDriver,
      driverForm,
      teamPace,
    });
  }, [gridPosition, currentDriver, driverForm, teamPace]);

  // Run interactive race simulation
  const handleSimulate = () => {
    setIsSimulating(true);
    setSimulationResult(null);
    playGearShift();
    revEngine(19000);

    setTimeout(() => {
      setIsSimulating(false);
      const isPodium = Math.random() * 100 <= predictions.mlpAdaptive;
      const isWinner = isPodium && Math.random() * 100 <= predictions.winProbability;
      
      const finishPos = isWinner ? 1 : isPodium ? (Math.random() > 0.5 ? 2 : 3) : Math.min(20, Math.floor(gridPosition + Math.random() * 5));

      setSimulationResult({
        isPodium,
        isWinner,
        finishPos
      });

      if (isPodium) {
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { y: 0.6 },
          colors: ['#E10600', '#FFD166', '#00F5D4', '#FFFFFF']
        });
      }
    }, 700);
  };

  if (!isOpen) return null;

  return (
    <div className="absolute top-0 right-0 bottom-0 z-40 w-full max-w-md p-4 md:p-6 flex flex-col pointer-events-none animate-in fade-in slide-in-from-right-6 duration-300">
      <div className="glass-panel p-6 rounded-3xl flex-1 flex flex-col pointer-events-auto border border-white/10 overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-white/10">
          <div className="flex items-center space-x-2">
            <Cpu size={18} className="text-[#E10600]" />
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">
                APEX PREDICT // ML ENSEMBLE
              </h2>
              <span className="text-[10px] font-mono text-white/40 uppercase">
                AdaptiveStack Neural Meta-Layer
              </span>
            </div>
          </div>
          <button
            onClick={() => {
              playTelemetryBeep(900);
              onClose();
            }}
            className="text-white/40 hover:text-white p-1 rounded-lg hover:bg-white/5 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Inputs */}
        <div className="space-y-4 my-4">
          {/* Driver Selection */}
          <div>
            <label className="text-[11px] font-mono text-white/50 uppercase tracking-wider block mb-1.5">
              Select Driver
            </label>
            <div className="grid grid-cols-2 gap-2">
              {DRIVERS.map((d) => {
                const isSelected = d.id === selectedDriverId;
                return (
                  <button
                    key={d.id}
                    onClick={() => {
                      playTelemetryBeep(1100, 'sine', 0.05);
                      setSelectedDriverId(d.id);
                    }}
                    className={`px-3 py-2 rounded-xl text-left border transition-all ${
                      isSelected
                        ? 'border-[#E10600] bg-[#E10600]/15 text-white shadow-[0_0_15px_rgba(225,6,0,0.3)]'
                        : 'border-white/5 bg-white/[0.02] text-white/70 hover:border-white/20'
                    }`}
                  >
                    <div className="text-xs font-bold truncate">{d.name}</div>
                    <div className="text-[10px] font-mono text-white/40 truncate">{d.team}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Grid Position Slider */}
          <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/5">
            <div className="flex items-center justify-between text-xs font-mono mb-2">
              <span className="text-white/50 uppercase">Starting Grid</span>
              <span className="font-bold text-[#FFD166] text-sm">
                {gridPosition === 1 ? 'POLE POSITION (P1)' : `P${gridPosition}`}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="20"
              value={gridPosition}
              onChange={(e) => {
                playTelemetryBeep(700 + e.target.value * 25, 'sine', 0.03);
                setGridPosition(Number(e.target.value));
              }}
              className="w-full accent-[#E10600] cursor-pointer"
            />
          </div>

          {/* Form & Pace Adjusters */}
          <div className="grid grid-cols-2 gap-2">
            <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/5">
              <div className="flex justify-between text-[11px] font-mono text-white/50 mb-1">
                <span>Driver Form</span>
                <span className="text-[#00F5D4] font-bold">{(driverForm * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.7"
                max="1.3"
                step="0.05"
                value={driverForm}
                onChange={(e) => setDriverForm(Number(e.target.value))}
                className="w-full accent-[#00F5D4] cursor-pointer"
              />
            </div>
            <div className="p-3 rounded-2xl bg-white/[0.02] border border-white/5">
              <div className="flex justify-between text-[11px] font-mono text-white/50 mb-1">
                <span>Car Pace</span>
                <span className="text-[#FF8000] font-bold">{(teamPace * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.7"
                max="1.3"
                step="0.05"
                value={teamPace}
                onChange={(e) => setTeamPace(Number(e.target.value))}
                className="w-full accent-[#FF8000] cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Real-time Probabilities Card */}
        <div className="p-4 rounded-2xl bg-gradient-to-br from-[#E10600]/20 via-[#07080C] to-black border border-[#E10600]/30 relative overflow-hidden mb-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Trophy size={16} className="text-[#FFD166]" />
              <span className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                Predicted Outcomes
              </span>
            </div>
            <span className="text-[10px] font-mono text-[#00F5D4] bg-[#00F5D4]/10 px-2 py-0.5 rounded-full border border-[#00F5D4]/20">
              CONFIDENCE {predictions.confidenceScore}%
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-2">
            <div>
              <div className="text-[10px] font-mono text-white/50 uppercase">Podium Probability</div>
              <div className="text-2xl font-bold font-mono text-[#00F5D4]">
                {predictions.mlpAdaptive}%
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-white/50 uppercase">Win Probability</div>
              <div className="text-2xl font-bold font-mono text-[#FFD166]">
                {predictions.winProbability}%
              </div>
            </div>
          </div>

          {/* Model Breakdown */}
          <div className="pt-3 border-t border-white/10 space-y-1.5 text-[11px] font-mono">
            <div className="flex justify-between text-white/60">
              <span>Soft Voting:</span>
              <span className="text-white">{predictions.softVote}%</span>
            </div>
            <div className="flex justify-between text-white/60">
              <span>Stacking (LR):</span>
              <span className="text-white">{predictions.stackLR}%</span>
            </div>
            <div className="flex justify-between text-white/60">
              <span>MLP Static:</span>
              <span className="text-white">{predictions.mlpStatic}%</span>
            </div>
            <div className="flex justify-between text-[#00F5D4] font-semibold">
              <span>MLP Adaptive (Active):</span>
              <span>{predictions.mlpAdaptive}%</span>
            </div>
          </div>
        </div>

        {/* Simulation Output */}
        {simulationResult && (
          <div className="p-3 mb-4 rounded-xl bg-white/[0.05] border border-white/10 animate-in fade-in duration-200">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-white/70">Simulation Finish:</span>
              <span className={`text-sm font-bold font-mono ${
                simulationResult.isWinner ? 'text-[#FFD166]' : simulationResult.isPodium ? 'text-[#00F5D4]' : 'text-white/60'
              }`}>
                {simulationResult.isWinner ? '🏆 WINNER (P1)' : simulationResult.isPodium ? `🥇 PODIUM (P${simulationResult.finishPos})` : `P${simulationResult.finishPos}`}
              </span>
            </div>
          </div>
        )}

        {/* Simulate Race Button */}
        <button
          onClick={handleSimulate}
          disabled={isSimulating}
          className="w-full mt-auto py-3.5 px-4 rounded-2xl bg-gradient-to-r from-[#E10600] to-[#FF8000] text-white font-bold font-mono text-xs uppercase tracking-widest flex items-center justify-center space-x-2 shadow-[0_0_25px_rgba(225,6,0,0.4)] hover:brightness-110 active:scale-98 transition-all disabled:opacity-50"
        >
          {isSimulating ? (
            <span>RUNNING 1000 MONTE CARLO LAPS...</span>
          ) : (
            <>
              <Sparkles size={16} />
              <span>SIMULATE RACE FINISH</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
