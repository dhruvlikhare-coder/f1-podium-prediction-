import React, { useState, useEffect, useRef } from 'react';
import F1Showcase3D from './components/F1Showcase3D';
import ShowcaseVideo from './components/ShowcaseVideo';
import AeroHUDOverlay from './components/AeroHUDOverlay';
import SkeletonLoadingOverlay from './components/SkeletonLoadingOverlay';
import FinalPredictionPage from './components/FinalPredictionPage';
import { ChevronDown, Sparkles, Volume2, VolumeX, ArrowRight, Quote } from 'lucide-react';
import { startV10Engine, stopV10Engine, revEngine, playTelemetryBeep } from './utils/audioSynth';

export default function App() {
  const [activeSection, setActiveSection] = useState(0);
  const [isAudioEnabled, setIsAudioEnabled] = useState(false);
  const [quoteIdx, setQuoteIdx] = useState(0);

  // States for Page 3 prediction sequence
  const [isPredictionRunning, setIsPredictionRunning] = useState(false);
  const [isPredictionCompleted, setIsPredictionCompleted] = useState(false);

  const isScrollingRef = useRef(false);

  const quotes = [
    {
      text: "Aerodynamics are for people who can't build engines.",
      author: "Enzo Ferrari"
    },
    {
      text: "To achieve anything in this game, you must be prepared to dabble on the boundary of disaster.",
      author: "Sir Stirling Moss"
    },
    {
      text: "When you're racing, you're living. Everything that happens before or after is just waiting.",
      author: "Steve McQueen"
    }
  ];

  // Mouse wheel listener
  useEffect(() => {
    const handleWheel = (e) => {
      if (isPredictionRunning || isPredictionCompleted) return;
      if (isScrollingRef.current) return;

      if (e.deltaY > 35) {
        if (activeSection < 2) {
          isScrollingRef.current = true;
          playTelemetryBeep(950, 'sine', 0.04);
          setActiveSection((prev) => Math.min(prev + 1, 2));
          setTimeout(() => {
            isScrollingRef.current = false;
          }, 850);
        }
      } else if (e.deltaY < -35) {
        if (activeSection > 0) {
          isScrollingRef.current = true;
          playTelemetryBeep(850, 'sine', 0.04);
          setActiveSection((prev) => Math.max(prev - 1, 0));
          setTimeout(() => {
            isScrollingRef.current = false;
          }, 850);
        }
      }
    };

    window.addEventListener('wheel', handleWheel, { passive: true });
    return () => window.removeEventListener('wheel', handleWheel);
  }, [activeSection, isPredictionRunning, isPredictionCompleted]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (isPredictionRunning || isPredictionCompleted) return;

      if (e.key === 'ArrowDown' || e.key === 'PageDown') {
        e.preventDefault();
        setActiveSection((prev) => Math.min(prev + 1, 2));
      } else if (e.key === 'ArrowUp' || e.key === 'PageUp') {
        e.preventDefault();
        setActiveSection((prev) => Math.max(prev - 1, 0));
      } else if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        revEngine(19000);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPredictionRunning, isPredictionCompleted]);

  const toggleAudio = () => {
    if (!isAudioEnabled) {
      startV10Engine();
      setIsAudioEnabled(true);
    } else {
      stopV10Engine();
      setIsAudioEnabled(false);
    }
  };

  const handleStartPrediction = () => {
    playTelemetryBeep(1200);
    revEngine(18500);
    setIsPredictionRunning(true);
  };

  const handleLoadingComplete = () => {
    setIsPredictionRunning(false);
    setIsPredictionCompleted(true);
  };

  const handleRestart = () => {
    setIsPredictionCompleted(false);
    setIsPredictionRunning(false);
    setActiveSection(0);
  };

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#050608] text-white select-none font-sans">
      {/* 3D WebGL Canvas Layer */}
      {!isPredictionCompleted && (
        <F1Showcase3D
          activeSection={activeSection}
          isPredictionRunning={isPredictionRunning}
        />
      )}

      {/* Subtle Studio Vignette */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-black/30 pointer-events-none z-10" />

      {/* Floating Top Header */}
      {!isPredictionCompleted && (
        <header
          className={`absolute top-0 left-0 right-0 z-30 p-6 md:p-8 flex items-center justify-between pointer-events-none transition-transform duration-700 ease-in-out ${
            isPredictionRunning ? '-translate-x-[150%] opacity-0' : 'translate-x-0 opacity-100'
          }`}
        >
          {/* Scuderia Badge */}
          <div className="pointer-events-auto flex items-center space-x-3">
            <div className="w-2.5 h-2.5 rounded-full bg-[#E10600] animate-ping" />
            <span className="text-[11px] font-mono tracking-[0.25em] text-white/70 uppercase font-semibold">
              SCUDERIA FERRARI // F2004
            </span>
          </div>

          {/* Audio Controls */}
          <div className="pointer-events-auto flex items-center space-x-3">
            <button
              onClick={() => revEngine(18800)}
              className="px-3.5 py-1.5 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-[11px] font-mono text-white/70 hover:text-white transition-all flex items-center space-x-1.5 shadow-sm cursor-pointer"
              title="Rev Ferrari V10 Engine"
            >
              <Sparkles size={12} className="text-[#E10600]" />
              <span className="hidden sm:inline">REV V10</span>
            </button>

            <button
              onClick={toggleAudio}
              className={`p-2 rounded-full border transition-all cursor-pointer ${
                isAudioEnabled
                  ? 'bg-[#E10600] border-[#E10600] text-white shadow-[0_0_15px_rgba(225,6,0,0.5)]'
                  : 'bg-white/5 border-white/10 text-white/50 hover:text-white'
              }`}
              title="Toggle Engine Audio"
            >
              {isAudioEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
            </button>
          </div>
        </header>
      )}

      {/* Side Navigation Dots */}
      {!isPredictionCompleted && (
        <nav
          className={`absolute right-6 top-1/2 -translate-y-1/2 z-30 flex flex-col space-y-4 pointer-events-auto transition-transform duration-700 ease-in-out ${
            isPredictionRunning ? 'translate-x-32 opacity-0' : 'translate-x-0 opacity-100'
          }`}
        >
          {[0, 1, 2].map((idx) => {
            const isActive = activeSection === idx;
            const labels = ['01 // HERO', '02 // SHOWCASE', '03 // PREDICT'];
            return (
              <button
                key={idx}
                onClick={() => {
                  playTelemetryBeep(900 + idx * 100);
                  setActiveSection(idx);
                }}
                className="group flex items-center justify-end space-x-3 cursor-pointer"
              >
                <span
                  className={`text-[10px] font-mono tracking-widest transition-all ${
                    isActive ? 'text-[#E2C8A2] font-bold opacity-100' : 'text-white/30 opacity-0 group-hover:opacity-75'
                  }`}
                >
                  {labels[idx]}
                </span>
                <span
                  className={`h-2 rounded-full transition-all duration-300 ${
                    isActive ? 'w-8 bg-[#E10600] shadow-[0_0_10px_#E10600]' : 'w-2 bg-white/20 group-hover:bg-white/50'
                  }`}
                />
              </button>
            );
          })}
        </nav>
      )}

      {/* ─── PAGE 1: HERO (Side View Positioned Exactly Like media_1794666958608_7923f954.jpg) ─ */}
      {!isPredictionCompleted && (
        <section
          className={`absolute inset-0 z-20 flex flex-col justify-between p-6 md:p-12 pointer-events-none transition-all duration-700 ease-in-out ${
            activeSection === 0 && !isPredictionRunning
              ? 'opacity-100 translate-y-0'
              : 'opacity-0 -translate-y-12 pointer-events-none'
          }`}
        >
          {/* Top Title & Ayrton Senna Quote */}
          <div className="pt-16 md:pt-20 max-w-4xl">
            <h1 className="text-6xl sm:text-7xl md:text-8xl lg:text-9xl font-display font-black tracking-tight text-white leading-none">
              FORMULA <span className="text-[#E2C8A2]">1</span>
            </h1>

            <div className="mt-4 md:mt-6 ml-1 md:ml-2 max-w-md text-white/80 font-sans text-sm md:text-base leading-relaxed">
              <p className="italic">
                &ldquo;If you no longer go for a gap that exists, you&rsquo;re no longer a racing driver.&rdquo;
              </p>
              <span className="block text-xs font-mono text-white/50 mt-1 uppercase tracking-wider">
                &mdash; Ayrton Senna
              </span>
            </div>
          </div>

          {/* Right Cursive Script: "Sport or Luxury" */}
          <div className="self-end mr-4 md:mr-16 mb-16 md:mb-20 text-right">
            <div className="font-cursive text-[#D91424] text-6xl sm:text-7xl md:text-8xl lg:text-9xl leading-none drop-shadow-[0_0_30px_rgba(217,20,36,0.4)] select-none">
              <div>Sport</div>
              <div className="text-3xl sm:text-4xl md:text-5xl text-[#D91424]/90 my-1 font-sans italic font-light lowercase tracking-wider">
                or
              </div>
              <div>Luxury</div>
            </div>
          </div>

          {/* Transition Prompt */}
          <div className="self-center pb-2 pointer-events-auto">
            <button
              onClick={() => setActiveSection(1)}
              className="flex flex-col items-center space-y-1 text-white/40 hover:text-white transition-colors cursor-pointer"
            >
              <span className="text-[10px] font-mono tracking-[0.3em] uppercase">TRANSITION</span>
              <ChevronDown size={18} className="animate-bounce" />
            </button>
          </div>
        </section>
      )}

      {/* ─── PAGE 2: TOP-DOWN VIEW + BIGGER VIDEO (0.8X AUDIO) + AERO HUD OVERLAY ─ */}
      {!isPredictionCompleted && (
        <section
          className={`absolute inset-0 z-20 flex items-center p-6 md:p-12 pointer-events-none transition-all duration-700 ease-in-out ${
            activeSection === 1 && !isPredictionRunning
              ? 'opacity-100 translate-y-0'
              : 'opacity-0 translate-y-12 pointer-events-none'
          }`}
        >
          <div className="w-full max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Left Column (7 cols): Bigger Video Box & Quotes */}
            <div className="lg:col-span-7 pointer-events-auto space-y-5">
              {/* Quotes Box */}
              <div
                onClick={() => setQuoteIdx((prev) => (prev + 1) % quotes.length)}
                className="cursor-pointer group glass-panel p-4 md:p-5 rounded-2xl border border-white/10 hover:border-[#E10600]/40 transition-all shadow-lg"
              >
                <div className="flex items-center space-x-2 text-[#E2C8A2] mb-1.5">
                  <Quote size={15} />
                  <span className="text-[10px] font-mono tracking-widest uppercase text-white/50">
                    CLICK FOR NEXT QUOTE
                  </span>
                </div>
                <blockquote className="text-sm md:text-base text-white font-sans font-medium leading-relaxed italic">
                  &ldquo;{quotes[quoteIdx].text}&rdquo;
                </blockquote>
                <div className="text-[11px] font-mono text-[#E10600] font-bold mt-1.5 uppercase tracking-wider">
                  &mdash; {quotes[quoteIdx].author}
                </div>
              </div>

              {/* Bigger Video Box Playing with Audio at 0.8x Speed */}
              <div>
                <ShowcaseVideo isVisible={activeSection === 1} />
              </div>
            </div>

            {/* Right Column (5 cols): Aero Telemetry HUD Overlay beside the 3D Car */}
            <div className="lg:col-span-5 flex justify-end pointer-events-none">
              <AeroHUDOverlay />
            </div>
          </div>

          {/* Proceed to Page 3 */}
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 pointer-events-auto">
            <button
              onClick={() => setActiveSection(2)}
              className="flex flex-col items-center space-y-1 text-white/40 hover:text-white transition-colors cursor-pointer"
            >
              <span className="text-[10px] font-mono tracking-[0.3em] uppercase">PAGE 03 // PREDICTION</span>
              <ChevronDown size={18} className="animate-bounce" />
            </button>
          </div>
        </section>
      )}

      {/* ─── PAGE 3: REAR TIRE FOCUS & START PREDICTION BUTTON ────────────── */}
      {!isPredictionCompleted && (
        <section
          className={`absolute inset-0 z-20 flex flex-col items-center justify-center p-6 pointer-events-none transition-all duration-700 ease-in-out ${
            activeSection === 2 && !isPredictionRunning
              ? 'opacity-100 translate-x-0'
              : isPredictionRunning
              ? '-translate-x-[150%] opacity-0 pointer-events-none'
              : 'opacity-0 translate-y-12 pointer-events-none'
          }`}
        >
          <div className="text-center pointer-events-auto flex flex-col items-center space-y-8">
            <div className="space-y-2">
              <div className="text-xs font-mono tracking-[0.4em] text-[#E10600] uppercase font-bold">
                MACHINE LEARNING APEX ENGINE
              </div>
              <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-black text-white tracking-tight">
                RACE OUTCOME & PODIUM PREDICTIONS
              </h2>
              <p className="text-xs sm:text-sm font-mono text-white/50 max-w-md mx-auto">
                Rear axle torque & tire degradation telemetry ready for Monte Carlo simulation
              </p>
            </div>

            {/* White Pill Button: START PREDICTION */}
            <button
              onClick={handleStartPrediction}
              className="group relative px-12 md:px-16 py-4 md:py-5 rounded-full bg-white text-[#5E4738] hover:text-black font-display font-black text-lg md:text-xl tracking-wider uppercase transition-all duration-300 hover:scale-105 active:scale-95 shadow-[0_0_50px_rgba(255,255,255,0.4)] hover:shadow-[0_0_70px_rgba(225,6,0,0.7)] flex items-center space-x-3 cursor-pointer"
            >
              <span>START PREDICTION</span>
              <ArrowRight size={20} className="group-hover:translate-x-1.5 transition-transform" />
            </button>
          </div>
        </section>
      )}

      {/* ─── 8-10s SKELETON WIREFRAME & 80% NEON BLUE LOADING PROGRESSION ─── */}
      {isPredictionRunning && (
        <SkeletonLoadingOverlay onComplete={handleLoadingComplete} />
      )}

      {/* ─── FINAL PREDICTION PODIUM PAGE (Exact Model Predictions: VER, NOR, LEC) ─ */}
      {isPredictionCompleted && (
        <FinalPredictionPage onRestart={handleRestart} />
      )}
    </div>
  );
}
