import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, Sparkles, Gauge } from 'lucide-react';

export default function ShowcaseVideo({ isVisible }) {
  const [isPlaying, setIsPlaying] = useState(true);
  const [isMuted, setIsMuted] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // Set video playback to normal speed (1.0x)
    video.playbackRate = 1.0;

    if (isVisible) {
      video.currentTime = 0;
      video.play().catch(() => {
        // If autoplay with audio is restricted by browser policy, mute and retry
        video.muted = true;
        setIsMuted(true);
        video.play().catch(() => {});
      });
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }, [isVisible]);

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.playbackRate = 1.0;
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    const nextMuted = !videoRef.current.muted;
    videoRef.current.muted = nextMuted;
    setIsMuted(nextMuted);
    if (!nextMuted) {
      videoRef.current.playbackRate = 1.0;
      videoRef.current.play().catch(() => {});
    }
  };

  return (
    <div className="relative w-full aspect-[16/10] md:aspect-video rounded-3xl overflow-hidden glass-panel border border-white/15 group shadow-[0_30px_70px_rgba(0,0,0,0.9)]">
      {/* Uploaded F1 Showcase Video playing on loop with audio at 0.8x speed */}
      <video
        ref={videoRef}
        src="/videos/f1_video.mp4"
        className="w-full h-full object-cover scale-[1.02]"
        autoPlay
        loop
        playsInline
        onPlay={() => {
          if (videoRef.current) videoRef.current.playbackRate = 1.0;
        }}
      />

      {/* Cinematic Vignette */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-transparent to-black/30 pointer-events-none" />

      {/* Top Telemetry Header */}
      <div className="absolute top-4 left-4 flex items-center space-x-2.5">
        <span className="w-2.5 h-2.5 rounded-full bg-[#E10600] animate-ping" />
        <span className="text-[11px] font-mono tracking-widest text-white uppercase font-bold">
          FERRARI F2004 // LIVE TRACK TELEMETRY
        </span>
      </div>

      {/* Normal Speed Indicator Tag */}
      <div className="absolute top-4 right-4 flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-black/60 backdrop-blur-md border border-white/15 text-[10px] font-mono text-[#00F5D4]">
        <Gauge size={12} className="text-[#00F5D4]" />
        <span>1.0X REALTIME</span>
      </div>

      {/* Audio Status Banner if Muted by browser */}
      {isMuted && (
        <button
          onClick={toggleMute}
          className="absolute top-14 right-4 z-20 px-3 py-1 rounded-full bg-[#E10600] text-white text-[10px] font-mono font-bold tracking-wider uppercase animate-pulse shadow-[0_0_15px_#E10600]"
        >
          ENABLE AUDIO 🔊
        </button>
      )}

      {/* Bottom Controls */}
      <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between pointer-events-auto">
        <div className="flex items-center space-x-2.5">
          <button
            onClick={togglePlay}
            className="w-10 h-10 rounded-full bg-white/15 backdrop-blur-md border border-white/25 flex items-center justify-center text-white hover:bg-white hover:text-black transition-all shadow-md"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
          </button>
          <button
            onClick={toggleMute}
            className={`w-10 h-10 rounded-full backdrop-blur-md border flex items-center justify-center transition-all shadow-md ${
              !isMuted
                ? 'bg-[#E10600] text-white border-[#E10600] shadow-[0_0_15px_rgba(225,6,0,0.6)]'
                : 'bg-white/15 text-white border-white/25 hover:bg-white hover:text-black'
            }`}
            title={isMuted ? 'Unmute Audio' : 'Mute Audio'}
          >
            {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
          </button>
        </div>

        <div className="text-[11px] font-mono text-white/70 flex items-center gap-1.5 bg-black/50 px-3.5 py-1.5 rounded-full border border-white/10">
          <Sparkles size={12} className="text-[#E2C8A2]" />
          <span>V10 RAW ONBOARD SYMPHONY</span>
        </div>
      </div>
    </div>
  );
}
