import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { HOTSPOTS } from '../data/f1Data';
import { playTelemetryBeep } from '../utils/audioSynth';

export default function F1Canvas3D({
  cameraPreset,
  selectedHotspot,
  onSelectHotspot,
  activeLivery,
  isWindTunnel,
  isWireframe,
  autoRotate,
  onModelLoaded
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const carModelRef = useRef(null);
  const carMaterialsRef = useRef([]);
  const windParticlesRef = useRef(null);
  const animFrameIdRef = useRef(null);
  const targetCamPosRef = useRef(null);
  const targetLookAtRef = useRef(null);

  const [loadingProgress, setLoadingProgress] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);
  const [hotspotScreenCoords, setHotspotScreenCoords] = useState({});

  // ─── Setup Three.js Scene ────────────────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // 1. Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x07080c);
    scene.fog = new THREE.FogExp2(0x07080c, 0.035);

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / container.clientHeight,
      0.1,
      100
    );
    camera.position.set(3.4, 1.8, 4.0);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
      alpha: true
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 - 0.04; // Don't clip through ground
    controls.minDistance = 1.2;
    controls.maxDistance = 14;
    controls.target.set(0, 0.35, 0);
    controlsRef.current = controls;

    // 5. Lighting Setup (Cinematic F1 Studio Lighting)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xddeeff, 0x111122, 1.2);
    scene.add(hemiLight);

    // Overhead Key Light
    const mainLight = new THREE.DirectionalLight(0xffffff, 2.8);
    mainLight.position.set(5, 8, 5);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.bias = -0.0001;
    scene.add(mainLight);

    // Scuderia Red Rim Light
    const redRimLight = new THREE.DirectionalLight(0xe10600, 2.2);
    redRimLight.position.set(-6, 3, -4);
    scene.add(redRimLight);

    // Telemetry Cyan Fill Light
    const cyanFillLight = new THREE.DirectionalLight(0x00f5d4, 1.4);
    cyanFillLight.position.set(4, 2, -5);
    scene.add(cyanFillLight);

    // Under-chassis soft floor light
    const underGlow = new THREE.PointLight(0xe10600, 1.0, 4);
    underGlow.position.set(0, 0.1, 0);
    scene.add(underGlow);

    // 6. Ground Telemetry Grid
    const gridHelper = new THREE.GridHelper(16, 32, 0xe10600, 0x1d2232);
    gridHelper.position.y = -0.01;
    scene.add(gridHelper);

    // Radial ground disc shadow
    const groundGeo = new THREE.CircleGeometry(7.5, 64);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x07080c,
      roughness: 0.85,
      metalness: 0.2
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.02;
    ground.receiveShadow = true;
    scene.add(ground);

    // Telemetry circular radar rings
    const ringGeo = new THREE.RingGeometry(2.8, 2.82, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xe10600,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.4
    });
    const radarRing = new THREE.Mesh(ringGeo, ringMat);
    radarRing.rotation.x = -Math.PI / 2;
    radarRing.position.y = 0.005;
    scene.add(radarRing);

    // 7. Wind Tunnel Particles (Streamlines)
    const particleCount = 450;
    const particleGeo = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    const particleSpeeds = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      particlePositions[i * 3 + 0] = (Math.random() - 0.5) * 2.2;     // X width
      particlePositions[i * 3 + 1] = Math.random() * 1.3 + 0.1;       // Y height
      particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 8.0;     // Z length
      particleSpeeds[i] = 0.15 + Math.random() * 0.25;
    }
    particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: 0x00f5d4,
      size: 0.045,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending
    });
    const windParticles = new THREE.Points(particleGeo, particleMat);
    windParticles.visible = false;
    scene.add(windParticles);
    windParticlesRef.current = windParticles;

    // 8. Load 3D Model: Ferrari F2004
    const loader = new GLTFLoader();
    loader.load(
      '/models/ferrari_f2004.glb',
      (gltf) => {
        const car = gltf.scene;
        carModelRef.current = car;

        // Auto-center and normalize scale
        const box = new THREE.Box3().setFromObject(car);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        // Target length ~4.6m (typical modern/classic F1 scale)
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 4.5 / maxDim;
        car.scale.set(scale, scale, scale);

        // Center on X/Z, rest perfectly on floor
        box.setFromObject(car);
        const newBox = new THREE.Box3().setFromObject(car);
        car.position.x = -center.x * scale;
        car.position.z = -center.z * scale;
        car.position.y = -newBox.min.y; // Sit on ground

        const materialsList = [];
        car.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.material) {
              // Enhance material shaders for automotive finish
              child.material.envMapIntensity = 1.6;
              child.material.needsUpdate = true;
              materialsList.push(child.material);
            }
          }
        });
        carMaterialsRef.current = materialsList;

        scene.add(car);
        setIsLoaded(true);
        if (onModelLoaded) onModelLoaded();
      },
      (xhr) => {
        if (xhr.total > 0) {
          setLoadingProgress(Math.round((xhr.loaded / xhr.total) * 100));
        } else {
          setLoadingProgress((prev) => Math.min(prev + 15, 95));
        }
      },
      (error) => {
        console.error('Error loading 3D model:', error);
        // Fallback procedural F1 silhouette if GLB fails
        const fallback = createFallbackF1Car();
        scene.add(fallback);
        carModelRef.current = fallback;
        setIsLoaded(true);
      }
    );

    // 9. Animation Loop
    const clock = new THREE.Clock();
    let ringPulse = 0;

    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      const delta = clock.getDelta();

      // Smooth Camera Animation Lerp
      if (targetCamPosRef.current && cameraRef.current) {
        cameraRef.current.position.lerp(targetCamPosRef.current, 0.05);
        if (cameraRef.current.position.distanceTo(targetCamPosRef.current) < 0.02) {
          targetCamPosRef.current = null;
        }
      }
      if (targetLookAtRef.current && controlsRef.current) {
        controlsRef.current.target.lerp(targetLookAtRef.current, 0.05);
        if (controlsRef.current.target.distanceTo(targetLookAtRef.current) < 0.02) {
          targetLookAtRef.current = null;
        }
      }

      // Wind Tunnel Streamlines animation
      if (windParticlesRef.current && windParticlesRef.current.visible) {
        const positions = windParticlesRef.current.geometry.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
          positions[i * 3 + 2] -= particleSpeeds[i]; // Move front to back
          if (positions[i * 3 + 2] < -4.0) {
            positions[i * 3 + 2] = 4.0; // Reset to nose
            positions[i * 3 + 0] = (Math.random() - 0.5) * 2.2;
            positions[i * 3 + 1] = Math.random() * 1.3 + 0.1;
          }
        }
        windParticlesRef.current.geometry.attributes.position.needsUpdate = true;
      }

      // Radar Ring Pulse
      ringPulse += delta * 1.5;
      radarRing.scale.setScalar(1.0 + Math.sin(ringPulse) * 0.05);

      // Project Hotspots to 2D Screen space
      if (cameraRef.current && container) {
        const coords = {};
        HOTSPOTS.forEach((spot) => {
          const vec = new THREE.Vector3(...spot.position);
          vec.project(cameraRef.current);
          // Check if hotspot is facing the camera
          const isBehind = vec.z > 1.0;
          if (!isBehind) {
            const x = ((vec.x + 1) * container.clientWidth) / 2;
            const y = ((-vec.y + 1) * container.clientHeight) / 2;
            coords[spot.id] = { x, y, visible: true };
          } else {
            coords[spot.id] = { visible: false };
          }
        });
        setHotspotScreenCoords(coords);
      }

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 10. Window Resize Handler
    const handleResize = () => {
      if (!container || !renderer || !camera) return;
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // ─── Update Camera Preset ────────────────────────────────────────────────
  useEffect(() => {
    if (!cameraPreset || !cameraRef.current || !controlsRef.current) return;
    targetCamPosRef.current = new THREE.Vector3(...cameraPreset.pos);
    targetLookAtRef.current = new THREE.Vector3(...cameraPreset.target);
  }, [cameraPreset]);

  // ─── Update Camera When Hotspot Clicked ───────────────────────────────────
  useEffect(() => {
    if (!selectedHotspot) return;
    targetCamPosRef.current = new THREE.Vector3(...selectedHotspot.camTarget);
    targetLookAtRef.current = new THREE.Vector3(...selectedHotspot.camLookAt);
  }, [selectedHotspot]);

  // ─── Update Auto-Rotate ──────────────────────────────────────────────────
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = autoRotate;
      controlsRef.current.autoRotateSpeed = 1.2;
    }
  }, [autoRotate]);

  // ─── Update Livery Paint Color ───────────────────────────────────────────
  useEffect(() => {
    if (!activeLivery || carMaterialsRef.current.length === 0) return;
    const targetColor = new THREE.Color(activeLivery.hex);
    carMaterialsRef.current.forEach((mat) => {
      // Modulate primary body paint
      if (mat.name && (mat.name.toLowerCase().includes('body') || mat.name.toLowerCase().includes('red') || mat.name.toLowerCase().includes('paint') || mat.name.toLowerCase().includes('car'))) {
        mat.color.copy(targetColor);
        mat.roughness = 0.22;
        mat.metalness = 0.75;
      }
    });
  }, [activeLivery]);

  // ─── Update Wind Tunnel Visibility ───────────────────────────────────────
  useEffect(() => {
    if (windParticlesRef.current) {
      windParticlesRef.current.visible = isWindTunnel;
    }
  }, [isWindTunnel]);

  // ─── Update Wireframe Mode ───────────────────────────────────────────────
  useEffect(() => {
    carMaterialsRef.current.forEach((mat) => {
      mat.wireframe = isWireframe;
    });
  }, [isWireframe]);

  // Fallback F1 3D Geometry if GLB fails
  function createFallbackF1Car() {
    const group = new THREE.Group();
    // Fuselage / Monocoque
    const bodyGeo = new THREE.BoxGeometry(0.8, 0.45, 3.8);
    const redMat = new THREE.MeshStandardMaterial({ color: 0xe10600, roughness: 0.2, metalness: 0.8 });
    const body = new THREE.Mesh(bodyGeo, redMat);
    body.position.y = 0.35;
    group.add(body);

    // Front Wing
    const fwGeo = new THREE.BoxGeometry(1.8, 0.08, 0.5);
    const blackMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.4 });
    const fw = new THREE.Mesh(fwGeo, blackMat);
    fw.position.set(0, 0.15, 1.9);
    group.add(fw);

    // Rear Wing
    const rwGeo = new THREE.BoxGeometry(1.2, 0.4, 0.3);
    const rw = new THREE.Mesh(rwGeo, redMat);
    rw.position.set(0, 0.7, -1.8);
    group.add(rw);

    // Wheels
    const wheelGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.32, 24);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.7 });
    const wheelPositions = [
      [0.85, 0.35, 1.3],
      [-0.85, 0.35, 1.3],
      [0.85, 0.35, -1.3],
      [-0.85, 0.35, -1.3]
    ];
    wheelPositions.forEach(([x, y, z]) => {
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.rotation.z = Math.PI / 2;
      w.position.set(x, y, z);
      group.add(w);
    });
    return group;
  }

  return (
    <div className="relative w-full h-full overflow-hidden select-none">
      {/* 3D WebGL Canvas */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Loading Overlay */}
      {!isLoaded && (
        <div className="absolute inset-0 bg-[#07080C] flex flex-col items-center justify-center z-50">
          <div className="text-4xl mb-4 animate-bounce">🏎️</div>
          <div className="text-xs uppercase tracking-[0.35em] text-[#E10600] font-bold font-mono mb-2">
            INITIALIZING SCUDERIA 3D ENVIRONMENT
          </div>
          <div className="w-64 h-1.5 bg-white/10 rounded-full overflow-hidden border border-white/5">
            <div
              className="h-full bg-gradient-to-r from-[#E10600] to-[#FF8000] transition-all duration-300"
              style={{ width: `${loadingProgress}%` }}
            />
          </div>
          <div className="text-[11px] font-mono text-white/50 mt-2">
            Loading Ferrari F2004 V10 Engine & Aero ({loadingProgress}%)
          </div>
        </div>
      )}

      {/* 3D Hotspot Interactive Markers Projected into Screen Space */}
      {isLoaded &&
        HOTSPOTS.map((spot) => {
          const coord = hotspotScreenCoords[spot.id];
          if (!coord || !coord.visible) return null;

          const isSelected = selectedHotspot?.id === spot.id;

          return (
            <div
              key={spot.id}
              onClick={() => {
                playTelemetryBeep(1200, 'sine', 0.1);
                onSelectHotspot(spot);
              }}
              style={{
                transform: `translate(${coord.x}px, ${coord.y}px)`,
              }}
              className="absolute -top-3.5 -left-3.5 z-20 group cursor-pointer"
            >
              <div className="relative flex items-center justify-center">
                {/* Pulsing ring */}
                <span
                  className={`absolute inline-flex h-7 w-7 rounded-full animate-ping opacity-75 ${
                    isSelected ? 'bg-[#00F5D4]' : 'bg-[#E10600]'
                  }`}
                />
                {/* Center dot */}
                <span
                  className={`relative inline-flex items-center justify-center rounded-full h-6 w-6 border text-[11px] font-mono font-bold transition-all duration-200 ${
                    isSelected
                      ? 'bg-[#00F5D4] text-black border-white shadow-[0_0_15px_#00F5D4]'
                      : 'bg-[#E10600] text-white border-white/40 group-hover:scale-125 group-hover:border-white shadow-[0_0_10px_#E10600]'
                  }`}
                >
                  +
                </span>

                {/* Floating tooltip on hover */}
                <div className="hidden group-hover:flex absolute left-8 top-1/2 -translate-y-1/2 px-2.5 py-1 glass-panel rounded-md whitespace-nowrap pointer-events-none z-30 shadow-lg border border-white/10">
                  <span className="text-[11px] font-mono text-[#00F5D4] uppercase tracking-wider">
                    {spot.title}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
    </div>
  );
}
