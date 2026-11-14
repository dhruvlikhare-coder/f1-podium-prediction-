import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export default function F1Showcase3D({ activeSection = 0, isPredictionRunning = false }) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const carGroupRef = useRef(null);
  const materialsRef = useRef([]);
  const animFrameIdRef = useRef(null);

  const [isLoaded, setIsLoaded] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);

  // Precision camera states:
  // Section 0 (Page 1): Matches media_1794666958608_7923f954.jpg side profile position
  // Section 1 (Page 2): Top-down overhead vertical orientation on right side
  // Section 2 (Page 3): Shift to side view, zoomed in focused specifically on rear tire
  const sectionStates = [
    {
      // Page 1: Exact framing of media_1794666958608_7923f954.jpg
      camPos: new THREE.Vector3(4.6, 0.9, 0.25),
      camLookAt: new THREE.Vector3(0, 0.38, 0.25),
      carPos: new THREE.Vector3(1.5, -0.00, 1.0),
      carRot: new THREE.Euler(0, Math.PI, 0),
      camUp: new THREE.Vector3(0, 1, 0),
    },
    {
      // Page 2: Top-down vertical overhead on right side of screen (nose pointing UP)
      camPos: new THREE.Vector3(1.8, 5.2, 0),
      camLookAt: new THREE.Vector3(1.8, 0, 0),
      carPos: new THREE.Vector3(1.8, 0, 0),
      carRot: new THREE.Euler(0, 0, 0),
      camUp: new THREE.Vector3(0, 0, 1),
    },
    {
      // Page 3: Side view focused tightly on the rear tire (at Z ~ -1.35)
      camPos: new THREE.Vector3(2.2, 0.42, -1.35),
      camLookAt: new THREE.Vector3(0, 0.42, -1.35),
      carPos: new THREE.Vector3(0, 0, 0),
      carRot: new THREE.Euler(0, 0, 0),
      camUp: new THREE.Vector3(0, 1, 0),
    }
  ];

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // 1. Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x050608);
    scene.fog = new THREE.FogExp2(0x050608, 0.04);

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(
      40,
      container.clientWidth / container.clientHeight,
      0.1,
      100
    );
    const initial = sectionStates[0];
    camera.position.copy(initial.camPos);
    camera.up.copy(initial.camUp);
    camera.lookAt(initial.camLookAt);
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

    // 4. Studio Lighting matching the rich red/dark atmosphere in PDF
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const hemiLight = new THREE.HemisphereLight(0xffeedd, 0x111122, 1.2);
    scene.add(hemiLight);

    // Overhead Key Light
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
    keyLight.position.set(4, 8, 5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.bias = -0.0001;
    scene.add(keyLight);

    // Crimson / Burgundy Studio Glow
    const redGlowLight = new THREE.PointLight(0xd91424, 4.5, 12, 1.2);
    redGlowLight.position.set(-2, 2.5, -2.5);
    scene.add(redGlowLight);

    const sideRedLight = new THREE.DirectionalLight(0xe10600, 2.4);
    sideRedLight.position.set(-6, 2, 2);
    scene.add(sideRedLight);

    const frontFill = new THREE.DirectionalLight(0xffe8d6, 1.2);
    frontFill.position.set(2, 2, 5);
    scene.add(frontFill);

    // 5. Reflective Floor Plane
    const floorGeo = new THREE.PlaneGeometry(30, 30);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x050608,
      roughness: 0.35,
      metalness: 0.65,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.01;
    floor.receiveShadow = true;
    scene.add(floor);

    // Subtle red studio backdrop gradient
    const backdropGeo = new THREE.CircleGeometry(9, 64);
    const backdropMat = new THREE.MeshBasicMaterial({
      color: 0x660810,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending
    });
    const backdrop = new THREE.Mesh(backdropGeo, backdropMat);
    backdrop.rotation.x = -Math.PI / 2;
    backdrop.position.set(-1, -0.005, -0.5);
    scene.add(backdrop);

    // 6. Car Container Group
    const carGroup = new THREE.Group();
    carGroup.position.copy(initial.carPos);
    carGroup.rotation.copy(initial.carRot);
    scene.add(carGroup);
    carGroupRef.current = carGroup;

    // 7. Load 3D Model: Ferrari F2004
    const loader = new GLTFLoader();
    loader.load(
      '/models/ferrari_f2004.glb',
      (gltf) => {
        const car = gltf.scene;

        const box = new THREE.Box3().setFromObject(car);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 4.3 / maxDim;

        car.scale.set(scale, scale, scale);
        car.position.set(-center.x * scale, -box.min.y * scale, -center.z * scale);

        const mats = [];
        car.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true;
            child.receiveShadow = true;
            if (child.material) {
              child.material.envMapIntensity = 1.8;
              child.material.needsUpdate = true;
              mats.push(child.material);
            }
          }
        });
        materialsRef.current = mats;

        carGroup.add(car);
        setIsLoaded(true);
      },
      (xhr) => {
        if (xhr.total > 0) {
          setLoadProgress(Math.round((xhr.loaded / xhr.total) * 100));
        } else {
          setLoadProgress((prev) => Math.min(prev + 15, 95));
        }
      },
      (error) => {
        console.error('Error loading car GLB:', error);
        const fallback = createFallbackCar();
        carGroup.add(fallback);
        setIsLoaded(true);
      }
    );

    // 8. Render Loop
    let clock = new THREE.Clock();

    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const time = clock.getElapsedTime();

      // Subtle breathing motion
      if (carGroupRef.current && isLoaded) {
        carGroupRef.current.position.y += Math.sin(time * 1.5) * 0.0002;
      }

      renderer.render(scene, camera);
    };
    animate();

    // 9. Resize Handler
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

  // ─── Smooth Transition on ActiveSection Change ───────────────────────────
  useEffect(() => {
    if (!cameraRef.current || !carGroupRef.current) return;

    const targetState = sectionStates[Math.min(activeSection, sectionStates.length - 1)];
    const camera = cameraRef.current;
    const carGroup = carGroupRef.current;

    let animId;
    const startCamPos = camera.position.clone();
    const startCamUp = camera.up.clone();
    const startCarPos = carGroup.position.clone();
    const startCarRot = new THREE.Euler().copy(carGroup.rotation);

    const duration = 1.0;
    const startTime = performance.now();

    const updateTransition = (now) => {
      const elapsed = (now - startTime) / 1000;
      const t = Math.min(elapsed / duration, 1.0);
      const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      // Lerp Camera Position
      camera.position.lerpVectors(startCamPos, targetState.camPos, ease);

      // Slerp Camera Up Vector
      camera.up.lerpVectors(startCamUp, targetState.camUp, ease).normalize();

      // LookAt Target
      const currentLookAt = new THREE.Vector3().lerpVectors(
        sectionStates[0].camLookAt,
        targetState.camLookAt,
        ease
      );
      camera.lookAt(currentLookAt);

      // Lerp Car Position
      carGroup.position.lerpVectors(startCarPos, targetState.carPos, ease);

      // Lerp Car Rotation
      carGroup.rotation.x = THREE.MathUtils.lerp(startCarRot.x, targetState.carRot.x, ease);
      carGroup.rotation.y = THREE.MathUtils.lerp(startCarRot.y, targetState.carRot.y, ease);
      carGroup.rotation.z = THREE.MathUtils.lerp(startCarRot.z, targetState.carRot.z, ease);

      if (t < 1.0) {
        animId = requestAnimationFrame(updateTransition);
      }
    };
    animId = requestAnimationFrame(updateTransition);

    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }, [activeSection]);

  // Reaction when Prediction Button is Pressed
  useEffect(() => {
    if (!carGroupRef.current) return;
    if (isPredictionRunning) {
      carGroupRef.current.rotation.x = -0.05;
      carGroupRef.current.position.y = 0.15;
      materialsRef.current.forEach((mat) => {
        mat.wireframe = true;
        mat.color = new THREE.Color(0x00f5d4);
      });
    } else {
      carGroupRef.current.rotation.x = 0;
      carGroupRef.current.position.y = activeSection === 0 ? -0.05 : 0;
      materialsRef.current.forEach((mat) => {
        mat.wireframe = false;
        mat.color = new THREE.Color(0xffffff);
      });
    }
  }, [isPredictionRunning, activeSection]);

  // Fallback procedural F1 Car
  function createFallbackCar() {
    const group = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0xd91424, roughness: 0.2, metalness: 0.8 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.5 });

    const body = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.45, 3.8), bodyMat);
    body.position.y = 0.35;
    group.add(body);

    const fw = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.08, 0.5), darkMat);
    fw.position.set(0, 0.15, 1.9);
    group.add(fw);

    const rw = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.4, 0.3), bodyMat);
    rw.position.set(0, 0.7, -1.8);
    group.add(rw);

    return group;
  }

  return (
    <div className="fixed inset-0 w-full h-full pointer-events-none z-0">
      <div ref={containerRef} className="w-full h-full" />

      {/* Loading Screen */}
      {!isLoaded && (
        <div className="absolute inset-0 bg-[#050608] flex flex-col items-center justify-center z-50 pointer-events-auto">
          <div className="text-3xl font-display font-black tracking-widest text-white mb-2">
            FORMULA <span className="text-[#E2C8A2]">1</span>
          </div>
          <div className="text-[11px] font-mono tracking-[0.3em] text-[#D91424] uppercase mb-4">
            LOADING FERRARI F2004 3D ASSETS
          </div>
          <div className="w-48 h-1 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[#D91424] to-[#E2C8A2] transition-all duration-300"
              style={{ width: `${loadProgress}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
