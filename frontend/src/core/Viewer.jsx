import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import GaussianRenderer from "./gaussian/GaussianRenderer";
import MeshSurface from "./mesh/MeshSurface";
import BackgroundGrid from "./BackgroundGrid";

function CameraFitter({ bboxMin, bboxMax, controlsRef }) {
  const { camera } = useThree();
  React.useEffect(() => {
    if (!bboxMin || !bboxMax) return;
    const min = bboxMin;
    const max = bboxMax;
    const center = [
      (min[0] + max[0]) / 2,
      (min[1] + max[1]) / 2,
      (min[2] + max[2]) / 2,
    ];
    const size = Math.max(
      max[0] - min[0],
      max[1] - min[1],
      max[2] - min[2]
    );
    const distance = size * 1.1 + 0.3;
    camera.position.set(center[0] + distance, center[1] + distance, center[2] + distance);
    camera.near = Math.max(0.01, distance / 100);
    camera.far = distance * 10;
    camera.updateProjectionMatrix();
    if (controlsRef.current) {
      controlsRef.current.target.set(center[0], center[1], center[2]);
      controlsRef.current.update();
    }
  }, [bboxMin, bboxMax, camera, controlsRef]);
  return null;
}

function RaycastConfig() {
  const { raycaster } = useThree();
  React.useEffect(() => {
    raycaster.params.Points.threshold = 0.15;
  }, [raycaster]);
  return null;
}

function decimateData(data, maxPoints) {
  if (!data || !data.positions) return data;
  const count = data.count || data.positions.length / 3;
  if (!maxPoints || count <= maxPoints) return data;

  const step = Math.ceil(count / maxPoints);
  const newCount = Math.floor(count / step);

  const positions = new Float32Array(newCount * 3);
  const normals = new Float32Array(newCount * 3);
  const colors = new Uint8Array(newCount * 3);
  const sizes = new Uint16Array(newCount);

  let j = 0;
  for (let i = 0; i < count && j < newCount; i += step) {
    const p = i * 3;
    const q = j * 3;
    positions[q] = data.positions[p];
    positions[q + 1] = data.positions[p + 1];
    positions[q + 2] = data.positions[p + 2];
    normals[q] = data.normals[p];
    normals[q + 1] = data.normals[p + 1];
    normals[q + 2] = data.normals[p + 2];
    colors[q] = data.colors[p];
    colors[q + 1] = data.colors[p + 1];
    colors[q + 2] = data.colors[p + 2];
    sizes[j] = data.sizes[i];
    j += 1;
  }

  return {
    positions,
    normals,
    colors,
    sizes,
    count: newCount,
  };
}

export default function Viewer({
  data,
  meshUrl,
  meshType,
  onSelect,
  inspectMode,
  clipEnabled,
  clipAxis,
  clipRatio,
  showFps,
  onFps,
  onRenderStats,
  annotations,
  showAnnotations,
  solidSurface,
  surfaceColor,
}) {
  const controlsRef = useRef(null);
  const bboxSize = useMemo(() => {
    if (!data?.bboxMin || !data?.bboxMax) return null;
    return Math.max(
      data.bboxMax[0] - data.bboxMin[0],
      data.bboxMax[1] - data.bboxMin[1],
      data.bboxMax[2] - data.bboxMin[2]
    );
  }, [data, onRenderStats]);

  const clipValue = useMemo(() => {
    if (!data?.bboxMin || !data?.bboxMax) return 0;
    const axis = clipAxis;
    const min = axis === "x" ? data.bboxMin[0] : axis === "y" ? data.bboxMin[1] : data.bboxMin[2];
    const max = axis === "x" ? data.bboxMax[0] : axis === "y" ? data.bboxMax[1] : data.bboxMax[2];
    return min + (max - min) * clipRatio;
  }, [data, clipAxis, clipRatio]);

  const [drawBudget, setDrawBudget] = useState(200000);
  const [renderData, setRenderData] = useState(null);
  const [meshReady, setMeshReady] = useState(false);
  const lastAdjustRef = useRef(0);
  const fpsSamples = useRef([]);
  const [edgeFillBoost, setEdgeFillBoost] = useState(1.0);
  const stableRef = useRef(0);
  const pickData = useMemo(
    () => (inspectMode ? decimateData(data, 80000) : null),
    [data, inspectMode]
  );

  useEffect(() => {
    if (!data) {
      setRenderData(null);
      return;
    }
    const activeCount = data.activeCount || data.count || 0;
    const initial = Math.min(200000, activeCount);
    setDrawBudget(initial);
    setRenderData(decimateData(data, initial));
    if (onRenderStats) {
      onRenderStats({
        total: data.count || activeCount,
        active: activeCount,
        rendered: Math.min(initial, activeCount),
      });
    }
  }, [data, onRenderStats]);


  function AdaptiveBudget() {
    const { clock } = useThree();
    useFrame((state, delta) => {
      if (!data) return;
      const activeCount = data.activeCount || data.count || 0;
      const now = clock.elapsedTime;

      fpsSamples.current.push(1 / Math.max(delta, 0.001));
      if (fpsSamples.current.length > 20) fpsSamples.current.shift();

      if (now - lastAdjustRef.current < 0.5) return;
      lastAdjustRef.current = now;

      const avgFps =
        fpsSamples.current.reduce((a, b) => a + b, 0) / fpsSamples.current.length;

      const maxCap = activeCount;
      let next = drawBudget;
      if (avgFps < 45) {
        stableRef.current = 0;
        next = Math.max(50000, Math.floor(drawBudget * 0.8));
      } else {
        stableRef.current += delta;
        if (avgFps > 58 && stableRef.current > 2.0) {
          next = Math.min(maxCap, Math.floor(drawBudget * 1.1));
          stableRef.current = 0;
        }
      }

      if (Math.abs(next - drawBudget) >= 20000) {
        setDrawBudget(next);
        const nextData = decimateData(data, next);
        setRenderData(nextData);
        if (onRenderStats) {
          const activeCount = data.activeCount || data.count || 0;
          onRenderStats({
            total: data.count || activeCount,
            active: activeCount,
            rendered: nextData?.count || 0,
          });
        }
      }
    });
    return null;
  }

  function FpsOverlay() {
    useFrame((state, delta) => {
      if (!showFps) return;
      const current = 1 / Math.max(delta, 0.001);
      const last = fpsSamples.current.length
        ? fpsSamples.current[fpsSamples.current.length - 1]
        : current;
      const smoothed = last * 0.85 + current * 0.15;
      fpsSamples.current.push(smoothed);
      if (fpsSamples.current.length > 60) fpsSamples.current.shift();
      if (fpsSamples.current.length >= 10 && onFps) {
        const avg =
          fpsSamples.current.reduce((a, b) => a + b, 0) / fpsSamples.current.length;
        onFps(avg);
      }
    });
    return null;
  }

  return (
    <>
      <Canvas
        dpr={[1, 2]}
        style={{ width: "100%", height: "100%", touchAction: "none" }}
      >
        <BackgroundGrid />
        <color attach="background" args={["#0d1014"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} />

        <RaycastConfig />
        {renderData && (!solidSurface || !meshReady) && (
          <GaussianRenderer
            data={renderData}
            onPick={inspectMode ? onSelect : null}
            pickData={pickData}
            enableSort={false}
            edgeFillBoost={edgeFillBoost}
            clipEnabled={clipEnabled}
            clipAxis={clipAxis}
            clipValue={clipValue}
            annotations={annotations}
            showAnnotations={showAnnotations}
          />
        )}
        {meshUrl && solidSurface && (
          <MeshSurface
            url={meshUrl}
            type={meshType}
            onLoaded={() => setMeshReady(true)}
            color={surfaceColor}
          />
        )}
        {data && (
          <CameraFitter
            bboxMin={data.bboxMin}
            bboxMax={data.bboxMax}
            controlsRef={controlsRef}
          />
        )}
        <AdaptiveBudget />
        <FpsOverlay />
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enableZoom
          enablePan
          enableDamping
          dampingFactor={0.08}
          zoomSpeed={0.9}
          minDistance={bboxSize ? bboxSize * 0.35 : undefined}
          maxDistance={bboxSize ? bboxSize * 8.0 : undefined}
          onChange={() => {
            if (!bboxSize || !controlsRef.current) return;
            const dist = controlsRef.current.getDistance();
            const t = Math.min(1.0, Math.max(0.0, (dist - bboxSize * 0.6) / (bboxSize * 2.2)));
            const base = 1.7;
            const falloff = 0.8;
            setEdgeFillBoost(base - t * falloff);
          }}
        />
      </Canvas>
    </>
  );
}
