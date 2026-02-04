import React, { useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import GaussianRenderer from "./gaussian/GaussianRenderer";

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
    const distance = size * 1.8 + 0.5;
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

export default function Viewer({ data, onSelect, inspectMode }) {
  const controlsRef = useRef(null);
  const pickData = useMemo(
    () => (inspectMode ? decimateData(data, 120000) : null),
    [data, inspectMode]
  );

  return (
    <>
      <Canvas style={{ width: "100%", height: "100%", touchAction: "none" }}>
        <color attach="background" args={["#0a0a0a"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} />

        <RaycastConfig />
        {data && (
          <GaussianRenderer
            data={data}
            onPick={inspectMode ? onSelect : null}
            pickData={pickData}
          />
        )}
        {data && (
          <CameraFitter
            bboxMin={data.bboxMin}
            bboxMax={data.bboxMax}
            controlsRef={controlsRef}
          />
        )}
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enableZoom
          enablePan
          enableDamping
          dampingFactor={0.08}
          zoomSpeed={0.9}
        />
      </Canvas>
    </>
  );
}
