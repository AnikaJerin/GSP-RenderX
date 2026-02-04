import React, { useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import GaussianRenderer from "./gaussian/GaussianRenderer";
import UploadPanel from "../ui/UploadPanel";

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

export default function Viewer() {
  const [gaussianData, setGaussianData] = useState(null);
  const controlsRef = useRef(null);

  return (
    <>
      <Canvas style={{ width: "100%", height: "100%" }}>
        <color attach="background" args={["#0a0a0a"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} />

        {gaussianData && <GaussianRenderer data={gaussianData} />}
        {gaussianData && (
          <CameraFitter
            bboxMin={gaussianData.bboxMin}
            bboxMax={gaussianData.bboxMax}
            controlsRef={controlsRef}
          />
        )}
        <OrbitControls ref={controlsRef} makeDefault />
      </Canvas>

      {/* Upload panel overlays UI */}
      <UploadPanel onFileUpload={setGaussianData} />
    </>
  );
}
