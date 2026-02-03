import React, { useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import GaussianRenderer from "./gaussian/GaussianRenderer";
import UploadPanel from "../ui/UploadPanel";

export default function Viewer() {
  const [gaussianData, setGaussianData] = useState(null);

  return (
    <>
      <Canvas style={{ width: "100%", height: "100%" }}>
        <color attach="background" args={["#0a0a0a"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} />

        <GaussianRenderer data={gaussianData} />
        <OrbitControls makeDefault />
      </Canvas>

      {/* Upload panel overlays UI */}
      <UploadPanel onFileUpload={setGaussianData} />
    </>
  );
}
