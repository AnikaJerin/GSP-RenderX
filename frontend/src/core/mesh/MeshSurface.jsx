import React, { useMemo } from "react";
import { useLoader } from "@react-three/fiber";
import * as THREE from "three";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";

export default function MeshSurface({ url, type, onLoaded, color }) {
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(color || "#eae7e2"),
        roughness: 0.55,
        metalness: 0.02,
        side: THREE.DoubleSide,
      }),
    [color]
  );

  const mesh = useLoader(
    type === "ply"
      ? PLYLoader
      : type === "stl"
      ? STLLoader
      : type === "obj"
      ? OBJLoader
      : GLTFLoader,
    url
  );

  React.useEffect(() => {
    if (onLoaded) onLoaded();
  }, [onLoaded, mesh]);

  if (type === "obj") {
    mesh.traverse((child) => {
      if (child.isMesh) {
        if (child.geometry && child.geometry.isBufferGeometry) {
          if (!child.geometry.attributes.normal) {
            child.geometry.computeVertexNormals();
          }
          child.geometry.normalizeNormals();
        }
        child.material = material;
      }
    });
    return <primitive object={mesh} />;
  }

  if (type === "gltf" || type === "glb") {
    return <primitive object={mesh.scene} />;
  }

  // PLY/STL return BufferGeometry
  if (mesh && mesh.isBufferGeometry) {
    if (!mesh.attributes.normal) {
      mesh.computeVertexNormals();
    }
    mesh.normalizeNormals();
  }
  return (
    <mesh geometry={mesh} material={material} castShadow receiveShadow />
  );
}
