import React, { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
// import vert from "../../shaders/gaussian.vert.glsl";
// import frag from "../../shaders/gaussian.frag.glsl";
const vertexShader = `
  attribute float size;

  varying vec3 vColor;
  varying vec3 vNormal;
  varying float vSize;

  void main() {
      vColor = color;     // provided by Three.js
      vNormal = normal;   // provided by Three.js
      vSize = size;

      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * (300.0 / -mvPosition.z);
      gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  varying vec3 vColor;
  varying vec3 vNormal;
  varying float vSize;

  void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = dot(coord, coord);

      float edgeBoost = mix(1.0, 1.6, clamp(0.012 / (vSize + 0.0001), 0.0, 1.0));
      float alpha = exp(-60.0 * dist) * edgeBoost;

      vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0));
      float diff = max(dot(normalize(vNormal), lightDir), 0.2);

      gl_FragColor = vec4(vColor * diff, alpha);

      if (alpha < 0.02) discard;
  }
`;
export default function GaussianRenderer({ data, onPick, pickData, enableSort }) {

  const geometry = useMemo(() => {
    if (!data || !data.positions || data.positions.length === 0) {
      return null;
    }
  
    const geo = new THREE.BufferGeometry();
  
    geo.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(data.normals, 3));
  
    // Uint8 → Float32
    const colorFloat = new Float32Array(data.colors.length);
    for (let i = 0; i < data.colors.length; i++) {
      colorFloat[i] = data.colors[i] / 255;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colorFloat, 3));
  
    // Uint16 → Float32
    const sizeFloat = new Float32Array(data.sizes.length);
    for (let i = 0; i < data.sizes.length; i++) {
      sizeFloat[i] = data.sizes[i] / 1000;
    }
    geo.setAttribute("size", new THREE.BufferAttribute(sizeFloat, 1));

    geo.computeBoundingSphere();
    return geo;
  }, [data]);

  useEffect(() => {
    if (!geometry) return;
    const count = data?.activeCount ?? data?.count ?? 0;
    geometry.setDrawRange(0, count);
  }, [geometry, data]);
  

  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        transparent: true,
        depthTest: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
        vertexColors: true,
      }),
    []
  );

  if (!geometry) return null;

  const handlePointerDown = (event) => {
    if (!onPick || event.index == null) return;
    const i = event.index;
    const p = i * 3;
    const position = [
      pickData ? pickData.positions[p] : data.positions[p],
      pickData ? pickData.positions[p + 1] : data.positions[p + 1],
      pickData ? pickData.positions[p + 2] : data.positions[p + 2],
    ];
    const color = [
      pickData ? pickData.colors[p] : data.colors[p],
      pickData ? pickData.colors[p + 1] : data.colors[p + 1],
      pickData ? pickData.colors[p + 2] : data.colors[p + 2],
    ];
    const size = (pickData ? pickData.sizes[i] : data.sizes[i]) / 1000;
    onPick({ index: i, position, color, size });
  };

  const indexRef = useRef(null);
  const sortThrottle = useRef(0);
  const { camera } = useThree();

  useFrame(({ clock }) => {
    if (!enableSort || !geometry) return;
    const count = geometry.getAttribute("position").count;
    if (count > 200000) return;
    if (clock.elapsedTime - sortThrottle.current < 0.5) return;
    sortThrottle.current = clock.elapsedTime;

    const pos = geometry.getAttribute("position").array;
    const indices = new Uint32Array(count);
    const depths = new Float32Array(count);

    const mv = new THREE.Matrix4();
    mv.multiplyMatrices(camera.matrixWorldInverse, geometry.matrixWorld);

    const v = new THREE.Vector3();
    for (let i = 0; i < count; i += 1) {
      const x = pos[i * 3];
      const y = pos[i * 3 + 1];
      const z = pos[i * 3 + 2];
      v.set(x, y, z).applyMatrix4(mv);
      depths[i] = v.z;
      indices[i] = i;
    }

    indices.sort((a, b) => depths[b] - depths[a]);
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    indexRef.current = indices;
  });

  return (
    <>
      <points geometry={geometry} material={material} />
      {pickData && onPick && <PickPoints data={pickData} onPick={handlePointerDown} />}
    </>
  );
}

function PickPoints({ data, onPick }) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
    return geo;
  }, [data]);

  const material = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.02,
        transparent: true,
        opacity: 0,
        depthWrite: false,
      }),
    []
  );

  return <points geometry={geometry} material={material} onPointerDown={onPick} />;
}





// import { useMemo } from "react";
// import * as THREE from "three";
// import { generateGaussianSphere } from "../../data/sampleGaussians";
// // import vert from "../../shaders/gaussian.vert.glsl";
// // import frag from "../../shaders/gaussian.frag.glsl";
// const vertexShader = `
// precision highp float;

// attribute float size;

// varying vec3 vColor;

// void main() {
//     vColor = color; // injected by Three.js

//     vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);

//     // Perspective-correct Gaussian splat size
//     gl_PointSize = size * (300.0 / -mvPosition.z);

//     gl_Position = projectionMatrix * mvPosition;
// }
// `;

// const fragmentShader = `
// precision highp float;

// varying vec3 vColor;

// void main() {
//     vec2 coord = gl_PointCoord - vec2(0.5);
//     float r2 = dot(coord, coord);

//     // Gaussian falloff
//     float alpha = exp(-r2 * 8.0);
//     alpha = clamp(alpha, 0.0, 0.8);

//     if (alpha < 0.01) discard;

//     gl_FragColor = vec4(vColor, alpha);
// }
// `;

// export default function GaussianRenderer() {
//   const { positions, colors, sizes } = useMemo(() => generateGaussianSphere(8000), []);

//   const geometry = useMemo(() => {
//     const geo = new THREE.BufferGeometry();
//     geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
//     geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
//     geo.setAttribute("size", new THREE.Float32BufferAttribute(sizes, 1));
//     return geo;
//   }, [positions, colors, sizes]);

//   const material = useMemo(() => {
//     return new THREE.ShaderMaterial({
//       vertexShader,
//       fragmentShader,
//       transparent: true,
//       depthTest: true,
//       depthWrite: false,
//       blending: THREE.NormalBlending,
//       vertexColors: true, // ✅ ADD THIS
//     });
//   }, []);  

//   return <points geometry={geometry} material={material} />;
// }
