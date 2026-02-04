import React, { useMemo } from "react";
import * as THREE from "three";
// import vert from "../../shaders/gaussian.vert.glsl";
// import frag from "../../shaders/gaussian.frag.glsl";
const vertexShader = `
  attribute float size;

  varying vec3 vColor;
  varying vec3 vNormal;

  void main() {
      vColor = color;     // provided by Three.js
      vNormal = normal;   // provided by Three.js

      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * (300.0 / -mvPosition.z);
      gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  varying vec3 vColor;
  varying vec3 vNormal;

  void main() {
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = dot(coord, coord);

      float alpha = exp(-60.0 * dist);

      vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0));
      float diff = max(dot(normalize(vNormal), lightDir), 0.2);

      gl_FragColor = vec4(vColor * diff, alpha);

      if (alpha < 0.02) discard;
  }
`;
export default function GaussianRenderer({ data, onPick, pickData }) {

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
