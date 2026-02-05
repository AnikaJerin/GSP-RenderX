import React, { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Html } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
// import vert from "../../shaders/gaussian.vert.glsl";
// import frag from "../../shaders/gaussian.frag.glsl";
const vertexShader = `
  attribute float size;

  varying vec3 vColor;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying vec3 vWorldPos;
  varying float vSize;
  varying float vDepth;
  uniform float uPointScale;
  uniform float uMinSize;
  uniform float uMaxSize;
  uniform float uViewportHeight;
  uniform float uNear;
  uniform float uFar;
  uniform float uFill;
  uniform float uEdgeBoost;
  uniform float uDensityBoost;
  uniform float uSizeRef;
  uniform float uClipEnabled;
  uniform float uClipAxis;
  uniform float uClipValue;

  void main() {
      vColor = color;     // provided by Three.js
      vNormal = normalMatrix * normal;   // view-space normal
      vSize = size;

      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      vDepth = -mvPosition.z;
      vViewDir = normalize(-mvPosition.xyz);
      vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
      float screenScale = uViewportHeight / 800.0;
      float viewDepth = max(-mvPosition.z, 0.001);
      float densityScale = mix(1.0, uDensityBoost, clamp(1.0 - (size / uSizeRef), 0.0, 1.0));
      float facing = abs(dot(normalize(vNormal), normalize(vViewDir)));
      float edgeScale = mix(uEdgeBoost, 1.0, facing);
      float pSize = size * densityScale * edgeScale * (uPointScale * screenScale / viewDepth);
      gl_PointSize = clamp(pSize, uMinSize, uMaxSize);
      gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  varying vec3 vColor;
  varying vec3 vNormal;
  varying vec3 vViewDir;
  varying float vSize;
  varying float vDepth;
  varying vec3 vWorldPos;
  uniform float uNear;
  uniform float uFar;
  uniform float uFill;
  uniform float uEdgeBoost;
  uniform float uClipEnabled;
  uniform float uClipAxis;
  uniform float uClipValue;

  void main() {
      if (uClipEnabled > 0.5) {
        float v = uClipAxis < 0.5 ? vWorldPos.x : (uClipAxis < 1.5 ? vWorldPos.y : vWorldPos.z);
        if (v > uClipValue) discard;
      }
      vec2 coord = gl_PointCoord - vec2(0.5);
      float dist = dot(coord, coord);

      float edgeBoost = mix(1.0, 2.8, clamp(0.012 / (vSize + 0.0001), 0.0, 1.0));
      float sharp = mix(180.0, 60.0, clamp(vSize / 0.02, 0.0, 1.0));
      float depthBoost = clamp((uFar - vDepth) / max(uFar - uNear, 0.001), 0.5, 1.6);
      float facing = abs(dot(normalize(vNormal), normalize(vViewDir)));
      float silhouette = pow(1.0 - facing, 2.1) * uEdgeBoost;
      float alpha = exp(-sharp * dist) * edgeBoost * depthBoost * uFill + silhouette * 0.45;

      vec3 lightDir = normalize(vec3(0.35, 0.85, 0.35));
      float diff = max(dot(normalize(vNormal), lightDir), 0.18);
      vec3 ambient = vec3(0.55);
      vec3 colorOut = vColor * (diff + ambient);

      gl_FragColor = vec4(colorOut, alpha);

      if (alpha < 0.004) discard;
  }
`;
export default function GaussianRenderer({
  data,
  onPick,
  pickData,
  enableSort,
  edgeFillBoost = 1.0,
  clipEnabled = false,
  clipAxis = "y",
  clipValue = 0,
  annotations = [],
  showAnnotations = true,
}) {
  const pointsRef = useRef(null);

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
  

  const materialFill = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        transparent: true,
        depthTest: true,
        depthWrite: true,
        blending: THREE.NormalBlending,
        vertexColors: true,
        uniforms: {
          uPointScale: { value: 460.0 },
          uMinSize: { value: 3.0 },
          uMaxSize: { value: 28.0 },
          uViewportHeight: { value: 800.0 },
          uNear: { value: 0.1 },
          uFar: { value: 100.0 },
          uFill: { value: 3.4 },
          uEdgeBoost: { value: 2.6 },
          uDensityBoost: { value: 2.1 },
          uSizeRef: { value: 0.02 },
          uClipEnabled: { value: 0.0 },
          uClipAxis: { value: 1.0 },
          uClipValue: { value: 0.0 },
          uMono: { value: 0.0 },
          uMonoColor: { value: new THREE.Color(0.86, 0.86, 0.86) },
        },
      }),
    []
  );

  const materialDetail = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        transparent: true,
        depthTest: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
        vertexColors: true,
        uniforms: {
          uPointScale: { value: 300.0 },
          uMinSize: { value: 1.7 },
          uMaxSize: { value: 10.5 },
          uViewportHeight: { value: 800.0 },
          uNear: { value: 0.1 },
          uFar: { value: 100.0 },
          uFill: { value: 1.6 },
          uEdgeBoost: { value: 2.3 },
          uDensityBoost: { value: 1.4 },
          uSizeRef: { value: 0.02 },
          uClipEnabled: { value: 0.0 },
          uClipAxis: { value: 1.0 },
          uClipValue: { value: 0.0 },
          uMono: { value: 0.0 },
          uMonoColor: { value: new THREE.Color(0.86, 0.86, 0.86) },
        },
      }),
    []
  );

  const materialEdge = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        transparent: true,
        depthTest: true,
        depthWrite: false,
        blending: THREE.NormalBlending,
        vertexColors: true,
        uniforms: {
          uPointScale: { value: 250.0 },
          uMinSize: { value: 1.4 },
          uMaxSize: { value: 8.0 },
          uViewportHeight: { value: 800.0 },
          uNear: { value: 0.1 },
          uFar: { value: 100.0 },
          uFill: { value: 0.9 },
          uEdgeBoost: { value: 3.0 },
          uDensityBoost: { value: 1.1 },
          uSizeRef: { value: 0.02 },
          uClipEnabled: { value: 0.0 },
          uClipAxis: { value: 1.0 },
          uClipValue: { value: 0.0 },
          uMono: { value: 0.0 },
          uMonoColor: { value: new THREE.Color(0.86, 0.86, 0.86) },
        },
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
  const { camera, size } = useThree();

  useEffect(() => {
    const materials = [materialFill, materialDetail, materialEdge];
    materials.forEach((mat) => {
      if (!mat) return;
      mat.uniforms.uViewportHeight.value = size.height;
      mat.uniforms.uNear.value = camera.near;
      mat.uniforms.uFar.value = camera.far;
    });
    materialFill.uniforms.uFill.value = 3.2 * edgeFillBoost;
    materialEdge.uniforms.uEdgeBoost.value = 3.0 * edgeFillBoost;
    const axisValue = clipAxis === "x" ? 0.0 : clipAxis === "y" ? 1.0 : 2.0;
    const clipOn = clipEnabled ? 1.0 : 0.0;
    [materialFill, materialDetail, materialEdge].forEach((mat) => {
      mat.uniforms.uClipEnabled.value = clipOn;
      mat.uniforms.uClipAxis.value = axisValue;
      mat.uniforms.uClipValue.value = clipValue;
    });
  }, [
    materialFill,
    materialDetail,
    materialEdge,
    size.height,
    camera.near,
    camera.far,
    edgeFillBoost,
    clipEnabled,
    clipAxis,
    clipValue,
  ]);

  useFrame(({ clock }) => {
    if (!enableSort || !geometry) return;
    if (!camera) return;
    if (!pointsRef.current) return;
    const count = geometry.getAttribute("position").count;
    if (count > 300000) return;
    if (clock.elapsedTime - sortThrottle.current < 0.5) return;
    sortThrottle.current = clock.elapsedTime;

    const pos = geometry.getAttribute("position").array;
    const indices = new Uint32Array(count);
    const depths = new Float32Array(count);

    const mv = new THREE.Matrix4();
    mv.multiplyMatrices(camera.matrixWorldInverse, pointsRef.current.matrixWorld);

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
      <points ref={pointsRef} geometry={geometry} material={materialFill} renderOrder={0} />
      <points geometry={geometry} material={materialDetail} renderOrder={1} />
      <points geometry={geometry} material={materialEdge} renderOrder={2} />
      {showAnnotations && annotations && annotations.length > 0 && (
        <AnnotationPoints annotations={annotations} />
      )}
      {showAnnotations && annotations && annotations.length > 0 && (
        <AnnotationLabels annotations={annotations} />
      )}
      {pickData && onPick && <PickPoints data={pickData} onPick={handlePointerDown} />}
    </>
  );
}

function AnnotationPoints({ annotations }) {
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(annotations.length * 3);
    annotations.forEach((ann, i) => {
      positions[i * 3] = ann.position[0];
      positions[i * 3 + 1] = ann.position[1];
      positions[i * 3 + 2] = ann.position[2];
    });
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return geo;
  }, [annotations]);

  const material = useMemo(
    () =>
      new THREE.PointsMaterial({
        size: 0.03,
        color: new THREE.Color(0.15, 1.0, 0.85),
        depthWrite: false,
      }),
    []
  );

  return <points geometry={geometry} material={material} renderOrder={3} />;
}

function AnnotationLabels({ annotations }) {
  return (
    <>
      {annotations.map((ann) => (
        <Html key={ann.id} position={ann.position} center>
          <div className="annotation-label">{ann.label}</div>
        </Html>
      ))}
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
