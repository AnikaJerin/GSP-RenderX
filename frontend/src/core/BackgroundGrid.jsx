import { useMemo } from "react";
import * as THREE from "three";

const bgVertex = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

const bgFragment = `
  precision highp float;
  varying vec2 vUv;

  float gridLine(vec2 uv, float scale, float width) {
    vec2 gv = abs(fract(uv * scale - 0.5) - 0.5) / fwidth(uv * scale);
    float line = min(gv.x, gv.y);
    return 1.0 - smoothstep(width, width + 1.0, line);
  }

  void main() {
    vec3 top = vec3(0.08, 0.09, 0.11);
    vec3 bottom = vec3(0.04, 0.045, 0.055);
    vec3 grad = mix(bottom, top, smoothstep(0.0, 1.0, vUv.y));

    float fine = gridLine(vUv, 36.0, 2.4);
    float coarse = gridLine(vUv, 9.0, 2.6);
    float gridMix = fine * 0.02 + coarse * 0.04;

    float vignette = smoothstep(0.95, 0.2, distance(vUv, vec2(0.5)));
    vec3 color = grad + vec3(gridMix) * vignette * 0.6;

    gl_FragColor = vec4(color, 1.0);
  }
`;

export default function BackgroundGrid() {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: bgVertex,
        fragmentShader: bgFragment,
        depthTest: false,
        depthWrite: false,
      }),
    []
  );

  return (
    <mesh renderOrder={-10} material={material} frustumCulled={false}>
      <planeGeometry args={[2, 2]} />
    </mesh>
  );
}
