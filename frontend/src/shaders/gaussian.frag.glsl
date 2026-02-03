precision mediump float;

varying vec3 vColor;

void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    float alpha = exp(-dist * dist * 8.0);

    if (alpha < 0.01) discard;

    gl_FragColor = vec4(vColor, alpha);
}
