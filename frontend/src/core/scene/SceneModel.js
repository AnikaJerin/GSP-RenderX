// Lightweight scene model for GSP-RenderX.
// NOTE: This is additive and does NOT change existing renderer behavior unless
// explicitly enabled via new Scene Engine UI controls.

export function createEmptyScene() {
  return {
    id: "scene-root",
    meta: {
      title: "Untitled Scene",
      createdAt: Date.now(),
      source: "manual",
    },
    entities: [],
    settings: {
      camera: {
        // High-level camera metadata (kept for future use).
        // Viewer continues to use its existing bbox-based fitting and controls.
        suggestedCenter: null,
      },
      rendering: {
        gaussianFill: 1.0,
        gaussianEdgeBoost: 1.0,
        gaussianDensityBoost: 1.0,
        meshVisibility: 1.0, // 0 = hidden, 1 = fully visible
        splatVisibility: 1.0,
      },
      dynamics: {
        enabled: false,
        timeScale: 1.0,
        // Temporal (4D) parameter in seconds; when set, it drives animation time.
        time: null,
        mode: "gravity", // "gravity" | "orbit" | "oscillate" | "none"
        gravity: [0, -9.8, 0],
        gravityStrength: 0.2,
        bounce: 0.35,
        floorOffset: 0.02,
        stiffness: 0.22,
      },
      overlays: {
        showStickFigure: true,
        nodeScale: 0.014,
        lineOpacity: 0.92,
      },
      training: {
        // Stub for future in-browser / remote training hooks.
        enabled: false,
        mode: "none", // "none" | "preview" | "full"
        reconstructionId: null,
      },
    },
  };
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function addNode(nodes, id, label, position) {
  nodes.push({ id, label, position });
}

function addChain(edges, ids) {
  for (let i = 0; i < ids.length - 1; i += 1) {
    edges.push([ids[i], ids[i + 1]]);
  }
}

function getGaussianPoints(gaussianData) {
  if (!gaussianData?.positions || gaussianData.positions.length < 3) {
    return [];
  }
  const points = [];
  for (let i = 0; i < gaussianData.positions.length; i += 3) {
    points.push([
      gaussianData.positions[i],
      gaussianData.positions[i + 1],
      gaussianData.positions[i + 2],
    ]);
  }
  return points;
}

function computeSliceStats(points, targetY, tolerance, fallbackCenterX, fallbackCenterZ) {
  const slice = points.filter((point) => Math.abs(point[1] - targetY) <= tolerance);
  if (!slice.length) {
    return {
      centerX: fallbackCenterX,
      centerZ: fallbackCenterZ,
      minX: fallbackCenterX,
      maxX: fallbackCenterX,
    };
  }

  const xs = slice.map((point) => point[0]).sort((a, b) => a - b);
  const zs = slice.map((point) => point[2]).sort((a, b) => a - b);
  const mid = Math.floor(xs.length / 2);
  return {
    centerX: xs[mid],
    centerZ: zs[mid],
    minX: xs[Math.floor(xs.length * 0.15)],
    maxX: xs[Math.floor(xs.length * 0.85)],
  };
}

function computeStickFigureLayout(gaussianData, bboxMin, bboxMax) {
  if (!bboxMin || !bboxMax) {
    return null;
  }

  const min = bboxMin;
  const max = bboxMax;
  const points = getGaussianPoints(gaussianData);
  const width = Math.max(max[0] - min[0], 0.001);
  const height = Math.max(max[1] - min[1], 0.001);
  const depth = Math.max(max[2] - min[2], 0.001);
  const defaultCenterX = (min[0] + max[0]) / 2;
  const defaultCenterZ = (min[2] + max[2]) / 2;
  const centerSlice = computeSliceStats(
    points,
    lerp(min[1], max[1], 0.52),
    height * 0.08,
    defaultCenterX,
    defaultCenterZ
  );
  const centerX = centerSlice.centerX;
  const centerZ = centerSlice.centerZ;
  const shoulderSlice = computeSliceStats(
    points,
    lerp(min[1], max[1], 0.72),
    height * 0.07,
    centerX,
    centerZ
  );
  const hipSlice = computeSliceStats(
    points,
    lerp(min[1], max[1], 0.36),
    height * 0.07,
    centerX,
    centerZ
  );
  const halfShoulder = Math.max((shoulderSlice.maxX - shoulderSlice.minX) * 0.5, width * 0.12);
  const halfHip = Math.max((hipSlice.maxX - hipSlice.minX) * 0.34, width * 0.1);
  const halfHead = width * 0.08;
  const armReach = Math.max(halfShoulder * 1.55, width * 0.3);
  const forearmReach = width * 0.16;
  const kneeOffset = width * 0.08;
  const depthOffset = depth * 0.035;
  const spineLevels = [0.94, 0.84, 0.74, 0.63, 0.53, 0.43, 0.34, 0.24, 0.1];
  const chestBands = [0.76, 0.66, 0.56];

  const nodes = [];
  const edges = [];

  const spineIds = spineLevels.map((t, index) => {
    const id = `spine-${index}`;
    const slice = computeSliceStats(points, lerp(min[1], max[1], t), height * 0.06, centerX, centerZ);
    addNode(nodes, id, `Spine ${index + 1}`, [slice.centerX, lerp(min[1], max[1], t), slice.centerZ]);
    return id;
  });
  addChain(edges, spineIds);

  const headSlice = computeSliceStats(points, lerp(min[1], max[1], 0.9), height * 0.05, centerX, centerZ);
  addNode(nodes, "head-left", "Head Left", [headSlice.centerX - halfHead, lerp(min[1], max[1], 0.9), headSlice.centerZ + depthOffset]);
  addNode(nodes, "head-center", "Head Center", [headSlice.centerX, lerp(min[1], max[1], 0.94), headSlice.centerZ + depthOffset * 1.4]);
  addNode(nodes, "head-right", "Head Right", [headSlice.centerX + halfHead, lerp(min[1], max[1], 0.9), headSlice.centerZ + depthOffset]);
  edges.push(["head-left", "head-center"], ["head-center", "head-right"], ["head-left", spineIds[1]], ["head-right", spineIds[1]]);

  chestBands.forEach((t, bandIndex) => {
    const leftId = `chest-left-${bandIndex}`;
    const rightId = `chest-right-${bandIndex}`;
    const bandY = lerp(min[1], max[1], t);
    const slice = computeSliceStats(points, bandY, height * 0.06, centerX, centerZ);
    const span = Math.max((slice.maxX - slice.minX) * 0.5, lerp(halfHip, halfShoulder, clamp01((t - 0.5) / 0.3)));
    addNode(nodes, leftId, `Chest Left ${bandIndex + 1}`, [slice.centerX - span, bandY, slice.centerZ + depthOffset * 0.5]);
    addNode(nodes, rightId, `Chest Right ${bandIndex + 1}`, [slice.centerX + span, bandY, slice.centerZ + depthOffset * 0.5]);
    edges.push([leftId, rightId]);
    edges.push([leftId, spineIds[bandIndex + 2]]);
    edges.push([rightId, spineIds[bandIndex + 2]]);
  });

  const leftArmIds = [
    ["left-shoulder", [centerX - halfShoulder, lerp(min[1], max[1], 0.73), centerZ]],
    ["left-upper-arm", [centerX - halfShoulder - armReach * 0.45, lerp(min[1], max[1], 0.64), centerZ]],
    ["left-forearm", [centerX - halfShoulder - armReach * 0.8, lerp(min[1], max[1], 0.55), centerZ - depthOffset]],
    ["left-hand", [centerX - halfShoulder - armReach - forearmReach, lerp(min[1], max[1], 0.48), centerZ - depthOffset]],
  ];
  const rightArmIds = [
    ["right-shoulder", [centerX + halfShoulder, lerp(min[1], max[1], 0.73), centerZ]],
    ["right-upper-arm", [centerX + halfShoulder + armReach * 0.45, lerp(min[1], max[1], 0.64), centerZ]],
    ["right-forearm", [centerX + halfShoulder + armReach * 0.8, lerp(min[1], max[1], 0.55), centerZ - depthOffset]],
    ["right-hand", [centerX + halfShoulder + armReach + forearmReach, lerp(min[1], max[1], 0.48), centerZ - depthOffset]],
  ];

  leftArmIds.forEach(([id, position], index) => addNode(nodes, id, `Left Arm ${index + 1}`, position));
  rightArmIds.forEach(([id, position], index) => addNode(nodes, id, `Right Arm ${index + 1}`, position));
  addChain(edges, leftArmIds.map(([id]) => id));
  addChain(edges, rightArmIds.map(([id]) => id));
  edges.push(["left-shoulder", spineIds[2]], ["right-shoulder", spineIds[2]]);

  const leftLegIds = [
    ["left-hip", [centerX - halfHip, lerp(min[1], max[1], 0.36), centerZ]],
    ["left-thigh", [centerX - halfHip - kneeOffset, lerp(min[1], max[1], 0.25), centerZ]],
    ["left-shin", [centerX - halfHip - kneeOffset * 0.7, lerp(min[1], max[1], 0.14), centerZ - depthOffset]],
    ["left-foot", [centerX - halfHip + kneeOffset * 0.3, lerp(min[1], max[1], 0.06), centerZ - depthOffset * 1.2]],
  ];
  const rightLegIds = [
    ["right-hip", [centerX + halfHip, lerp(min[1], max[1], 0.36), centerZ]],
    ["right-thigh", [centerX + halfHip + kneeOffset, lerp(min[1], max[1], 0.25), centerZ]],
    ["right-shin", [centerX + halfHip + kneeOffset * 0.7, lerp(min[1], max[1], 0.14), centerZ - depthOffset]],
    ["right-foot", [centerX + halfHip - kneeOffset * 0.3, lerp(min[1], max[1], 0.06), centerZ - depthOffset * 1.2]],
  ];

  leftLegIds.forEach(([id, position], index) => addNode(nodes, id, `Left Leg ${index + 1}`, position));
  rightLegIds.forEach(([id, position], index) => addNode(nodes, id, `Right Leg ${index + 1}`, position));
  addChain(edges, leftLegIds.map(([id]) => id));
  addChain(edges, rightLegIds.map(([id]) => id));
  edges.push(["left-hip", spineIds[6]], ["right-hip", spineIds[6]], ["left-hip", "right-hip"]);

  return {
    type: "stick-figure",
    nodeCount: nodes.length,
    edgeCount: edges.length,
    connectivity: edges.map(([from, to]) => ({ from, to })),
    nodes,
    metrics: {
      width,
      height,
      depth,
      compactness: clamp01(width / height),
    },
  };
}

export function buildSceneFromGaussianData(gaussianData, meshUrl, meshType) {
  const scene = createEmptyScene();
  if (!gaussianData) {
    return scene;
  }

  const bboxMin = gaussianData.bboxMin || null;
  const bboxMax = gaussianData.bboxMax || null;
  let center = [0, 0, 0];
  if (bboxMin && bboxMax) {
    center = [
      (bboxMin[0] + bboxMax[0]) / 2,
      (bboxMin[1] + bboxMax[1]) / 2,
      (bboxMin[2] + bboxMax[2]) / 2,
    ];
  }

  const gaussianEntity = {
    id: "gaussian-cloud-0",
    name: "Gaussian Cloud",
    components: {
      transform: {
        position: [0, 0, 0],
        rotationEuler: [0, 0, 0],
        scale: [1, 1, 1],
      },
      gaussianCloud: {
        source: "gaussianData",
        // Reference is out-of-band; actual buffers stay in existing renderer.
        dataKey: "primary",
        bboxMin,
        bboxMax,
        center,
      },
      animation: {
        enabled: false,
        clip: "idle", // "idle" | "orbit" | "bounce"
        speed: 1.0,
        amplitude: 1.0,
      },
      physics: {
        enabled: true,
        mode: "gravity", // "floating" | "gravity"
        mass: 1.0,
        restitution: 0.4,
      },
    },
  };

  scene.entities.push(gaussianEntity);
  const stickFigure = computeStickFigureLayout(gaussianData, bboxMin, bboxMax);
  if (stickFigure) {
    scene.entities.push({
      id: "stick-figure-0",
      name: "Stick Figure",
      components: {
        overlay: {
          visible: true,
          color: "#8af8c3",
        },
        stickFigure,
      },
    });
  }

  if (meshUrl) {
    scene.entities.push({
      id: "mesh-surface-0",
      name: "Reconstructed Mesh",
      components: {
        transform: {
          position: [0, 0, 0],
          rotationEuler: [0, 0, 0],
          scale: [1, 1, 1],
        },
        meshSurface: {
          url: meshUrl,
          type: meshType || null,
        },
        animation: {
          enabled: false,
          clip: "idle",
          speed: 1.0,
          amplitude: 1.0,
        },
      },
    });
  }

  // Camera suggestion – actual application is optional and handled by Viewer.
  scene.settings.camera = {
    ...scene.settings.camera,
    suggestedCenter: center,
  };

  if (bboxMin && bboxMax) {
    scene.settings.dynamics = {
      ...scene.settings.dynamics,
      floorOffset: Math.max((bboxMax[1] - bboxMin[1]) * 0.02, 0.01),
    };
  }

  return scene;
}

export function enableDynamics(scene, enabled, options = {}) {
  const next = {
    ...scene,
    settings: {
      ...scene.settings,
      dynamics: {
        ...scene.settings.dynamics,
        enabled,
        ...options,
      },
    },
  };
  return next;
}

export function updateRenderingSettings(scene, partial) {
  return {
    ...scene,
    settings: {
      ...scene.settings,
      rendering: {
        ...scene.settings.rendering,
        ...partial,
      },
    },
  };
}

export function updateDynamicsSettings(scene, partial) {
  return {
    ...scene,
    settings: {
      ...scene.settings,
      dynamics: {
        ...scene.settings.dynamics,
        ...partial,
      },
    },
  };
}

export function updateOverlaySettings(scene, partial) {
  return {
    ...scene,
    settings: {
      ...scene.settings,
      overlays: {
        ...scene.settings.overlays,
        ...partial,
      },
    },
  };
}

export function updateTrainingSettings(scene, partial) {
  return {
    ...scene,
    settings: {
      ...scene.settings,
      training: {
        ...scene.settings.training,
        ...partial,
      },
    },
  };
}

export function toggleEntityAnimation(scene, entityId, enabled, overrides = {}) {
  const entities = scene.entities.map((ent) => {
    if (ent.id !== entityId) return ent;
    const currentAnim = ent.components.animation || {
      enabled: false,
      clip: "idle",
      speed: 1.0,
      amplitude: 1.0,
    };
    return {
      ...ent,
      components: {
        ...ent.components,
        animation: {
          ...currentAnim,
          enabled,
          ...overrides,
        },
      },
    };
  });
  return { ...scene, entities };
}
