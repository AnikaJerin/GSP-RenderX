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
        mode: "orbit", // "orbit" | "oscillate" | "none"
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
        enabled: false,
        mode: "floating", // "floating" | "gravity"
        mass: 1.0,
        restitution: 0.4,
      },
    },
  };

  scene.entities.push(gaussianEntity);

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

