# Option A Colab Workflow

This document is the handoff between the deep-learning work in Colab and the local GSP app in this repository.

Actual Colab-side research code now lives in:

- [research README](/Users/anika/Documents/GSP-RenderX/research/option_a/README.md)
- [train.py](/Users/anika/Documents/GSP-RenderX/research/option_a/train.py)
- [infer_export.py](/Users/anika/Documents/GSP-RenderX/research/option_a/infer_export.py)

## Goal

Train the heavy vessel model in Colab, export one prediction bundle as `.npz`, then inspect topology and render the centerline-aware Gaussian result locally.

## What Runs Where

- `Colab`: training, inference, checkpointing, dataset preprocessing
- `Local repo`: upload `.npz`, analyze topology, convert to GSP, inspect the result in the viewer

## Input To The Model

The current local app assumes your Colab model processes a single 3D vessel case at a time.

Recommended model input:

- `volume`: `float32`, shape `(D, H, W)` or `(1, D, H, W)` before batching
- `spacing`: voxel spacing as `(sx, sy, sz)`
- `modality`: `TOF-MRA`

Recommended training target:

- `vessel_mask`: binary vessel segmentation mask with the same spatial shape

Optional future targets:

- `centerline_mask`
- `radius_map`
- `uncertainty_map`

## Output Contract

The local app accepts `.npy` or `.npz`, but `.npz` is the intended format.

Required keys in `.npz`:

- `vessel_mask` or `vessel_prob`

Optional keys:

- `centerline_mask`
- `radius_map`
- `uncertainty_map`
- `spacing`
- `metadata_json`

Shape rules:

- Every 3D array must have the same shape
- `vessel_mask`: binary or integer array
- `vessel_prob`: `float32`, expected in `[0, 1]`
- `centerline_mask`: binary or integer array
- `radius_map`: `float32`
- `uncertainty_map`: `float32`, preferred in `[0, 1]`
- `spacing`: array-like with 3 numbers

## Minimal Colab Export Code

If you only have vessel probabilities:

```python
import numpy as np

vessel_prob = pred_probs.astype(np.float32)        # shape: (D, H, W)
vessel_mask = (vessel_prob > 0.5).astype(np.uint8)
spacing = np.array([0.45, 0.45, 0.60], dtype=np.float32)

np.savez_compressed(
    "case_001_prediction.npz",
    vessel_prob=vessel_prob,
    vessel_mask=vessel_mask,
    spacing=spacing,
)
```

If you also predict centerlines and uncertainty:

```python
import numpy as np

np.savez_compressed(
    "case_001_prediction.npz",
    vessel_prob=vessel_prob.astype(np.float32),
    vessel_mask=(vessel_prob > 0.5).astype(np.uint8),
    centerline_mask=centerline_pred.astype(np.uint8),
    radius_map=radius_pred.astype(np.float32),
    uncertainty_map=uncertainty_pred.astype(np.float32),
    spacing=np.array([0.45, 0.45, 0.60], dtype=np.float32),
    metadata_json=np.array(
        "{'model':'baseline_unet','dataset':'TopCoW-MRA','case':'001'}",
        dtype=object,
    ),
)
```

## Reusable Helper

This repo now includes a helper you can mirror into Colab:

- [vessel_export.py](/Users/anika/Documents/GSP-RenderX/backend/app/services/vessel_export.py)

Equivalent usage:

```python
from backend.app.services.vessel_export import save_vessel_bundle

save_vessel_bundle(
    "case_001_prediction.npz",
    vessel_mask=vessel_mask,
    vessel_prob=vessel_prob,
    centerline_mask=centerline_mask,
    radius_map=radius_map,
    uncertainty_map=uncertainty_map,
    spacing=(0.45, 0.45, 0.60),
    metadata={"model": "baseline_unet", "dataset": "TopCoW-MRA", "case": "001"},
)
```

If importing from this repo is inconvenient in Colab, just use the raw `np.savez_compressed(...)` form above.

## Local Run Commands

Start the backend from the repo root:

```bash
cd /Users/anika/Documents/GSP-RenderX
backend/gsp-render-venv/bin/python -m uvicorn backend.app.main:app --reload
```

Start the frontend in another terminal:

```bash
cd /Users/anika/Documents/GSP-RenderX/frontend
npm run dev
```

Open:

- `http://127.0.0.1:5173`

## How To See Output Right Now

If you want to verify the vessel path before training anything, generate a synthetic test case:

```bash
cd /Users/anika/Documents/GSP-RenderX
backend/gsp-render-venv/bin/python tools/make_sample_vessel_case.py --out sample_vessel_case.npz
```

Then in the UI:

1. Open the app in the browser
2. Use `Vessel Centerline Workflow`
3. Upload `sample_vessel_case.npz`
4. Click `Analyze Vessel Topology`
5. Click `Convert Vessel to GSP`

That should render a branching synthetic vessel and populate the vessel research panel.

## First Colab Baseline

For the first experiment, keep the model simple:

- `Model`: `nnUNet` or plain `3D U-Net`
- `Input`: `TOF-MRA`
- `Target`: `vessel_mask`
- `Loss`: `Dice + BCE`, then add `clDice` later
- `Export`: one `.npz` per validation case

Do not start with graph decoding or uncertainty learning. First make sure:

- predictions upload cleanly
- the viewer exposes topology failures
- your metric scripts agree with what you see qualitatively

## Suggested File Naming

Use predictable names:

- `topcow_mra_001_prediction.npz`
- `topcow_mra_014_prediction.npz`
- `costa_case_008_prediction.npz`

## What The Local App Currently Does With Your Bundle

When you upload a vessel bundle, the backend will:

1. read `vessel_mask` or `vessel_prob`
2. derive or use `centerline_mask`
3. derive or use `radius_map`
4. compute topology-oriented structural features
5. convert centerline points into Gaussian splats
6. send a `.gsp` file to the frontend viewer

Current implementation entry points:

- [upload.py](/Users/anika/Documents/GSP-RenderX/backend/app/api/upload.py)
- [vessel_loader.py](/Users/anika/Documents/GSP-RenderX/backend/app/services/vessel_loader.py)
- [vessel_centerline.py](/Users/anika/Documents/GSP-RenderX/backend/app/services/vessel_centerline.py)

## Immediate Next Step

The fastest real milestone is:

1. train or fake one TopCoW prediction in Colab
2. export one `.npz`
3. upload it locally
4. inspect a failure case
5. only then iterate on topology-aware loss
