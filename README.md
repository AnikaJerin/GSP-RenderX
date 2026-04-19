# GSP-RenderX
GSP-RenderX is a web-based 3D engine replacing heavy STLs with a custom, compressed GSP format. It uses Edge-Aware Gaussian Splatting to reconstruct objects in real-time. By mapping particles to geometric constraints, it delivers sharp, CAD-quality visuals and ultra-fast, lightweight rendering via Three.js.

## Option A Workflow

For the vessel-centerline research workflow:

- local app + upload flow: `backend` and `frontend` in this repo
- heavy training/inference: Colab
- handoff contract: [OPTION_A_COLAB_WORKFLOW.md](/Users/anika/Documents/GSP-RenderX/docs/OPTION_A_COLAB_WORKFLOW.md)

If you just want to test the vessel path quickly, generate a synthetic bundle locally:

```bash
cd /Users/anika/Documents/GSP-RenderX
backend/gsp-render-venv/bin/python tools/make_sample_vessel_case.py --out sample_vessel_case.npz
```
