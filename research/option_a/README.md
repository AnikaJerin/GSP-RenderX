# Option A Research Code

This folder contains the actual Colab-side research code for the first publishable version of Option A.

## Method

`TopologyCenterlineUNet`

- 3D residual U-Net backbone
- vessel segmentation head
- centerline auxiliary head
- branchpoint head
- radius head
- centerline-conditioned fusion before final segmentation
- topology-sensitive training loss with `Dice + BCE + centerline BCE + branchpoint BCE + radius SmoothL1 + soft clDice`

This version is meant to move beyond "segmentation plus visualization" by making learned structure drive both the segmentation and the exported Gaussian attributes.

## Files

- `topcow_data.py`: TopCoW MRA loader, normalization, centerline proxy target generation
- `model.py`: dual-head 3D U-Net
- `losses.py`: topology-aware training losses
- `metrics.py`: Dice, clDice, endpoint and branchpoint reporting
- `train.py`: training entrypoint
- `infer_export.py`: checkpoint inference and `.npz` export for the local app

## Colab Install

```bash
pip install -r /content/GSP-RenderX/research/option_a/requirements-colab.txt
```

## Train

```bash
python -m research.option_a.train \
  --data-root "/content/drive/MyDrive/TopCoW" \
  --output-dir "/content/drive/MyDrive/gsp_option_a_runs/run_001" \
  --dataset topcow \
  --modality mr \
  --epochs 120 \
  --batch-size 1 \
  --patch-size 96 96 96 \
  --num-workers 2 \
  --device cuda
```

For your `TopBrain` MR layout:

```bash
python -m research.option_a.train \
  --data-root "/content/drive/MyDrive/TopBrain_Data_Release_Batches1n2_081425" \
  --output-dir "/content/drive/MyDrive/gsp_option_a_runs/topbrain_mr_run_001" \
  --dataset topbrain \
  --modality mr \
  --epochs 120 \
  --batch-size 1 \
  --patch-size 96 96 96 \
  --num-workers 2 \
  --device cuda
```

If you want to check discovery before training:

```bash
python -m research.option_a.inspect_topcow \
  --data-root "/content/drive/MyDrive/TopBrain_Data_Release_Batches1n2_081425" \
  --dataset topbrain \
  --modality mr
```

## Export One Case For Local Viewer

```bash
python -m research.option_a.infer_export \
  --checkpoint "/content/drive/MyDrive/gsp_option_a_runs/run_001/best.pt" \
  --image "/content/drive/MyDrive/TopCoW/imagesTr/topcow_mr_001_0000.nii.gz" \
  --out "/content/drive/MyDrive/gsp_exports/topcow_mr_001_prediction.npz" \
  --device cuda
```

Then download that `.npz` or access it from your machine and upload it into the local viewer.
