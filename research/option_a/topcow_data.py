from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


def normalize_mra(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    foreground = volume > np.percentile(volume, 35)
    if np.count_nonzero(foreground) < 16:
        foreground = np.ones_like(volume, dtype=bool)
    mean = float(volume[foreground].mean())
    std = float(volume[foreground].std()) + 1e-6
    return (volume - mean) / std


def resize_volume_nearest(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    zoom = [t / s for t, s in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, zoom=zoom, order=0)


def resize_volume_linear(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    zoom = [t / s for t, s in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, zoom=zoom, order=1)


def compute_centerline_proxy(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    if np.count_nonzero(mask) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    distance = ndimage.distance_transform_edt(mask.astype(bool), sampling=spacing)
    local_max = distance == ndimage.maximum_filter(distance, size=3, mode="nearest")
    centerline = (mask > 0) & local_max & (distance > max(spacing) * 0.35)

    if np.count_nonzero(centerline) == 0:
        relaxed = distance == ndimage.maximum_filter(distance, size=5, mode="nearest")
        centerline = (mask > 0) & relaxed & (distance > 0)

    neighbor_kernel = np.ones((3, 3, 3), dtype=np.uint8)
    neighbors = ndimage.convolve(centerline.astype(np.uint8), neighbor_kernel, mode="constant")
    neighbors = neighbors - centerline.astype(np.uint8)
    centerline = centerline & (neighbors > 0)

    if np.count_nonzero(centerline) == 0:
        centerline = mask > 0

    return centerline.astype(np.uint8)


def random_crop_around_mask(
    image: np.ndarray,
    mask: np.ndarray,
    centerline: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = np.array(image.shape)
    patch = np.array(patch_size)

    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        center = shape // 2
    else:
        center = coords[random.randrange(len(coords))]

    low = np.maximum(0, center - patch // 2)
    high = np.minimum(shape, low + patch)
    low = np.maximum(0, high - patch)

    slices = tuple(slice(int(l), int(h)) for l, h in zip(low, high))
    img_crop = image[slices]
    mask_crop = mask[slices]
    center_crop = centerline[slices]

    pad = [(0, max(0, p - c)) for p, c in zip(patch_size, img_crop.shape)]
    if any(x[1] > 0 for x in pad):
        img_crop = np.pad(img_crop, pad, mode="constant")
        mask_crop = np.pad(mask_crop, pad, mode="constant")
        center_crop = np.pad(center_crop, pad, mode="constant")

    return img_crop, mask_crop, center_crop


def maybe_flip(image: np.ndarray, mask: np.ndarray, centerline: np.ndarray):
    for axis in range(3):
        if random.random() < 0.5:
            image = np.flip(image, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()
            centerline = np.flip(centerline, axis=axis).copy()
    return image, mask, centerline


@dataclass
class TopCoWCase:
    case_id: str
    image_path: Path
    label_path: Path


def discover_topcow_cases(
    root: str | Path,
    split: str = "train",
    modality: str = "mr",
) -> list[TopCoWCase]:
    root = Path(root)
    images_dir = root / "imagesTr"
    labels_dir = root / "cow_seg_labelsTr"

    pattern = f"topcow_{modality}_*_0000.nii.gz"
    image_paths = sorted(images_dir.glob(pattern))
    if not image_paths:
        raise FileNotFoundError(f"No TopCoW images found in {images_dir} with pattern {pattern}")

    all_cases: list[TopCoWCase] = []
    for image_path in image_paths:
        parts = image_path.name.split("_")
        case_id = parts[2]
        label_name = image_path.name.replace("_0000.nii.gz", ".nii.gz").replace(
            f"topcow_{modality}_", f"topcow_{modality}_seg_"
        )
        label_path = labels_dir / label_name
        if not label_path.exists():
            alt = labels_dir / image_path.name.replace("_0000.nii.gz", ".nii.gz")
            if alt.exists():
                label_path = alt
            else:
                raise FileNotFoundError(f"Missing label for {image_path.name}")
        all_cases.append(TopCoWCase(case_id=case_id, image_path=image_path, label_path=label_path))

    random.Random(7).shuffle(all_cases)
    n = len(all_cases)
    train_end = max(1, int(round(n * 0.8)))
    val_end = max(train_end + 1, int(round(n * 0.9)))
    if split == "train":
        return all_cases[:train_end]
    if split == "val":
        return all_cases[train_end:val_end]
    if split == "test":
        return all_cases[val_end:]
    raise ValueError(f"Unsupported split: {split}")


class TopCoWMRADataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: tuple[int, int, int] = (96, 96, 96),
        use_patches: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.cases = discover_topcow_cases(root, split=split, modality="mr")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int):
        case = self.cases[index]

        image_nii = nib.load(str(case.image_path))
        label_nii = nib.load(str(case.label_path))

        image = normalize_mra(np.asarray(image_nii.get_fdata(), dtype=np.float32))
        label = np.asarray(label_nii.get_fdata(), dtype=np.int16)
        vessel_mask = (label > 0).astype(np.uint8)
        spacing = tuple(float(x) for x in image_nii.header.get_zooms()[:3])
        centerline = compute_centerline_proxy(vessel_mask, spacing=spacing)

        if self.use_patches and self.split == "train":
            image, vessel_mask, centerline = random_crop_around_mask(
                image,
                vessel_mask,
                centerline,
                patch_size=self.patch_size,
            )
            image, vessel_mask, centerline = maybe_flip(image, vessel_mask, centerline)

        image_t = torch.from_numpy(image[None, ...].astype(np.float32))
        vessel_t = torch.from_numpy(vessel_mask[None, ...].astype(np.float32))
        center_t = torch.from_numpy(centerline[None, ...].astype(np.float32))

        return {
            "image": image_t,
            "mask": vessel_t,
            "centerline": center_t,
            "case_id": case.case_id,
            "spacing": torch.tensor(spacing, dtype=torch.float32),
            "image_path": str(case.image_path),
            "label_path": str(case.label_path),
        }
