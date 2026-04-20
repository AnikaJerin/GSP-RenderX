from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research.option_a.losses import topology_centerline_loss
from research.option_a.metrics import topology_report
from research.option_a.model import TopologyCenterlineUNet
from research.option_a.topcow_data import TopCoWMRADataset


def seed_everything(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = True,
    max_batches: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running: dict[str, float] = {}
    num_batches = 0

    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            enabled=use_amp and device.type == "cuda",
        ):
            outputs = model(batch["image"])
            loss, stats = topology_centerline_loss(
                outputs,
                target_mask=batch["mask"],
                target_centerline=batch["centerline"],
                target_radius=batch.get("radius_map"),
                target_branchpoints=batch.get("branchpoints"),
            )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break

    return {k: v / max(num_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate_casewise(
    model: torch.nn.Module,
    dataset: TopCoWMRADataset,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    torch.cuda.empty_cache() # Clear training leftovers
    reports = []

    for sample in tqdm(dataset, leave=False):
        # image shape is likely [C, D, H, W]
        image = sample["image"][None, ...].to(device) 
        
        outputs = model(image)
        
        # IMMEDIATELY move to CPU to free up GPU space
        seg_logits = outputs["seg_logits"].detach().cpu()
        del outputs # Delete the large dictionary from GPU
        
        seg_prob = torch.sigmoid(seg_logits).squeeze().numpy()
        # seg_prob = torch.sigmoid(outputs["seg_logits"]).squeeze().cpu().numpy()
        pred = (seg_prob > 0.5).astype(np.uint8)
        target = sample["mask"].squeeze().cpu().numpy().astype(np.uint8)
        spacing = tuple(float(x) for x in sample["spacing"].cpu().numpy().tolist())
        reports.append(topology_report(pred, target, spacing=spacing))

    keys = reports[0].keys() if reports else []
    return {k: float(np.mean([r[k] for r in reports])) for k in keys}


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if isinstance(payload, dict) and "epoch" in payload:
        return int(payload["epoch"])
    return 0


def main():
    parser = argparse.ArgumentParser(description="Train Option A on TopCoW/TopBrain vessel data")
    parser.add_argument("--data-root", required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", default="artifacts/option_a", help="Where checkpoints and logs go")
    parser.add_argument("--dataset", default="topcow", choices=["topcow", "topbrain"])
    parser.add_argument("--modality", default="mr", choices=["mr", "ct", "cta", "mra"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="", help="Resume from checkpoint path; defaults to output-dir/latest.pt")
    parser.add_argument("--save-every", type=int, default=1, help="Save latest checkpoint every N epochs")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Optional cap on train batches per epoch")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    seed_everything(7)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_ds = TopCoWMRADataset(
        args.data_root,
        split="train",
        patch_size=tuple(args.patch_size),
        use_patches=True,
        dataset_name=args.dataset,
        modality=args.modality,
    )
    val_ds = TopCoWMRADataset(
    args.data_root,
    split="val",
    patch_size=tuple(args.patch_size),
    use_patches=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TopologyCenterlineUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    best_cldice = -1.0
    history = []
    start_epoch = 1

    resume_path = Path(args.resume) if args.resume else out_dir / "latest.pt"
    if resume_path.exists():
        loaded_epoch = load_checkpoint(resume_path, model, optimizer=optimizer)
        start_epoch = loaded_epoch + 1
        print(json.dumps({"resume_from": str(resume_path), "start_epoch": start_epoch}))

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=not args.no_amp,
            max_batches=args.max_train_batches or None,
        )
        val_stats = {}
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **train_stats,
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        history.append(row)

        if epoch % args.save_every == 0:
            latest_path = out_dir / "latest.pt"
            save_checkpoint(latest_path, model, optimizer, epoch)
        save_checkpoint(out_dir / "best.pt", model, optimizer, epoch)

        print(json.dumps(row))

        with open(out_dir / "history.json", "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)


if __name__ == "__main__":
    main()
