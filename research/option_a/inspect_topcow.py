from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.option_a.topcow_data import (
    _candidate_image_dirs,
    _candidate_label_dirs,
    discover_topcow_cases,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect a TopCoW folder layout.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="topcow", choices=["topcow", "topbrain"])
    parser.add_argument("--modality", default="mr", choices=["mr", "ct", "cta", "mra"])
    args = parser.parse_args()

    root = Path(args.data_root)
    report = {
        "root_exists": root.exists(),
        "candidate_image_dirs": [
            str(p) for p in _candidate_image_dirs(root, dataset_name=args.dataset, modality=args.modality)
        ],
        "candidate_label_dirs": [
            str(p) for p in _candidate_label_dirs(root, dataset_name=args.dataset, modality=args.modality)
        ],
    }

    try:
        cases = discover_topcow_cases(root, dataset_name=args.dataset, modality=args.modality)
        report["num_cases"] = len(cases)
        report["sample_cases"] = [
            {
                "case_id": c.case_id,
                "image_path": str(c.image_path),
                "label_path": str(c.label_path),
            }
            for c in cases[:5]
        ]
    except Exception as exc:
        report["discovery_error"] = str(exc)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
