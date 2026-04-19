from __future__ import annotations

import json
import os
from typing import Any, Dict


def get_learned_mesh_manifest_template() -> Dict[str, Any]:
    """
    Stable contract for future Colab-trained mesh allocators.
    """
    return {
        "model_name": "geometry_budget_allocator_v1",
        "problem_focus": "edge_and_thin_structure_fidelity_under_budget",
        "input_feature_names": [
            "vertex_count",
            "face_count",
            "bbox_extent_x",
            "bbox_extent_y",
            "bbox_extent_z",
            "bbox_diagonal",
            "sharp_edge_count",
            "sharp_edge_ratio",
            "mean_edge_length",
            "thin_structure_score",
            "mean_adjacency_angle_deg",
        ],
        "target_names": [
            "surface_size_scale",
            "edge_size_scale",
            "edge_oversample_scale",
            "budget_priority",
        ],
        "checkpoint_format": "json_or_pt_path",
        "notes": (
            "This manifest describes the future learned mesh model that will map "
            "mesh structural features to Gaussian allocation policy."
        ),
    }


def get_learned_vessel_manifest_template() -> Dict[str, Any]:
    """
    Stable contract for Colab-trained vessel centerline models.
    """
    return {
        "model_name": "vessel_centerline_topology_v1",
        "problem_focus": "topology_preserving_centerline_reconstruction",
        "supported_inputs": [
            "tof_mra_volume",
            "dsa_volume_or_phase_stack",
            "predicted_vessel_probability_volume",
        ],
        "accepted_upload_formats": {
            ".npy": "Single 3D vessel mask/probability volume.",
            ".npz": {
                "preferred_keys": [
                    "vessel_mask",
                    "vessel_prob",
                    "centerline_mask",
                    "radius_map",
                    "uncertainty_map",
                    "spacing",
                ],
                "required_any_of": ["vessel_mask", "vessel_prob", "volume", "segmentation"],
            },
        },
        "notes": (
            "Train the heavy model in Colab, then export an .npz bundle containing "
            "the predicted vessel mask or probabilities, and optionally centerline, "
            "radius, uncertainty, and spacing arrays for local visualization."
        ),
        "target_outputs": [
            "vessel_mask",
            "centerline_mask",
            "radius_map",
            "branchpoint_logits_or_count",
            "endpoint_logits_or_count",
        ],
    }


def get_learned_mesh_model_state() -> Dict[str, Any]:
    """
    Report whether a learned mesh checkpoint/config is available locally.
    This keeps the backend honest about when it is doing real model inference
    versus a heuristic fallback.
    """
    model_path = os.getenv("LEARNED_MESH_MODEL_PATH", "").strip()
    if not model_path:
        return {
            "ready": False,
            "status": "missing_path",
            "detail": "Set LEARNED_MESH_MODEL_PATH to a model manifest or checkpoint path.",
            "expected_manifest": get_learned_mesh_manifest_template(),
        }

    if not os.path.exists(model_path):
        return {
            "ready": False,
            "status": "missing_file",
            "model_path": model_path,
            "detail": "Configured learned mesh model path does not exist.",
            "expected_manifest": get_learned_mesh_manifest_template(),
        }

    state: Dict[str, Any] = {
        "ready": True,
        "status": "configured",
        "model_path": model_path,
        "expected_manifest": get_learned_mesh_manifest_template(),
    }

    if model_path.endswith(".json"):
        try:
            with open(model_path, "r", encoding="utf-8") as fh:
                state["manifest"] = json.load(fh)
        except Exception as exc:
            state["ready"] = False
            state["status"] = "invalid_manifest"
            state["detail"] = f"Could not parse learned mesh manifest: {exc}"

    return state


def get_learned_vessel_model_state() -> Dict[str, Any]:
    """
    Report whether a learned vessel checkpoint/config is available locally.
    The local app can visualize Colab-exported predictions even without this,
    but this state makes the contract explicit.
    """
    model_path = os.getenv("LEARNED_VESSEL_MODEL_PATH", "").strip()
    if not model_path:
        return {
            "ready": False,
            "status": "missing_path",
            "detail": "Set LEARNED_VESSEL_MODEL_PATH to a model manifest or checkpoint path.",
            "expected_manifest": get_learned_vessel_manifest_template(),
        }

    if not os.path.exists(model_path):
        return {
            "ready": False,
            "status": "missing_file",
            "model_path": model_path,
            "detail": "Configured learned vessel model path does not exist.",
            "expected_manifest": get_learned_vessel_manifest_template(),
        }

    state: Dict[str, Any] = {
        "ready": True,
        "status": "configured",
        "model_path": model_path,
        "expected_manifest": get_learned_vessel_manifest_template(),
    }

    if model_path.endswith(".json"):
        try:
            with open(model_path, "r", encoding="utf-8") as fh:
                state["manifest"] = json.load(fh)
        except Exception as exc:
            state["ready"] = False
            state["status"] = "invalid_manifest"
            state["detail"] = f"Could not parse learned vessel manifest: {exc}"

    return state
