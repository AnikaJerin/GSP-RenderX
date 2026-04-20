import os
import uuid

import trimesh

STATIC_DIR = "static"


def _apply_prompt_scaling(mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
    prompt_l = prompt.lower()
    scale = [1.0, 1.0, 1.0]
    if "tall" in prompt_l or "vertical" in prompt_l:
        scale[2] = 1.45
    if "wide" in prompt_l or "broad" in prompt_l:
        scale[0] = 1.35
        scale[1] = 1.2
    if "flat" in prompt_l or "thin" in prompt_l:
        scale[2] = 0.55
    mesh.apply_scale(scale)
    return mesh


def _compose(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    mesh = trimesh.util.concatenate(parts)
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    return mesh


def mesh_from_prompt(prompt: str) -> trimesh.Trimesh:
    prompt_l = prompt.strip().lower()
    if not prompt_l:
        raise ValueError("Prompt is empty.")

    if "table" in prompt_l:
        top = trimesh.creation.box(extents=(1.3, 0.8, 0.12))
        top.apply_translation((0, 0, 0.55))
        legs = []
        for sx in (-0.55, 0.55):
            for sy in (-0.3, 0.3):
                leg = trimesh.creation.box(extents=(0.12, 0.12, 1.0))
                leg.apply_translation((sx, sy, 0.0))
                legs.append(leg)
        mesh = _compose([top, *legs])
    elif "chair" in prompt_l:
        seat = trimesh.creation.box(extents=(0.9, 0.9, 0.12))
        seat.apply_translation((0, 0, 0.15))
        back = trimesh.creation.box(extents=(0.9, 0.12, 1.0))
        back.apply_translation((0, -0.39, 0.65))
        legs = []
        for sx in (-0.35, 0.35):
            for sy in (-0.35, 0.35):
                leg = trimesh.creation.box(extents=(0.1, 0.1, 0.6))
                leg.apply_translation((sx, sy, -0.24))
                legs.append(leg)
        mesh = _compose([seat, back, *legs])
    elif "rocket" in prompt_l:
        body = trimesh.creation.cylinder(radius=0.28, height=1.6, sections=32)
        nose = trimesh.creation.cone(radius=0.28, height=0.5, sections=32)
        nose.apply_translation((0, 0, 1.05))
        fins = []
        for sx, sy in ((0.22, 0), (-0.22, 0), (0, 0.22), (0, -0.22)):
            fin = trimesh.creation.box(extents=(0.08 if sx == 0 else 0.24, 0.08 if sy == 0 else 0.24, 0.35))
            fin.apply_translation((sx, sy, -0.62))
            fins.append(fin)
        mesh = _compose([body, nose, *fins])
    elif "tree" in prompt_l:
        trunk = trimesh.creation.cylinder(radius=0.16, height=1.2, sections=24)
        trunk.apply_translation((0, 0, -0.1))
        crown = trimesh.creation.icosphere(subdivisions=2, radius=0.75)
        crown.apply_translation((0, 0, 0.9))
        mesh = _compose([trunk, crown])
    elif "snowman" in prompt_l:
        base = trimesh.creation.icosphere(subdivisions=2, radius=0.58)
        middle = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        middle.apply_translation((0, 0, 0.8))
        head = trimesh.creation.icosphere(subdivisions=2, radius=0.28)
        head.apply_translation((0, 0, 1.38))
        mesh = _compose([base, middle, head])
    elif "cylinder" in prompt_l or "pillar" in prompt_l or "tube" in prompt_l:
        mesh = trimesh.creation.cylinder(radius=0.42, height=1.8, sections=32)
    elif "cone" in prompt_l or "pyramid" in prompt_l:
        mesh = trimesh.creation.cone(radius=0.6, height=1.4, sections=32)
    elif "capsule" in prompt_l or "human" in prompt_l or "character" in prompt_l:
        mesh = trimesh.creation.capsule(radius=0.35, height=1.2)
    elif "sphere" in prompt_l or "ball" in prompt_l:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.8)
    else:
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    mesh = _apply_prompt_scaling(mesh, prompt_l)
    return mesh


def export_prompt_mesh(prompt: str) -> dict:
    os.makedirs(STATIC_DIR, exist_ok=True)
    mesh = mesh_from_prompt(prompt)
    mesh_name = f"{uuid.uuid4().hex}.ply"
    mesh_path = os.path.join(STATIC_DIR, mesh_name)
    mesh.export(mesh_path)
    return {
        "mesh_path": mesh_path,
        "mesh_url": f"/static/{mesh_name}",
        "mesh_type": "ply",
        "source_prompt": prompt,
    }
