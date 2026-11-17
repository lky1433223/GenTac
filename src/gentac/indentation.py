"""Run sphere indentation on the fixed GenTac bench and save sensor-frame outputs."""

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from trimesh.triangles import closest_point

from .markers import bind_markers
from .sensor import MINI_GEL, MINI_MARKERS, GelSpec, MarkerGrid

DEFAULT_RADIUS_M = 0.003
DEFAULT_HYDROELASTIC_MODULUS_PA = 2_000_000.0
_PRESET_REFERENCE = (
    "https://github.com/univtac/UniVTAC/blob/main/third_party/TacEx/source/"
    "tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py"
)


def run(
    output: Path,
    *,
    depth_m: float,
    steps: int,
    save_every: int,
    gel: GelSpec = MINI_GEL,
    markers: MarkerGrid = MINI_MARKERS,
    radius_m: float = DEFAULT_RADIUS_M,
    hydroelastic_modulus_pa: float = DEFAULT_HYDROELASTIC_MODULUS_PA,
    overwrite: bool = False,
    capture_camera: bool = False,
) -> dict:
    """Simulate a fixed, world-aligned sensor and save its surface and marker trajectories."""
    if not np.isfinite(depth_m) or not 0 < depth_m < gel.size_m[2]:
        raise ValueError(
            "Indentation depth must be positive and less than the gel thickness"
        )
    if steps < 2 or save_every < 1:
        raise ValueError("Use at least two steps and a positive save interval")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} already exists. Choose a new directory or pass --overwrite."
        )
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)

    # Load Genesis after argument checks, so CLI help works without initializing the solver.
    from .sap import simulate

    markers_rest = markers.positions(gel)
    state = simulate(
        gel,
        depth_m=depth_m,
        steps=steps,
        save_every=save_every,
        radius_m=radius_m,
        hydroelastic_modulus_pa=hydroelastic_modulus_pa,
        capture_camera=capture_camera,
    )
    rest, positions = state["rest"], state["positions"]
    faces, marker_faces, marker_weights = bind_markers(
        rest, state["tets"], markers_rest
    )
    markers_xyz = np.einsum("fkvi,kv->fki", positions[:, marker_faces], marker_weights)
    marker_disp = markers_xyz - markers_xyz[0]
    surface_vertices, face_indices = np.unique(faces, return_inverse=True)
    surface_rest = rest[surface_vertices]
    surface_xyz = positions[:, surface_vertices]
    surface_faces = face_indices.reshape(faces.shape)

    centers = state["indenter_center"]
    radius = float(state["sphere_radius"])
    if not np.isfinite(surface_xyz).all() or not np.isfinite(markers_xyz).all():
        raise RuntimeError("FEM returned non-finite positions")
    origin = np.zeros((len(surface_faces), 3))
    overlap = []
    for vertices, center in zip(surface_xyz, centers, strict=True):
        # Query in millimeters to keep trimesh's absolute tolerances small
        # relative to the triangles. Save the result in SI.
        nearest = closest_point((vertices[surface_faces] - center) * 1_000, origin)
        distance_m = float(np.linalg.norm(nearest, axis=1).min() / 1_000)
        overlap.append(max(0.0, radius - distance_m))
    contact_face = np.isclose(surface_rest[:, 2], gel.size_m[2], atol=1e-8, rtol=0)
    max_down = float(
        np.max(surface_rest[None, contact_face, 2] - surface_xyz[:, contact_face, 2])
    )
    peak_marker_xy = float(np.max(np.linalg.norm(marker_disp[..., :2], axis=-1)))
    if max_down <= 1e-7:
        raise RuntimeError("The contact face did not measurably deform")

    report = {
        "backend": "genesis-fem-sap",
        "frame": "sensor",
        "mount": {
            "parent": "world",
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "gel": asdict(gel),
        "gel_volume_m3": gel.volume_m3,
        "marker_layout": asdict(markers),
        "preset_dimensions_and_marker_counts_reference": _PRESET_REFERENCE
        if gel.size_m == MINI_GEL.size_m
        and (markers.columns, markers.rows) == (MINI_MARKERS.columns, MINI_MARKERS.rows)
        else None,
        "preset_assumptions": "planar gel, bottom-only bonding, marker margin, and uncalibrated mechanical properties",
        "requested_depth_m": depth_m,
        "sphere_radius_m": radius,
        "hydroelastic_modulus_pa": hydroelastic_modulus_pa,
        "max_surface_down_m": max_down,
        "max_marker_in_plane_m": peak_marker_xy,
        "max_surface_overlap_m": max(overlap),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp)
        np.savez_compressed(
            staging / "state.npz",
            step=state["step"],
            time_s=state["time_s"],
            surface_rest_m=surface_rest,
            surface_xyz_m=surface_xyz,
            surface_faces=surface_faces,
            marker_id=np.arange(len(markers_rest)),
            marker_rest_m=markers_xyz[0],
            marker_xyz_m=markers_xyz,
            marker_disp_m=marker_disp,
            indenter_center_m=centers,
            indenter_radius_m=radius,
            surface_overlap_m=overlap,
            **({"camera_rgb": state["camera_rgb"]} if capture_camera else {}),
        )
        (staging / "run.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        output.mkdir(exist_ok=overwrite)
        for artifact in staging.iterdir():
            artifact.replace(output / artifact.name)
    return report
