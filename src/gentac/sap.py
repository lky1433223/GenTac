"""Fixed-bench spherical indentation with Genesis FEM and SAP contact."""

from pathlib import Path
from tempfile import TemporaryDirectory

import genesis as gs
import numpy as np
from numpy.typing import NDArray

from .sensor import GelSpec, make_gel_mesh

STEP_S = 0.002
GAP_M = 0.0001


def simulate(
    spec: GelSpec,
    *,
    depth_m: float,
    steps: int,
    save_every: int,
    radius_m: float,
    hydroelastic_modulus_pa: float,
    capture_camera: bool = False,
) -> dict[str, NDArray]:
    """Return a sphere-press trajectory in the fixed sensor frame.

    The stationary mount places the sensor frame at the world origin and
    anchors the gel's bonded face.
    """
    width, length, height = spec.size_m
    if spec.bonded_face != "bottom":
        raise ValueError("The fixed bench supports a bottom bonded face")
    if steps < 2 or save_every < 1:
        raise ValueError("Use at least two steps and a positive save interval")
    if not np.isfinite((depth_m, radius_m, hydroelastic_modulus_pa)).all():
        raise ValueError(
            "Indentation, sphere radius and hydroelastic modulus must be finite"
        )
    if not 0 < depth_m < min(height, radius_m):
        raise ValueError(
            "Indentation must be positive and smaller than the gel thickness and sphere radius"
        )
    if hydroelastic_modulus_pa <= 0:
        raise ValueError("Hydroelastic modulus must be positive")

    mesh = make_gel_mesh(spec)
    one_cell_volume = width * length * height / (spec.surface_grid - 1) ** 2
    if not gs._initialized:
        gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    with TemporaryDirectory(prefix="gentac-sap-") as scratch:
        mesh_path = Path(scratch) / "gel.obj"
        mesh.export(mesh_path)
        scene = gs.Scene(
            show_viewer=False,
            renderer=gs.renderers.Rasterizer() if capture_camera else None,
            vis_options=gs.options.VisOptions(
                shadow=False, ambient_light=(0.28, 0.28, 0.28)
            )
            if capture_camera
            else None,
            sim_options=gs.options.SimOptions(
                dt=STEP_S, substeps=2, gravity=(0.0, 0.0, 0.0)
            ),
            fem_options=gs.options.FEMOptions(
                use_implicit_solver=True,
                enable_vertex_constraints=True,
                damping_alpha=2.0,
                damping_beta=0.005,
            ),
            coupler_options=gs.options.SAPCouplerOptions(
                enable_rigid_fem_contact=True,
                enable_fem_self_tet_contact=False,
                rigid_floor_contact_type="none",
                fem_floor_contact_type="none",
                rigid_rigid_contact_type="none",
                sap_taud=0.001,
                # The absolute solver tolerances otherwise swallow millimeter-scale impulses.
                sap_convergence_atol=1e-14,
                pcg_threshold=1e-14,
            ),
        )
        try:
            gel = scene.add_entity(
                gs.morphs.Mesh(
                    file=str(mesh_path),
                    nobisect=False,
                    maxvolume=one_cell_volume / 12,
                    decimate=False,
                    convexify=False,
                ),
                material=gs.materials.FEM.Elastic(
                    E=spec.young_pa,
                    nu=spec.poisson,
                    rho=spec.density,
                    friction_mu=spec.friction,
                    model=spec.model,
                    hydroelastic_modulus=hydroelastic_modulus_pa,
                ),
                surface=gs.surfaces.Default(color=(0.25, 0.8, 0.72, 1.0), smooth=True)
                if capture_camera
                else None,
            )
            start_z = height + radius_m + GAP_M
            sphere = scene.add_entity(
                gs.morphs.Sphere(radius=radius_m, pos=(0.0, 0.0, start_z), fixed=True),
                material=gs.materials.Rigid(coup_friction=spec.friction),
                surface=gs.surfaces.Default(color=(0.94, 0.58, 0.24, 1.0), smooth=True)
                if capture_camera
                else None,
            )
            camera = None
            if capture_camera:
                span = max(width, length, height + 2 * radius_m)
                camera = scene.add_camera(
                    res=(800, 700),
                    pos=(1.3 * span, -1.7 * span, 1.27 * span),
                    lookat=(0, 0, height * 0.9),
                    fov=32,
                    near=0.04 * span,
                    far=8 * span,
                    GUI=False,
                )
            scene.build()

            rest = gel.get_state().pos.detach().cpu().numpy().reshape(-1, 3).copy()
            tets = gel.get_el2v().detach().cpu().numpy().copy()
            bonded = np.flatnonzero(np.isclose(rest[:, 2], 0.0, atol=1e-9)).astype(
                np.int32
            )
            if not bonded.size:
                raise RuntimeError("The FEM mesh has no vertices on the bonded face")
            gel.set_vertex_constraints(bonded, target_poss=rest[bonded])

            frames = [rest]
            centers = [sphere.get_pos().detach().cpu().numpy().reshape(3).copy()]
            frame_steps = [0]
            camera_frames = []
            if camera is not None:
                camera_frames.append(np.asarray(camera.render(rgb=True)[0]).copy())
            for step in range(1, steps + 1):
                phase = min(step / (0.8 * steps), 1.0)
                travel = (depth_m + GAP_M) * phase**2 * (3 - 2 * phase)
                sphere.set_pos((0.0, 0.0, start_z - travel))
                scene.step()
                if step % save_every == 0 or step == steps:
                    frames.append(
                        gel.get_state().pos.detach().cpu().numpy().reshape(-1, 3).copy()
                    )
                    centers.append(
                        sphere.get_pos().detach().cpu().numpy().reshape(3).copy()
                    )
                    frame_steps.append(step)
                    if camera is not None:
                        camera_frames.append(
                            np.asarray(camera.render(rgb=True)[0]).copy()
                        )
        finally:
            scene.destroy()

    positions = np.stack(frames)
    indenter_center = np.stack(centers)
    if not (np.isfinite(positions).all() and np.isfinite(indenter_center).all()):
        raise RuntimeError("The SAP indentation produced non-finite positions")
    if np.max(np.linalg.norm(positions[-1] - rest, axis=1)) < 1e-9:
        raise RuntimeError("The SAP indentation produced no measurable gel response")

    state = {
        "rest": rest,
        "tets": tets,
        "positions": positions,
        "step": np.asarray(frame_steps),
        "time_s": STEP_S * np.asarray(frame_steps),
        "indenter_center": indenter_center,
        "sphere_radius": np.asarray(radius_m),
    }

    if capture_camera:
        state["camera_rgb"] = np.stack(camera_frames)
    return state
