"""Fixed-bench spherical indentation with Genesis FEM and SAP contact."""

from pathlib import Path
from tempfile import TemporaryDirectory

import genesis as gs
import numpy as np
from numpy.typing import NDArray

from .sensor import MINI_HOUSING, GelSpec, HousingSpec, make_gel_mesh

STEP_S = 0.002
GAP_M = 0.0001
STAGE_KP = 40_000.0
STAGE_KV = 60.0
STAGE_MAX_FORCE_N = 30.0
STAGE_LINK_MASS_KG = 0.02


def _stage_urdf(radius_m: float) -> str:
    inertia = 0.4 * STAGE_LINK_MASS_KG * radius_m**2
    return f"""<robot name="indentation_stage">
      <link name="base"/>
      <link name="carriage">
        <inertial><mass value="{STAGE_LINK_MASS_KG}"/><inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial>
      </link>
      <link name="indenter">
        <inertial><mass value="{STAGE_LINK_MASS_KG}"/><inertia ixx="{inertia}" iyy="{inertia}" izz="{inertia}" ixy="0" ixz="0" iyz="0"/></inertial>
        <visual><geometry><sphere radius="{radius_m}"/></geometry></visual>
        <collision><geometry><sphere radius="{radius_m}"/></geometry></collision>
      </link>
      <joint name="slide_x" type="prismatic">
        <parent link="base"/><child link="carriage"/><axis xyz="1 0 0"/>
        <limit lower="-1" upper="1" effort="{STAGE_MAX_FORCE_N}" velocity="1"/>
      </joint>
      <joint name="slide_z" type="prismatic">
        <parent link="carriage"/><child link="indenter"/><axis xyz="0 0 1"/>
        <limit lower="-1" upper="1" effort="{STAGE_MAX_FORCE_N}" velocity="1"/>
      </joint>
    </robot>"""


def _smooth_ramp(phase: float) -> float:
    phase = min(max(phase, 0.0), 1.0)
    return phase**2 * (3 - 2 * phase)


def simulate(
    spec: GelSpec,
    *,
    depth_m: float,
    steps: int,
    save_every: int,
    radius_m: float,
    hydroelastic_modulus_pa: float,
    capture_camera: bool = False,
    housing: HousingSpec = MINI_HOUSING,
    slide_m: float = 0.0,
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
    if not np.isfinite(slide_m) or abs(slide_m) + radius_m > width / 2:
        raise ValueError("Slide must keep the sphere footprint within the gel along X")
    if hydroelastic_modulus_pa <= 0:
        raise ValueError("Hydroelastic modulus must be positive")

    mesh = make_gel_mesh(spec)
    housing_size = housing.size_m(spec)
    housing_center = (0.0, 0.0, -housing_size[2] / 2)
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
            # The rigid backing transfers motion through the bond constraint.
            # ponytail: use a solid visual backing until a shell needs collision geometry.
            housing_body = scene.add_entity(
                gs.morphs.Box(
                    size=housing_size, pos=housing_center, fixed=True, collision=False
                ),
                material=gs.materials.Rigid(),
                surface=gs.surfaces.Default(color=(0.32, 0.37, 0.48, 1.0)),
                name="sensor_housing",
            )
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
            stage_path = Path(scratch) / "indentation_stage.urdf"
            stage_path.write_text(_stage_urdf(radius_m), encoding="utf-8")
            stage = scene.add_entity(
                gs.morphs.URDF(
                    file=str(stage_path), pos=(0.0, 0.0, start_z), fixed=True
                ),
                material=gs.materials.Rigid(coup_friction=spec.friction),
                surface=gs.surfaces.Default(color=(0.94, 0.58, 0.24, 1.0), smooth=True)
                if capture_camera
                else None,
            )
            camera = None
            if capture_camera:
                span = max(*housing_size[:2], housing_size[2] + height + 2 * radius_m)
                camera = scene.add_camera(
                    res=(800, 700),
                    pos=(1.3 * span, -1.7 * span, 1.27 * span),
                    lookat=(0, 0, (height - housing_size[2]) / 2),
                    fov=32,
                    near=0.04 * span,
                    far=8 * span,
                    GUI=False,
                )
            scene.build()
            sphere = stage.get_link("indenter")
            dofs = [
                stage.get_joint(name).dofs_idx_local[0]
                for name in ("slide_x", "slide_z")
            ]
            stage.set_dofs_kp([STAGE_KP, STAGE_KP], dofs)
            stage.set_dofs_kv([STAGE_KV, STAGE_KV], dofs)
            stage.set_dofs_force_range(
                [-STAGE_MAX_FORCE_N] * 2, [STAGE_MAX_FORCE_N] * 2, dofs
            )

            rest = gel.get_state().pos.detach().cpu().numpy().reshape(-1, 3).copy()
            tets = gel.get_el2v().detach().cpu().numpy().copy()
            bonded = np.flatnonzero(np.isclose(rest[:, 2], 0.0, atol=1e-9)).astype(
                np.int32
            )
            if not bonded.size:
                raise RuntimeError("The FEM mesh has no vertices on the bonded face")
            gel.set_vertex_constraints(bonded, link=housing_body.base_link)

            frames = [rest]
            centers = [sphere.get_pos().detach().cpu().numpy().reshape(3).copy()]
            target_centers = [np.array([0.0, 0.0, start_z])]
            velocities = [np.zeros(3)]
            frame_steps = [0]
            camera_frames = []
            if camera is not None:
                camera_frames.append(np.asarray(camera.render(rgb=True)[0]).copy())
            for step in range(1, steps + 1):
                fraction = step / steps
                press = _smooth_ramp(fraction / (0.45 if slide_m else 0.8))
                travel = (depth_m + GAP_M) * press
                lateral = slide_m * _smooth_ramp((fraction - 0.55) / 0.30)
                stage.control_dofs_position([lateral, -travel], dofs)
                scene.step()
                if step % save_every == 0 or step == steps:
                    frames.append(
                        gel.get_state().pos.detach().cpu().numpy().reshape(-1, 3).copy()
                    )
                    centers.append(
                        sphere.get_pos().detach().cpu().numpy().reshape(3).copy()
                    )
                    target_centers.append(np.array([lateral, 0.0, start_z - travel]))
                    vx, vz = (
                        stage.get_dofs_velocity(dofs).detach().cpu().numpy().reshape(2)
                    )
                    velocities.append(np.array([vx, 0.0, vz]))
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
        "indenter_target_center": np.stack(target_centers),
        "indenter_velocity": np.stack(velocities),
        "sphere_radius": np.asarray(radius_m),
        "housing_size_m": np.asarray(housing_size),
        "housing_center_m": np.asarray(housing_center),
    }

    if capture_camera:
        state["camera_rgb"] = np.stack(camera_frames)
    return state
