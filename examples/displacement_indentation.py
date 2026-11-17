"""Run a displacement-controlled FEM indentation and capture the gel surface."""

import argparse
import tempfile
from pathlib import Path

import genesis as gs
import numpy as np
import trimesh
from PIL import Image

PAD_L = PAD_W = 0.020
PAD_H = 0.005
PAD_GRID = 25
DT = 1 / 120
SPHERE_R = 0.0025
CUBE_S = 0.004
CYL_R = 0.002
CYL_H = 0.008
WEDGE_S = 0.004
SHAPES = ("sphere", "cube", "cylinder", "wedge")

def build_pad_mesh(
    size: tuple[float, float, float] = (PAD_L, PAD_W, PAD_H), n: int = PAD_GRID
) -> trimesh.Trimesh:
    """Closed box with a single welded vertex set and planar UVs.

    Only the top and bottom grids carry vertices; the side walls reuse their
    boundary loops, so the surface is watertight and manifold, which is what
    tetgen needs for a clean tetrahedralisation. Every vertex gets the planar
    projection of its own (x, y) as UV, so the marker lattice on the contact
    face is undistorted and no UV seam splits the mesh.
    """
    lx, ly, h = size
    top = np.zeros((n, n), dtype=np.int64)
    bottom = np.zeros((n, n), dtype=np.int64)
    verts: list[list[float]] = []
    uvs: list[list[float]] = []

    def add(x, y, z):
        verts.append([x, y, z])
        uvs.append([(x + lx / 2) / lx, (y + ly / 2) / ly])
        return len(verts) - 1

    for j in range(n):
        for i in range(n):
            x = -lx / 2 + lx * i / (n - 1)
            y = -ly / 2 + ly * j / (n - 1)
            bottom[j, i] = add(x, y, 0.0)
            top[j, i] = add(x, y, h)

    faces: list[list[int]] = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b = top[j, i], top[j, i + 1]
            c, d = top[j + 1, i], top[j + 1, i + 1]
            faces += [[a, b, d], [a, d, c]]
            a, b = bottom[j, i], bottom[j, i + 1]
            c, d = bottom[j + 1, i], bottom[j + 1, i + 1]
            faces += [[a, d, b], [a, c, d]]

    # Side walls stitch the two boundary loops together.
    for k in range(n - 1):
        pairs = [
            (top[0, k], top[0, k + 1], bottom[0, k], bottom[0, k + 1]),
            (top[n - 1, k], top[n - 1, k + 1], bottom[n - 1, k], bottom[n - 1, k + 1]),
            (top[k, 0], top[k + 1, 0], bottom[k, 0], bottom[k + 1, 0]),
            (top[k, n - 1], top[k + 1, n - 1], bottom[k, n - 1], bottom[k + 1, n - 1]),
        ]
        for t0, t1, b0, b1 in pairs:
            faces += [[t0, b0, b1], [t0, b1, t1]]

    mesh = trimesh.Trimesh(
        np.asarray(verts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        process=False,
        visual=trimesh.visual.TextureVisuals(
            uv=np.asarray(uvs, dtype=np.float64),
            material=trimesh.visual.material.SimpleMaterial(
                image=Image.new("RGB", (8, 8), (205, 205, 205))
            ),
        ),
    )
    trimesh.repair.fix_normals(mesh)
    return mesh


def make_marker_texture(
    size: int = 512, cells: int = 20, bg: int = 205, fg: int = 35
) -> np.ndarray:
    """Regular dot lattice: stretch and shear show up as a deformed lattice."""
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    cell = size / cells
    radius = cell * 0.22
    for j in range(cells):
        for i in range(cells):
            cx = (i + 0.5) * cell
            cy = (j + 0.5) * cell
            img[(xx - cx) ** 2 + (yy - cy) ** 2 < radius**2] = (fg, fg, fg)
    return img


def make_indenter(shape: str) -> trimesh.Trimesh:
    """Indenter mesh in its own frame, contact feature at local z = 0."""
    if shape == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=SPHERE_R)
        mesh.apply_translation((0.0, 0.0, SPHERE_R))
        return mesh
    if shape == "cube":
        mesh = trimesh.creation.box(extents=(CUBE_S, CUBE_S, CUBE_S))
        mesh.apply_translation((0.0, 0.0, CUBE_S / 2))
        return mesh
    if shape == "cylinder":
        mesh = trimesh.creation.cylinder(radius=CYL_R, height=CYL_H, sections=48)
        # Axis along Y, so the pad is pressed by the curved side (line contact).
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, (1.0, 0.0, 0.0))
        )
        mesh.apply_translation((0.0, 0.0, CYL_R))
        return mesh
    if shape == "wedge":
        # A box turned 45 degrees about Y presents a knife edge along Y to the pad.
        mesh = trimesh.creation.box(extents=(WEDGE_S, WEDGE_S, WEDGE_S))
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 4, (0.0, 1.0, 0.0))
        )
        mesh.apply_translation((0.0, 0.0, WEDGE_S / np.sqrt(2.0)))
        return mesh
    raise ValueError(f"unknown indenter shape {shape!r}")


def indenter_lower_z(shape: str, xy: np.ndarray) -> np.ndarray:
    """Height above the tip at each (x, y), or infinity outside the footprint."""
    x, y = xy[:, 0], xy[:, 1]
    out = np.full(x.shape, np.inf, dtype=np.float64)
    if shape == "sphere":
        r = np.hypot(x, y)
        inside = r < SPHERE_R
        out[inside] = SPHERE_R - np.sqrt(SPHERE_R**2 - r[inside] ** 2)
    elif shape == "cube":
        inside = (np.abs(x) <= CUBE_S / 2) & (np.abs(y) <= CUBE_S / 2)
        out[inside] = 0
    elif shape == "cylinder":
        inside = (np.abs(x) <= CYL_R) & (np.abs(y) <= CYL_H / 2)
        out[inside] = (
            CYL_R - np.sqrt(np.maximum(CYL_R**2 - x[inside] ** 2, 0.0))
        )
    elif shape == "wedge":
        half = WEDGE_S / np.sqrt(2.0)
        inside = (np.abs(x) <= half) & (np.abs(y) <= WEDGE_S / 2)
        out[inside] = np.abs(x[inside])
    return out


def positions(pad):
    return pad.get_state().pos.detach().cpu().numpy().reshape(-1, 3).copy()


def run(shape, args, assets):
    """Build one scene and save synchronized images and FEM state."""
    pad_obj = assets / "pad.obj"
    marker_png = assets / "markers.png"
    build_pad_mesh().export(pad_obj)
    Image.fromarray(make_marker_texture()).save(marker_png)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, substeps=4, gravity=(0, 0, 0)),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            enable_vertex_constraints=True,
            damping_alpha=1.0,
            damping_beta=0.005,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_fem=False),
        vis_options=gs.options.VisOptions(
            shadow=False,
            ambient_light=(0.15, 0.15, 0.15),
            lights=[
                {"type": "directional", "dir": (-1, -0.25, -0.3),
                 "color": (1, 1, 1), "intensity": 4.0},
                {"type": "directional", "dir": (0.6, 0.9, -0.3),
                 "color": (0.4, 0.4, 0.4), "intensity": 1.0},
            ],
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=args.vis,
    )
    pad = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(pad_obj), nobisect=False, maxvolume=1.2e-9,
            decimate=False, convexify=False,
        ),
        material=gs.materials.FEM.Elastic(E=5e4, nu=0.45, rho=1000, model="linear_corotated"),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(image_path=str(marker_png)),
            vis_mode="visual",
            smooth=True,
        ),
    )
    gap = 0.0002
    indenter = None
    if args.show_indenter:
        indenter_obj = assets / f"{shape}.obj"
        make_indenter(shape).export(indenter_obj)
        indenter = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=str(indenter_obj), pos=(0, 0, PAD_H + gap),
                fixed=True, collision=False, decimate=False, convexify=False,
            ),
            surface=gs.surfaces.Default(color=(0.25, 0.65, 0.95, 1.0)),
        )

    camera = scene.add_camera(
        res=(args.res, args.res), pos=(0.005, -0.014, 0.050),
        lookat=(0, 0, PAD_H), up=(0, 0, 1), fov=30,
        near=0.005, far=0.2, GUI=False,
    )
    scene.build()

    rest = positions(pad)
    bottom = np.flatnonzero(np.isclose(rest[:, 2], 0, atol=1e-7)).astype(np.int32)
    top = np.flatnonzero(np.isclose(rest[:, 2], PAD_H, atol=1e-7)).astype(np.int32)
    if not bottom.size or not top.size:
        raise RuntimeError("Could not find the gel's bottom and top faces")
    pad.set_vertex_constraints(bottom, target_poss=None)
    top_xy = rest[top, :2]
    lower_surface = indenter_lower_z(shape, top_xy)

    frame_steps, tip_heights, displacement, depth_frames, rgb_frames = [], [], [], [], []
    for step in range(args.steps + 1):
        progress = min(step / (args.steps * 0.8), 1.0)
        tip_z = PAD_H + gap - (gap + args.depth_mm * 1e-3) * progress**2 * (3 - 2 * progress)
        if step:
            if indenter is not None:
                indenter.set_pos((0, 0, tip_z))
            # The ramp is monotonic, so the constrained patch can only grow.
            surface_z = tip_z + lower_surface
            pressed = surface_z < PAD_H
            if pressed.any():
                targets = np.column_stack((top_xy[pressed], surface_z[pressed]))
                pad.set_vertex_constraints(top[pressed], target_poss=targets)
            scene.step()

        if step % args.save_every == 0 or step == args.steps:
            rgb, depth, _, _ = camera.render(rgb=True, depth=True)
            rgb_frames.append(np.asarray(rgb).copy())
            depth_frames.append(np.asarray(depth).copy())
            displacement.append(positions(pad)[top] - rest[top])
            frame_steps.append(step)
            tip_heights.append(tip_z)

    final = positions(pad)
    measured = -float(displacement[-1][:, 2].min())
    if not np.isfinite(displacement).all() or not np.isclose(measured, args.depth_mm * 1e-3, atol=2e-6):
        raise RuntimeError(f"{shape}: gel displacement does not match the requested indentation")

    output = args.out / shape
    output.mkdir(parents=True, exist_ok=True)
    for old_frame in output.glob("rgb_[0-9]*.png"):
        if old_frame.stem.removeprefix("rgb_").isdigit():
            old_frame.unlink()
    for step, rgb in zip(frame_steps, rgb_frames):
        Image.fromarray(rgb).save(output / f"rgb_{step:04d}.png")
    np.savez_compressed(
        output / "state.npz", step=frame_steps, time_s=np.asarray(frame_steps) * DT,
        indenter_tip_z=tip_heights, rest=rest, pos=final, top_rest_idx=top,
        top_disp=displacement, depth=depth_frames,
    )
    Image.fromarray(np.concatenate((rgb_frames[0], rgb_frames[-1]), axis=1)).save(
        output / "summary.png"
    )
    print(f"{shape}: {measured * 1e3:.3f} mm indentation, {len(frame_steps)} frames → {output}", flush=True)
    scene.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=(*SHAPES, "all"), default="sphere")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--depth-mm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--res", type=int, default=480)
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--show-indenter", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.save_every, args.res) < 1:
        parser.error("--steps, --save-every and --res must be positive")
    if not 0 < args.depth_mm <= 1.5:
        parser.error("--depth-mm must be in (0, 1.5]")

    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    for shape in SHAPES if args.shape == "all" else (args.shape,):
        with tempfile.TemporaryDirectory(prefix="gentac-") as assets:
            run(shape, args, Path(assets))


if __name__ == "__main__":
    main()
