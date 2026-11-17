"""Render diagnostics, FEM geometry, and material textures from a saved run."""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.tri import LinearTriInterpolator, Triangulation
from PIL import Image


def plot_run(
    run_dir: Path, *, frame: int = -1, texture: Path | None = None
) -> Path:
    """Write plots for a sampled frame and return their new output directory."""
    with np.load(run_dir / "state.npz", allow_pickle=False) as state:
        rest = state["surface_rest_m"] * 1000
        surface = state["surface_xyz_m"] * 1000
        faces = state["surface_faces"]
        time = state["time_s"]
        steps = state["step"]
        markers = state["marker_xyz_m"] * 1000
        displacement = state["marker_disp_m"] * 1000
        overlap = state["surface_overlap_m"] * 1000

    index = range(len(time))[frame]
    top = np.isclose(rest[:, 2], rest[:, 2].max(), atol=1e-5, rtol=0)
    top_faces = faces[top[faces].all(axis=1)]
    if not len(top_faces):
        raise ValueError("The saved run has no planar rest contact face")
    vertices, triangles = np.unique(top_faces, return_inverse=True)
    triangles = triangles.reshape(-1, 3)
    rest_xy = rest[vertices, :2]
    current = surface[index]
    xy = current[vertices, :2]
    down = rest[vertices, 2] - surface[:, vertices, 2]
    rest_markers = markers[0]
    image = None
    if texture is not None:
        with Image.open(texture) as source:
            image = np.asarray(source.convert("RGB"))

    out = run_dir / f"plots-step-{steps[index]:06d}"
    out.mkdir()
    title = f"Step {steps[index]} · {time[index]:.3f} s"

    fig = Figure(figsize=(10, 7), layout="constrained")
    FigureCanvasAgg(fig)
    fig.suptitle(title)
    curves, field, vectors, traces = fig.subplots(2, 2).ravel()
    curves.plot(time, down.max(axis=1), label="Peak surface depression")
    curves.plot(time, np.linalg.norm(displacement[:, :, :2], axis=2).max(axis=1),
                label="Peak in-plane marker motion")
    curves.plot(time, overlap, label="Sphere overlap")
    curves.axvline(time[index], color="0.5", linestyle=":")
    curves.set(xlabel="Time (s)", ylabel="Distance (mm)")
    curves.legend(fontsize="small")
    colors = field.tripcolor(*rest_xy.T, triangles, down[index], shading="gouraud")
    fig.colorbar(colors, ax=field, label="Depression (mm)")
    field.set_title("Surface depression in rest coordinates")
    vectors.quiver(*rest_markers[:, :2].T, *displacement[index, :, :2].T,
                   angles="xy", scale_units="xy", scale=0.05, color="tab:blue")
    vectors.scatter(*rest_markers[:, :2].T, s=4, color="0.3")
    vectors.set_title("In-plane marker motion (arrows 20×)")
    traces.plot(time, markers[:, :, 2] - rest_markers[:, 2], alpha=0.25, color="tab:blue")
    traces.axvline(time[index], color="0.5", linestyle=":")
    traces.set(xlabel="Time (s)", ylabel="Marker ΔZ (mm)", title="Marker trajectories")
    for axis in (field, vectors):
        axis.set(xlabel="X (mm)", ylabel="Y (mm)", aspect="equal")
    fig.savefig(out / "diagnostics.png", dpi=160)

    fig = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(fig)
    fig.suptitle(title)
    bounds = np.vstack((rest, current))
    for panel, (xyz, points, label) in enumerate(
        ((rest, rest_markers, "Rest"), (current, markers[index], "Deformed")), 1
    ):
        axis = fig.add_subplot(1, 2, panel, projection="3d")
        axis.plot_trisurf(*xyz.T, triangles=faces, color="steelblue",
                          edgecolor="0.35", linewidth=0.12, alpha=0.75)
        axis.scatter(*points.T, s=5, color="darkorange", depthshade=False)
        axis.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", title=label,
                 xlim=(bounds[:, 0].min(), bounds[:, 0].max()),
                 ylim=(bounds[:, 1].min(), bounds[:, 1].max()),
                 zlim=(bounds[:, 2].min(), bounds[:, 2].max()))
        axis.set_box_aspect(np.ptp(bounds, axis=0))
    fig.savefig(out / "simulation.png", dpi=160)

    fig = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(fig)
    fig.suptitle(title)
    lower = np.minimum(rest_xy.min(axis=0), xy.min(axis=0))
    upper = np.maximum(rest_xy.max(axis=0), xy.max(axis=0))
    extent = (lower[0], upper[0], lower[1], upper[1])
    size = 600
    pixel = (np.arange(size) + 0.5) / size
    gx, gy = np.meshgrid(*(lower[i] + pixel * (upper[i] - lower[i]) for i in range(2)))
    uv = (rest_xy - rest_xy.min(axis=0)) / np.ptp(rest_xy, axis=0)
    for axis, coords, label in zip(fig.subplots(1, 2), (rest_xy, xy),
                                   ("Rest", "Deformed"), strict=True):
        mesh = Triangulation(*coords.T, triangles)
        u, v = [LinearTriInterpolator(mesh, values)(gx, gy) for values in uv.T]
        mask = np.ma.getmaskarray(u)
        u, v = np.clip(u.filled(0), 0, 1), np.clip(v.filled(0), 0, 1)
        if image is None:
            checks = (np.floor(u * 24) + np.floor(v * 20)) % 2
            rgb = np.where(checks[..., None] == 0, [46, 89, 128], [226, 237, 246])
        else:
            row = np.minimum(((1 - v) * image.shape[0]).astype(int), image.shape[0] - 1)
            col = np.minimum((u * image.shape[1]).astype(int), image.shape[1] - 1)
            rgb = image[row, col]
        rgba = np.dstack((rgb, np.where(mask, 0, 255))).astype(np.uint8)
        axis.imshow(rgba, extent=extent, origin="lower")
        axis.set(xlabel="X (mm)", ylabel="Y (mm)", title=label, aspect="equal")
    fig.savefig(out / "texture.png", dpi=160)
    return out
