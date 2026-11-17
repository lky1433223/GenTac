"""Render diagnostics, FEM geometry, and material textures from a saved run."""

from collections.abc import Callable
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib import colormaps
from matplotlib.animation import FFMpegWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.tri import LinearTriInterpolator, Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


def _texture_rgba(
    xy: np.ndarray,
    triangles: np.ndarray,
    uv: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    image: np.ndarray | None,
) -> np.ndarray:
    mesh = Triangulation(*xy.T, triangles)
    u, v = [LinearTriInterpolator(mesh, values)(gx, gy) for values in uv.T]
    mask = np.ma.getmaskarray(u)
    u, v = np.clip(u.filled(0), 0, 1), np.clip(v.filled(0), 0, 1)
    if image is None:
        checks = (np.floor(u * 48) + np.floor(v * 40)) % 2
        rgb = np.where(checks[..., None] == 0, [23, 60, 76], [225, 241, 238])
    else:
        row = np.minimum(((1 - v) * image.shape[0]).astype(int), image.shape[0] - 1)
        col = np.minimum((u * image.shape[1]).astype(int), image.shape[1] - 1)
        rgb = image[row, col]
    return np.dstack((rgb, np.where(mask, 0, 255))).astype(np.uint8)


def _write_video(
    fig: Figure, path: Path, count: int, draw: Callable[[int], None]
) -> None:
    writer = FFMpegWriter(
        fps=10,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    with writer.saving(fig, str(path), dpi=100):
        for sample in range(count):
            draw(sample)
            writer.grab_frame()
        for _ in range(13):
            writer.grab_frame()


def plot_run(
    run_dir: Path,
    *,
    frame: int = -1,
    texture: Path | None = None,
    video: bool = False,
    camera_texture: bool = False,
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
        centers = state["indenter_center_m"] * 1000
        radius = float(state["indenter_radius_m"]) * 1000
        camera_rgb = None
        if camera_texture:
            if "camera_rgb" not in state:
                raise ValueError(
                    "This run has no Genesis RGB frames. Rerun run_indentation.py with --camera."
                )
            camera_rgb = state["camera_rgb"]
            if (
                camera_rgb.ndim != 4
                or camera_rgb.shape[-1] != 3
                or len(camera_rgb) != len(time)
            ):
                raise ValueError(
                    "Genesis RGB frames do not match the saved simulation samples"
                )

    if (video or camera_texture) and not FFMpegWriter.isAvailable():
        raise RuntimeError("MP4 export requires ffmpeg on PATH")
    index = range(len(time))[frame]
    top = np.isclose(rest[:, 2], rest[:, 2].max(), atol=1e-5, rtol=0)
    is_top = top[faces].all(axis=1)
    top_faces = faces[is_top]
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
    curves.plot(
        time,
        np.linalg.norm(displacement[:, :, :2], axis=2).max(axis=1),
        label="Peak in-plane marker motion",
    )
    curves.plot(time, overlap, label="Sphere overlap")
    curves.axvline(time[index], color="0.5", linestyle=":")
    curves.set(xlabel="Time (s)", ylabel="Distance (mm)")
    curves.legend(fontsize="small")
    colors = field.tripcolor(*rest_xy.T, triangles, down[index], shading="gouraud")
    fig.colorbar(colors, ax=field, label="Depression (mm)")
    field.set_title("Surface depression in rest coordinates")
    vectors.quiver(
        *rest_markers[:, :2].T,
        *displacement[index, :, :2].T,
        angles="xy",
        scale_units="xy",
        scale=0.05,
        color="tab:blue",
    )
    vectors.scatter(*rest_markers[:, :2].T, s=4, color="0.3")
    vectors.set_title("In-plane marker motion (arrows 20×)")
    traces.plot(
        time, markers[:, :, 2] - rest_markers[:, 2], alpha=0.25, color="tab:blue"
    )
    traces.axvline(time[index], color="0.5", linestyle=":")
    traces.set(xlabel="Time (s)", ylabel="Marker ΔZ (mm)", title="Marker trajectories")
    for axis in (field, vectors):
        axis.set(xlabel="X (mm)", ylabel="Y (mm)", aspect="equal")
    fig.savefig(out / "diagnostics.png", dpi=160)

    fig = Figure(figsize=(12.8, 7.2), facecolor="white")
    FigureCanvasAgg(fig)
    view = fig.add_axes(
        (0.02, 0.10, 0.53, 0.72), projection="3d", computed_zorder=False
    )
    plan = fig.add_axes((0.57, 0.20, 0.34, 0.57))
    colorbar = fig.add_axes((0.93, 0.22, 0.014, 0.54))
    fig.suptitle(
        "GenTac | Sphere indentation",
        x=0.055,
        y=0.95,
        ha="left",
        fontsize=24,
        fontweight="bold",
    )
    caption = fig.text(0.055, 0.875, "", fontsize=12)
    fig.text(0.57, 0.10, "Arrows show in-plane displacement at 20× scale.", fontsize=10)
    norm = Normalize(min(0.0, float(down.min())), max(1e-6, float(down.max())))
    cmap = colormaps["magma"]
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar,
        label="Downward displacement (mm)",
    )
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 32), np.linspace(0, np.pi, 18))
    sphere = radius * np.stack(
        (np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)), axis=-1
    )
    lower = np.minimum(surface.min(axis=(0, 1)), (centers - radius).min(axis=0))
    upper = np.maximum(surface.max(axis=(0, 1)), (centers + radius).max(axis=0))
    lower -= 0.04 * (upper - lower)
    upper += 0.04 * (upper - lower)

    def draw(sample: int) -> None:
        view.clear()
        plan.clear()
        caption.set_text(f"Step {steps[sample]} · Simulation time {time[sample]:.3f} s")
        points = surface[sample]
        current_markers = markers[sample]
        colors = cmap(norm(down[sample, triangles].mean(axis=1)))
        view.add_collection3d(
            Poly3DCollection(
                points[faces[~is_top]],
                facecolors="#a7b0bd",
                edgecolors="none",
                zorder=1,
            )
        )
        view.add_collection3d(
            Poly3DCollection(
                points[top_faces], facecolors=colors, edgecolors="none", zorder=2
            )
        )
        view.scatter(*current_markers.T, c="#59f1eb", s=9, depthshade=False, zorder=3)
        view.plot_wireframe(
            *(sphere + centers[sample]).transpose(2, 0, 1),
            rstride=2,
            cstride=3,
            color="#2689c7",
            linewidth=0.8,
            zorder=4,
        )
        view.set(
            xlim=(lower[0], upper[0]),
            ylim=(lower[1], upper[1]),
            zlim=(lower[2], upper[2]),
            title="Deformed gel and indenter",
        )
        view.set_box_aspect(upper - lower)
        view.view_init(elev=27, azim=-57)
        view.set_axis_off()

        plan.add_collection(
            PolyCollection(points[top_faces, :2], facecolors=colors, edgecolors="none")
        )
        plan.quiver(
            *rest_markers[:, :2].T,
            *(20 * displacement[sample, :, :2]).T,
            angles="xy",
            scale_units="xy",
            scale=1,
            units="xy",
            width=0.10,
            color="#59f1eb",
        )
        plan.scatter(*current_markers[:, :2].T, s=8, color="#59f1eb")
        plan.set(
            xlim=(lower[0], upper[0]),
            ylim=(lower[1], upper[1]),
            xlabel="X (mm)",
            ylabel="Y (mm)",
            aspect="equal",
            title="Top view and marker motion",
        )

    draw(index)
    fig.savefig(out / "simulation.png", dpi=100)
    if video:
        _write_video(fig, out / "indentation.mp4", len(time), draw)

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
    for axis, coords, label in zip(
        fig.subplots(1, 2), (rest_xy, xy), ("Rest", "Deformed"), strict=True
    ):
        rgba = _texture_rgba(coords, triangles, uv, gx, gy, image)
        axis.imshow(rgba, extent=extent, origin="lower")
        axis.set(xlabel="X (mm)", ylabel="Y (mm)", title=label, aspect="equal")
    fig.savefig(out / "texture.png", dpi=160)
    if camera_rgb is not None:
        fig = Figure(figsize=(12.8, 7.2), facecolor="#f7f9fa")
        FigureCanvasAgg(fig)
        fig.suptitle(
            "GenTac | Sphere indentation",
            x=0.052,
            y=0.95,
            ha="left",
            fontsize=22,
            fontweight="bold",
        )
        caption = fig.text(0.052, 0.875, "", fontsize=11, color="#435463")
        cam = fig.add_axes((0.05, 0.13, 0.50, 0.67))
        plan = fig.add_axes((0.61, 0.15, 0.34, 0.60))
        cam.set_title("Genesis scene camera", pad=12)
        cam.axis("off")
        plan.set_title("Deformed material texture", pad=12)
        fig.text(
            0.61,
            0.055,
            "Orange dots: surface markers at true displacement scale.",
            fontsize=9,
            color="#435463",
        )

        # Keep the same close-up bounds for every frame.
        center = rest_xy.mean(axis=0)
        half_span = min(4.8, float(np.ptp(rest_xy, axis=0).min()) * 0.48)
        pixel = (np.arange(600) + 0.5) / 600
        gx, gy = np.meshgrid(*(x + (2 * pixel - 1) * half_span for x in center))
        left, bottom = center - half_span
        right, top = center + half_span
        plan.set(
            xlim=(left, right),
            ylim=(bottom, top),
            xlabel="X (mm)",
            ylabel="Y (mm)",
            aspect="equal",
        )
        camera_image = cam.imshow(camera_rgb[0])
        texture_view = plan.imshow(
            np.zeros((600, 600, 4), dtype=np.uint8),
            extent=(left, right, bottom, top),
            origin="lower",
        )
        dots = plan.scatter(
            [], [], s=18, color="#ef8042", edgecolor="#612812", linewidth=0.4
        )
        marker_motion = np.linalg.norm(displacement[..., :2], axis=-1).max(axis=1)

        def draw_camera(sample: int) -> None:
            camera_image.set_data(camera_rgb[sample])
            texture_view.set_data(
                _texture_rgba(
                    surface[sample, vertices, :2], triangles, uv, gx, gy, image
                )
            )
            dots.set_offsets(markers[sample, :, :2])
            caption.set_text(
                f"Step {steps[sample]} · Simulation time {time[sample]:.3f} s"
                f" · Max marker XY motion {marker_motion[sample]:.3f} mm"
            )

        _write_video(fig, out / "genesis-and-texture.mp4", len(time), draw_camera)
        draw_camera(index)
        fig.savefig(out / "genesis-and-texture.png", dpi=100)
    return out


def plot_combined_video(run_dir: Path, *, texture: Path | None = None) -> Path:
    """Stack the synchronized camera and FEM panels into one MP4."""
    output = run_dir / "combined-video"
    if output.exists():
        raise FileExistsError(f"{output} already exists")
    state_path = (run_dir / "state.npz").resolve(strict=True)
    with TemporaryDirectory(dir=run_dir) as scratch:
        staging = Path(scratch)
        (staging / "state.npz").symlink_to(state_path)
        clips = plot_run(staging, texture=texture, video=True, camera_texture=True)
        video = staging / "indentation-combined.mp4"
        result = run(
            [
                FFMpegWriter.bin_path(),
                "-v",
                "error",
                "-i",
                str(clips / "genesis-and-texture.mp4"),
                "-i",
                str(clips / "indentation.mp4"),
                "-filter_complex",
                "vstack=inputs=2:shortest=1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                result.stderr.strip() or "FFmpeg could not combine the videos"
            )
        output.mkdir()
        return video.replace(output / video.name)
