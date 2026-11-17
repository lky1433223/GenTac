"""Project the FEM contact surface into a virtual GelSight-style internal camera."""

from dataclasses import dataclass

import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraSpec:
    """Prototype optical parameters in sensor coordinates and SI units."""

    width_px: int = 320
    height_px: int = 240
    standoff_m: float = 0.020
    marker_diameter_m: float = 0.0007

    def __post_init__(self) -> None:
        """Reject camera settings that cannot produce a valid image."""
        if min(self.width_px, self.height_px) < 1:
            raise ValueError("Tactile camera image size must be positive")
        lengths = np.asarray((self.standoff_m, self.marker_diameter_m))
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError(
                "Tactile camera standoff and marker diameter must be finite and positive"
            )


def render_tactile(
    rest: NDArray,
    surface: NDArray,
    faces: NDArray,
    markers: NDArray,
    camera: CameraSpec,
) -> tuple[dict[str, NDArray], dict]:
    """Render the reflective surface from below and return RGB and aligned geometry."""
    top = np.isclose(rest[:, 2], rest[:, 2].max(), rtol=0, atol=1e-8)
    vertices, triangles = np.unique(faces[top[faces].all(axis=1)], return_inverse=True)
    if not vertices.size:
        raise ValueError("The tactile camera requires a planar rest contact surface")
    triangles = triangles.reshape(-1, 3)
    material_xy = rest[vertices, :2]
    width, height = camera.width_px, camera.height_px
    principal = np.array([width / 2, height / 2])
    camera_z = -camera.standoff_m
    focal = (
        0.94
        * (rest[:, 2].max() - camera_z)
        * np.min(np.array([width, height]) / np.ptp(material_xy, axis=0))
    )
    px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    azimuth = np.arange(3) * (2 * np.pi / 3)
    lights = np.column_stack(
        (0.8 * np.cos(azimuth), 0.8 * np.sin(azimuth), np.full(3, -0.6))
    )
    marker_x, marker_y = (np.unique(markers[0, :, i]) for i in (0, 1))
    # Use the layout centers rounded below solver precision to recover regular grid axes.
    marker_x, marker_y = (np.unique(a.round(9)) for a in (marker_x, marker_y))
    rgb_frames, depth_frames, normals_frames = [], [], []
    for frame in surface[:, vertices]:
        z = frame[:, 2] - camera_z
        if np.any(z <= 0):
            raise ValueError("The contact surface crosses the tactile camera plane")
        projected = focal * frame[:, :2] / z[:, None] + principal
        mesh = Triangulation(*projected.T, triangles)
        sample = mesh.get_trifinder()(px, py)
        valid = sample >= 0

        def interpolate(values: NDArray) -> NDArray:
            return LinearTriInterpolator(mesh, values)(px, py).filled(np.nan)

        inv_depth = interpolate(1 / z)
        depth = 1 / inv_depth
        face_xyz = frame[triangles]
        normals = np.cross(
            face_xyz[:, 2] - face_xyz[:, 0], face_xyz[:, 1] - face_xyz[:, 0]
        )
        surface_normals = np.zeros_like(frame)
        np.add.at(surface_normals, triangles.ravel(), np.repeat(normals, 3, axis=0))
        smooth = np.stack([interpolate(n / z) for n in surface_normals.T], axis=-1)
        smooth /= np.linalg.norm(smooth, axis=-1, keepdims=True)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        shading = 0.18 + 0.74 * np.clip(smooth[valid] @ lights.T, 0, 1)
        u, v = (
            interpolate(coord / z)[valid] / inv_depth[valid] for coord in material_xy.T
        )
        xi = np.searchsorted((marker_x[:-1] + marker_x[1:]) / 2, u)
        yi = np.searchsorted((marker_y[:-1] + marker_y[1:]) / 2, v)
        ink = (u - marker_x[xi]) ** 2 + (v - marker_y[yi]) ** 2 <= (
            camera.marker_diameter_m / 2
        ) ** 2
        shading[ink] *= 0.15
        rgb[valid] = np.rint(shading * 255).astype(np.uint8)

        normal_image = np.full((height, width, 3), np.nan, dtype=np.float32)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        normal_image[valid] = normals[sample[valid]]
        rgb_frames.append(rgb)
        depth_frames.append(depth.astype(np.float32))
        normals_frames.append(normal_image)

    marker_uv = focal * markers[..., :2] / (markers[..., 2:] - camera_z) + principal
    depth = np.stack(depth_frames)
    return (
        {
            "tactile_rgb": np.stack(rgb_frames),
            "tactile_depth_m": depth,
            "tactile_depth_delta_m": depth - depth[0],
            "tactile_normal": np.stack(normals_frames),
            "tactile_marker_uv_px": marker_uv,
            "tactile_marker_disp_px": marker_uv - marker_uv[0],
        },
        {
            "model": "pinhole Lambertian prototype; no refraction or occlusion by the mount",
            "image_size_px": [width, height],
            "focal_length_px": float(focal),
            "principal_point_px": principal.tolist(),
            "position_m": [0.0, 0.0, camera_z],
            "axes": {"right": "+X", "down": "+Y", "forward": "+Z"},
            "marker_diameter_m": camera.marker_diameter_m,
        },
    )
