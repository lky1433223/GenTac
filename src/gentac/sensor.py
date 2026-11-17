"""Static geometry and material parameters for the first GenTac sensor."""

from dataclasses import dataclass

import numpy as np
import trimesh
from numpy.typing import NDArray


@dataclass(frozen=True)
class GelSpec:
    """SI parameters for a rectangular gel with a planar contact face."""

    size_m: tuple[float, float, float]
    young_pa: float
    poisson: float
    density: float
    friction: float
    surface_grid: int
    model: str = "linear_corotated"
    bonded_face: str = "bottom"

    @property
    def volume_m3(self) -> float:
        """Return the volume of the undeformed gel."""
        return float(np.prod(self.size_m))


@dataclass(frozen=True)
class HousingSpec:
    """Dimensions of the rigid backing below the sensor-frame origin, in meters."""

    overhang_m: float
    thickness_m: float

    def size_m(self, gel: GelSpec) -> tuple[float, float, float]:
        """Return the backing size for the given gel footprint."""
        if (
            not np.isfinite((self.overhang_m, self.thickness_m)).all()
            or self.overhang_m < 0
            or self.thickness_m <= 0
        ):
            raise ValueError(
                "Housing needs a nonnegative overhang and positive thickness"
            )
        width, length, _ = gel.size_m
        return (
            width + 2 * self.overhang_m,
            length + 2 * self.overhang_m,
            self.thickness_m,
        )


@dataclass(frozen=True)
class MarkerGrid:
    """Rectangular layout of material points on the undeformed contact face."""

    columns: int
    rows: int
    margin_m: float

    def positions(self, gel: GelSpec) -> NDArray[np.float64]:
        """Return marker centers in row-major order in the sensor frame."""
        width, length, height = gel.size_m
        if (
            self.columns < 2
            or self.rows < 2
            or not 0 < self.margin_m < min(width, length) / 2
        ):
            raise ValueError(
                "Marker grid needs at least two points per axis and a margin inside the gel"
            )
        x = np.linspace(
            -width / 2 + self.margin_m, width / 2 - self.margin_m, self.columns
        )
        y = np.linspace(
            -length / 2 + self.margin_m, length / 2 - self.margin_m, self.rows
        )
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, height)))


# Dimensions and grid counts follow UniVTAC's TacEx GelSightMiniCfg.
# Shape, bonding, marker margin, and material values are provisional.
MINI_GEL = GelSpec(
    size_m=(0.02525, 0.02075, 0.00450),
    young_pa=50_000.0,
    poisson=0.45,
    density=1_000.0,
    friction=0.6,
    surface_grid=17,
)
MINI_MARKERS = MarkerGrid(columns=11, rows=9, margin_m=0.002)
# Provisional backing dimensions for the fixed bench.
MINI_HOUSING = HousingSpec(overhang_m=0.002, thickness_m=0.006)


def make_gel_mesh(spec: GelSpec) -> trimesh.Trimesh:
    """Create a welded box with refined contact and bonded faces for tetrahedralization."""
    width, length, height = spec.size_m
    n = spec.surface_grid
    if n < 2 or not np.isfinite(spec.size_m).all() or min(spec.size_m) <= 0:
        raise ValueError(
            "Gel dimensions must be finite and positive, and surface_grid at least two"
        )
    x, y = np.meshgrid(
        np.linspace(-width / 2, width / 2, n), np.linspace(-length / 2, length / 2, n)
    )
    z = np.zeros(n * n)
    bottom = np.arange(n * n).reshape(n, n)
    top = bottom + n * n
    vertices = np.column_stack(
        (np.tile(x.ravel(), 2), np.tile(y.ravel(), 2), np.r_[z, z + height])
    )

    a, b = bottom[:-1, :-1].ravel(), bottom[:-1, 1:].ravel()
    c, d = bottom[1:, :-1].ravel(), bottom[1:, 1:].ravel()
    floor = np.concatenate((np.column_stack((a, d, b)), np.column_stack((a, c, d))))
    ceiling = floor[:, ::-1] + n * n
    walls = []
    for edge, upper in (
        (bottom[0], top[0]),
        (bottom[:, -1], top[:, -1]),
        (bottom[-1, ::-1], top[-1, ::-1]),
        (bottom[::-1, 0], top[::-1, 0]),
    ):
        walls.extend(
            (
                np.column_stack((edge[:-1], edge[1:], upper[1:])),
                np.column_stack((edge[:-1], upper[1:], upper[:-1])),
            )
        )

    return trimesh.Trimesh(
        vertices=vertices, faces=np.concatenate((floor, ceiling, *walls)), process=False
    )
