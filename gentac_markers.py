"""Extract the FEM surface and bind material points to its rest geometry."""

import numpy as np
from numpy.typing import NDArray


def bind_markers(
    rest: NDArray[np.floating],
    tets: NDArray[np.integer],
    marker_xyz: NDArray[np.floating],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Return outward surface triangles and the support and weights of each top marker.

    All indices refer to `rest` and marker rows retain the input order. Evaluate
    the same weights on deformed vertices to follow the material points.
    """
    rest = np.asarray(rest, dtype=np.float64)
    tets = np.asarray(tets)
    marker_xyz = np.asarray(marker_xyz, dtype=np.float64)
    if rest.ndim != 2 or rest.shape[1] != 3 or not np.isfinite(rest).all():
        raise ValueError("Rest vertices must be a finite (N, 3) array")
    if (
        tets.ndim != 2
        or tets.shape[1] != 4
        or not tets.size
        or tets.dtype.kind not in "iu"
        or np.any((tets < 0) | (tets >= len(rest)))
    ):
        raise ValueError("Tetrahedra must be a nonempty (M, 4) array of vertex indices")
    if (
        marker_xyz.ndim != 2
        or marker_xyz.shape[1] != 3
        or not np.isfinite(marker_xyz).all()
    ):
        raise ValueError("Marker positions must be a finite (K, 3) array")

    # Faces are ordered by the index of the opposite vertex in each tetrahedron.
    faces = tets[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("Tetrahedral faces must have at most two incident tetrahedra")
    boundary = first[counts == 1]
    surface = faces[boundary].astype(np.intp)
    triangles = rest[surface]
    first_edge = triangles[:, 1] - triangles[:, 0]
    second_edge = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(first_edge, second_edge)
    opposite = rest[tets.ravel()[boundary]] - triangles[:, 0]
    orientation = np.einsum("ij,ij->i", normals, opposite)
    if np.any(orientation == 0):
        raise ValueError("Degenerate tetrahedron incident to gel surface")
    inward = orientation > 0
    surface[inward] = surface[inward][:, [0, 2, 1]]

    marker_faces = np.empty((len(marker_xyz), 3), dtype=np.intp)
    weights = np.empty((len(marker_xyz), 3), dtype=np.float64)
    if not len(marker_xyz):
        return surface, marker_faces, weights

    # Float32 rest meshes can round the nominal top plane by a few nanometres.
    top_z = rest[:, 2].max()
    plane_tol = np.ptp(rest, axis=0).max() * 1e-6
    on_plane = np.all(np.abs(rest[surface, 2] - top_z) <= plane_tol, axis=1)
    upward = np.where(inward, -normals[:, 2], normals[:, 2]) > 0
    top_faces = surface[on_plane & upward]
    if not len(top_faces):
        raise ValueError("Gel mesh has no planar top surface for marker binding")
    if np.any(np.abs(marker_xyz[:, 2] - top_z) > plane_tol):
        raise ValueError("Marker positions must lie on the gel's rest top plane")

    # ponytail: scan top triangles once per marker at initialization. Use a
    # spatial index if dense marker layouts make this scan expensive.
    xy = rest[top_faces, :2]
    edges = (xy[:, 1:] - xy[:, :1]).transpose(0, 2, 1)
    inverse = np.linalg.inv(edges)
    for index, point in enumerate(marker_xyz):
        vw = np.einsum("fij,fj->fi", inverse, point[:2] - xy[:, 0])
        barycentric = np.column_stack((1 - vw.sum(axis=1), vw))
        candidates = np.flatnonzero(np.all(barycentric >= -1e-8, axis=1))
        if not len(candidates):
            raise ValueError(f"Marker {index} is outside the gel's rest top surface")
        face = candidates[0]
        marker_faces[index] = top_faces[face]
        weights[index] = barycentric[face]

    return surface, marker_faces, weights
