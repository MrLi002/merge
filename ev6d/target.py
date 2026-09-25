"""Target segmentation policy, separate from the optical-flow grid.

Projected cuboid silhouette + metric depth gate is an engineering assumption.
It needs a known model and an initialized pose, never ground-truth future poses.
"""
import cv2
import numpy as np

from .geometry import project, quat_matrix


def cuboid_vertices(size):
    return np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], dtype=float)*np.asarray(size)/2


def model_target_mask(position, quaternion, size, K, shape, depth, margin_px=3, depth_margin_m=.08):
    vertices = cuboid_vertices(size) @ quat_matrix(quaternion).T + position
    mask = np.zeros(shape, dtype=np.uint8)
    if np.any(vertices[:, 2] <= 1e-4):
        return mask.astype(bool)
    uv = project(vertices, K)
    hull = cv2.convexHull(np.rint(uv).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    if margin_px:
        k = 2*int(margin_px)+1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    good_z = np.isfinite(depth) & (depth > 0)
    good_z &= (depth >= vertices[:, 2].min()-depth_margin_m) & (depth <= vertices[:, 2].max()+depth_margin_m)
    return mask.astype(bool) & good_z


def filter_target_events(events, mask):
    if not len(events):
        return events
    xy = events[:, 1:3].astype(int)
    return events[mask[xy[:, 1], xy[:, 0]]]
