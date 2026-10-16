"""Normalized eight-point estimation; convention: x_right.T @ F @ x_left = 0."""
import numpy as np


def homogeneous(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Expected finite N x 2 coordinates")
    return np.column_stack((p, np.ones(len(p))))


def normalize_points(points):
    p = np.asarray(points, dtype=np.float64)
    homogeneous(p)  # Validate before centering.
    center = p.mean(axis=0)
    mean_distance = np.linalg.norm(p - center, axis=1).mean()
    if mean_distance < 1e-12:
        raise ValueError("Coincident points cannot be normalized")
    scale = np.sqrt(2) / mean_distance
    T = np.array([[scale, 0, -scale * center[0]],
                  [0, scale, -scale * center[1]], [0, 0, 1.]])
    return (homogeneous(p) @ T.T)[:, :2], T


def eight_point(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    homogeneous(left)
    homogeneous(right)
    if left.shape != right.shape or len(left) < 8:
        raise ValueError("At least eight paired points are required")
    a, T1 = normalize_points(left)
    b, T2 = normalize_points(right)
    u, v = a.T
    up, vp = b.T
    A = np.column_stack((up*u, up*v, up, vp*u, vp*v, vp, u, v, np.ones(len(a))))
    # Full Vh is essential for an 8 x 9 minimal sample: reduced SVD drops the null vector.
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    if s[7] < 1e-10 * s[0]:
        raise ValueError("Degenerate correspondence configuration")
    F = vh[-1].reshape(3, 3)
    U, s, Vh = np.linalg.svd(F)
    s[-1] = 0
    F = T2.T @ (U @ np.diag(s) @ Vh) @ T1
    F /= np.linalg.norm(F)
    if F.flat[np.argmax(np.abs(F))] < 0:
        F = -F
    return F


def sampson_squared(F, left, right):
    x, xp = homogeneous(left), homogeneous(right)
    lines_right, lines_left = x @ F.T, xp @ F
    residual = np.sum(xp * lines_right, axis=1)
    denominator = (lines_right[:, :2]**2).sum(axis=1) + (lines_left[:, :2]**2).sum(axis=1)
    return residual**2 / np.maximum(denominator, 1e-30)


def line_distances(F, left, right):
    x, xp = homogeneous(left), homogeneous(right)
    lr, ll = x @ F.T, xp @ F
    residual = np.abs(np.sum(xp * lr, axis=1))
    return (residual / np.maximum(np.linalg.norm(ll[:, :2], axis=1), 1e-15),
            residual / np.maximum(np.linalg.norm(lr[:, :2], axis=1), 1e-15))


def epipoles(F):
    U, _, Vh = np.linalg.svd(F)
    def canonical(e):
        return e/e[2] if abs(e[2]) > 1e-10 else e/np.linalg.norm(e)
    return canonical(Vh[-1]), canonical(U[:, -1])


def evaluate(F, left, right):
    errors = sampson_squared(F, left, right)
    dl, dr = line_distances(F, left, right)
    e1, e2 = epipoles(F)
    return dict(count=len(left), sampson_mean_px2=float(errors.mean()),
                sampson_median_px2=float(np.median(errors)),
                sampson_root_mean_px=float(np.sqrt(errors).mean()),
                line_left_mean_px=float(dl.mean()), line_right_mean_px=float(dr.mean()),
                algebraic_abs_mean=float(np.abs(np.sum(homogeneous(right) *
                    (homogeneous(left) @ F.T), axis=1)).mean()),
                singular_values=np.linalg.svd(F, compute_uv=False).tolist(),
                epipole_left=e1.tolist(), epipole_right=e2.tolist(),
                epipole_left_null_residual=float(np.linalg.norm(F @ e1)/np.linalg.norm(e1)),
                epipole_right_null_residual=float(np.linalg.norm(F.T @ e2)/np.linalg.norm(e2)))
