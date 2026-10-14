#!/usr/bin/env python3
"""Zhang planar calibration on the provided chessboard dataset.

The physical square size is not included with the dataset, so translations are
reported in square-length units. Intrinsics, distortion, and rotations do not
depend on that choice.
"""

from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).absolute().parent

# Prefer packages from the active Python environment.  Fall back to the
# optional local dependency directory used by the README only when needed.
try:
    import cv2
    import numpy as np
except ImportError:
    sys.path.insert(0, str(ROOT / ".deps"))
    import cv2
    import numpy as np

DATA = ROOT / "标定实验" / "标定实验"
OUT = ROOT / "outputs" / "zhang"
PATTERN = (6, 4)  # columns, rows of inner corners
IMAGE_SIZE = (720, 576)  # dominant resolution in the dataset
SQUARE_SIZE = 1.0
MAX_VIEWS_PER_GROUP = 2


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def imwrite(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    buffer.tofile(str(path))


def collect_observations():
    template = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    template[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_SIZE
    detected = []
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    for path in sorted(DATA.glob("*/*"), key=lambda p: (int(p.parent.name), p.name.lower())):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        gray = imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None or (gray.shape[1], gray.shape[0]) != IMAGE_SIZE:
            continue
        found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
        detected.append((template.copy(), corners.astype(np.float32), path))
    selected = []
    for group in sorted({item[2].parent.name for item in detected}, key=int):
        candidates = [item for item in detected if item[2].parent.name == group]
        if len(candidates) <= MAX_VIEWS_PER_GROUP:
            selected.extend(candidates)
        else:
            indices = np.linspace(0, len(candidates) - 1, MAX_VIEWS_PER_GROUP).round().astype(int)
            selected.extend(candidates[i] for i in indices)
    object_points = [item[0] for item in selected]
    image_points = [item[1] for item in selected]
    paths = [item[2] for item in selected]
    return object_points, image_points, paths, len(detected)


def calibrate(obj, img):
    flags = 0  # estimate k1, k2, p1, p2, and k3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, IMAGE_SIZE, None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9),
    )
    return rms, K, dist, rvecs, tvecs, flags


def per_view_errors(obj, img, K, dist, rvecs, tvecs):
    values = []
    for op, ip, rv, tv in zip(obj, img, rvecs, tvecs):
        projected, _ = cv2.projectPoints(op, rv, tv, K, dist)
        delta = projected.reshape(-1, 2) - ip.reshape(-1, 2)
        values.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return np.asarray(values)


def reject_outliers(obj, img, paths):
    # Two robust passes. This removes gross corner-localization failures while
    # keeping challenging edge views needed to constrain radial distortion.
    kept = np.arange(len(paths))
    rejected = []
    for _ in range(2):
        rms, K, dist, rvecs, tvecs, flags = calibrate(obj, img)
        errors = per_view_errors(obj, img, K, dist, rvecs, tvecs)
        med = float(np.median(errors))
        mad = float(np.median(np.abs(errors - med)))
        threshold = max(1.5, med + 3.0 * 1.4826 * mad)
        good = errors <= threshold
        if good.all():
            break
        rejected.extend((paths[i], float(errors[i])) for i in np.flatnonzero(~good))
        obj = [x for x, ok in zip(obj, good) if ok]
        img = [x for x, ok in zip(img, good) if ok]
        paths = [x for x, ok in zip(paths, good) if ok]
        kept = kept[good]
    rms, K, dist, rvecs, tvecs, flags = calibrate(obj, img)
    errors = per_view_errors(obj, img, K, dist, rvecs, tvecs)
    return obj, img, paths, rejected, rms, K, dist, rvecs, tvecs, errors, flags


def make_corner_sheet(paths, points, errors):
    order = np.argsort(errors)
    picks = list(order[:4]) + list(order[-4:])
    tiles = []
    for index in picks:
        image = imread(paths[index])
        cv2.drawChessboardCorners(image, PATTERN, points[index], True)
        label = f"{paths[index].parent.name}/{paths[index].name}  RMSE={errors[index]:.3f}px"
        cv2.rectangle(image, (0, 0), (IMAGE_SIZE[0], 28), (255, 255, 255), -1)
        cv2.putText(image, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(image, (360, 288), interpolation=cv2.INTER_AREA))
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    imwrite(OUT / "棋盘格角点检测.jpg", sheet)


def make_undistortion_example(path, K, dist):
    image = imread(path)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, IMAGE_SIZE, 0.7, IMAGE_SIZE)
    corrected = cv2.undistort(image, K, dist, None, new_K)
    comparison = np.hstack([image, corrected])
    cv2.putText(comparison, "Original", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(comparison, "Undistorted", (IMAGE_SIZE[0] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    imwrite(OUT / "畸变校正对比.jpg", comparison)
    return new_K, roi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    obj, img, paths, detected_count = collect_observations()
    if len(paths) < 3:
        raise RuntimeError(f"Only {len(paths)} valid images; at least 3 are required")
    initial_count = len(paths)
    (obj, img, paths, rejected, rms, K, dist, rvecs, tvecs, errors, flags) = reject_outliers(obj, img, paths)
    make_corner_sheet(paths, img, errors)
    sample_index = int(np.argsort(errors)[len(errors) // 2])
    new_K, roi = make_undistortion_example(paths[sample_index], K, dist)

    rows = []
    for path, error, rv, tv in zip(paths, errors, rvecs, tvecs):
        rows.append([path.parent.name, path.name, error, *rv.ravel(), *tv.ravel()])
    with (OUT / "逐图误差与外参.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "file", "rmse_px", "rvec_x", "rvec_y", "rvec_z",
                         "t_x_square", "t_y_square", "t_z_square"])
        writer.writerows(rows)

    result = {
        "method": "Zhang planar calibration via OpenCV calibrateCamera",
        "dataset_scope": "numbered subdirectories only; calibration_images.zip is a separate 1920x1080, 7x7-inner-corner dataset and is not mixed with this camera model",
        "image_size": list(IMAGE_SIZE),
        "pattern_inner_corners": list(PATTERN),
        "square_size": SQUARE_SIZE,
        "square_unit_note": "physical square size absent; translations are in square-length units",
        "selected_views_before_rejection": initial_count,
        "all_detected_views_at_target_resolution": detected_count,
        "selection_strategy": f"sort candidates lexicographically by filename, then take {MAX_VIEWS_PER_GROUP} evenly spaced indices per numbered group",
        "used_views": len(paths),
        "rejected_views": [{"file": str(p.relative_to(ROOT)), "rmse_px": e} for p, e in rejected],
        "rms_px": float(rms),
        "mean_view_rmse_px": float(errors.mean()),
        "median_view_rmse_px": float(np.median(errors)),
        "max_view_rmse_px": float(errors.max()),
        "camera_matrix": K.tolist(),
        "distortion_opencv_k1_k2_p1_p2_k3": dist.ravel().tolist(),
        "flags": int(flags),
        "sample_image_for_undistortion": str(paths[sample_index].relative_to(ROOT)),
        "new_camera_matrix_alpha_0_7": new_K.tolist(),
        "valid_roi": list(map(int, roi)),
    }
    (OUT / "标定结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(OUT / "calibration_parameters.npz", camera_matrix=K, distortion=dist,
             new_camera_matrix=new_K, image_size=np.asarray(IMAGE_SIZE))
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Zhang calibration on the chessboard image dataset")
    parser.add_argument("--data", type=Path, default=DATA,
                        help="directory containing numbered image subdirectories")
    parser.add_argument("--output", type=Path, default=OUT,
                        help="directory for calibration results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA = args.data.expanduser().resolve()
    OUT = args.output.expanduser().resolve()
    main()
