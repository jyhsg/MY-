#!/usr/bin/env python3
"""Single-image calibration from two orthogonal chessboard directions.

Assumptions: zero skew, square pixels (fx=fy=f), and principal point at the
image center. The chessboard supplies two mutually orthogonal families of
parallel 3-D lines. Their directions determine a rotation up to axis-sign
conventions, but translation and absolute scale cannot be obtained from
vanishing points alone.
"""

from pathlib import Path
import argparse
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
OUT = ROOT / "outputs" / "vanishing_point"
PATTERN = (6, 4)
IMAGE_SIZE = (720, 576)
FOCAL_MIN_WIDTH_RATIO = 0.15
FOCAL_MAX_WIDTH_RATIO = 2.5
MAX_CONCURRENCE_RESIDUAL_PX = 2.0
VP_MAX_EXTENT_WEIGHT = 0.0008


def imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def imwrite(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    buffer.tofile(str(path))


def fit_line(points):
    """Normalized total least-squares fit of ax+by+c=0."""
    center = points.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))
    normalized = (points - center) / max(scale, 1e-9)
    _, _, vt = np.linalg.svd(np.column_stack([normalized, np.ones(len(points))]))
    line_n = vt[-1]
    T = np.array([[1 / scale, 0, -center[0] / scale],
                  [0, 1 / scale, -center[1] / scale], [0, 0, 1]])
    line = T.T @ line_n
    return line / np.linalg.norm(line[:2])


def fit_vanishing_point(lines):
    _, _, vt = np.linalg.svd(np.asarray(lines))
    point = vt[-1]
    if abs(point[2]) < 1e-10:
        return None
    return point / point[2]


def detect_and_measure(path):
    gray = imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None or (gray.shape[1], gray.shape[0]) != IMAGE_SIZE:
        return None
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria).reshape(PATTERN[1], PATTERN[0], 2)
    row_lines = [fit_line(row) for row in corners]
    col_lines = [fit_line(corners[:, j, :]) for j in range(PATTERN[0])]
    vx, vy = fit_vanishing_point(row_lines), fit_vanishing_point(col_lines)
    if vx is None or vy is None:
        return None
    center = np.array([IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0])
    f2 = -float(np.dot(vx[:2] - center, vy[:2] - center))
    if f2 <= 0:
        return None
    f = float(np.sqrt(f2))
    if not (FOCAL_MIN_WIDTH_RATIO * IMAGE_SIZE[0] <= f <= FOCAL_MAX_WIDTH_RATIO * IMAGE_SIZE[0]):
        return None
    residuals = [abs(line @ vx) for line in row_lines] + [abs(line @ vy) for line in col_lines]
    mean_residual = float(np.mean(residuals))
    if mean_residual > MAX_CONCURRENCE_RESIDUAL_PX:
        return None
    # Prefer low concurrence error while preventing either vanishing point
    # from becoming so distant that the extended-coordinate plot is unreadable.
    vp_distances = [np.linalg.norm(vx[:2] - center), np.linalg.norm(vy[:2] - center)]
    max_vp_distance = float(max(vp_distances))
    score = mean_residual + VP_MAX_EXTENT_WEIGHT * max_vp_distance
    return {
        "path": path, "corners": corners, "row_lines": row_lines, "col_lines": col_lines,
        "vx": vx, "vy": vy, "f": f, "mean_residual": mean_residual,
        "max_vp_distance": max_vp_distance, "score": score,
    }


def rotation_from_vps(vx, vy, K):
    invK = np.linalg.inv(K)
    r1 = invK @ vx
    r2 = invK @ vy
    r1 /= np.linalg.norm(r1)
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R0 = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(R0)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def draw_infinite_line(image, line, color):
    a, b, c = line
    candidates = []
    w, h = IMAGE_SIZE
    if abs(b) > 1e-9:
        candidates.extend([(0, int(round(-c / b))), (w - 1, int(round(-(a * (w - 1) + c) / b)))])
    if abs(a) > 1e-9:
        candidates.extend([(int(round(-c / a)), 0), (int(round(-(b * (h - 1) + c) / a)), h - 1)])
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            ok, p1, p2 = cv2.clipLine((0, 0, w, h), candidates[i], candidates[j])
            if ok and p1 != p2:
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
                return


def make_visualization(result):
    image = imread(result["path"])
    overlay = image.copy()
    for line in result["row_lines"]:
        draw_infinite_line(overlay, line, (30, 60, 240))
    for line in result["col_lines"]:
        draw_infinite_line(overlay, line, (40, 210, 60))
    for point in result["corners"].reshape(-1, 2):
        cv2.circle(overlay, tuple(np.rint(point).astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)

    vx, vy = result["vx"], result["vy"]
    vanishing_line = np.cross(vx, vy)
    vanishing_line /= np.linalg.norm(vanishing_line[:2])

    # The two vanishing points are outside the source image.  Keep the source
    # view at full resolution and add a coordinate overview that contains the
    # image rectangle, principal point, both vanishing points, and their line.
    image_w, image_h = IMAGE_SIZE
    overview_w = 600
    footer_h = 150
    canvas = np.full((image_h + footer_h, image_w + overview_w, 3), 250, dtype=np.uint8)
    canvas[:image_h, :image_w] = cv2.addWeighted(image, 0.35, overlay, 0.65, 0)
    canvas[:image_h, image_w:] = (246, 246, 246)

    world_points = np.array([
        [0.0, 0.0],
        [float(image_w), float(image_h)],
        vx[:2],
        vy[:2],
    ])
    world_min = world_points.min(axis=0)
    world_max = world_points.max(axis=0)
    world_span = np.maximum(world_max - world_min, 1.0)
    world_min -= 0.08 * world_span
    world_max += 0.08 * world_span
    world_span = world_max - world_min

    panel_left = image_w
    panel_top = 68
    panel_padding = 42
    available_w = overview_w - 2 * panel_padding
    available_h = image_h - panel_top - panel_padding
    scale = min(available_w / world_span[0], available_h / world_span[1])
    drawing_w, drawing_h = world_span * scale
    offset_x = panel_left + (overview_w - drawing_w) / 2.0
    offset_y = panel_top + (available_h - drawing_h) / 2.0

    def map_world(point):
        mapped = (np.asarray(point, dtype=float) - world_min) * scale
        return tuple(np.rint(mapped + [offset_x, offset_y]).astype(int))

    cv2.putText(canvas, "Extended image-coordinate view", (panel_left + 22, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Vanishing points are outside the 720 x 576 image", (panel_left + 22, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (75, 75, 75), 1, cv2.LINE_AA)

    # Coordinate axes, image boundary, and the board-plane vanishing line.
    axis_color = (195, 195, 195)
    if world_min[0] <= 0 <= world_max[0]:
        cv2.line(canvas, map_world((0, world_min[1])), map_world((0, world_max[1])),
                 axis_color, 1, cv2.LINE_AA)
    if world_min[1] <= 0 <= world_max[1]:
        cv2.line(canvas, map_world((world_min[0], 0)), map_world((world_max[0], 0)),
                 axis_color, 1, cv2.LINE_AA)

    image_tl = map_world((0, 0))
    image_br = map_world((image_w, image_h))
    cv2.rectangle(canvas, image_tl, image_br, (225, 235, 248), -1)
    cv2.rectangle(canvas, image_tl, image_br, (70, 95, 125), 2, cv2.LINE_AA)
    cv2.putText(canvas, "source image", (image_tl[0] + 5, image_tl[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (55, 70, 90), 1, cv2.LINE_AA)

    mapped_vx = map_world(vx[:2])
    mapped_vy = map_world(vy[:2])
    mapped_center = map_world((image_w / 2.0, image_h / 2.0))
    cv2.line(canvas, mapped_vx, mapped_vy, (180, 70, 150), 2, cv2.LINE_AA)
    cv2.arrowedLine(canvas, mapped_center, mapped_vx, (30, 60, 240), 1,
                    cv2.LINE_AA, tipLength=0.025)
    cv2.arrowedLine(canvas, mapped_center, mapped_vy, (40, 180, 50), 1,
                    cv2.LINE_AA, tipLength=0.025)

    cv2.drawMarker(canvas, mapped_center, (20, 20, 20), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
    cv2.circle(canvas, mapped_vx, 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, mapped_vx, 6, (30, 60, 240), -1, cv2.LINE_AA)
    cv2.circle(canvas, mapped_vy, 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, mapped_vy, 6, (40, 180, 50), -1, cv2.LINE_AA)
    def put_vp_label(text, point, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.49
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
        x = point[0] + 10
        y = point[1] - 10
        if x + text_w > canvas.shape[1] - 8:
            x = point[0] - text_w - 10
            y = point[1] + text_h + 10
        if y - text_h < 8:
            y = point[1] + text_h + 10
        cv2.putText(canvas, text, (x, y), font, font_scale, color, 1, cv2.LINE_AA)

    put_vp_label(f"VP_x ({vx[0]:.1f}, {vx[1]:.1f})", mapped_vx, (30, 60, 210))
    put_vp_label(f"VP_y ({vy[0]:.1f}, {vy[1]:.1f})", mapped_vy, (30, 145, 40))
    cv2.putText(canvas, "principal point", (mapped_center[0] + 8, mapped_center[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    lines = [
        "Red/green: two orthogonal families of parallel board lines",
        f"VP_x=({vx[0]:.1f}, {vx[1]:.1f}), VP_y=({vy[0]:.1f}, {vy[1]:.1f})  [both outside source image]",
        f"Board vanishing line: {vanishing_line[0]:.4f}u + {vanishing_line[1]:.4f}v + {vanishing_line[2]:.2f}=0",
        f"f={result['f']:.2f} px, mean line-concurrence residual={result['mean_residual']:.3f} px",
        f"K=[[{result['f']:.2f}, 0, {image_w / 2:.0f}], [0, {result['f']:.2f}, {image_h / 2:.0f}], [0, 0, 1]]; skew=0, fx=fy",
    ]
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (12, image_h + 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.57, (20, 20, 20), 1, cv2.LINE_AA)
    imwrite(OUT / "单张消失线标定图.jpg", canvas)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = []
    for path in sorted(DATA.glob("*/*"), key=lambda p: (int(p.parent.name), p.name.lower())):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            measured = detect_and_measure(path)
            if measured is not None:
                candidates.append(measured)
    if not candidates:
        raise RuntimeError("No valid single-image vanishing-point candidate")
    result = min(candidates, key=lambda item: item["score"])
    cx, cy = IMAGE_SIZE[0] / 2.0, IMAGE_SIZE[1] / 2.0
    K = np.array([[result["f"], 0, cx], [0, result["f"], cy], [0, 0, 1.0]])
    R = rotation_from_vps(result["vx"], result["vy"], K)
    vanishing_line = np.cross(result["vx"], result["vy"])
    vanishing_line /= np.linalg.norm(vanishing_line[:2])
    d1 = np.linalg.inv(K) @ result["vx"]
    d2 = np.linalg.inv(K) @ result["vy"]
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    orthogonality_error = float(abs(d1 @ d2))
    make_visualization(result)
    report = {
        "method": "two orthogonal vanishing points from one chessboard image",
        "dataset_scope": "numbered subdirectories only; calibration_images.zip is a separate 1920x1080, 7x7-inner-corner dataset",
        "source_image": str(result["path"].relative_to(ROOT)),
        "image_size": list(IMAGE_SIZE),
        "assumptions": ["principal point equals image center", "zero skew", "fx = fy", "board axes are orthogonal"],
        "vanishing_point_x": result["vx"].tolist(),
        "vanishing_point_y": result["vy"].tolist(),
        "board_plane_vanishing_line_au_bv_c_eq_0": vanishing_line.tolist(),
        "mean_line_concurrence_residual_px": result["mean_residual"],
        "candidate_selection": {
            "accepted_focal_length_px": [
                FOCAL_MIN_WIDTH_RATIO * IMAGE_SIZE[0],
                FOCAL_MAX_WIDTH_RATIO * IMAGE_SIZE[0],
            ],
            "maximum_mean_line_concurrence_residual_px": MAX_CONCURRENCE_RESIDUAL_PX,
            "score": "mean_line_concurrence_residual_px + 0.0008 * max(distance(vx, center), distance(vy, center))",
            "max_vanishing_point_distance_px": result["max_vp_distance"],
        },
        "focal_length_px": result["f"],
        "camera_matrix": K.tolist(),
        "rotation_world_board_axes_to_camera": R.tolist(),
        "orthogonality_absolute_dot_error": orthogonality_error,
        "unobservable": ["translation", "absolute scale"],
        "warning": "raw image has strong radial distortion; this simplified pinhole estimate is expected to be biased",
    }
    (OUT / "消失点标定结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Single-image calibration from two orthogonal vanishing points")
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
