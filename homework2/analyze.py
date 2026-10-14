"""Single-view metrology for 1.jpg.

Coordinate convention: image origin at top-left, x rightward, y downward.
The fixed scale is 1.70 m for every marked pedestrian. Dependencies: NumPy,
Pillow and Matplotlib. Run from any directory: python path/to/analyze.py.
"""

from pathlib import Path
import csv
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ``resolve()`` can fail on some Windows/Python combinations when the path
# contains non-ASCII characters; ``absolute()`` is sufficient here.
ROOT = Path(__file__).absolute().parent
IMAGE_PATH = ROOT / "1.jpg"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Manually selected structural lines. Each tuple is (x1, y1, x2, y2).
# X: long direction of the main facade; Y: depth direction of its right facade.
LINES_X = [
    (568, 165, 802, 156),
    (568, 209, 802, 208),
    (568, 253, 802, 262),
    (344, 177, 458, 174),
    (344, 221, 458, 222),
    (344, 265, 458, 270),
]
LINES_Y = [
    (831.7, 125.4, 881.1, 136.1),
    (836.8, 134.6, 883.3, 143.4),
    (922.8, 143.3, 960.4, 150.8),
    (836.8, 186.1, 880.7, 190.4),
    (831.9, 237.5, 878.1, 236.6),
    (842.0, 247.1, 870.7, 245.6),
]

# These short Y segments stay on a single facade patch instead of connecting
# mismatched window corners across a projecting tower. Y and Z edge endpoints
# were assisted by OpenCV LSD during review, visually checked, then rounded to
# 0.1 px. They are saved constants, so OpenCV is NOT needed to reproduce results.
# Decimal precision is not a claim of subpixel measurement accuracy.
# Z uses long architectural vertical edges. The old lamp segment
# (589,302)-(590,466) incorrectly continued along an adjacent monument.
LINES_Z = [
    (829.4, 134.4, 825.9, 220.6),
    (524.9, 156.9, 524.2, 258.1),
    (486.2, 159.4, 486.1, 254.4),
    (471.6, 160.6, 472.1, 274.4),
    (886.4, 136.9, 880.8, 245.7),
    (921.0, 151.9, 916.0, 233.2),
]

# Complete same-model lamp at the right edge. Use the top of its globe and its
# visible ground contact. The old point (676, 681) was on a cropped foreground
# pole and therefore was not a valid lamp base.
REFERENCE_LAMP_TOP = (1006, 367)
REFERENCE_LAMP_BASE = (997, 470)
TARGET_TOP = (293, 389)  # top of globe on the purple-arrow lamp

# Three scale readings sharing the SAME geometry: not independent experiments.
PEOPLE = [
    ("左侧黑衣行人", (452, 516), (450, 570)),
    ("左侧橙衣行人", (474, 522), (476, 575)),
    ("中部黑衣行人", (625, 465), (620, 512)),
]
ASSUMED_PERSON_HEIGHT_M = 1.70
PIXEL_NOISE_STD = 1.0
PERTURBATION_TRIALS = 2000
RANDOM_SEED = 20261012


def font(size=22):
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def hpoint(xy):
    return np.array([xy[0], xy[1], 1.0], dtype=float)


def normalize_point(point):
    if abs(point[2]) < 1e-12 * np.linalg.norm(point):
        # A ground-plane direction can legitimately have its VP at infinity.
        return point / np.linalg.norm(point[:2])
    return point / point[2]


def homogeneous_line(segment):
    line = np.cross(hpoint(segment[:2]), hpoint(segment[2:]))
    return line / np.hypot(line[0], line[1])


def join(point_a, point_b):
    """Homogeneous line through two homogeneous points."""
    return np.cross(point_a, point_b)


def meet(line_a, line_b):
    """Homogeneous intersection, including legitimate points at infinity."""
    return normalize_point(np.cross(line_a, line_b))


def fit_vanishing_point(segments):
    """Least-squares finite intersection minimizing point-to-line distances.

    Each line is normalized so that sqrt(a^2+b^2)=1. With the finite-chart
    convention w=1, solving a*x+b*y=-c directly avoids the scale-dependent
    homogeneous-vector constraint that is especially unstable for distant VPs.
    """
    lines = np.stack([homogeneous_line(segment) for segment in segments])
    xy, _, _, _ = np.linalg.lstsq(lines[:, :2], -lines[:, 2], rcond=None)
    return np.array([xy[0], xy[1], 1.0])


def y_on_line(line, x):
    return -(line[0] * x + line[2]) / line[1]


def image_length(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def project_top_to_vertical(top, base, vz):
    """Orthogonally project a marked top onto the line joining base and Vz.

    Manual points are rarely exactly collinear. Keep ground contact fixed,
    correct only the top; every subsequent cross-ratio then uses one line.
    """
    axis = vz[:2] - base[:2] if abs(vz[2]) > 1e-12 else vz[:2]
    axis = axis / np.linalg.norm(axis)
    xy = base[:2] + np.dot(top[:2] - base[:2], axis) * axis
    return hpoint(xy)


def angular_residual_deg(segment, vanishing_point):
    """Angle between a marked segment and its ray to the fitted VP."""
    direction = np.array(
        [segment[2] - segment[0], segment[3] - segment[1]], dtype=float
    )
    midpoint = np.array(
        [(segment[0] + segment[2]) / 2, (segment[1] + segment[3]) / 2]
    )
    ray = vanishing_point[:2] - midpoint
    cosine = abs(np.dot(direction, ray)) / (
        np.linalg.norm(direction) * np.linalg.norm(ray)
    )
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def rms_angular_residual(segments, vanishing_point):
    values = [angular_residual_deg(segment, vanishing_point) for segment in segments]
    return float(math.sqrt(np.mean(np.square(values))))


def transfer_reference_height(ref_top, ref_base, target_base, horizon, vz):
    """Transfer a known vertical height to another ground-plane position."""
    ground_vp = meet(join(ref_base, target_base), horizon)
    equal_height_top = meet(join(ground_vp, ref_top), join(target_base, vz))
    return equal_height_top, ground_vp


def projective_height_ratio(bottom, top, reference_top, vz):
    """Height(top)/Height(reference_top) from a vertical-line cross-ratio."""
    axis = vz[:2] - bottom[:2] if abs(vz[2]) > 1e-12 else vz[:2]
    axis = axis / np.linalg.norm(axis)
    t = np.dot(top[:2] - bottom[:2], axis)
    r = np.dot(reference_top[:2] - bottom[:2], axis)
    if abs(vz[2]) < 1e-12:
        return float(t / r)
    v = np.dot(vz[:2] - bottom[:2], axis)
    # Signed 1-D cross-ratio: do not hide impossible geometry with abs().
    return float(t * (v - r) / (r * (v - t)))


def height_readings(horizon, vz, lamp_top_xy, lamp_base_xy, people):
    base = hpoint(lamp_base_xy)
    top = project_top_to_vertical(hpoint(lamp_top_xy), base, vz)
    rows, transferred, corrected_people = [], [], []
    for label, person_top_xy, person_base_xy in people:
        person_base = hpoint(person_base_xy)
        person_top = project_top_to_vertical(hpoint(person_top_xy), person_base, vz)
        equal_top, ground_vp = transfer_reference_height(
            person_top, person_base, base, horizon, vz
        )
        ratio = projective_height_ratio(base, top, equal_top, vz)
        rows.append({
            "reference": label,
            "person_pixel_height": image_length(person_top_xy, person_base_xy),
            "corrected_person_pixel_height": image_length(person_top, person_base),
            "person_top_correction_px": image_length(person_top, person_top_xy),
            "transferred_1.70m_point_x": float(equal_top[0]),
            "transferred_1.70m_point_y": float(equal_top[1]),
            "lamp_to_person_ratio": ratio,
            "lamp_height_m_at_1.70m": ratio * ASSUMED_PERSON_HEIGHT_M,
            "ground_direction_vp_x": float(ground_vp[0]),
            "ground_direction_vp_y": float(ground_vp[1]),
        })
        transferred.append(equal_top)
        corrected_people.append(person_top)
    return rows, transferred, corrected_people, top


def pixel_perturbation_analysis():
    """Synthetic annotation sensitivity, NOT new observations or confidence CI.

    The 1.70 m scale never changes. Add independent N(0,1 px^2) error to ALL
    selected line endpoints and measurement points in each trial; recompute
    geometry and the three-reading median. VPs are jointly recomputed, so their
    dependence is preserved. No arbitrary filtering of inconvenient trials.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    heights = []
    horizon_y = []
    for _ in range(PERTURBATION_TRIALS):
        groups = [np.asarray(g) + rng.normal(0, PIXEL_NOISE_STD, (len(g), 4))
                  for g in (LINES_X, LINES_Y, LINES_Z)]
        vx, vy, vz = [fit_vanishing_point(g) for g in groups]
        line = np.cross(vx, vy)
        top = np.asarray(REFERENCE_LAMP_TOP) + rng.normal(0, PIXEL_NOISE_STD, 2)
        base = np.asarray(REFERENCE_LAMP_BASE) + rng.normal(0, PIXEL_NOISE_STD, 2)
        people = [(label, np.asarray(t) + rng.normal(0, PIXEL_NOISE_STD, 2),
                   np.asarray(b) + rng.normal(0, PIXEL_NOISE_STD, 2))
                  for label, t, b in PEOPLE]
        readings, _, _, _ = height_readings(line, vz, top, base, people)
        heights.append(np.median([r["lamp_height_m_at_1.70m"] for r in readings]))
        horizon_y.append(y_on_line(line, 512))
    heights = np.asarray(heights)
    data = {
        "trials": PERTURBATION_TRIALS,
        "random_seed": RANDOM_SEED,
        "coordinate_noise_std_px": PIXEL_NOISE_STD,
        "fixed_person_height_m": ASSUMED_PERSON_HEIGHT_M,
        "median_height_m": float(np.median(heights)),
        "height_std_m": float(np.std(heights)),
        "height_percentiles_2.5_97.5_m": np.percentile(heights, [2.5, 97.5]).tolist(),
        "horizon_y_at_512_percentiles_2.5_97.5_px": np.percentile(horizon_y, [2.5, 97.5]).tolist(),
        "interpretation": "Sensitivity under specified synthetic pixel error; not empirical confidence or total uncertainty.",
    }
    return data, heights


def reconstruct_equal_height_base(ref_top, ref_base, target_top, horizon, vz):
    """Recover the hidden base of an object known to be as tall as ref object."""
    top_plane_vp = meet(join(ref_top, target_top), horizon)
    target_base = meet(join(top_plane_vp, ref_base), join(target_top, vz))
    return target_base, top_plane_vp


def leave_one_out_horizon_range(lines_x, lines_y, x_coordinate):
    values = []
    for index_x in range(len(lines_x)):
        vx_i = fit_vanishing_point(lines_x[:index_x] + lines_x[index_x + 1 :])
        for index_y in range(len(lines_y)):
            vy_i = fit_vanishing_point(lines_y[:index_y] + lines_y[index_y + 1 :])
            horizon_i = np.cross(vx_i, vy_i)
            horizon_i /= np.hypot(horizon_i[0], horizon_i[1])
            values.append(y_on_line(horizon_i, x_coordinate))
    return float(min(values)), float(max(values)), float(np.std(values))


def draw_cross(draw, xy, color, size=8, width=3):
    x, y = xy
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)


def main():
    source = Image.open(IMAGE_PATH).convert("RGB")
    image_width, image_height = source.size

    vx = fit_vanishing_point(LINES_X)
    vy = fit_vanishing_point(LINES_Y)
    vz = fit_vanishing_point(LINES_Z)
    horizon = np.cross(vx, vy)
    horizon /= np.hypot(horizon[0], horizon[1])

    reference_base = hpoint(REFERENCE_LAMP_BASE)
    rows, equal_height_points, corrected_people, reference_top = height_readings(
        horizon, vz, REFERENCE_LAMP_TOP, REFERENCE_LAMP_BASE, PEOPLE
    )

    height_values = np.array([row["lamp_height_m_at_1.70m"] for row in rows])
    height_m = float(np.median(height_values))
    height_min_refs = float(np.min(height_values))
    height_max_refs = float(np.max(height_values))

    target_base_h, top_plane_vp = reconstruct_equal_height_base(
        reference_top, reference_base, hpoint(TARGET_TOP), horizon, vz
    )
    target_base = (float(target_base_h[0]), float(target_base_h[1]))

    # Same transfer with vertical direction at infinity: model sensitivity only.
    affine_rows, _, _, _ = height_readings(
        horizon, np.array([0.0, 1.0, 0.0]),
        REFERENCE_LAMP_TOP, REFERENCE_LAMP_BASE, PEOPLE
    )
    affine_height_m = float(np.median([r["lamp_height_m_at_1.70m"] for r in affine_rows]))
    perturbation, perturbed_heights = pixel_perturbation_analysis()

    horizon_center_range = leave_one_out_horizon_range(
        LINES_X, LINES_Y, image_width / 2
    )
    diagnostics = {
        "rms_angular_residual_deg": {
            "x_direction": rms_angular_residual(LINES_X, vx),
            "y_direction": rms_angular_residual(LINES_Y, vy),
            "vertical_direction": rms_angular_residual(LINES_Z, vz),
        },
        "leave_one_out_horizon_y_at_image_center": {
            "min": horizon_center_range[0],
            "max": horizon_center_range[1],
            "std": horizon_center_range[2],
        },
        "affine_cross_check_height_m": affine_height_m,
        "finite_vp_minus_affine_m": height_m - affine_height_m,
        "reference_top_projection_shift_px": image_length(reference_top, REFERENCE_LAMP_TOP),
        "pixel_perturbation": perturbation,
    }

    with (OUT / "height_estimates.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "coordinate_system": "origin top-left; x right; y down; pixels",
        "image_size": [image_width, image_height],
        "vanishing_point_x": [float(vx[0]), float(vx[1])],
        "vanishing_point_y": [float(vy[0]), float(vy[1])],
        "vanishing_point_vertical": [float(vz[0]), float(vz[1])],
        "vanishing_line_abc": [float(value) for value in horizon],
        "vanishing_line_slope_intercept": {
            "slope": float(-horizon[0] / horizon[1]),
            "intercept": float(-horizon[2] / horizon[1]),
        },
        "reference_lamp_top": list(REFERENCE_LAMP_TOP),
        "reference_lamp_base": list(REFERENCE_LAMP_BASE),
        "reference_lamp_top_corrected": reference_top[:2].tolist(),
        "purple_arrow_lamp_top": list(TARGET_TOP),
        "purple_arrow_lamp_reconstructed_base": list(target_base),
        "top_plane_direction_vp": [float(top_plane_vp[0]), float(top_plane_vp[1])],
        "assumed_person_height_m": ASSUMED_PERSON_HEIGHT_M,
        "estimated_lamp_height_m_median": height_m,
        "individual_estimate_range_m": [height_min_refs, height_max_refs],
        "height_interpretation": "Conditional estimate: people fixed at 1.70 m, common ground plane, equal-height lamp model.",
        "diagnostics": diagnostics,
    }
    (OUT / "result_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    annotations = {
        "image_size": [image_width, image_height],
        "coordinate_system": summary["coordinate_system"],
        "lines_x": LINES_X, "lines_y": LINES_Y, "lines_z": LINES_Z,
        "people": [{"label": n, "top_raw": list(t), "base": list(b),
                    "top_corrected": p[:2].tolist(), "fixed_height_m": ASSUMED_PERSON_HEIGHT_M}
                   for (n, t, b), p in zip(PEOPLE, corrected_people)],
        "reference_top_raw": list(REFERENCE_LAMP_TOP),
        "reference_top_corrected": reference_top[:2].tolist(),
        "reference_base": list(REFERENCE_LAMP_BASE),
        "target_top": list(TARGET_TOP),
        "target_base_inferred_NOT_observed": list(target_base),
        "line_annotation_source": "X manual; Y/Z assisted edge endpoints visually checked during review; all constants are reproducible.",
    }
    (OUT / "annotations.json").write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "pixel_perturbation.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["synthetic_trial", "fixed_person_height_m", "lamp_height_m"])
        writer.writerows((i + 1, ASSUMED_PERSON_HEIGHT_M, float(h)) for i, h in enumerate(perturbed_heights))

    # Figure 1: expanded canvas displays both off-image horizontal VPs.
    offset_x = int(max(0, -vx[0]) + 250)
    canvas_width = int(offset_x + max(image_width, vy[0]) + 350)
    canvas = Image.new("RGB", (canvas_width, image_height + 75), "#f5f5f2")
    canvas.paste(source, (offset_x, 35))
    draw = ImageDraw.Draw(canvas)
    for segments, color in ((LINES_X, "#d23b3b"), (LINES_Y, "#276fbf")):
        for segment in segments:
            line = homogeneous_line(segment)
            xa, xb = -offset_x, canvas_width - offset_x
            draw.line(
                (
                    xa + offset_x,
                    y_on_line(line, xa) + 35,
                    xb + offset_x,
                    y_on_line(line, xb) + 35,
                ),
                fill=color,
                width=2,
            )
    for point, color, label in (
        (vx, "#d23b3b", "Vx"),
        (vy, "#276fbf", "Vy"),
    ):
        display_point = (point[0] + offset_x, point[1] + 35)
        draw_cross(draw, display_point, color, size=12, width=4)
        draw.text(
            (display_point[0] + 14, display_point[1] - 28),
            f"{label}=({point[0]:.0f}, {point[1]:.0f})",
            fill=color,
            font=font(22),
        )
    xa, xb = -offset_x, canvas_width - offset_x
    draw.line(
        (
            xa + offset_x,
            y_on_line(horizon, xa) + 35,
            xb + offset_x,
            y_on_line(horizon, xb) + 35,
        ),
        fill="#8a2be2",
        width=5,
    )
    draw.text(
        (offset_x + 18, 55),
        "红/蓝：两组水平结构线  紫：消失线",
        fill="white",
        stroke_fill="black",
        stroke_width=2,
        font=font(22),
    )
    canvas.save(OUT / "vanishing_geometry.png")

    # Figure 2: support points and the reconstructed hidden target base.
    overlay = source.copy()
    draw = ImageDraw.Draw(overlay)
    draw.line(
        (0, y_on_line(horizon, 0), image_width, y_on_line(horizon, image_width)),
        fill="#d000ff",
        width=4,
    )
    draw.line((*reference_top[:2], *REFERENCE_LAMP_BASE), fill="#00e5ff", width=4)
    draw_cross(draw, REFERENCE_LAMP_TOP, "#00e5ff", 7, 3)
    draw_cross(draw, REFERENCE_LAMP_BASE, "#00e5ff", 7, 3)
    for (_, top_xy, base_xy), equal_top, corrected_top in zip(PEOPLE, equal_height_points, corrected_people):
        draw.line((*corrected_top[:2], *base_xy), fill="#ffd400", width=3)
        draw_cross(draw, top_xy, "#ffd400", 5, 2)
        draw_cross(draw, base_xy, "#ffd400", 5, 2)
        draw_cross(draw, equal_top[:2], "#42f56c", 4, 2)
    draw.line((*TARGET_TOP, *target_base), fill="#ff4d4d", width=5)
    draw_cross(draw, TARGET_TOP, "#ff4d4d", 7, 3)
    draw_cross(draw, target_base, "#ff4d4d", 7, 3)
    draw.rectangle((12, 12, 720, 116), fill=(0, 0, 0, 170))
    draw.text(
        (24, 18),
        "紫：消失线  青：参考灯  黄：行人尺度固定1.70 m",
        fill="white",
        font=font(19),
    )
    draw.text(
        (24, 50),
        f"红：目标路灯（遮挡底点重建≈({target_base[0]:.1f}, {target_base[1]:.1f})）",
        fill="white",
        font=font(19),
    )
    draw.text(
        (24, 82),
        f"等高假设下目标高度：{height_m:.2f} m；绿：转移后的1.70 m点",
        fill="white",
        font=font(19),
    )
    overlay.save(OUT / "measurement_overlay.png")

    # Figure: actual selected finite segments, never confused with extensions.
    marked = source.copy()
    draw = ImageDraw.Draw(marked)
    for segments, color, prefix in ((LINES_X, "#ff3333", "X"),
                                    (LINES_Y, "#00bbff", "Y"),
                                    (LINES_Z, "#ffff00", "Z")):
        for i, seg in enumerate(segments, 1):
            draw.line(seg, fill=color, width=2)
            label_xy = (seg[2] + 3, seg[3] - 14) if prefix == "Z" else (seg[0] + 3, seg[1] - 14)
            draw.text(label_xy, f"{prefix}{i}", fill=color,
                      stroke_fill="black", stroke_width=1, font=font(13))
    marked.save(OUT / "selected_segments.png")
    for name, box in {
        "side_segments_crop.png": (820, 105, 975, 285),
        "vertical_segments_crop.png": (465, 150, 545, 280),
    }.items():
        marked.crop(box).resize(((box[2]-box[0])*4, (box[3]-box[1])*4)).save(OUT / name)

    # Scientific charts: fixed 1.70 m scale, no body-height range analysis.
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 11})
    fig, ax = plt.subplots(figsize=(7.4, 4.1), layout="constrained")
    colors = ["#4c78a8", "#ed8a22", "#50a26c"]
    for i, (row, color) in enumerate(zip(rows, colors)):
        value = row["lamp_height_m_at_1.70m"]
        ax.scatter(i, value, s=90, color=color, zorder=3)
        ax.annotate(f"{value:.3f}", (i, value), xytext=(0, 12),
                    textcoords="offset points", ha="center")
    ax.axhline(height_m, color="#8759ad", linestyle="--", linewidth=1.5,
               label=f"中位数 {height_m:.3f} m")
    ax.set(xticks=range(3), xticklabels=[r["reference"] for r in rows],
           ylabel="路灯高度 / m", xlim=(-0.45, 2.45),
           ylim=(height_min_refs - 0.20, height_max_refs + 0.25),
           title="固定行人高度 1.70 m：三组参照结果")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUT / "height_sensitivity.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.1), layout="constrained")
    ax.hist(perturbed_heights, bins=38, color="#4c78a8", alpha=0.85,
            edgecolor="white")
    ax.axvline(height_m, color="#b14943", linewidth=1.6, label=f"原始标注 {height_m:.3f} m")
    for j, bound in enumerate(perturbation["height_percentiles_2.5_97.5_m"]):
        ax.axvline(bound, color="#8759ad", linestyle="--", linewidth=1.3,
                   label="模拟取点的中间95%区间" if j == 0 else None)
    ax.set(xlabel="重算后路灯高度 / m", ylabel="模拟次数",
           title=f"固定1.70 m尺度，坐标噪声σ=1 px，{PERTURBATION_TRIALS}次扰动")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "pixel_error_analysis.png", dpi=170)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
