"""Publication figures; exact image coordinates and clipping at image borders."""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .fundamental import epipoles, homogeneous

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'savefig.dpi': 180})


def save(fig, path):
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def clip_line(line, xmin, xmax, ymin, ymax):
    """Intersect an implicit line with arbitrary plotting bounds."""
    a, b, c = line
    points = []
    if abs(b) > 1e-12:
        for u in (xmin, xmax):
            v = -(a*u+c)/b
            if ymin <= v <= ymax:
                points.append((u, v))
    if abs(a) > 1e-12:
        for v in (ymin, ymax):
            u = -(b*v+c)/a
            if xmin <= u <= xmax:
                points.append((u, v))
    unique = []
    for p in points:
        if not any(np.linalg.norm(np.array(p)-q) < 1e-6 for q in unique):
            unique.append(p)
    return np.array(unique[:2]) if len(unique) >= 2 else None


def line_segment(line, width, height):
    return clip_line(line, 0, width-1, 0, height-1)


def epipolar_figure(images, F, left, right, path, count=16, expanded=False, view=None):
    e1, e2 = epipoles(F)
    indices = np.linspace(0, len(left)-1, min(count, len(left)), dtype=int)
    lines = (homogeneous(right[indices]) @ F, homogeneous(left[indices]) @ F.T)
    panels = list(zip(images, (left[indices], right[indices]), lines,
                      (e1, e2), ('Left image', 'Right image')))
    if view is not None:
        if view not in ('left', 'right'):
            raise ValueError("view must be 'left', 'right', or None")
        panels = [panels[0 if view == 'left' else 1]]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(8.6, 4.2) if len(panels) == 1 else (10.6, 4.1))
    for ax, (im, pts, ls, ep, title) in zip(np.atleast_1d(axes), panels):
        h, w = im.shape[:2]
        ax.imshow(im, cmap='gray', vmin=0, vmax=255, extent=(-.5, w-.5, h-.5, -.5))
        for i, (p, line) in enumerate(zip(pts, ls)):
            seg = line_segment(line, w, h)
            if seg is not None:
                ax.plot(seg[:, 0], seg[:, 1], color='#155ee8', lw=.8, alpha=.85)
            ax.plot(*p, 'o', ms=3.2, color='#ffe22c', mec='black', mew=.35)
            if count <= 20 and not expanded:
                ax.annotate(str(i+1), p, xytext=(3, 3), textcoords='offset points', fontsize=6,
                            color='#fff27a', bbox=dict(facecolor='black', alpha=.4, pad=.2))
        finite = abs(ep[2]) > 1e-10
        if finite:
            ax.plot(ep[0], ep[1], 'o', color='red', ms=6, label='Epipole', zorder=5)
        if expanded and finite:
            xmin, xmax = min(0, ep[0]), max(w, ep[0])
            ymin, ymax = min(0, ep[1]), max(h, ep[1])
            dx = .08*(xmax-xmin)
            dy = max(.08*(ymax-ymin), .20*(xmax-xmin))
            ax.set_xlim(xmin-dx, xmax+dx)
            ax.set_ylim(ymax+dy, ymin-dy)
            # Extend the F-derived lines themselves; noisy points need not lie on them.
            for line in ls:
                seg = clip_line(line, xmin-dx, xmax+dx, ymin-dy, ymax+dy)
                if seg is not None:
                    ax.plot(seg[:, 0], seg[:, 1], color='#155ee8', lw=.55, alpha=.6)
        else:
            ax.set_xlim(-.5, w-.5)
            ax.set_ylim(h-.5, -.5)
        ax.set_title(title)
        ax.set_xlabel('x / pixel')
        ax.set_ylabel('y / pixel')
        ax.set_aspect('equal')
        text = f'e = ({ep[0]:.2f}, {ep[1]:.2f}, {ep[2]:.0f})' if finite else 'Epipole at infinity'
        ax.text(.02, .02, text, transform=ax.transAxes, fontsize=8, color='red',
                bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    fig.tight_layout()
    save(fig, path)


def matches_figure(images, left, right, mask, path, title):
    h = max(im.shape[0] for im in images)
    w = images[0].shape[1]
    canvas = np.full((h, w+images[1].shape[1]), 255, np.uint8)
    canvas[:images[0].shape[0], :w] = images[0]
    canvas[:images[1].shape[0], w:] = images[1]
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    if mask is None:
        # No geometric model has classified these candidates yet.
        groups = [(np.arange(len(left)), '#2878b5', 'Candidate match')]
    else:
        groups = [(np.flatnonzero(mask == good), color, label) for good, color, label in
                  ((False, '#ef5350', 'Outlier'), (True, '#22ba79', 'Inlier'))]
    for ids, color, label in groups:
        for j, i in enumerate(ids):
            a, b = left[i], right[i] + [w, 0]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=.45, alpha=.65,
                    label=label if j == 0 else None)
            ax.scatter([a[0], b[0]], [a[1], b[1]], s=5, color=color)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.axis('off')
    save(fig, path)


def keypoints_figure(images, k1, k2, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for ax, im, keys, name in zip(axes, images, (k1, k2), ('Left', 'Right')):
        drawn = cv2.drawKeypoints(im, keys, None, color=(70, 220, 255),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f'{name}: {len(keys)} SIFT keypoints')
        ax.axis('off')
    fig.tight_layout()
    save(fig, path)


