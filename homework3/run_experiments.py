"""Run both compulsory experiments and export figures, matrices and real metrics.

Usage: python run_experiments.py --mode all
All paths are resolved relative to this script, not the terminal directory.
"""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
from src.fundamental import eight_point, evaluate
from src.ransac import estimate_ransac
from src.feature_matching import detect_and_match
from src.visualization import epipolar_figure, matches_figure, keypoints_figure

ROOT = Path(__file__).resolve().parent


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


def prepare_data():
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    if not all((data / n).exists() for n in ('left_original.jpg', 'right_original.jpg', 'h_matches.txt')):
        candidates = [p for p in ROOT.glob('19-20*/**/*.docx') if not p.name.startswith('~$')]
        if len(candidates) != 1:
            raise RuntimeError('Expected one assignment DOCX, or an existing prepared data directory')
        with zipfile.ZipFile(candidates[0]) as doc:
            for member, name in [('word/media/image1.jpeg', 'left_original.jpg'),
                                 ('word/media/image2.jpeg', 'right_original.jpg')]:
                (data / name).write_bytes(doc.read(member))
        source = list(candidates[0].parent.glob('h_matches*.txt'))
        if len(source) != 1:
            raise RuntimeError('Expected one h_matches text file')
        shutil.copyfile(source[0], data / 'h_matches.txt')
    images = [read_image(data / n) for n in ('left_original.jpg', 'right_original.jpg')]
    matches = np.loadtxt(data / 'h_matches.txt')
    if matches.ndim != 2 or matches.shape[1] != 4 or not np.isfinite(matches).all():
        raise ValueError('Expected a finite N x 4 match file')
    # The numerical coordinate ranges establish [xL,yL,xR,yR], despite the handout wording.
    for i, im in enumerate(images):
        xy = matches[:, 2*i:2*i+2]
        if (xy < 0).any() or (xy[:, 0] >= im.shape[1]).any() or (xy[:, 1] >= im.shape[0]).any():
            raise ValueError('Matches fall outside the original image; verify x/y order')
    return images, matches[:, :2], matches[:, 2:]


def export_model(folder, F, left, right, mask=None):
    np.savetxt(folder / 'F.txt', F, fmt='%.12e')
    result = {'F': F.tolist(), 'all_points': evaluate(F, left, right)}
    if mask is not None:
        result['inlier_points'] = evaluate(F, left[mask], right[mask])
    return result


def run_given(images, left, right):
    folder = ROOT / 'outputs/given_matches'
    folder.mkdir(parents=True, exist_ok=True)
    F = eight_point(left, right)
    results = export_model(folder, F, left, right)
    epipolar_figure(images, F, left, right, folder / 'epilines.png')
    epipolar_figure(images, F, left, right, folder / 'epilines_all.png', count=len(left))
    for view in ('left', 'right'):
        epipolar_figure(images, F, left, right, folder / f'epipole_{view}.png',
                       expanded=True, view=view)
    return results


def run_auto(images, args):
    folder = ROOT / 'outputs/automatic_matching'
    folder.mkdir(parents=True, exist_ok=True)
    k1, k2, forward, mutual, left, right = detect_and_match(*images, ratio=args.ratio)
    F, mask, details = estimate_ransac(left, right, args.threshold, seed=args.seed)
    results = dict(model=export_model(folder, F, left, right, mask),
                   ransac=details, keypoints_left=len(k1), keypoints_right=len(k2),
                   ratio_forward=len(forward), ratio_mutual=len(mutual), ratio=args.ratio)
    np.savetxt(folder / 'matches_with_inlier_mask.txt', np.column_stack((left, right, mask)),
               header='xL yL xR yR inlier', fmt='%.8f')
    keypoints_figure(images, k1, k2, folder / 'keypoints.png')
    fl = np.array([k1[m.queryIdx].pt for m in forward])
    fr = np.array([k2[m.trainIdx].pt for m in forward])
    matches_figure(images, fl, fr, None, folder / 'ratio_matches.png',
                   f'Forward Lowe ratio test: {len(fl)} matches (before mutual check)')
    matches_figure(images, left, right, mask, folder / 'ransac_matches.png',
                   f'Mutual matches: {len(left)} pairs; {mask.sum()} inliers, {(~mask).sum()} outliers')
    epipolar_figure(images, F, left[mask], right[mask], folder / 'epilines.png')
    epipolar_figure(images, F, left[mask], right[mask], folder / 'epilines_all.png', count=int(mask.sum()))
    return results


def tex_matrix(F):
    return r'\begin{bmatrix}' + r'\\'.join(' & '.join(f'{v:.7f}' for v in row) for row in F) + r'\end{bmatrix}'


def export_tex(results):
    """Numerical tables are generated from actual runs to avoid manual transcription."""
    given, auto = results['given'], results['automatic']
    commands = []
    def cmd(name, value):
        commands.append('\\newcommand{\\' + name + '}{' + str(value) + '}')
    cmd('GivenF', tex_matrix(given['F']))
    cmd('AutoF', tex_matrix(auto['model']['F']))
    for prefix, stats in [('Given', given['all_points']),
                          ('Auto', auto['model']['inlier_points'])]:
        cmd(prefix+'Mean', f"{stats['sampson_mean_px2']:.5f}")
        cmd(prefix+'Median', f"{stats['sampson_median_px2']:.5f}")
        cmd(prefix+'LeftDist', f"{stats['line_left_mean_px']:.4f}")
        cmd(prefix+'RightDist', f"{stats['line_right_mean_px']:.4f}")
        for side, key in [('Left', 'epipole_left'), ('Right', 'epipole_right')]:
            cmd(prefix+side+'Ep', ', '.join(f'{v:.3f}' for v in stats[key]))
    cmd('NumLeftKeys', auto['keypoints_left'])
    cmd('NumRightKeys', auto['keypoints_right'])
    cmd('NumRatio', auto['ratio_forward'])
    cmd('NumMutual', auto['ratio_mutual'])
    cmd('NumAutoInliers', auto['ransac']['inliers'])
    cmd('NumAutoOutliers', auto['ransac']['matches']-auto['ransac']['inliers'])
    cmd('AutoInlierPercent', f"{100*auto['ransac']['inlier_ratio']:.2f}")
    cmd('AutoIterations', auto['ransac']['iterations'])
    (ROOT / 'report').mkdir(exist_ok=True)
    (ROOT / 'report/results.tex').write_text('\n'.join(commands)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all', 'given', 'auto'], default='all')
    parser.add_argument('--ratio', type=float, default=.75)
    parser.add_argument('--threshold', type=float, default=1.5, help='sqrt(Sampson distance), pixels')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(args.seed)
    images, left, right = prepare_data()
    results = {'versions': {'numpy': np.__version__, 'opencv': cv2.__version__},
               'coordinate_order': 'xL yL xR yR; values used as supplied, no origin offset',
               'image_shapes': [list(im.shape) for im in images]}
    if args.mode in ('given', 'all'):
        results['given'] = run_given(images, left, right)
    if args.mode in ('auto', 'all'):
        results['automatic'] = run_auto(images, args)
    destination = ROOT / 'outputs' / ('metrics.json' if args.mode == 'all' else f'metrics_{args.mode}.json')
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.mode == 'all':
        export_tex(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
