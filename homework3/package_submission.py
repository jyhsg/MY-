"""Package only deliverables, excluding caches, TeX intermediates and Word lock files."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    required = ['report/main.pdf', 'report/main.tex', 'report/results.tex',
                'outputs/metrics.json', 'data/h_matches.txt']
    for path in required:
        if not (ROOT / path).is_file():
            raise SystemExit(f'Missing deliverable: {path}')
    archive = ROOT / 'submission_compulsory.zip'
    paths = [ROOT / n for n in ('README.md', 'requirements.txt', 'run_experiments.py',
                                'build_report.py', 'package_submission.py')]
    for name in ('src', 'data', 'outputs'):
        paths.extend(p for p in (ROOT / name).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc')
    paths.extend(ROOT / n for n in ('report/main.tex', 'report/results.tex', 'report/main.pdf'))
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):
            z.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP integrity check failed')
        print(f'{archive.name}: {len(z.namelist())} files; integrity verified')


if __name__ == '__main__':
    main()
