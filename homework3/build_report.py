"""Compile the report twice; XeLaTeX must be available on PATH."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    executable = shutil.which('xelatex')
    if not executable:
        raise SystemExit('XeLaTeX is missing; install TeX Live with Chinese support.')
    report = ROOT / 'report'
    if not (report / 'results.tex').exists():
        raise SystemExit('Run run_experiments.py --mode all first.')
    for _ in range(2):
        subprocess.run([executable, '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
                       cwd=report, check=True)
    print(f'Report compiled: {report / "main.pdf"}')


if __name__ == '__main__':
    main()
