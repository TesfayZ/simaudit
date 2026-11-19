"""Record the release checkout and the two locally tested distribution files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = Path('evidence/release_receipt.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, default=ROOT / 'dist')
    args = parser.parse_args()
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    archives = [f'simaudit-{version}-py3-none-any.whl', f'simaudit-{version}.tar.gz']
    if sorted(p.name for p in args.dist.iterdir()) != sorted(archives):
        parser.error(f'expected exactly {archives} in {args.dist}')
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT
    ).decode().split('\0')
    files = {name: digest(ROOT / name) for name in sorted(set(names))
             if name and Path(name) != RECEIPT and (ROOT / name).is_file()}
    receipt = dict(package_version=version, checkout_sha256=files,
                   distribution_sha256={name: digest(args.dist / name) for name in archives})
    (ROOT / RECEIPT).write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
