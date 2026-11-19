"""Check the release checkout, measured workflow sources and optional original archives."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, help='also require both original archives in this directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / 'evidence/release_receipt.json').read_text())
    workflow = json.loads((root / 'evidence/workflow_cases.json').read_text())
    version = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
    if version != receipt['package_version']:
        raise SystemExit(f'Release receipt does not cover version {version}')
    sources = {p.relative_to(root).as_posix() for p in (root / 'src/simaudit').rglob('*.py')}
    measured = {p for p in workflow['sha256'] if p.startswith('src/simaudit/')}
    if sources != measured:
        raise SystemExit('Package module inventory differs from the workflow measurement')
    manifests = [(root, receipt['checkout_sha256']), (root, workflow['sha256'])]
    if args.dist:
        manifests.append((args.dist, receipt['distribution_sha256']))
    for directory, hashes in manifests:
        for name, expected in hashes.items():
            path = directory / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise SystemExit(f'Missing or changed file: {path}')
    print(f"Verified {receipt['package_version']} checkout and workflow sources" +
          (' and both distribution files.' if args.dist else '.'))


if __name__ == '__main__':
    main()
