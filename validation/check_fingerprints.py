"""Check the energy classifier against the DAFD shipped-Acrobot per-trial archive."""
import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from simaudit.energy import _classify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paper', type=Path, help='dafd repository at revision 1e69c46')
    args = parser.parse_args()
    counts = Counter()
    with (args.paper / 'Results/shipped_acrobot_fingerprint_trials.csv').open() as stream:
        for row in csv.DictReader(stream):
            rates = [float(row[f'r{level}']) for level in (1, 2, 4, 8)]
            scale = 1 + float(row['E_abs_mean'])
            floor = 50 * np.finfo(float).eps * scale / 0.2
            label, _, _ = _classify(rates, floor, scale / 2, 4)
            if label != row['cls']:
                raise ValueError(f"{row['case']} trial {row['trial']}: {label} != {row['cls']}")
            counts[label] += 1
    if sum(counts.values()) != 1200:
        raise ValueError(f'Expected 1200 archived trials, read {sum(counts.values())}')
    print(f'All 1200 archived labels match: {dict(counts)}')


if __name__ == '__main__':
    main()
