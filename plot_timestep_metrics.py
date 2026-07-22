"""Render per-timestep metric plots from a metrics.csv written by
eval_timestep_metrics.py. Pure csv + matplotlib -- no torch, so it can run
anywhere matplotlib is installed (login node, laptop, ...).

Usage:
    python plot_timestep_metrics.py [timestep_metrics/metrics.csv] [--out-dir DIR]

Writes accuracy.png / cross_entropy.png / l2_distance.png next to the CSV
(or into --out-dir).
"""
import argparse
import csv
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# resolution-chunk schedule of VAR._t_to_size: 256 | 128 | 64 | 32 | 16
CHUNKS = ((0, 90, 256), (90, 120, 128), (120, 150, 64), (150, 180, 32), (180, 200, 16))

METRICS = (
    ('accuracy', 'token accuracy', 'accuracy.png'),
    ('cross_entropy', 'cross entropy (nats/token)', 'cross_entropy.png'),
    ('l2_distance', 'L2 distance (codebook embedding)', 'l2_distance.png'),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv_path', nargs='?', default='timestep_metrics/metrics.csv')
    p.add_argument('--out-dir', type=str, default=None, help='output directory (default: the CSV\'s directory)')
    return p.parse_args()


def plot_metric(ts, values, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, values, marker='o', markersize=3, linewidth=1.5)
    for left, right, size in CHUNKS:
        if right < 200:
            ax.axvline(right, color='gray', linestyle='--', linewidth=0.8)
        ax.text((left + right) / 2, 1.02, f'{size}px', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8, color='gray')
    ax.set_xlim(-2, 202)
    ax.set_xlabel('blur timestep t')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else (osp.dirname(args.csv_path) or '.')
    os.makedirs(out_dir, exist_ok=True)

    with open(args.csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'no data rows in {args.csv_path}')
    rows.sort(key=lambda r: int(r['t']))
    ts = [int(r['t']) for r in rows]

    for column, ylabel, filename in METRICS:
        values = [float(r[column]) for r in rows]
        plot_metric(ts, values, ylabel, osp.join(out_dir, filename))


if __name__ == '__main__':
    main()
