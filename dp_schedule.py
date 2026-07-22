"""DP-based generation-schedule selection from per-timestep teacher-forcing metrics.

Shared by compute_fid.py (--dp) and inference.py (--dp). The input CSV is the
metrics.csv written by eval_timestep_metrics.py.
"""
import csv

import numpy as np


def dp_schedule(csv_path: str, metric: str, num_steps: int, K: int):
    """Compute a generation schedule t_0=K-1 > t_1 > ... > t_{N-1}=0 by dynamic programming.

    Cost model: the CSV gives a per-timestep one-shot prediction error e(t)
    (cross_entropy / l2_distance directly, accuracy as 1-accuracy), linearly
    interpolated onto every integer t. A schedule "relies on" the prediction made
    at node t_i for the whole jump down to t_{i+1}, so its total cost is the
    left-endpoint quadrature

        sum_i e(t_i) * (t_i - t_{i+1}),

    which the DP minimizes over all strictly decreasing N-node paths with both
    endpoints fixed. The optimum places nodes densely where the model is weak
    (large e) and jumps far where it is strong. t=0 is a virtual terminal node:
    autoregressive_infer_cfg drops trailing t=0 entries, so N nodes execute N-1
    forwards, the last at the smallest nonzero chosen t.

    Returns (schedule as a python list of ints, optimal cost).
    """
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows, f'no data rows in {csv_path}'
    rows.sort(key=lambda r: int(r['t']))
    ts = np.array([int(r['t']) for r in rows])
    vals = np.array([float(r[metric]) for r in rows])
    assert 0 <= ts[0] and ts[-1] <= K - 1, f'CSV timesteps outside [0, {K-1}]'

    err = (1.0 - vals) if metric == 'accuracy' else vals
    # np.interp clamps outside the measured range, so a CSV missing t=0 / t=K-1
    # still yields a full cost table (flat extrapolation).
    cost = np.interp(np.arange(K), ts, err)

    N = num_steps
    T = K - 1
    assert 2 <= N <= K, f'--num-steps must be in [2, {K}] for a DP schedule, got {N}'

    idx = np.arange(K)
    gap = idx[:, None] - idx[None, :]                # gap[v, u] = v - u
    edge = cost[:, None] * gap                       # edge[v, u] = cost of jump v -> u
    invalid = gap <= 0                               # only strictly decreasing jumps

    # dp[u] = min cost of a path T -> u with exactly k edges; parent for backtracking
    dp = np.full(K, np.inf)
    dp[T] = 0.0
    parent = np.zeros((N - 1, K), dtype=np.int64)
    for k in range(N - 1):
        cand = dp[:, None] + edge                    # cand[v, u]
        cand[invalid] = np.inf
        parent[k] = cand.argmin(axis=0)
        dp = cand.min(axis=0)
    assert np.isfinite(dp[0]), 'DP found no feasible schedule (should not happen for N <= K)'

    schedule = [0]
    u = 0
    for k in range(N - 2, -1, -1):
        u = int(parent[k, u])
        schedule.append(u)
    assert u == T, f'backtracking did not reach t={T}'
    return schedule[::-1], float(dp[0])
