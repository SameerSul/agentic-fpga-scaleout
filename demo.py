"""End-to-end demo: an agent generates chiplet RTL, the flow signs it off and
measures a chiplet profile, the profile deploys onto simulated FPGA boards of
three classes, and the interconnect fabric scales the workload out across
homogeneous and heterogeneous clusters.

Run: python3 demo.py    (writes results.json)"""
import json
import random

from chiplet_flow import run_flow
from boards import BOARDS, fit, FIT_FRACTION
from fabric import make_cluster, fabric_stats, start, mm, relu
from collectives import ring_allreduce, run_workers

M, D, F = 64, 64, 512  # MLP shapes: x is MxD, W1 is DxF, W2 is FxD


def table(title, headers, rows):
    print('\n' + title)
    cells = [headers] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    line = '  '.join('-' * x for x in w)
    print('  '.join(h.ljust(x) for h, x in zip(headers, w)))
    print(line)
    for r in cells[1:]:
        print('  '.join(c.rjust(x) for c, x in zip(r, w)))


def fmt_ns(ns):
    if ns >= 1e6:
        return '%.2f ms' % (ns / 1e6)
    if ns >= 1e3:
        return '%.1f us' % (ns / 1e3)
    return '%.0f ns' % ns


def stage1_flow(results):
    print('=' * 72)
    print('Stage 1: agentic RTL generation and signoff (iverilog, yosys, OpenSTA)')
    print('=' * 72)
    report, profile = run_flow(verbose=True)
    assert report['converged'] and profile is not None, 'flow did not converge'
    results['flow'] = {'converged': True,
                       'iterations': report['iterations_used']}
    return profile


def stage2_profile(profile, results):
    print('\n' + '=' * 72)
    print('Stage 2: measured chiplet profile (the contract between the halves)')
    print('=' * 72)
    rows = [
        ['cycles_per_mac', '%.3f' % profile['cycles_per_mac'],
         'measured by testbench burst (span cycles / MACs)'],
        ['latency_cycles', profile['latency_cycles'], 'measured by testbench'],
        ['cell_count', profile['cell_count'], 'yosys stat, generic liberty'],
        ['area', profile['area'], 'yosys stat, generic liberty'],
        ['fmax_estimate_mhz', '%.1f' % profile['fmax_estimate_mhz'],
         'OpenSTA: 1000 / (period - worst slack)'],
        ['sim_checks_passed', profile['sim_checks_passed'],
         'self-checking testbench vs golden model'],
    ]
    table('chiplet_profile.json (method: %s)' % profile['fmax_method'],
          ['field', 'value', 'provenance'], rows)
    results['chiplet_profile'] = profile


def stage3_fit(profile, results):
    print('\n' + '=' * 72)
    print('Stage 3: the same profile deploys on any board class (boards.py)')
    print('=' * 72)
    rows, out = [], []
    for name in ('artix7_small', 'zynq_us_mid', 'versal_large'):
        f = fit(name, profile)
        rows.append([f['board_class'], f['instances'],
                     '%.1f' % f['clock_mhz'],
                     '%.0f%%' % (100 * f['utilization']),
                     '%.2f' % (f['macs_per_s'] / 1e9),
                     '%g / %d' % (f['link_gbps'], f['num_links'])])
        out.append(f)
    table('fit(board, chiplet_profile), usable fabric fraction = %.0f%%'
          % (100 * FIT_FRACTION),
          ['board class', 'instances', 'clock MHz', 'util',
           'GMAC/s', 'link Gbps / lanes'], rows)
    print('   no upstream change: instances and clock derive from the one '
          'measured profile')
    results['fit'] = out
    return {name: fit(name, profile)
            for name in ('artix7_small', 'zynq_us_mid', 'versal_large')}


def make_shards(x, W1, W2, cols):
    """Column-shard W1 and row-shard W2 with per-board column counts."""
    W1s, W2s, off = [], [], 0
    for c in cols:
        W1s.append([row[off:off + c] for row in W1])
        W2s.append(W2[off:off + c])
        off += c
    assert off == len(W2)
    return W1s, W2s


def mlp_run(fits, cols=None, seed=7):
    """Tensor-parallel MLP: y = relu(x @ W1) @ W2, W1 column-sharded,
    W2 row-sharded, ring all-reduce fused after the second matmul.
    cols optionally gives a per-board shard width (work-proportional split).
    Returns (time_ns, max_err_vs_ref, per-board busy fractions, output)."""
    n = len(fits)
    rng = random.Random(seed)
    rm = lambda r, c: [[rng.uniform(-0.5, 0.5) for _ in range(c)]
                       for _ in range(r)]
    x, W1, W2 = rm(M, D), rm(D, F), rm(F, D)
    ref = mm([[relu(v) for v in r] for r in mm(x, W1)], W2)
    flat_ref = [v for row in ref for v in row]
    sim, boards = make_cluster(fits)
    if cols is None:
        cols = [F // n] * n
    W1s, W2s = make_shards(x, W1, W2, cols)
    outs = [[0.0] * (M * D) for _ in range(n)]
    ar = ring_allreduce(boards, outs)

    def worker(i):
        b = boards[i]
        h = yield from b.matmul(x, W1s[i])
        h = yield from b.elementwise(h, relu)
        y = yield from b.matmul(h, W2s[i])
        outs[i][:] = [v for row in y for v in row]
        yield from ar[i]

    t = run_workers(sim, [worker(i) for i in range(n)])
    err = max(max(abs(a - b) for a, b in zip(o, flat_ref)) for o in outs)
    assert err < 1e-9, err
    st = fabric_stats(boards)
    assert st['overflow_drops'] == 0
    util = [b.busy_ns / t for b in boards]
    return t, err, util, outs[0]


def stage4_scaleout(fits_by_name, results):
    print('\n' + '=' * 72)
    print('Stage 4: tensor-parallel MLP scaleout on the mid board class')
    print('=' * 72)
    mid = fits_by_name['zynq_us_mid']
    macs = 2 * M * D * F  # two matmuls
    rows, out, t1 = [], [], None
    for n in (1, 2, 4, 8):
        t, err, _, _ = mlp_run([mid] * n)
        if n == 1:
            t1 = t
        rows.append([n, fmt_ns(t), '%.1f' % (2 * macs / t),
                     '%.2fx' % (t1 / t), '%.1e' % err])
        out.append({'boards': n, 'time_ns': t, 'gflops': 2 * macs / t,
                    'speedup': t1 / t, 'max_err': err})
    table('MLP x:%dx%d W1:%dx%d W2:%dx%d, fused ring all-reduce, '
          'zynq_us_mid boards' % (M, D, D, F, F, D),
          ['boards', 'time', 'GFLOP/s', 'speedup', 'max err vs ref'], rows)
    print('   sharded output matches the single-board reference on every run')
    results['scaleout'] = out


def stage5_hetero(fits_by_name, results):
    print('\n' + '=' * 72)
    print('Stage 5: heterogeneous cluster, same chiplet, mixed board classes')
    print('=' * 72)
    small, large = fits_by_name['artix7_small'], fits_by_name['versal_large']
    fits = [small, small, large, large]

    t_eq, err_eq, util_eq, _ = mlp_run(fits)
    rows = [[f['board'], F // len(fits), '%.0f%%' % (100 * u)]
            for f, u in zip(fits, util_eq)]
    table('equal shard split (2 small + 2 large), time %s, max err %.1e'
          % (fmt_ns(t_eq), err_eq),
          ['board', 'columns', 'compute utilization'], rows)

    # Work-proportional split: columns proportional to each board's MACs/s.
    wsum = sum(f['macs_per_s'] for f in fits)
    cols = [max(1, round(F * f['macs_per_s'] / wsum)) for f in fits]
    cols[-1] += F - sum(cols)  # largest board absorbs rounding
    t_pr, err_pr, util_pr, _ = mlp_run(fits, cols=cols)
    rows = [[f['board'], c, '%.0f%%' % (100 * u)]
            for f, c, u in zip(fits, cols, util_pr)]
    table('work-proportional split, time %s, max err %.1e'
          % (fmt_ns(t_pr), err_pr),
          ['board', 'columns', 'compute utilization'], rows)

    t_ref, _, _, _ = mlp_run([fits_by_name['zynq_us_mid']] * 4)
    print('   proportional split is %.2fx faster than equal split on the '
          'mixed cluster (slow boards no longer gate the ring)'
          % (t_eq / t_pr))
    results['hetero'] = {
        'boards': [f['board'] for f in fits],
        'equal_split': {'time_ns': t_eq, 'max_err': err_eq,
                        'utilization': util_eq},
        'proportional_split': {'time_ns': t_pr, 'max_err': err_pr,
                               'columns': cols, 'utilization': util_pr},
        'recovery_factor': t_eq / t_pr,
        'mid_homogeneous_4b_time_ns': t_ref,
    }


def main():
    print('Agentic FPGA scaleout: chiplet RTL to multi-board fabric, one run')
    results = {}
    profile = stage1_flow(results)
    stage2_profile(profile, results)
    fits_by_name = stage3_fit(profile, results)
    stage4_scaleout(fits_by_name, results)
    stage5_hetero(fits_by_name, results)
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nresults written to results.json')


if __name__ == '__main__':
    main()
