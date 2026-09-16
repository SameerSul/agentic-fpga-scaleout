"""Correctness tests. Run: python3 tests.py
Runs the agentic flow once (needs iverilog/vvp), then exercises the fit layer
and the fabric against exact references. Pure Python 3 stdlib."""
import random

from chiplet_flow import run_flow
from boards import BOARDS, fit
from fabric import make_cluster, fabric_stats, mm, relu, RX_CAP
from collectives import ring_allreduce, run_workers
from demo import mlp_run

PASSED = [0]


def check(name, cond):
    print('%-55s %s' % (name, 'PASS' if cond else 'FAIL'))
    assert cond, name
    PASSED[0] += 1


def get_profile():
    report, profile = run_flow(verbose=False)
    return report, profile


def test_flow_and_profile(report, profile):
    check('flow converges', report['converged'])
    check('flow converges in 3 iterations', report['iterations_used'] == 3)
    fields = ('cycles_per_mac', 'latency_cycles', 'cell_count', 'area',
              'fmax_estimate_mhz', 'sim_checks_passed')
    ok = profile is not None and all(
        profile.get(k) is not None and profile[k] > 0 for k in fields)
    check('profile fields all present and positive', ok)
    check('cycles_per_mac measured near pipelined ideal',
          0.9 <= profile['cycles_per_mac'] <= 2.0)


def test_fit_monotonic(profile):
    names = sorted(BOARDS, key=lambda n: BOARDS[n]['lut_capacity_proxy'])
    fits = [fit(n, profile) for n in names]
    inst = [f['instances'] for f in fits]
    thr = [f['macs_per_s'] for f in fits]
    check('fit monotonic: bigger board fits more instances',
          all(a < b for a, b in zip(inst, inst[1:])))
    check('fit monotonic: bigger board has higher MACs/s',
          all(a < b for a, b in zip(thr, thr[1:])))
    check('fit respects board clock cap',
          all(f['clock_mhz'] <= BOARDS[n]['max_chiplet_clock_mhz']
              for n, f in zip(names, fits)))


def test_allreduce(profile):
    mid = fit('zynq_us_mid', profile)
    for n in (2, 3, 5, 8):
        sim, boards = make_cluster([mid] * n)
        L = 1000
        for i, b in enumerate(boards):
            b.mem = [(i + 1) * 0.25 + k * 1e-3 for k in range(L)]
        exp = [sum((i + 1) * 0.25 + k * 1e-3 for i in range(n))
               for k in range(L)]
        run_workers(sim, ring_allreduce(boards, [b.mem for b in boards]))
        err = max(abs(b.mem[k] - exp[k]) for b in boards for k in range(L))
        st = fabric_stats(boards)
        check('ring allreduce n=%d exact vs reference sum' % n,
              err < 1e-9 and st['overflow_drops'] == 0)
    # Heterogeneous ring: mixed link rates, still exact.
    fits = [fit('artix7_small', profile), fit('versal_large', profile),
            fit('zynq_us_mid', profile)]
    sim, boards = make_cluster(fits)
    L = 777
    for i, b in enumerate(boards):
        b.mem = [(i + 2) * 0.5 - k * 1e-3 for k in range(L)]
    exp = [sum((i + 2) * 0.5 - k * 1e-3 for i in range(3)) for k in range(L)]
    run_workers(sim, ring_allreduce(boards, [b.mem for b in boards]))
    err = max(abs(b.mem[k] - exp[k]) for b in boards for k in range(L))
    check('ring allreduce heterogeneous 3-board exact', err < 1e-9)


def test_hetero_bit_identical(profile):
    """Same n, same shard split: the heterogeneous cluster must produce
    bit-identical MLP output to the homogeneous one (only timing differs)."""
    small, mid, large = (fit(n, profile) for n in
                         ('artix7_small', 'zynq_us_mid', 'versal_large'))
    t_h, err_h, _, out_h = mlp_run([mid] * 4)
    t_x, err_x, _, out_x = mlp_run([small, small, large, large])
    check('heterogeneous MLP output bit-identical to homogeneous',
          out_x == out_h)
    check('heterogeneous cluster slower with equal shards (honest model)',
          t_x > t_h)


def test_link_reliability(profile):
    mid = fit('zynq_us_mid', profile)
    sim, boards = make_cluster([mid, mid], ber=1e-5)
    vals = [random.Random(3).uniform(-1, 1) for _ in range(8192)]  # 64 KB
    out = {}

    def rxp():
        got = yield from boards[1].recv(0, 't')
        out['ok'] = (got == vals)

    run_workers(sim, [boards[0].send(1, vals, 't'), rxp()])
    st = fabric_stats(boards)
    check('BER injection: corruption detected and retransmitted',
          st['crc_drops'] > 0 and st['retransmits'] > 0)
    check('BER injection: transfer bit-exact after recovery', out['ok'])
    check('rx buffer bounded under recovery',
          st['overflow_drops'] == 0 and st['max_rx_occupancy'] <= RX_CAP)


if __name__ == '__main__':
    report, profile = get_profile()
    test_flow_and_profile(report, profile)
    test_fit_monotonic(profile)
    test_allreduce(profile)
    test_hetero_bit_identical(profile)
    test_link_reliability(profile)
    print('all %d tests passed' % PASSED[0])
