"""Correctness tests. Run: python3 tests.py
Runs both agentic flows once (needs iverilog/vvp), then exercises the fit
layer, the fabric, and the model sizing against exact references. Pure
Python 3 stdlib."""
import random
import zlib

from chiplet_flow import run_flow, FABRIC_JOB
from boards import BOARDS, fit
from fabric import make_cluster, fabric_stats, mm, relu, RX_CAP
from collectives import ring_allreduce, run_workers
from demo import mlp_run
from sizing import load_model_spec, size_fabric, simulate_decode

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


def crc32_word_serial(data):
    """Python mirror of the generated RTL: 32-bit word-serial, bytes packed
    little-endian, reflected poly 0xEDB88320, init and final XOR all-ones.
    Defined for payloads that are a multiple of 4 bytes, like the fabric's
    packet payloads."""
    c = 0xFFFFFFFF
    for i in range(0, len(data), 4):
        x = c ^ int.from_bytes(data[i:i + 4], 'little')
        for _ in range(32):
            x = (x >> 1) ^ (0xEDB88320 if x & 1 else 0)
        c = x
    return c ^ 0xFFFFFFFF


def test_fabric_flow():
    report, fp = run_flow(FABRIC_JOB, verbose=False)
    check('fabric endpoint flow converges', report['converged'])
    check('fabric endpoint flow converges in 2 iterations',
          report['iterations_used'] == 2)
    fields = ('cycles_per_byte', 'latency_cycles', 'cell_count', 'area',
              'fmax_estimate_mhz', 'sim_checks_passed', 'endpoint_gbps')
    ok = fp is not None and all(
        fp.get(k) is not None and fp[k] > 0 for k in fields)
    check('fabric profile fields all present and positive', ok)
    rng = random.Random(9)
    ok = all(crc32_word_serial(p) == zlib.crc32(p)
             for p in (bytes(rng.randrange(256) for _ in range(4 * k))
                       for k in (1, 2, 7, 33, 256)))
    check('word-serial CRC32 reference matches zlib on random payloads', ok)
    return fp


def test_sizing(profile, fp):
    ms = load_model_spec()
    mid = fit('zynq_us_mid', profile, fp)

    def boards_needed(m):
        ch = size_fabric(m, mid)['chosen']
        return ch['boards'] if ch else float('inf')

    targets = [100, 300, 500, 1000, 2000]
    need = [boards_needed(dict(ms, target_tokens_per_s=t)) for t in targets]
    check('sizing monotonic: higher target rate needs more boards',
          all(a <= b for a, b in zip(need, need[1:])))
    need = [boards_needed(dict(ms, d_model=d)) for d in (384, 768, 1536)]
    check('sizing monotonic: bigger model needs more boards',
          all(a <= b for a, b in zip(need, need[1:])))

    ch = size_fabric(ms, mid)['chosen']
    check('sizing finds a config that meets the target on the mid class',
          ch is not None
          and ch['predicted_tok_per_s'] >= ms['target_tokens_per_s'])
    r = simulate_decode([mid] * ch['boards'], ms, tokens=8)
    ratio = r['tok_per_s'] / ch['predicted_tok_per_s']
    check('analytic prediction within 15%% of fabric simulation '
          '(ratio %.2f)' % ratio, 0.85 <= ratio <= 1.15)
    check('decode issues n_layer * 2 all-reduces per token',
          r['collectives_per_token'] == ms['n_layer'] * 2)
    check('decode activations identical on every board',
          r['vecs_equal_across_boards'])


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
    fp = test_fabric_flow()
    test_sizing(profile, fp)
    test_link_reliability(profile)
    print('all %d tests passed' % PASSED[0])
