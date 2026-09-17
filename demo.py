"""End-to-end demo: the LLM is the input workload. Agents generate and sign
off two pieces of RTL (the compute chiplet and the fabric endpoint), the
measured profiles deploy onto simulated FPGA board classes, sizing derives
the right fabric configuration to host the model at its target token rate,
and the chosen cluster hosts decode on the discrete-event fabric. Scaling is
adding more boards running the same synthesized fabric.

Run: python3 demo.py    (writes results.json)"""
import json
import random

from chiplet_flow import run_flow, CHIPLET_JOB, FABRIC_JOB
from specgen import generate
from boards import fit, FIT_FRACTION
from fabric import make_cluster, fabric_stats, mm, relu
from collectives import ring_allreduce, run_workers
from sizing import (load_model_spec, model_summary, size_fabric,
                    simulate_decode)

M, D, F = 64, 64, 512  # reduced-dim MLP shapes for the numerics check
BOARD_ORDER = ('artix7_small', 'zynq_us_mid', 'versal_large')


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


def banner(n, text):
    print('\n' + '=' * 72)
    print('Stage %d: %s' % (n, text))
    print('=' * 72)


def stage1_model(results):
    banner(1, 'the input LLM (model_spec.json drives everything downstream)')
    ms = load_model_spec()
    s = model_summary(ms)
    rows = [
        ['model', ms['name'], ms['description']],
        ['layers x d_model x d_ff',
         '%d x %d x %d' % (ms['n_layer'], ms['d_model'], ms['d_ff']),
         'decoder-only transformer'],
        ['MACs per decode token', '%.1fM' % (s['per_token_macs'] / 1e6),
         'n_layer * (4*d^2 attn + 2*d*d_ff MLP)'],
        ['all-reduces per token', s['allreduces_per_token'],
         'Megatron TP: one per attn block, one per MLP block'],
        ['bytes per all-reduce', s['allreduce_bytes'],
         '[1, d_model] activation, %d-byte elements' % ms['dtype_bytes']],
        ['comm bytes per token', s['comm_bytes_per_token'],
         'what the fabric must carry per token'],
        ['target', '%d tok/s' % ms['target_tokens_per_s'],
         'the rate the fabric is sized to'],
    ]
    table('the workload the agents must build hardware for',
          ['quantity', 'value', 'meaning'], rows)
    results['model'] = {'spec': ms, 'summary': s}
    return ms, s


def stage2_chiplet(ms, results):
    banner(2, 'derive the chiplet spec from the model, then agents '
              'generate it')
    spec = generate(ms)
    p, dv = spec['parameters'], spec['derivation']
    rows = [
        ['MAC datapath', '%d x %d bits' % (p['data_width'], p['data_width']),
         'the model quantization: weight_bits=%d, activation_bits=%d'
         % (dv['weight_bits'], dv['activation_bits'])],
        ['accumulator', '%d bits' % p['acc_width'],
         '%d product bits + %d guard bits, overflow-free by construction'
         % (dv['weight_bits'] + dv['activation_bits'], dv['guard_bits'])],
        ['guard bits', dv['guard_bits'],
         'ceil(log2(%d)), the longest dot-product reduction (d_ff)'
         % dv['reduction_depth']],
    ]
    table('%s: spec_mac.json and tb_mac.v both generated from the model'
          % spec['name'], ['quantity', 'value', 'derivation'], rows)
    print()
    report, profile = run_flow(CHIPLET_JOB, verbose=True)
    assert report['converged'] and profile is not None
    results['chiplet_spec'] = spec
    results['chiplet_flow'] = {'converged': True,
                               'iterations': report['iterations_used']}
    results['chiplet_profile'] = profile
    return profile


def stage3_endpoint(results):
    banner(3, 'agents generate the fabric endpoint (CRC32 datapath, '
              'same signoff)')
    report, profile = run_flow(FABRIC_JOB, verbose=True)
    assert report['converged'] and profile is not None
    print('\n   the fabric is synthesized, not assumed: the endpoint that '
          'checks every')
    print('   packet in stage 6 is this RTL, and its measured rate caps the '
          'link below')
    results['fabric_flow'] = {'converged': True,
                              'iterations': report['iterations_used']}
    results['fabric_profile'] = profile
    return profile


def stage4_fit(cp, fp, results):
    banner(4, 'both profiles deploy on any board class (boards.py)')
    rows, out = [], {}
    for name in BOARD_ORDER:
        f = fit(name, cp, fp)
        rows.append([f['board_class'], f['instances'],
                     '%.1f' % f['clock_mhz'],
                     '%.2f' % (f['macs_per_s'] / 1e9),
                     '%.1f' % f['transceiver_gbps'],
                     '%.2f' % f['endpoint_gbps'],
                     '%.2f' % f['link_gbps']])
        out[name] = f
    table('fit(board, chiplet_profile, fabric_profile), usable fabric '
          'fraction = %.0f%%' % (100 * FIT_FRACTION),
          ['board class', 'instances', 'clock MHz', 'GMAC/s',
           'xcvr Gbps', 'endpoint Gbps', 'link Gbps'], rows)
    print('   link rate = min(transceiver, synthesized endpoint): the '
          'endpoint gates')
    print('   the mid and large boards, an honest measured limit, not a '
          'datasheet number')
    results['fit'] = list(out.values())
    return out


def stage5_sizing(ms, fits_by_name, results):
    banner(5, 'the right fabric: smallest cluster per board class that '
              'hosts the model')
    sizings, rows = {}, []
    for name in BOARD_ORDER:
        sz = size_fabric(ms, fits_by_name[name])
        sizings[name] = sz
        ch = sz['chosen']
        if ch is None:
            best = max(sz['sweep'], key=lambda p: p['predicted_tok_per_s'])
            rows.append([name, 'unreachable',
                         '%.0f @ n=%d' % (best['predicted_tok_per_s'],
                                          best['boards']), ''])
        else:
            rows.append([name, ch['boards'],
                         '%.0f' % ch['predicted_tok_per_s'],
                         '%.0f%%' % (100 * ch['comm_fraction'])])
    table('analytic sizing vs the %d tok/s target'
          % ms['target_tokens_per_s'],
          ['board class', 'boards needed', 'predicted tok/s',
           'comm fraction'], rows)

    mid = sizings['zynq_us_mid']
    rows = [[p['boards'], '%.0f' % p['predicted_tok_per_s'],
             fmt_ns(p['compute_ns_per_token']),
             fmt_ns(p['comm_ns_per_token']),
             'chosen' if mid['chosen'] and p['boards'] ==
             mid['chosen']['boards'] else '']
            for p in mid['sweep']]
    table('candidate sweep on the mid class (compute shrinks with n, '
          'all-reduce grows)',
          ['boards', 'pred tok/s', 'compute/token', 'comm/token', ''], rows)
    results['sizing'] = sizings
    return sizings


def stage6_host(ms, fits_by_name, sizings, results):
    banner(6, 'host the model: decode on the chosen fabric, every '
              'all-reduce real traffic')
    name = 'zynq_us_mid'
    ch = sizings[name]['chosen']
    n = ch['boards']
    r = simulate_decode([fits_by_name[name]] * n, ms, tokens=8)
    ratio = r['tok_per_s'] / ch['predicted_tok_per_s']
    rows = [
        ['boards', n, 'chosen by sizing in stage 5'],
        ['tokens decoded', r['tokens'], 'full n_layer loop per token'],
        ['measured tok/s', '%.0f' % r['tok_per_s'],
         'discrete-event fabric, packetized + CRC + credits'],
        ['predicted tok/s', '%.0f' % ch['predicted_tok_per_s'],
         'analytic model from stage 5'],
        ['prediction ratio', '%.2f' % ratio, 'measured / predicted'],
        ['collectives per token', '%.0f' % r['collectives_per_token'],
         'n_layer * 2 = %d expected' % (ms['n_layer'] * 2)],
        ['wire bytes', r['fabric']['wire_bytes'],
         'headers and CRC included'],
        ['activations identical', r['vecs_equal_across_boards'],
         'every board holds the same reduced vector'],
    ]
    table('%d x %s hosting %s' % (n, name, ms['name']),
          ['quantity', 'value', 'provenance'], rows)
    ok = r['tok_per_s'] >= ms['target_tokens_per_s']
    print('   target %s: %.0f tok/s measured vs %d required'
          % ('met' if ok else 'MISSED', r['tok_per_s'],
             ms['target_tokens_per_s']))
    results['host'] = {'board': name, 'boards': n,
                       'measured_tok_per_s': r['tok_per_s'],
                       'predicted_tok_per_s': ch['predicted_tok_per_s'],
                       'prediction_ratio': ratio,
                       'target_met': ok, 'fabric': r['fabric']}
    return r


def stage7_scale(ms, fits_by_name, results):
    banner(7, 'scale by adding boards, and survive a lossy fabric')
    name = 'zynq_us_mid'
    f = fits_by_name[name]
    rows, out = [], []
    for n in (2, 4, 8, 16):
        r = simulate_decode([f] * n, ms, tokens=4)
        rows.append([n, '%.0f' % r['tok_per_s'],
                     fmt_ns(r['token_ns']),
                     r['vecs_equal_across_boards']])
        out.append({'boards': n, 'tok_per_s': r['tok_per_s'],
                    'token_ns': r['token_ns']})
    table('same synthesized fabric, more %s boards' % name,
          ['boards', 'tok/s', 'time/token', 'activations identical'], rows)
    print('   throughput scales until the ring all-reduce term (2*(n-1) '
          'steps) pushes back')

    n = 8
    clean = next(o for o in out if o['boards'] == n)
    r = simulate_decode([f] * n, ms, tokens=4, ber=1e-6)
    st = r['fabric']
    print('\n   BER 1e-6 on every link, %d boards: %d CRC drops, %d '
          'retransmits,' % (n, st['crc_drops'], st['retransmits']))
    print('   activations still identical on every board = %s, throughput '
          '%.0f tok/s' % (r['vecs_equal_across_boards'], r['tok_per_s']))
    print('   (%.1fx slower than the clean fabric: reliability costs '
          'latency, never bits)' % (clean['tok_per_s'] / r['tok_per_s']))
    assert r['vecs_equal_across_boards']
    results['scaling'] = out
    results['ber'] = {'ber': 1e-6, 'boards': n,
                      'tok_per_s': r['tok_per_s'],
                      'crc_drops': st['crc_drops'],
                      'retransmits': st['retransmits'],
                      'bit_exact': r['vecs_equal_across_boards']}


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


def stage8_numerics(fits_by_name, results):
    banner(8, 'numerics check at reduced dimensions (real arithmetic '
              'through the fabric)')
    mid = fits_by_name['zynq_us_mid']
    t_h, err_h, _, out_h = mlp_run([mid] * 4)
    print('   homogeneous 4-board MLP %dx%d W1 %dx%d W2 %dx%d: time %s, '
          'max err vs single-board reference %.1e'
          % (M, D, D, F, F, D, fmt_ns(t_h), err_h))

    fits = [fits_by_name['artix7_small'], fits_by_name['artix7_small'],
            fits_by_name['versal_large'], fits_by_name['versal_large']]
    t_eq, err_eq, _, out_eq = mlp_run(fits)
    wsum = sum(f['macs_per_s'] for f in fits)
    cols = [max(1, round(F * f['macs_per_s'] / wsum)) for f in fits]
    cols[-1] += F - sum(cols)
    t_pr, err_pr, _, _ = mlp_run(fits, cols=cols)
    print('   heterogeneous 2 small + 2 large: bit-identical output to '
          'homogeneous = %s' % (out_eq == out_h))
    print('   work-proportional shards (%s) recover %.2fx over the equal '
          'split' % (cols, t_eq / t_pr))
    results['numerics'] = {
        'homogeneous': {'time_ns': t_h, 'max_err': err_h},
        'hetero_equal': {'time_ns': t_eq, 'max_err': err_eq},
        'hetero_proportional': {'time_ns': t_pr, 'max_err': err_pr,
                                'columns': cols},
        'bit_identical_across_clusters': out_eq == out_h,
        'recovery_factor': t_eq / t_pr,
    }


def main():
    print('Agentic FPGA scaleout: given an LLM, generate and synthesize '
          'the fabric to host it')
    results = {}
    ms, _ = stage1_model(results)
    cp = stage2_chiplet(ms, results)
    fp = stage3_endpoint(results)
    fits_by_name = stage4_fit(cp, fp, results)
    sizings = stage5_sizing(ms, fits_by_name, results)
    stage6_host(ms, fits_by_name, sizings, results)
    stage7_scale(ms, fits_by_name, results)
    stage8_numerics(fits_by_name, results)
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nresults written to results.json')


if __name__ == '__main__':
    main()
