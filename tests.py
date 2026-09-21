"""Correctness tests. Run: python3 tests.py
Runs both agentic flows once (needs iverilog/vvp), then exercises the spec
derivation, the fit layer, the fabric, and the model sizing against exact
references. Pure Python 3 stdlib."""
import math
import os
import random
import zlib

from chiplet_flow import (run_flow, run_endpoint_flow, make_agent, ROOT)
from agent import RuleBasedAgent
from llm_agent import build_prompt, extract_verilog, condense_feedback
from specgen import (derive_chiplet_spec, derive_endpoint_spec,
                     endpoint_options, generate)
from boards import BOARDS, fit, TRANSPORTS
from fpga import synth_fpga
from fabric import make_cluster, fabric_stats, mm, relu, RX_CAP
from collectives import ring_allreduce, run_workers
from demo import mlp_run
from sizing import (load_model_spec, model_summary, predict_config,
                    size_fabric, simulate_decode)

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
    names = sorted(BOARDS, key=lambda n: BOARDS[n]['dsps'])
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
    mid = fit('kc705', profile)
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
    fits = [fit('arty_a7_100t', profile), fit('alveo_u250', profile),
            fit('kc705', profile)]
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
                         ('arty_a7_100t', 'kc705', 'alveo_u250'))
    t_h, err_h, _, out_h = mlp_run([mid] * 4)
    t_x, err_x, _, out_x = mlp_run([small, small, large, large])
    check('heterogeneous MLP output bit-identical to homogeneous',
          out_x == out_h)
    check('heterogeneous cluster slower with equal shards (honest model)',
          t_x > t_h)


def test_specgen(profile):
    ms = load_model_spec()
    spec = derive_chiplet_spec(ms)
    p = spec['parameters']
    aw = (ms['weight_bits'] + ms['activation_bits']
          + math.ceil(math.log2(max(ms['d_model'], ms['d_ff']))))
    check('chiplet datapath width follows the model quantization',
          p['data_width'] == max(ms['weight_bits'], ms['activation_bits']))
    check('accumulator width covers the longest reduction exactly',
          p['acc_width'] == aw)
    check('signed-off profile carries the derived widths',
          profile['data_width'] == p['data_width']
          and profile['acc_width'] == p['acc_width'])
    wide = derive_chiplet_spec(dict(ms, d_ff=4 * ms['d_ff']))
    check('deeper reduction derives a wider accumulator',
          wide['parameters']['acc_width'] == p['acc_width'] + 2)


def test_model_driven_flow(profile):
    """A different model must yield different signed-off hardware through the
    identical loop: re-quantize to 4 bits and run the full flow again."""
    ms = load_model_spec()
    q4 = dict(ms, name='gpt2_q4', weight_bits=4, activation_bits=4)
    job = {'spec_file': 'spec_q4.json', 'tb_file': 'tb_q4.v',
           'rtl_file': 'mac_q4.v', 'profile_file': 'profile_q4.json',
           'report_file': 'report_q4.json'}
    generate(q4, spec_file=job['spec_file'], tb_file=job['tb_file'])
    report, prof = run_flow(job, verbose=False)
    check('4-bit quantized model: flow converges on the derived spec',
          report['converged'] and report['iterations_used'] == 3)
    check('4-bit quantized model: profile has 4-bit datapath, 20-bit acc',
          prof['data_width'] == 4 and prof['acc_width'] == 4 + 4 + 12)
    check('4-bit chiplet is smaller than the 8-bit chiplet',
          prof['cell_count'] < profile['cell_count'])
    for f in job.values():
        if f != 'mac_q4.v':
            os.remove(os.path.join(ROOT, f))


def test_llm_agent_offline():
    """The LLM agent's deterministic parts, no network: agent selection,
    prompt assembly, and Verilog extraction from messy model output."""
    check('default agent is the deterministic rule-based one',
          isinstance(make_agent(), RuleBasedAgent))
    ms = load_model_spec()
    spec = derive_chiplet_spec(ms)
    fb = [{'stage': 'sim', 'status': 'fail', 'iteration': 1,
           'mismatches': [{'test': 'wide_product',
                           'expected_acc': '65328', 'got_acc': '304'}]},
          {'stage': 'synth', 'status': 'fail', 'iteration': 2,
           'errors': ['ERROR: syntax error near always_ff']}]
    p = build_prompt(spec, fb, 'module mac();\nendmodule')
    check('prompt carries spec, previous attempt, and parsed feedback',
          spec['top_module'] in p and 'PREVIOUS ATTEMPT' in p
          and 'wide_product' in p and 'always_ff' in p
          and 'Verilog-2005' in p)
    check('feedback condenser keeps only the recent, trimmed records',
          len(condense_feedback(fb)) == 2
          and 'testbench_mismatches' in condense_feedback(fb)[0])
    fenced = 'Sure!\n```verilog\nmodule mac (input clk);\nendmodule\n```\ndone'
    bare = 'preamble text\nmodule mac (input clk);\nendmodule\ntrailing prose'
    check('verilog extraction strips fences and surrounding prose',
          extract_verilog(fenced) == 'module mac (input clk);\nendmodule\n'
          and extract_verilog(bare) == 'module mac (input clk);\nendmodule\n')


def test_fpga_backend(profile, fp):
    """Real device mapping, not generic cells: the two generated blocks land
    on different resources, which is what makes a single capacity proxy
    wrong and multi-resource fit necessary."""
    c, e = profile.get('fpga'), fp.get('fpga')
    check('chiplet profile carries real FPGA resources',
          c is not None and c['luts'] > 0 and c['ffs'] > 0)
    check('the MAC infers exactly one DSP slice', c['dsps'] == 1)
    check('the endpoint is pure LUT logic, no DSP',
          e is not None and e['luts'] > 0 and e['dsps'] == 0)
    bad = os.path.join(ROOT, 'build', 'lint_probe.v')
    with open(bad, 'w') as f:
        f.write('module lint_probe(input clk, input en, input [3:0] d,\n'
                '  output reg [3:0] q, output reg [3:0] l);\n'
                '  always @(posedge clk) q <= d;\n'
                '  always @(*) if (en) l = d;\n'
                'endmodule\n')
    res = synth_fpga('lint_probe.v', 'lint_probe',
                     os.path.join(ROOT, 'build'))
    kinds = [f['kind'] for f in res.get('lint', [])]
    check('FPGA lint catches an inferred latch', 'inferred_latch' in kinds)
    os.remove(bad)


def test_endpoint_derivation(fp):
    """The endpoint datapath is derived from the link rate it must sustain,
    and the signed-off design actually sustains it."""
    widths = [derive_endpoint_spec(g)['parameters']['bytes_per_cycle']
              for g in (1.0, 10.0, 25.0, 100.0)]
    check('endpoint width grows with the link rate',
          all(a <= b for a, b in zip(widths, widths[1:])))
    for g in (1.0, 10.0, 25.0):
        rate, opts = endpoint_options(g)
        ok = all(w * 8 * clk / 1000.0 >= rate - 1e-9 for w, clk in opts)
        check('every %g Gbps datapath option sustains the rate' % g, ok)
    d = fp.get('derivation', {})
    check('signed-off endpoint sustains its target link rate',
          fp['endpoint_gbps'] >= d.get('target_link_gbps', 0) - 1e-9)


def test_transports(profile, fp):
    """Wiring choice is a measurable trade, not a preference."""
    a = fit('alveo_u250', profile, fp, transport='aurora')
    d = fit('alveo_u250', profile, fp, transport='ethernet_direct')
    s = fit('alveo_u250', profile, fp, transport='ethernet_switched')
    check('Ethernet costs latency per hop versus Aurora',
          d['link_prop_ns'] > a['link_prop_ns'])
    check('a switch hop costs more latency than direct attach',
          s['link_prop_ns'] > d['link_prop_ns'])
    check('Ethernet framing costs more wire bytes per packet',
          d['frame_overhead_bytes'] > a['frame_overhead_bytes'])
    check('the Ethernet MAC costs real logic, so fewer chiplets fit',
          d['instances'] <= a['instances'])
    ms = load_model_spec()
    ta, td = (predict_config(ms, f, 8)['predicted_tok_per_s'] for f in (a, d))
    check('Aurora predicts higher throughput at 8 boards', ta > td)
    # Transport must not change the arithmetic, only the timing.
    ra = simulate_decode([a] * 4, ms, tokens=2)
    rd = simulate_decode([d] * 4, ms, tokens=2)
    check('every transport produces identical activations',
          ra['final_vec'] == rd['final_vec']
          and rd['vecs_equal_across_boards'])


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
    report, fp = run_endpoint_flow(verbose=False)
    check('fabric endpoint flow converges', report['converged'])
    check('endpoint converges in 2 iterations once the datapath closes',
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
    lg = fit('alveo_u250', profile, fp)
    mid = fit('kc705', profile, fp)
    s = model_summary(ms)

    check('weight traffic identity: bytes/token = block MACs * wb/8',
          s['weight_bytes'] == s['per_token_macs'] * ms['weight_bits'] / 8.0)
    check('KV cache bytes follow n_layer * 2 * d_model * seq_len',
          s['kv_cache_bytes'] == ms['n_layer'] * 2 * ms['d_model']
          * ms['seq_len'] * ms['activation_bits'] / 8.0)

    def boards_needed(m, f):
        ch = size_fabric(m, f)['chosen']
        return ch['boards'] if ch else float('inf')

    targets = [100, 300, 500, 1000]
    need = [boards_needed(dict(ms, target_tokens_per_s=t), lg)
            for t in targets]
    check('sizing monotonic: higher target rate needs more boards',
          all(a <= b for a, b in zip(need, need[1:])))
    need = [boards_needed(dict(ms, d_model=d), lg) for d in (384, 768, 1536)]
    check('sizing monotonic: bigger model needs more boards',
          all(a <= b for a, b in zip(need, need[1:])))
    tok_ns = [predict_config(dict(ms, seq_len=q), lg, 4)['token_ns']
              for q in (256, 1024, 4096)]
    check('sizing monotonic: longer context costs token time',
          all(a <= b for a, b in zip(tok_ns, tok_ns[1:])))

    check('small class cannot host this model at any board count',
          size_fabric(ms, fit('arty_a7_100t', profile, fp))['chosen'] is None)
    ch = size_fabric(ms, lg)['chosen']
    check('sizing meets the target on the large class',
          ch is not None
          and ch['predicted_tok_per_s'] >= ms['target_tokens_per_s'])
    one = predict_config(ms, lg, 1)
    check('streaming weights from DDR is memory-bound',
          not one['sram_resident'] and one['bound'] == 'memory')
    resident = [p for p in size_fabric(ms, lg)['sweep'] if p['sram_resident']]
    check('SRAM residency jumps throughput by more than 3x',
          resident and resident[0]['predicted_tok_per_s']
          > 3 * one['predicted_tok_per_s'])
    sweep = size_fabric(ms, lg)['sweep']
    peak = max(range(len(sweep)), key=lambda i: sweep[i]['predicted_tok_per_s'])
    check('throughput peaks then falls as the ring term takes over',
          peak < len(sweep) - 1
          and sweep[-1]['bound'] == 'comm')

    r = simulate_decode([lg] * ch['boards'], ms, tokens=8)
    ratio = r['tok_per_s'] / ch['predicted_tok_per_s']
    check('analytic prediction within 15%% of fabric simulation '
          '(ratio %.2f)' % ratio, 0.85 <= ratio <= 1.15)
    check('decode issues n_layer * 2 all-reduces per token',
          r['collectives_per_token'] == ms['n_layer'] * 2)
    check('decode activations identical on every board',
          r['vecs_equal_across_boards'])


def test_link_reliability(profile):
    mid = fit('kc705', profile)
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
    test_specgen(profile)
    test_model_driven_flow(profile)
    test_llm_agent_offline()
    test_fit_monotonic(profile)
    test_allreduce(profile)
    test_hetero_bit_identical(profile)
    fp = test_fabric_flow()
    test_fpga_backend(profile, fp)
    test_endpoint_derivation(fp)
    test_transports(profile, fp)
    test_sizing(profile, fp)
    test_link_reliability(profile)
    print('all %d tests passed' % PASSED[0])
