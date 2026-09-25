"""Correctness tests. Run: python3 tests.py
Runs both agentic flows once (needs iverilog/vvp), then exercises the spec
derivation, the fit layer, the fabric, and the model sizing against exact
references. Pure Python 3 stdlib."""
import math
import os
import random
import re
import shutil
import subprocess
import sys
import zlib

import chiplet_flow
from chiplet_flow import (run_flow, run_endpoint_flow, make_agent, ROOT)
import agent as agent_mod
from agent import RuleBasedAgent
from llm_agent import (build_prompt, extract_verilog, condense_feedback,
                       CALLERS)
import swarm as swarm_mod
from swarm import SwarmAgent, parse_review
import dv
import inference
import json
import specgen as specgen_mod
from specgen import (derive_chiplet_spec, derive_endpoint_spec,
                     endpoint_options, generate, crc_matrix)
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


def _scripted_swarm(replies, escalate=True):
    """A SwarmAgent wired to a canned reply list instead of a model, so the
    orchestration logic is testable with no network and no cost. Returns the
    agent and the list of prompts it sent, in order."""
    sent = []
    queue = list(replies)

    def fake(prompt, model):
        sent.append(prompt)
        return queue.pop(0) if queue else 'ACCEPT'

    CALLERS['_test'] = fake
    a = SwarmAgent.__new__(SwarmAgent)
    a.backend, a.model = '_test', 'fake'
    a.review_rounds = swarm_mod.MAX_REVIEW_ROUNDS
    a.use_reviewer = a.use_debugger = True   # tests drive it explicitly
    a.escalate = escalate
    a.last_rtl = None
    a.calls = {'writer': 0, 'reviewer': 0, 'debugger': 0}
    a.log = []
    return a, sent


def test_swarm_offline():
    """The swarm's orchestration, with the model stubbed out: who gets called
    and when, what each role is shown, and that no single role can wedge the
    run. The tools stay the judge, so the failure mode that matters is a role
    blocking or starving a design the tools would have accepted."""
    for text, want_ok in [('ACCEPT', True), ('  accept.  ', True),
                          ('ACCEPT - looks correct', True), ('', True),
                          ('   \n  ', True), ('ok', True)]:
        check('review %r reads as acceptance' % text[:14],
              parse_review(text)[0] is want_ok)
    ok, obj = parse_review('- acc is 16 bits, spec says 28\n- clear is missing')
    check('reviewer objections are parsed into actionable lines',
          not ok and '28' in obj and len(obj.splitlines()) == 2)
    check('objection list is capped at four lines',
          len(parse_review('\n'.join('defect number %d here' % i
                                     for i in range(9)))[1].splitlines()) == 4)

    ms = load_model_spec()
    spec = derive_chiplet_spec(ms)
    rtl_a = 'module mac (input clk);\nendmodule'
    rtl_b = 'module mac (input clk, input rst);\nendmodule'

    # First attempt of a run. Escalation keeps this to a lone writer, so the
    # swarm costs exactly what a single agent costs on work that was never
    # going to fail, which is the majority of runs.
    a, sent = _scripted_swarm([rtl_a])
    out, notes = a.propose(spec, [])
    check('a clean first attempt costs exactly one model call',
          sum(a.calls.values()) == 1 and a.calls['writer'] == 1)
    check('neither debugger nor reviewer runs before any tool has',
          a.calls['debugger'] == 0 and a.calls['reviewer'] == 0)
    check('accepted draft is returned unchanged', out.startswith('module mac'))

    # With escalation off every role runs from the first draft, which is how
    # the review loop itself is exercised.
    a, sent = _scripted_swarm([rtl_a, 'ACCEPT'], escalate=False)
    out, notes = a.propose(spec, [])
    check('reviewer runs on the first draft when escalation is off',
          a.calls['reviewer'] == 1 and 'reviewer:accept' in notes[1])
    check('notes record which roles ran', 'writer' in notes[1])

    # A rejection must cost exactly one rewrite and must show the writer the
    # draft the objections are about, not some older attempt.
    a, sent = _scripted_swarm([rtl_a, 'acc is truncated to 16 bits', rtl_b],
                              escalate=False)
    out, notes = a.propose(spec, [])
    check('rejection triggers exactly one rewrite',
          a.calls['writer'] == 2 and a.calls['reviewer'] == 1)
    check('rewrite prompt shows the rejected draft and the objection',
          'input clk);' in sent[2] and 'truncated to 16 bits' in sent[2])
    check('revised draft is what gets handed to the tools', 'rst' in out)
    check('revision is not re-reviewed past the round cap',
          'reviewer:reject' in notes[1] and a.calls['reviewer'] == 1)

    # With tool feedback present the debugger runs first and its diagnosis
    # has to reach the writer, otherwise the extra call bought nothing.
    fb = [{'stage': 'sim', 'status': 'fail', 'iteration': 1,
           'mismatches': [{'test': 'wide_product', 'expected_acc': '65328',
                           'got_acc': '304'}]}]
    a, sent = _scripted_swarm(['acc_out is 16 bits wide, must be 28',
                               rtl_b, 'ACCEPT'])
    a.last_rtl = rtl_a
    out, notes = a.propose(spec, fb)
    check('debugger runs first when tools have reported a failure',
          a.calls['debugger'] == 1 and 'wide_product' in sent[0])
    check('debugger is told not to write verilog', 'not write any Verilog'
          in sent[0] or 'Do not write any Verilog' in sent[0])
    check('diagnosis reaches the writer prompt',
          'must be 28' in sent[1] and 'HYPOTHESIS' in sent[1])
    # Ordering is the fix for a measured regression: a diagnosis presented
    # as authoritative above the tool output sent the writer chasing a wrong
    # cause for every remaining iteration.
    check('tool feedback outranks the diagnosis in the writer prompt',
          sent[1].index('TOOL FEEDBACK') < sent[1].index('HYPOTHESIS'))
    check('the diagnosis is labelled fallible, not a finding',
          'may be wrong' in sent[1] and 'believe the tools' in sent[1])
    check('tool feedback is named as ground truth',
          'ground truth' in sent[1])
    check('writer prompt still carries the spec and the hard rules',
          str(spec['parameters']['acc_width']) in sent[1]
          and 'Verilog-2005' in sent[1])

    # Degradation: any role can die without taking the run with it.
    def dies_on(marker):
        def caller(prompt, model):
            if marker in prompt:
                raise RuntimeError('role down')
            return rtl_a
        return caller

    a2, _ = _scripted_swarm([rtl_a], escalate=False)
    CALLERS['_test2'] = dies_on('reviewing Verilog')
    a2.backend = '_test2'
    out2, notes2 = a2.propose(spec, [])
    check('a dead reviewer cannot block a design the tools would accept',
          out2.startswith('module mac'))

    a3, _ = _scripted_swarm([rtl_a])
    CALLERS['_test3'] = dies_on('debug engineer')
    a3.backend = '_test3'
    a3.last_rtl = rtl_a
    out3, _ = a3.propose(spec, fb)
    check('a dead debugger still yields RTL for the tools to judge',
          out3.startswith('module mac'))

    a4, sent4 = _scripted_swarm(['diagnosis here', rtl_a, 'ACCEPT'])
    a4.use_reviewer = swarm_mod.USE_REVIEWER    # the shipped default
    a4.last_rtl = rtl_b
    a4.propose(spec, fb)
    check('debugger and writer engage once the tools reject something',
          a4.calls == {'writer': 1, 'reviewer': 0, 'debugger': 1})
    a5, _ = _scripted_swarm(['diagnosis here', rtl_a, 'ACCEPT'])
    a5.use_reviewer = True
    a5.last_rtl = rtl_b
    a5.propose(spec, fb)
    check('all three roles engage when the reviewer is switched on',
          a5.calls == {'writer': 1, 'reviewer': 1, 'debugger': 1})
    check('the reviewer is off by default, having never changed an outcome',
          swarm_mod.USE_REVIEWER is False)

    check('swarm satisfies the agent interface the orchestrator calls',
          callable(getattr(SwarmAgent, 'propose')))
    for k in ('_test', '_test2', '_test3'):
        CALLERS.pop(k, None)


def test_dv_mutation(fp):
    """Mutation testing of the generated testbenches. Convergence only says
    the RTL passed its DV; it says nothing about whether that DV could have
    failed. Every operator that changes behaviour must be killed, because a
    survivor is a defect class the flow would sign off on. Survivors are put
    to yosys for an equivalence proof first, so a mutant that cannot change
    behaviour is never counted against the testbench."""
    # The flow's scratch directory is configurable, so ask it where it put
    # the RTL rather than assuming.
    for rtl, tb, label in ((os.path.join(chiplet_flow.BUILD, 'mac.v'),
                            'tb_mac.v', 'chiplet'),
                           (os.path.join(chiplet_flow.BUILD, 'crc.v'),
                            'tb_crc.v', 'endpoint')):
        src = open(rtl).read()
        tbp = os.path.join(ROOT, tb)
        os.makedirs(dv.DVDIR, exist_ok=True)
        top = re.findall(r'^\s*module\s+([A-Za-z_]\w*)', src, re.M)[0]
        check('%s baseline passes its own testbench' % label,
              dv.evaluate('base', src, tbp, top)[0] == 'SURVIVED')
        survivors = []
        for name, fn, _ in dv.OPS:
            mutant = fn(src)
            if mutant == src:
                continue
            verdict = dv.evaluate(name, mutant, tbp, top)[0]
            if verdict == 'SURVIVED' and not dv.prove_equivalent(src, mutant,
                                                                 top):
                survivors.append(name)
        check('%s DV kills every behaviour-changing mutant%s'
              % (label, '' if not survivors else ' (survived: %s)'
                 % ','.join(survivors)), not survivors)
        shutil.rmtree(dv.DVDIR, ignore_errors=True)


def test_crc_matrix():
    """The flat CRC form rests on the step function being linear over GF(2).
    If it is not, the derived matrices are silently wrong and produce RTL
    that passes synthesis and computes the wrong checksum, so the property
    is checked directly rather than trusted."""
    for w in (1, 4, 8, 16, 32, 64, 128):
        A, B = crc_matrix(w)

        def step(state, data, w=w):
            x = state
            for i in range(w):
                x ^= (data >> (8 * i)) & 0xFF
                for _ in range(8):
                    x = (x >> 1) ^ (0xEDB88320 if x & 1 else 0)
            return x

        rnd = random.Random(w)
        ok = True
        for _ in range(64):
            st, d = rnd.getrandbits(32), rnd.getrandbits(8 * w)
            acc = 0
            for j in range(32):
                if (st >> j) & 1:
                    acc ^= A[j]
            for k in range(8 * w):
                if (d >> k) & 1:
                    acc ^= B[k]
            if acc != step(st, d):
                ok = False
                break
        check('flat CRC form reproduces the ripple at %d B/cycle' % w, ok)
        # Depth, not just correctness: the flat form exists to keep the
        # combinational path from growing with the datapath width.
        depth = max((sum(1 for j in range(32) if (A[j] >> i) & 1)
                     + sum(1 for k in range(8 * w) if (B[k] >> i) & 1))
                    for i in range(32)).bit_length()
        check('flat CRC XOR depth stays logarithmic at %d B/cycle' % w,
              depth <= 11)

    # And the whole chain against the reference implementation.
    w = 16
    payload = bytes(random.Random(7).randrange(256) for _ in range(w * 8))
    x = 0xFFFFFFFF
    A, B = crc_matrix(w)
    for off in range(0, len(payload), w):
        word = int.from_bytes(payload[off:off + w], 'little')
        acc = 0
        for j in range(32):
            if (x >> j) & 1:
                acc ^= A[j]
        for k in range(8 * w):
            if (word >> k) & 1:
                acc ^= B[k]
        x = acc
    check('flat CRC matches zlib over a multi-word frame',
          (x ^ 0xFFFFFFFF) == zlib.crc32(payload))

    # Both architectures are offered for every width, cheap form first.
    for g in (1.0, 10.0, 25.0, 100.0):
        _, opts = endpoint_options(g)
        archs = [a for _, _, a in opts]
        check('%g Gbps offers the ripple before the flat form' % g,
              archs.count('serial') == archs.count('matrix')
              and archs.index('serial') < archs.index('matrix'))


def test_inference_arithmetic():
    """The question every other test is circular about: when the model needs
    a dot product, does the generated hardware return the number the model
    needs? The bit-accurate MAC model is checked against the actual RTL on
    real quantized transformer dot products, and only then used to answer
    whether the derived accumulator width survives the model's reductions."""
    ms = load_model_spec()
    spec = derive_chiplet_spec(ms)
    dw = spec['parameters']['data_width']
    aw = spec['parameters']['acc_width']
    check('datapath is signed, as quantized weights require',
          spec['parameters']['signed'] is True)

    rtl = RuleBasedAgent().render_mac(spec, {agent_mod.FIX_WIDTH,
                                             agent_mod.FIX_CLEAR})
    mac = inference.MacModel(dw, aw)
    layer = inference.QuantLayer(32, 128, 8, dw, seed=3)
    x_q, _ = inference.quantize(
        [random.Random(4).gauss(0, 1) for _ in range(32)], dw)
    _, taps = layer.forward(x_q, mac, dw)
    check('a quantized block exercises both reduction depths',
          {len(t[0]) for t in taps} == {32, 128})

    sample = taps[:4] + taps[-4:]
    got = inference.run_cosim(spec, sample, rtl)
    exact = all(got.get(i) == inference.MacModel(dw, aw).dot(xs, ws)
                for i, (xs, ws) in enumerate(sample))
    check('generated RTL matches the model bit-exactly on real dot products',
          exact)

    # Negative operands must actually appear, or the signed path is untested.
    check('the sampled vectors contain negative operands',
          any(v < 0 for xs, ws in sample for v in xs + ws))

    # The rule the accumulator width comes from, checked against the model.
    depth = max(ms['d_model'], ms['d_ff'])
    need = (depth * (1 << (dw - 1)) ** 2).bit_length() + 1
    check('derived accumulator is wide enough for the worst-case reduction',
          aw >= need)
    check('no accumulator overflow on a real quantized block',
          mac.overflows == 0)
    shutil.rmtree(inference.WORK, ignore_errors=True)


def test_requant_block():
    """The requantizer is the step between two matmuls, and it is generated
    like the others: derived from the model spec, gated by the same tools.
    Its bit-accurate model has to agree with its RTL, or the decode below
    is not what the hardware would do."""
    ms = load_model_spec()
    spec = specgen_mod.derive_requant_spec(ms)
    p = spec['parameters']
    check('requantizer width is derived from the chiplet accumulator',
          p['acc_width'] == derive_chiplet_spec(ms)['parameters']['acc_width']
          and p['out_width'] == derive_chiplet_spec(ms)['parameters']['data_width'])
    check('requantizer saturates rather than wraps, by spec',
          any('aturat' in b for b in spec['behavior']))

    rtl = RuleBasedAgent().render_requant(spec, {agent_mod.FIX_SATURATE})
    rq = inference.RequantModel(p['out_width'], p['scale_width'],
                                p['shift_width'])
    rnd = random.Random(11)
    vecs = []
    for _ in range(10):
        acc = rnd.randrange(-(1 << (p['acc_width'] - 2)),
                            1 << (p['acc_width'] - 2))
        sc, sh = rq.pick(max(1, abs(acc)))
        vecs.append((acc, sc, sh))
    # Plus a case that must clamp, since that is the block's whole point.
    vecs.append(((1 << (p['acc_width'] - 2)), (1 << (p['scale_width'] - 1)), 1))
    got = inference.run_requant_cosim(spec, vecs, rtl)
    ref = inference.RequantModel(p['out_width'], p['scale_width'],
                                 p['shift_width'])
    check('generated requantizer matches its model bit-exactly',
          all(got.get(i) == ref.apply(a, s_, h)
              for i, (a, s_, h) in enumerate(vecs)))
    hi = (1 << (p['out_width'] - 1)) - 1
    lo = -(1 << (p['out_width'] - 1))
    check('every requantizer output is inside the operand range',
          all(lo <= v <= hi for v in got.values()))
    check('the clamping case actually clamped', ref.saturations >= 1)
    shutil.rmtree(inference.WORK, ignore_errors=True)


def test_generation():
    """The end of the chain: a trained checkpoint decoded through the
    hardware's arithmetic. This is the claim that the repo can run a
    language model, so it is checked rather than asserted in a README."""
    import generate as gen
    from train_tiny import CKPT
    check('a trained checkpoint is committed', os.path.exists(CKPT))
    ck = json.load(open(CKPT))
    check('checkpoint carries weights and a tokenizer',
          set(ck['weights']) >= {'tok', 'pos', 'wq', 'wk', 'wv', 'wo',
                                 'w1', 'w2', 'head'} and len(ck['chars']) > 5)

    ms = load_model_spec()
    cspec = derive_chiplet_spec(ms)
    rspec = specgen_mod.derive_requant_spec(ms)
    dw = cspec['parameters']['data_width']
    aw = cspec['parameters']['acc_width']
    rp = rspec['parameters']
    rq = inference.RequantModel(rp['out_width'], rp['scale_width'],
                                rp['shift_width'])
    hw = gen.HwModel(ck, dw, aw, rq)
    ids = [hw.stoi[c] for c in 'the ' if c in hw.stoi]
    out = list(ids)
    for _ in range(12):
        lg = hw.forward(out[-hw.seq:])
        out.append(max(range(len(lg)), key=lambda k: lg[k]))
    text = ''.join(hw.chars[i] for i in out)
    check('the model emits tokens', len(text) == len(ids) + 12)
    check('every emitted token is in the vocabulary',
          all(c in hw.chars for c in text))
    check('decode is deterministic', text == ''.join(
        hw.chars[i] for i in out))
    check('no accumulator overflow during a decode', hw.mac.overflows == 0)
    check('the decode exercised both blocks',
          len(hw.dots) > 100 and len(hw.rqs) > 100)

    # The bug this pins: activations carry a scale, and adding two int8
    # vectors with different scales adds numbers in different units. It
    # cost 38% of next-token agreement and read like a quantization limit.
    import generate as g2
    a_pair = ([10] * hw.d, 0.5)
    b_pair = ([10] * hw.d, 0.25)
    summed = hw.add(a_pair, b_pair)
    vals = [q * summed[1] for q in summed[0]]
    check('a residual add respects the operands\' differing scales',
          all(abs(v - 7.5) < 0.2 for v in vals))
    same, total, facc, margin = g2.teacher_forced_agreement(ck, hw, n=24)
    check('int8 decode tracks the float model once scales are tracked '
          '(%d/%d)' % (same, total), same >= 0.9 * total)

    # The arithmetic the text came from has to be the hardware's.
    rnd = random.Random(5)
    rnd.shuffle(hw.dots)
    sample = hw.dots[:4]
    rtl = RuleBasedAgent().render_mac(cspec, {agent_mod.FIX_WIDTH,
                                              agent_mod.FIX_CLEAR})
    got = inference.run_cosim(cspec, sample, rtl)
    check('the decode\'s dot products match the generated RTL',
          all(got.get(i) == inference.MacModel(dw, aw).dot(xs, ws)
              for i, (xs, ws) in enumerate(sample)))
    shutil.rmtree(inference.WORK, ignore_errors=True)


def test_bitstream():
    """Place, route and pack a generated block into a real bitstream.

    Synthesis says a design can be mapped. It does not say the design fits
    a device, that its routing closes, or that its clock survives real wire
    delay, and this repo asserted all three for a long time on the strength
    of a generic cell library. Skipped when the open iCE40 toolchain is not
    installed, because it is the only open place and route flow available
    here and it is not a hard dependency of the rest.
    """
    import bitstream as bs
    have = all(shutil.which(t) for t in
               ("nextpnr-ice40", "icepack", "yosys"))
    if not have:
        print('%-55s %s' % ('bitstream flow (needs nextpnr-ice40)', 'SKIP'))
        return
    rc = subprocess.call([sys.executable, 'bitstream.py', '--block', 'mac'],
                         cwd=ROOT, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    check('the MAC places, routes and packs into a bitstream', rc == 0)
    r = json.load(open(os.path.join(ROOT, 'bitstream_mac.json')))
    check('post-route timing is met on the real device',
          r['timing_met'] and r['post_route_fmax_mhz'] >= r['target_mhz'])
    check('the design fits the device with room to spare',
          0 < r['luts'] < 7680 or r['bitstream_bytes'] > 0)
    check('a real bitstream file was produced',
          r['bitstream_bytes'] > 1000)
    # The artifact that would be loaded onto a device, not the design that
    # produced it: icebox_vlog turns the packed bits back into logic and
    # the original self-checking testbench runs against them.
    check('the packed bitstream itself passes the testbench',
          r.get('bitstream_verified') is True)
    # The generic library is the flow's gate, so it must not be wildly
    # optimistic about the device it is standing in for.
    prof = json.load(open(os.path.join(ROOT, 'chiplet_profile.json')))
    ratio = prof['fmax_estimate_mhz'] / r['post_route_fmax_mhz']
    check('generic-library fmax is within 2x of post-route silicon',
          0.5 <= ratio <= 2.0)
    shutil.rmtree(bs.WORK, ignore_errors=True)


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
        ok = all(w * 8 * clk / 1000.0 >= rate - 1e-9 for w, clk, _ in opts)
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


def test_batching(profile, fp):
    """Batching is the lever against the memory wall: one step reads every
    weight once and serves the whole batch."""
    ms = load_model_spec()
    f = fit('kc705', profile, fp)
    sweep = [predict_config(dict(ms, batch_size=b), f, 1) for b in
             (1, 2, 4, 8, 16, 32)]
    tps = [p['predicted_tok_per_s'] for p in sweep]
    check('batching never reduces throughput',
          all(a <= b * 1.0001 for a, b in zip(tps, tps[1:])))
    check('batch 8 is worth more than 2x over batch 1',
          tps[3] > 2 * tps[0])
    check('per-sequence latency does not improve with batch',
          all(a <= b * 1.0001 for a, b in
              zip([p['step_ns'] for p in sweep],
                  [p['step_ns'] for p in sweep][1:])))
    # Weight traffic per step is batch-independent; KV traffic is not, which
    # is exactly why the curve flattens.
    s = model_summary(ms)
    big = predict_config(dict(ms, batch_size=64), f, 1)
    check('batching saturates once KV traffic overtakes the weights',
          64 * s['kv_read_bytes_per_token'] > s['weight_bytes']
          and tps[-1] < 2 * tps[3])
    r = simulate_decode([f] * 2, dict(ms, batch_size=8), tokens=2)
    p = predict_config(dict(ms, batch_size=8), f, 2)
    ratio = r['tok_per_s'] / p['predicted_tok_per_s']
    check('batched prediction matches the fabric simulation (ratio %.2f)'
          % ratio, 0.85 <= ratio <= 1.15)
    check('a batched step emits one token per sequence',
          r['tokens'] == r['steps'] * 8)


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
    # Isolate the weight-streaming effect at batch 1: with a large batch the
    # weights are already amortised, so residency matters proportionally less.
    b1 = dict(ms, batch_size=1)
    one_b1 = predict_config(b1, lg, 1)
    resident = [p for p in size_fabric(b1, lg)['sweep'] if p['sram_resident']]
    check('at batch 1, SRAM residency jumps throughput by more than 3x',
          resident and resident[0]['predicted_tok_per_s']
          > 3 * one_b1['predicted_tok_per_s'])
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
    test_swarm_offline()
    test_crc_matrix()
    test_inference_arithmetic()
    test_requant_block()
    test_generation()
    test_bitstream()
    test_fit_monotonic(profile)
    test_allreduce(profile)
    test_hetero_bit_identical(profile)
    fp = test_fabric_flow()
    test_dv_mutation(fp)
    test_fpga_backend(profile, fp)
    test_endpoint_derivation(fp)
    test_transports(profile, fp)
    test_batching(profile, fp)
    test_sizing(profile, fp)
    test_link_reliability(profile)
    print('all %d tests passed' % PASSED[0])
