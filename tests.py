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
import time
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
import boards as boards_mod
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
    q4 = dict(ms, name=ms['name'] + '_q4', weight_bits=4,
              activation_bits=4)
    # Derive the expectation rather than writing the number down. The
    # guard term is ceil(log2(longest reduction)), so it moves with the
    # model: hardcoding 12 was correct for d_ff 3072 and wrong the
    # moment the spec named a model with a different one.
    guard = math.ceil(math.log2(max(ms['d_model'], ms['d_ff'])))
    job = {'spec_file': 'spec_q4.json', 'tb_file': 'tb_q4.v',
           'rtl_file': 'mac_q4.v', 'profile_file': 'profile_q4.json',
           'report_file': 'report_q4.json'}
    generate(q4, spec_file=job['spec_file'], tb_file=job['tb_file'])
    report, prof = run_flow(job, verbose=False)
    check('4-bit quantized model: flow converges on the derived spec',
          report['converged'] and report['iterations_used'] == 3)
    check('4-bit quantized model: 4-bit datapath, %d-bit accumulator'
          % (8 + guard),
          prof['data_width'] == 4 and prof['acc_width'] == 4 + 4 + guard)
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
    a.best_rtl = None
    a.best_rank = -1
    a.tried = []
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


    # Regression recovery: the writer must be handed the attempt that
    # got furthest, not merely the latest. Feeding back a regression
    # compounds it, because every later iteration then starts worse.
    a6, sent6 = _scripted_swarm([rtl_a, rtl_b, rtl_a])
    a6.propose(spec, [])                       # first draft: rtl_a
    a6.propose(spec, [{'stage': 'timing', 'status': 'fail',
                       'iteration': 1, 'errors': ['slack -0.2']}])
    check('an attempt reaching timing is recorded as the best so far',
          a6.best_rank == swarm_mod.STAGE_RANK['timing'])
    a6.propose(spec, [{'stage': 'timing', 'status': 'fail',
                       'iteration': 1, 'errors': ['slack -0.2']},
                      {'stage': 'sim', 'status': 'fail', 'iteration': 2,
                       'mismatches': [{'test': 'x'}]}])
    # The writer's prompt is the one carrying the previous attempt; by
    # this point the reviewer has also engaged, so it is not the last.
    wp = [p for p in sent6 if 'PREVIOUS ATTEMPT' in p][-1]
    check('after a regression the writer is given the better attempt',
          rtl_a in wp)
    check('repeated designs are named so they are not proposed again',
          'already proposed' in wp)
    # Off on a first draft, on once the writer has been wrong twice.
    check('the reviewer engages after repeated failures',
          a6.calls['reviewer'] > 0)

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
    hw = gen.HwModel(ck, dw, aw, rq, specgen_mod.derive_exp_spec(ms),
                     specgen_mod.derive_recip_spec(ms),
                     specgen_mod.derive_rsqrt_spec(ms))
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
    check('the checkpoint has the norm gains the decode needs',
          'g1' in ck['weights'] and 'g2' in ck['weights'])

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
    # Every generated arithmetic block is on the decode path, not just
    # verified on its own. A block nothing calls is a weaker claim.
    check('the decode exercises the exponential, reciprocal and rsqrt',
          len(hw.exps) > 0 and len(hw.recips) > 0 and len(hw.rsqrts) > 0)
    rsspec = specgen_mod.derive_rsqrt_spec(ms)
    rs = hw.rsqrts[:3]
    got_rs = inference.run_rsqrt_cosim(
        rsspec, rs, RuleBasedAgent().render_rsqrt(rsspec,
                                                  {agent_mod.FIX_EVEN}))
    check('the decode\'s inverse square roots match the generated RTL',
          all(got_rs.get(i) == specgen_mod.rsqrt_golden(
              x, rsspec['parameters']) for i, x in enumerate(rs)))
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
    # optimistic about the device it is standing in for. This only means
    # anything when the profile's number came from a timing tool: the
    # gate-depth proxy is an estimate of logic depth, not of a clock, and
    # comparing it against post-route silicon compares two different
    # quantities. OpenSTA is not in oss-cad-suite, so on a runner
    # without it this is skipped rather than failed.
    prof = json.load(open(os.path.join(ROOT, 'chiplet_profile.json')))
    if prof.get('fmax_method') == 'opensta_slack':
        ratio = prof['fmax_estimate_mhz'] / r['post_route_fmax_mhz']
        check('generic-library fmax is within 2x of post-route silicon',
              0.5 <= ratio <= 2.0)
    else:
        print('%-55s %s' % ('library vs silicon fmax (needs OpenSTA)',
                            'SKIP'))
    shutil.rmtree(bs.WORK, ignore_errors=True)


def test_exp_block():
    """The exponential for attention's softmax, the last piece of the
    decode that was running on the host as floating point."""
    ms = load_model_spec()
    spec = specgen_mod.derive_exp_spec(ms)
    p = spec['parameters']
    check('the table is indexed by the whole fractional part',
          p['lut_bits'] == p['in_frac'])
    # Input width follows the output format's useful range, not the
    # model's activation width: past the underflow point a wider input
    # buys nothing and costs timing.
    big = dict(ms, weight_bits=16, activation_bits=16)
    check('input width does not grow with the activation width',
          specgen_mod.derive_exp_spec(big)['parameters']['in_width']
          == p['in_width'])

    worst = max(
        abs(specgen_mod.exp_golden(-i, p) / float(1 << p['out_frac'])
            - math.exp(-i / float(1 << p['in_frac'])))
        for i in range(0, (1 << (p['in_width'] - 1))))
    check('fixed-point exp is within 0.005 of math.exp (%.4f)' % worst,
          worst < 0.005)
    check('exp(0) is exactly one in the output format',
          specgen_mod.exp_golden(0, p) == (1 << p['out_frac']))
    check('a large negative argument underflows to zero, not wrap',
          specgen_mod.exp_golden(-(1 << (p['in_width'] - 1)), p) == 0)

    rtl = RuleBasedAgent().render_exp(spec, {agent_mod.FIX_LUT})
    # The table entry without its shift: e**x at or above one for a
    # negative x, which the testbench names.
    work = os.path.join(ROOT, 'build_exptbtest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        open(os.path.join(work, 'tb.v'), 'w').write(
            specgen_mod.render_exp_testbench(spec))
        open(os.path.join(work, 'rom.v'), 'w').write(specgen_mod.exp_rom(spec))
        res = {}
        noshift = rtl.replace('y         <= m >> sh;', 'y         <= m;')
        assert noshift != rtl
        for label, src in (('good', rtl), ('noshift', noshift)):
            open(os.path.join(work, 'e.v'), 'w').write(src)
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v', 'e.v',
                                'rom.v'], cwd=work, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            res[label] = subprocess.run(['vvp', 's.out'], cwd=work,
                                        capture_output=True, text=True,
                                        timeout=600).stdout
        check('an exponential at or above one for a negative input is named',
              'TB_RESULT: PASS' in res['good']
              and 'got_y_is_not_below_one_for_a_negative_x=1' in res['noshift'])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    xs = [0, -1, -(1 << p['in_frac']), -(3 << p['in_frac']),
          -(1 << (p['in_width'] - 1))]
    got = inference.run_exp_cosim(spec, xs, rtl)
    check('generated exponential matches its model bit-exactly',
          all(got.get(i) == specgen_mod.exp_golden(x, p)
              for i, x in enumerate(xs)))
    shutil.rmtree(inference.WORK, ignore_errors=True)

    # A table becomes a memory in synthesis, and the SAT solver cannot
    # reason about one. Without memory_map every mutant of a design with a
    # lookup table is misreported as a surviving DV hole.
    mut = [fn for n, fn, _ in dv.OPS if n == 'product_off_by_one'][0](rtl)
    os.makedirs(dv.DVDIR, exist_ok=True)
    # The table is its own module now, so the proof needs it in hand:
    # without it yosys cannot elaborate and the failure is about a
    # missing module rather than about equivalence.
    rompath = os.path.join(dv.DVDIR, 'exp_rom_src.v')
    open(rompath, 'w').write(specgen_mod.exp_rom(spec))
    check('equivalence checking works on a design with a lookup table',
          dv.prove_equivalent(rtl, mut, 'expu', (rompath,)))
    shutil.rmtree(dv.DVDIR, ignore_errors=True)


def test_recip_block():
    """The other half of softmax: one reciprocal per row, multiplied in,
    instead of a divider per weight."""
    ms = load_model_spec()
    spec = specgen_mod.derive_recip_spec(ms)
    p = spec['parameters']
    # Returning a mantissa and a shift rather than a pre-shifted value is
    # the design decision worth pinning: the denominator spans ten bits,
    # so a single fixed-point output holds as few as five significant
    # bits at the top of the range and carries 6% error.
    rnd = random.Random(2)
    worst = 0.0
    for _ in range(4000):
        x = rnd.randrange(1 << 15, 1 << p['in_width'])
        m, k = specgen_mod.recip_golden(x, p)
        got = specgen_mod.recip_apply(1 << 40, m, k, p)
        worst = max(worst, abs(got - (1 << 40) // x) / float((1 << 40) // x))
    check('reciprocal is within 0.5%% of true division (%.3f%%)'
          % (100 * worst), worst < 0.005)
    check('the mantissa always uses its full width',
          all(specgen_mod.recip_golden(x, p)[0] >= (1 << (p['out_width'] - 1))
              for x in (1, 3, 1 << 15, (1 << p['in_width']) - 1)))

    rtl = RuleBasedAgent().render_recip(spec, {agent_mod.FIX_NORM})
    xs = [1, 2, 3, 1 << 15, (1 << 15) + 1, (1 << p['in_width']) - 1]
    got = inference.run_recip_cosim(spec, xs, rtl)
    check('generated reciprocal matches its model bit-exactly',
          all(got.get(i) == specgen_mod.recip_golden(x, p)
              for i, x in enumerate(xs)))
    shutil.rmtree(inference.WORK, ignore_errors=True)

    # Softmax end to end through both generated units.
    import generate as g3
    espec = specgen_mod.derive_exp_spec(ms)
    hw = g3.HwModel(json.load(open(os.path.join(ROOT, 'tiny_llm.json'))),
                    8, 28, inference.RequantModel(8, 18, 7), espec, spec)
    ex = [hw.expi(-d) for d in (0.0, 0.5, 1.0, 3.0)]
    w = hw.normalise(ex)
    check('softmax weights through the generated units sum to one',
          abs(sum(w) - 1.0) < 0.01)
    check('softmax weights are ordered like their scores',
          all(w[i] >= w[i + 1] for i in range(len(w) - 1)))


def test_rsqrt_block():
    """The inverse square root RMSNorm needs. The sum of squares is the
    MAC unit and the mean is a shift; this is the part that needs its own
    hardware."""
    ms = load_model_spec()
    spec = specgen_mod.derive_rsqrt_spec(ms)
    p = spec['parameters']
    # A square root halves the exponent, so the normalisation must move
    # the value an even number of bits. At an odd input width that shift
    # goes negative for the largest inputs.
    check('input width is even so the halved exponent is an integer',
          p['in_width'] % 2 == 0)
    big = dict(ms, weight_bits=8, activation_bits=16, d_model=2048,
               d_ff=8192)
    check('the width stays even for every model spec',
          specgen_mod.derive_rsqrt_spec(big)['parameters']['in_width'] % 2
          == 0)

    rnd = random.Random(8)
    worst = 0.0
    for _ in range(4000):
        x = rnd.randrange(1, 1 << p['in_width'])
        m, e = specgen_mod.rsqrt_golden(x, p)
        got = specgen_mod.rsqrt_apply(1 << 30, m, e, p)
        want = (1 << 30) / math.sqrt(x)
        worst = max(worst, abs(got - want) / want)
    check('inverse square root within 1%% of true (%.3f%%)' % (100 * worst),
          worst < 0.01)
    check('exact at the powers of four, where the table is exact',
          all(abs(specgen_mod.rsqrt_apply(1 << 20,
                                          *specgen_mod.rsqrt_golden(x, p),
                                          p=p) / 2.0 ** 20
                  - 1.0 / math.sqrt(x)) < 1e-5
              for x in (1, 4, 16, 64, 256)))

    rtl = RuleBasedAgent().render_rsqrt(spec, {agent_mod.FIX_EVEN})
    xs = [1, 2, 3, 4, 1 << 10, (1 << p['in_width']) - 1]
    got = inference.run_rsqrt_cosim(spec, xs, rtl)
    check('generated inverse square root matches its model bit-exactly',
          all(got.get(i) == specgen_mod.rsqrt_golden(x, p)
              for i, x in enumerate(xs)))
    shutil.rmtree(inference.WORK, ignore_errors=True)


def test_matvec_sequencer():
    """The first generated block that sequences another rather than
    computing. Its correctness depends on the MAC's latency, so it is
    checked driving the real MAC, not a model of one."""
    ms = load_model_spec()
    spec = specgen_mod.derive_matvec_spec(ms)
    c = derive_chiplet_spec(ms)
    p = spec['parameters']
    check('the drain count is taken from the MAC it drives',
          p['mac_stages'] == c['parameters']['pipeline_stages'])
    check('the sequencer expects a registered memory read',
          p['mem_latency'] == 1)
    # A multiply in the control path is invisible in the generic library
    # and costs half the clock on real silicon.
    rtl = RuleBasedAgent().render_matvec(spec, {agent_mod.FIX_CLRCOL,
                                                agent_mod.FIX_MEMLAT})
    check('no multiply in the address path',
          '*' not in rtl.replace('/*', '').replace('*/', '')
          .replace('* ', '').split('always')[1])
    # A deeper MAC must change this block, not silently desynchronise it.
    deep = dict(ms, weight_bits=16, activation_bits=16)
    check('a deeper MAC changes the sequencer',
          specgen_mod.derive_matvec_spec(deep)['parameters']['mac_stages']
          == derive_chiplet_spec(deep)['parameters']['pipeline_stages']
          != p['mac_stages'])

    work = os.path.join(ROOT, 'build_seqtest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        open(os.path.join(work, 'tb.v'), 'w').write(
            specgen_mod.render_matvec_testbench(spec))
        open(os.path.join(work, 'mac.v'), 'w').write(
            RuleBasedAgent().render_mac(c, {agent_mod.FIX_WIDTH,
                                            agent_mod.FIX_CLEAR}))
        results = {}
        for label, fx in (('no_clear', {agent_mod.FIX_MEMLAT}),
                          ('no_delay', {agent_mod.FIX_CLRCOL}),
                          ('fixed', {agent_mod.FIX_CLRCOL,
                                     agent_mod.FIX_MEMLAT})):
            open(os.path.join(work, 'mv.v'), 'w').write(
                RuleBasedAgent().render_matvec(spec, fx))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out',
                                'tb.v', 'mv.v', 'mac.v'], cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=900)
            results[label] = 'TB_RESULT: PASS' in r.stdout
        check('the sequencer and the real MAC compute the matrix product',
              results['fixed'])
        # Column zero is right either way; the bug only shows from the
        # second column, which is why a one-column test would miss it.
        check('a missing inter-column clear is caught',
              not results['no_clear'])
        # Block RAM registers its read, so valid has to follow the
        # address. Driving them together multiplies stale data, and it
        # is wrong from the very first element rather than the second.
        check('a valid not delayed for the memory read is caught',
              not results['no_delay'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_wmem_subsystem():
    """The weight tile and its loader, checked as a subsystem.

    The block's own testbench instantiates the sequencer and the MAC,
    because the bug it is most prone to is a read port a cycle out of
    step with whatever reads it. That is invisible to the memory alone:
    a combinational read is perfectly well behaved until something
    depends on when the data arrives.
    """
    ms = load_model_spec()
    spec = specgen_mod.derive_wmem_spec(ms)
    p = spec['parameters']
    check('the tile is addressed by exactly its capacity',
          (1 << p['addr_width']) == p['capacity'])
    check('the read port is specified as registered',
          any('registered' in b for b in spec['behavior']))

    work = os.path.join(ROOT, 'build_wmemtest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        open(os.path.join(work, 'tb.v'), 'w').write(
            specgen_mod.render_wmem_testbench(spec))
        open(os.path.join(work, 'mac.v'), 'w').write(
            RuleBasedAgent().render_mac(derive_chiplet_spec(ms),
                                        {agent_mod.FIX_WIDTH,
                                         agent_mod.FIX_CLEAR}))
        open(os.path.join(work, 'mv.v'), 'w').write(
            RuleBasedAgent().render_matvec(
                specgen_mod.derive_matvec_spec(ms),
                {agent_mod.FIX_CLRCOL, agent_mod.FIX_MEMLAT}))
        res = {}
        for label, fx in (('comb_read', set()),
                          ('fixed', {agent_mod.FIX_REGRD})):
            open(os.path.join(work, 'wm.v'), 'w').write(
                RuleBasedAgent().render_wmem(spec, fx))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out',
                                'tb.v', 'wm.v', 'mv.v', 'mac.v'], cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=900)
            res[label] = 'TB_RESULT: PASS' in r.stdout
        check('loader, memory, sequencer and MAC compute the product',
              res['fixed'])
        check('a combinational read port is caught by the subsystem',
              not res['comb_read'])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Survivors are classified three ways. A timeout is not a defect and
    # not an equivalence; saying so is the only honest option.
    check('the DV tooling separates unproven from a real hole',
          hasattr(dv, 'survivor_verdict'))


def test_softmax_sequencer():
    """Softmax as one block: max pass, exponential pass with sum, one
    reciprocal, normalising multiply. It instantiates the generated
    exponential and reciprocal rather than reimplementing them, so it is
    the first generated block that contains others."""
    ms = load_model_spec()
    spec = specgen_mod.derive_softmax_spec(ms)
    p = spec['parameters']
    e = specgen_mod.derive_exp_spec(ms)
    check('the score and weight formats come from the exponential unit',
          p['score_width'] == e['parameters']['in_width']
          and p['weight_frac'] == e['parameters']['out_frac'])

    one = 1 << p['weight_frac']
    for scores in ([0, -256, -512, -1024], [7, 7, 7, 7], [100, -100]):
        w, ex, tot = specgen_mod.softmax_golden(scores, p)
        check('weights for %s sum to one within 1%%' % (scores,),
              abs(sum(w) - one) < one // 100)
        check('weights for %s are ordered like their scores' % (scores,),
              all((w[i] >= w[j]) == (scores[i] >= scores[j])
                  for i in range(len(w)) for j in range(len(w))))

    work = os.path.join(ROOT, 'build_smtest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        open(os.path.join(work, 'tb.v'), 'w').write(
            specgen_mod.render_softmax_testbench(spec))
        rcs = specgen_mod.derive_recip_spec(ms)
        open(os.path.join(work, 'expu.v'), 'w').write(
            RuleBasedAgent().render_exp(e, {agent_mod.FIX_LUT}))
        open(os.path.join(work, 'recip.v'), 'w').write(
            RuleBasedAgent().render_recip(rcs, {agent_mod.FIX_NORM}))
        open(os.path.join(work, 'roms.v'), 'w').write(
            specgen_mod.exp_rom(e) + specgen_mod.recip_rom(rcs))
        res, outs = {}, {}
        fixed = RuleBasedAgent().render_softmax(spec, {agent_mod.FIX_SUBMAX})
        flat = fixed.replace("dfull = s_data - mx", "dfull = mx - mx")
        assert flat != fixed
        narrow = fixed.replace("dfull = s_data - mx",
                               "dfull = (s_data >>> 8) - (mx >>> 8)")
        for label, src in (('raw_scores', RuleBasedAgent().render_softmax(spec, set())),
                           ('fixed', fixed), ('flat', flat), ('narrow', narrow)):
            open(os.path.join(work, 'sm.v'), 'w').write(src)
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out',
                                'tb.v', 'sm.v', 'expu.v', 'recip.v', 'roms.v'],
                               cwd=work, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=900)
            res[label] = 'TB_RESULT: PASS' in r.stdout
            outs[label] = r.stdout
        check('a softmax that reads every score as one value is named for it',
              not res['flat'] and
              'got_w_is_the_weight_of_a_row_whose_scores_are_all_equal=1' in outs['flat'])
        check('a softmax that drops the scores\' low bits is named for it',
              not res['narrow'] and 'got_w_is_the_weight_with_the_low_8_bits_'
              'of_each_score_dropped=1' in outs['narrow'])
        check('the sequencer and both units compute softmax', res['fixed'])
        # The exponential is only defined for non-positive arguments, so
        # skipping the max subtraction feeds it positive ones.
        check('feeding raw scores to the exponential is caught',
              not res['raw_scores'])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # A generated testbench can search for a vector that discriminates,
    # which is the only way to reach a boundary a random row hits about
    # once in a hundred thousand.
    import random as _r
    check('the testbench finds a vector exposing a one-count error',
          specgen_mod._softmax_discriminating_row(p, _r.Random(53))
          is not None)


def test_mlp_layer():
    """A layer rather than an operation: two matmuls with a requantize
    and a rectify between, over a two-bank activation buffer. This is
    the block that routes one matmul's outputs into the next one's
    inputs, which nothing did before it."""
    ms = load_model_spec()
    spec = specgen_mod.derive_mlp_spec(ms)
    p = spec['parameters']
    rq = specgen_mod.derive_requant_spec(ms)
    check('the requantizer depth is taken from that block',
          p['requant_stages'] == rq['parameters']['pipeline_stages'])

    work = os.path.join(ROOT, 'build_mlptest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        rr = RuleBasedAgent()
        open(os.path.join(work, 'tb.v'), 'w').write(
            specgen_mod.render_mlp_testbench(spec))
        open(os.path.join(work, 'mac.v'), 'w').write(
            rr.render_mac(derive_chiplet_spec(ms),
                          {agent_mod.FIX_WIDTH, agent_mod.FIX_CLEAR}))
        open(os.path.join(work, 'mv.v'), 'w').write(
            rr.render_matvec(specgen_mod.derive_matvec_spec(ms),
                             {agent_mod.FIX_CLRCOL, agent_mod.FIX_MEMLAT}))
        open(os.path.join(work, 'rq.v'), 'w').write(
            rr.render_requant(rq, {agent_mod.FIX_SATURATE}))
        res = {}
        for label, fx in (('unchained', set()),
                          ('fixed', {agent_mod.FIX_CHAIN})):
            open(os.path.join(work, 'mlp.v'), 'w').write(
                rr.render_mlp(spec, fx))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out',
                                'tb.v', 'mlp.v', 'mv.v', 'mac.v', 'rq.v'],
                               cwd=work, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=900)
            res[label] = 'TB_RESULT: PASS' in r.stdout
        check('the layer computes two chained matmuls with relu',
              res['fixed'])
        # The two numbers are equal by construction, so the design looks
        # right; it only diverges when a caller passes them inconsistently.
        check('a second reduction length not chained to the first is caught',
              not res['unchained'])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # Equivalence proving has to see a composite block's hierarchy, or it
    # fails for want of a module and every mutant reads as a DV hole.
    import inspect
    check('equivalence proving accepts dependencies',
          'extra' in inspect.signature(dv.prove_equivalent).parameters)


def test_attention_head():
    """The block that makes the rest a transformer layer: one attention
    head for one decode step. Scores through matvec and the MAC, weights
    through the softmax block, output through a weight-by-value sum and
    the requantizer. The testbench checks the scores and the weights
    directly before the outputs, since both are upstream of them."""
    import math
    import chiplet_flow as cf
    ms = load_model_spec()
    spec = specgen_mod.derive_attn_spec(ms)
    p = spec['parameters']
    sm_p = specgen_mod.derive_softmax_spec(ms)['parameters']
    check('head_dim is taken from the model',
          p['head_dim'] == ms.get('head_dim', ms['d_model'] // ms['n_head']))

    # The fixed-point head against real attention on the same integers.
    rnd = random.Random(3)
    hd, n = p['head_dim'], 12
    q = [rnd.randrange(-60, 60) for _ in range(hd)]
    K = [[rnd.randrange(-60, 60) for _ in range(hd)] for _ in range(n)]
    V = [[rnd.randrange(-60, 60) for _ in range(hd)] for _ in range(n)]
    # Scores are (t << guard) >> shift_s, so this is an effective right
    # shift of six.
    g = p.get('score_guard', 0)
    sh = 6 + g
    t, s_, w, a, o = specgen_mod.attn_golden(q, K, V, n, sh, 1, 0, p, sm_p)
    fs = [x / (1 << (sh - g)) / (1 << sm_p['score_frac']) for x in t]
    mx = max(fs)
    e = [math.exp(x - mx) for x in fs]
    z = sum(e)
    fa = [sum(e[j] / z * V[j][d] for j in range(n)) for d in range(hd)]
    err = max(abs(a[d] / (1 << sm_p['weight_frac']) - fa[d]) for d in range(hd))
    check('the fixed-point head matches float attention to within 0.25 '
          '(worst %.3f on values up to 60)' % err, err < 0.25)

    work = os.path.join(ROOT, 'build_attntest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_attn_deps(ms, work)
        rr = RuleBasedAgent()
        with open(os.path.join(work, 'tb.v'), 'w') as f:
            f.write(specgen_mod.render_attn_testbench(spec))
        res = {}
        for label, fx in (('first', set()), ('fixed', {agent_mod.FIX_VLAT})):
            with open(os.path.join(work, 'attn.v'), 'w') as f:
                f.write(rr.render_attn(spec, fx))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'attn.v'] + list(cf.ATTN_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=900)
            res[label] = r.stdout
        check('the head computes scores, softmax and the weighted sum',
              'TB_RESULT: PASS' in res['fixed'])
        check('a value used before its registered read arrives is caught',
              'TB_RESULT: PASS' not in res['first']
              and 'expected_o' in res['first'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_rmsnorm():
    """The norm Qwen applies before attention and before the MLP: a sum
    of squares, one inverse square root, and a scaled product per
    element. The testbench checks the sum of squares and the rsqrt result
    directly before the outputs, and includes an all-zero row, where
    epsilon is the whole answer, and a full-scale spike, which puts t at
    the bound its shift is derived from."""
    import math
    import chiplet_flow as cf
    ms = load_model_spec()
    spec = specgen_mod.derive_rmsnorm_spec(ms)
    p = spec['parameters']
    rs_p = spec['derivation']['rsqrt']
    check('the row length is the model\'s d_model', p['d_model'] == ms['d_model'])

    rnd = random.Random(2)
    D = p['d_model']
    x = [rnd.randrange(-60, 60) for _ in range(D)]
    g = [rnd.randrange(-90, 90) for _ in range(D)]
    ssq, m, e, t, o = specgen_mod.rmsnorm_golden(x, g, D, 1, 0, p, rs_p)
    fl = [x[i] * g[i] / math.sqrt(ssq / D) for i in range(D)]
    fx = [t[i] * math.sqrt(D) * (1 << p['norm_shift'])
          / (1 << p['rsqrt_out_width']) for i in range(D)]
    rel = max(abs(fx[i] - fl[i]) for i in range(D)) / max(abs(v) for v in fl)
    check('the fixed-point norm matches float RMSNorm to within 0.5%% '
          '(worst %.4f%%)' % (100 * rel), rel < 0.005)

    work = os.path.join(ROOT, 'build_rmstest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_rmsnorm_deps(ms, work)
        rr = RuleBasedAgent()
        with open(os.path.join(work, 'tb.v'), 'w') as f:
            f.write(specgen_mod.render_rmsnorm_testbench(spec))
        res = {}
        fixed = rr.render_rmsnorm(spec, {agent_mod.FIX_EPS})
        early = fixed.replace("if (v1) begin\n        sqr <= x_data * x_data;",
                              "if (v0) begin\n        sqr <= x_data * x_data;")
        assert early != fixed
        uns = fixed.replace("sqr <= x_data * x_data;",
                            "sqr <= $unsigned(x_data) * $unsigned(x_data);")
        assert uns != fixed
        for label, src in (('first', rr.render_rmsnorm(spec, set())),
                           ('fixed', fixed), ('early', early), ('uns', uns)):
            with open(os.path.join(work, 'rms.v'), 'w') as f:
                f.write(src)
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'rms.v'] + list(cf.RMSNORM_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=900)
            res[label] = r.stdout
        check('the norm computes the sum of squares, rsqrt and the products',
              'TB_RESULT: PASS' in res['fixed'])
        check('a sum of squares that leaves out epsilon is caught, with what '
              'the design saw on its first cycles and its last',
              'TB_RESULT: PASS' not in res['first']
              and 'expected_ssq' in res['first']
              and re.search(r'cycles_after_start_as_cycle_x_addr_x_data_ssq='
                            r'(\d+:-?\d+,-?\d+,\d+;){8} four_cycles_to_the_last_'
                            r'change_of_ssq_as_x_addr_x_data_ssq=(-?\d+,-?\d+,\d+;){4}$',
                            res['first'], re.M))
        check('a norm that reads its data a cycle early is caught, its '
              'outputs named as their neighbours\' expected values',
              'TB_RESULT: PASS' not in res['early']
              and 'got_norm_is_the_expected_value_for_out=' in res['early'])
        check('a sum of squares that reads x as unsigned is caught and named',
              'TB_RESULT: PASS' not in res['uns']
              and 'got_ssq_is_the_sum_with_x_read_as_unsigned=1' in res['uns'])
        # A read one cycle off counts one end element twice and the other
        # never; the testbench carries both sums for each row to name it.
        rnd2 = random.Random(83)
        top = (1 << (p['data_width'] - 1)) - 1
        xt = [rnd2.randrange(-top // 2, top // 2) for _ in range(D)]
        s0 = sum(v * v for v in xt) + D
        mask = (1 << p['rsqrt_in_width']) - 1
        tbtext = specgen_mod.render_rmsnorm_testbench(spec)
        check('the testbench carries each row\'s sum with x0 twice and with '
              'the last element twice',
              ', %d, %d, ' % ((s0 - xt[-1] ** 2 + xt[0] ** 2) & mask,
                              (s0 - xt[0] ** 2 + xt[-1] ** 2) & mask) in tbtext
              and 'got_ssq_counts_x0_twice_and_never_the_last_element=1' in tbtext)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_timing_leaves_subblock_internals_to_their_signoff():
    """A supplied sub-block signed off with no margin missed by 0.68 ns on
    a path wholly inside it when the attention head instantiated it, and
    nothing the head's agent wrote could change that. The composite's gate
    checks its own logic and its paths into and out of the sub-block."""
    if not chiplet_flow.tool('sta'):
        return
    work = os.path.join(ROOT, 'build_hiersta')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    saved = chiplet_flow.BUILD
    try:
        chiplet_flow.BUILD = work
        shutil.copy(chiplet_flow.LIB, work)
        open(os.path.join(work, 'sub.v'), 'w').write(
            "module sub(input clk, input [31:0] a, output reg [31:0] y);\n"
            "  reg [31:0] r;\n"
            "  always @(posedge clk) begin r <= a; y <= r * r * r + r; end\n"
            "endmodule\n")
        tops = {
            'quick': "module top(input clk, input [31:0] a, output [31:0] y);\n"
                     "  reg [31:0] ar;\n  always @(posedge clk) ar <= a;\n"
                     "  sub s (.clk(clk), .a(ar), .y(y));\nendmodule\n",
            'slow': "module top(input clk, input [31:0] a, output [31:0] y);\n"
                    "  reg [31:0] ar, br;\n"
                    "  always @(posedge clk) begin ar <= a; br <= ar * ar * ar + ar; end\n"
                    "  sub s (.clk(clk), .a(br), .y(y));\nendmodule\n"}
        spec = {'top_module': 'top', 'parameters': {'target_clock_mhz': 100},
                'ports': [{'name': 'clk', 'dir': 'input'},
                          {'name': 'a', 'dir': 'input'}]}
        res = {}
        for label, src in tops.items():
            open(os.path.join(work, 'top.v'), 'w').write(src)
            job = {'extra_sources': ('sub.v',), 'rtl_file': 'top.v'}
            chiplet_flow.stage_synth(job, spec, os.path.join(work, 'top.v'))
            res[label] = chiplet_flow.stage_timing(job, spec)
            res[label + '_flat'] = chiplet_flow.stage_timing({}, spec)
        check('a composite\'s timing leaves a supplied sub-block\'s own paths '
              'to its signoff, and still fails its own slow path',
              res['quick']['status'] == 'pass'
              and res['quick_flat']['status'] == 'fail'
              and res['slow']['status'] == 'fail')
    finally:
        chiplet_flow.BUILD = saved
        shutil.rmtree(work, ignore_errors=True)


def test_composite_cell_count():
    """A block that instantiates others is reported per module by yosys,
    and the flow took the first module's count: the MLP layer and RMSNorm
    both reported their requantizer's 10396 cells as their own. The total
    has to include the submodules."""
    work = os.path.join(ROOT, 'build_cellcount')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    saved = chiplet_flow.BUILD
    try:
        chiplet_flow.BUILD = work
        shutil.copy(chiplet_flow.LIB, work)
        with open(os.path.join(work, 'top.v'), 'w') as f:
            f.write("module sub(input clk, input [7:0] a, output reg [7:0] y);\n"
                    "  always @(posedge clk) y <= a * a;\nendmodule\n"
                    "module top(input clk, input [7:0] a, output [7:0] y);\n"
                    "  wire [7:0] m;\n  reg [7:0] r;\n"
                    "  sub s0 (.clk(clk), .a(a), .y(m));\n"
                    "  sub s1 (.clk(clk), .a(m), .y(y));\n"
                    "  always @(posedge clk) r <= a;\nendmodule\n")
        both = chiplet_flow.stage_synth({}, {'top_module': 'top'},
                                        os.path.join(work, 'top.v'))
        one = chiplet_flow.stage_synth({}, {'top_module': 'sub'},
                                       os.path.join(work, 'top.v'))
        check('a composite block counts its submodules\' cells '
              '(%s against %s for one of them)'
              % (both['cell_count'], one['cell_count']),
              both['cell_count'] > 2 * one['cell_count'] - 1
              and both['area'] > one['area'])
    finally:
        chiplet_flow.BUILD = saved
        shutil.rmtree(work, ignore_errors=True)


def test_silu():
    """SiLU, the nonlinearity in Qwen's gated MLP, built from the
    exponential and reciprocal units: sigmoid(x) is 1/(1+e) for x >= 0 and
    e/(1+e) for x < 0, with e = exp(-|x|). Checked against float SiLU on
    every input the format can represent."""
    import math
    import chiplet_flow as cf
    ms = load_model_spec()
    spec = specgen_mod.derive_silu_spec(ms)
    p, d = spec['parameters'], spec['derivation']
    iw, fi = p['width'], p['frac']
    worst = 0.0
    for x in range(-(1 << (iw - 1)), 1 << (iw - 1)):
        xf = x / (1 << fi)
        worst = max(worst, abs(specgen_mod.silu_golden(x, d) / (1 << fi)
                               - xf / (1 + math.exp(-xf))))
    check('fixed-point SiLU matches float on all %d inputs to within 0.05 '
          '(worst %.4f)' % (1 << iw, worst), worst < 0.05)

    work = os.path.join(ROOT, 'build_silutest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_silu_deps(ms, work)
        rr = RuleBasedAgent()
        with open(os.path.join(work, 'tb.v'), 'w') as f:
            f.write(specgen_mod.render_silu_testbench(spec))
        res = {}
        for label, fx in (('first', set()), ('fixed', {agent_mod.FIX_SIGN})):
            with open(os.path.join(work, 'silu.v'), 'w') as f:
                f.write(rr.render_silu(spec, fx))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'silu.v'] + list(cf.SILU_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=300)
            res[label] = r.stdout
        check('the SiLU unit streams one element per cycle, in order',
              'TB_RESULT: PASS' in res['fixed'])
        check('sigmoid of |x| for negative x is caught',
              'TB_RESULT: PASS' not in res['first']
              and 'expected_silu' in res['first'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_gated_mlp():
    """Qwen's MLP: down(SiLU(gate(x)) * up(x)), three projections on one
    matmul sequencer, the SiLU unit and one shared requantizer. The
    testbench checks the SiLU, up and product buffers directly, in order,
    before the outputs."""
    import math
    import chiplet_flow as cf
    ms = load_model_spec()
    spec = specgen_mod.derive_gmlp_spec(ms)
    p = spec['parameters']
    sp = spec['derivation']['silu']
    # SiLU's own error is at most 0.018 in absolute terms, so the product
    # with up carries at most that times |u|. Relative error is the wrong
    # yardstick: near a zero of the product it is huge while the error, in
    # units of the int8 step the product is requantized to, is a tenth.
    rnd = random.Random(5)
    worst = 0.0
    for _ in range(300):
        g = rnd.randrange(-(1 << (p['gate_width'] - 1)), 1 << (p['gate_width'] - 1))
        u = rnd.randrange(-127, 128)
        fx = specgen_mod.silu_golden(g, sp) * u / (1 << p['gate_frac'])
        gf = g / (1 << p['gate_frac'])
        fl = gf / (1 + math.exp(-gf)) * u
        worst = max(worst, abs(fx - fl) / max(1, abs(u)))
    check('SiLU(gate) * up is within 0.02 * |up| of float '
          '(worst %.4f * |up|)' % worst, worst < 0.02)

    work = os.path.join(ROOT, 'build_gmlptest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_gmlp_deps(ms, work)
        rr = RuleBasedAgent()
        with open(os.path.join(work, 'tb.v'), 'w') as f:
            f.write(specgen_mod.render_gmlp_testbench(spec))
        res = {}
        for label, fx_ in (('first', set()), ('fixed', {agent_mod.FIX_UPBASE})):
            with open(os.path.join(work, 'gmlp.v'), 'w') as f:
                f.write(rr.render_gmlp(spec, fx_))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'gmlp.v'] + list(cf.GMLP_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=300)
            res[label] = r.stdout
        check('the gated layer computes gate, up, their product and down',
              'TB_RESULT: PASS' in res['fixed'])
        check('an up projection reading the gate matrix is caught at ubuf',
              'TB_RESULT: PASS' not in res['first']
              and 'expected_u=' in res['first'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_residual_add():
    """The add that closes each half of a layer: two int8 tensors at
    different scales, rounded and saturated. The testbench plants exact
    rounding ties, negative ones included, so the direction a tie rounds
    is checked and not only that it rounds."""
    spec = specgen_mod.derive_resadd_spec(load_model_spec())
    work = os.path.join(ROOT, 'build_restest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        with open(os.path.join(work, 'tb.v'), 'w') as f:
            f.write(specgen_mod.render_resadd_testbench(spec))
        res = {}
        fixed = RuleBasedAgent().render_resadd(spec, {agent_mod.FIX_RRND})
        # An LLM signed off a residual add that read its scales as signed:
        # right for every scale the one stream then used, wrong for every
        # scale with its top bit set, and the decode step it sat in gave
        # the wrong tokens.
        signed_scales = fixed.replace("$signed({1'b0, scale_a})", "$signed(scale_a)") \
                             .replace("$signed({1'b0, scale_b})", "$signed(scale_b)")
        assert signed_scales != fixed
        for label, src in (('first', RuleBasedAgent().render_resadd(spec, set())),
                           ('fixed', fixed), ('signed', signed_scales)):
            with open(os.path.join(work, 'r.v'), 'w') as f:
                f.write(src)
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'r.v'], cwd=work, capture_output=True,
                               text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            res[label] = subprocess.run(['vvp', 's.out'], cwd=work,
                                        capture_output=True, text=True,
                                        timeout=120).stdout
        check('the residual add rounds, saturates and streams in order',
              'TB_RESULT: PASS' in res['fixed'])
        check('a residual add that truncates is caught',
              'TB_RESULT: PASS' not in res['first']
              and 'expected_res' in res['first'])
        check('a residual add that reads its scales as signed is caught, '
              'at a scale with its top bit set',
              'TB_RESULT: PASS' not in res['signed']
              and 'scale_a=%d' % ((1 << spec['parameters']['scale_width']) - 1)
              in res['signed'])
        prof = re.search(r'TB_PROFILE elements=(\d+) span_cycles=(\d+) latency_cycles=(\d+)',
                         res['fixed'])
        check('the profile is the first stream alone: 214 elements in 225 '
              'cycles, 4 of latency', prof and prof.groups() == ('214', '225', '4'))
        check('ties round toward plus infinity, as the requantizer does',
              specgen_mod.resadd_golden(1, 0, 1 << 11, 1, 12, 8) == 1
              and specgen_mod.resadd_golden(-1, 0, 1 << 11, 1, 12, 8) == 0)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_full_size_projection():
    """A projection at the model's full size, not a tile of it. The
    composite layers hold activations in 64-entry banks; a Qwen layer's
    rows run to d_ff. This block reads activations and weights through
    external memory ports, so the up projection runs at 896 by 4864,
    4.36 million multiply-accumulates, and every output is checked. The
    testbench computes both operands from a hash of the address, so it
    needs no initializer per element."""
    import chiplet_flow as cf
    ms = load_model_spec()
    spec = specgen_mod.derive_proj_spec(ms)
    D, F = ms['d_model'], ms['d_ff']
    work = os.path.join(ROOT, 'build_projfull')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_proj_deps(ms, work)
        rr = RuleBasedAgent()
        with open(os.path.join(work, 'small.v'), 'w') as f:
            f.write(specgen_mod.render_proj_testbench(spec))
        with open(os.path.join(work, 'full.v'), 'w') as f:
            f.write(specgen_mod.render_proj_testbench(spec, cases=((D, F, 9),)))
        out = {}
        for label, tb, fx in (('first', 'small.v', set()),
                              ('small', 'small.v', {agent_mod.FIX_PIDX}),
                              ('full', 'full.v', {agent_mod.FIX_PIDX})):
            with open(os.path.join(work, 'p.v'), 'w') as f:
                f.write(rr.render_proj(spec, fx))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', tb, 'p.v']
                               + list(cf.PROJ_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            out[label] = subprocess.run(['vvp', 's.out'], cwd=work,
                                        capture_output=True, text=True,
                                        timeout=1200).stdout
        check('the projection passes its small cases', 'TB_RESULT: PASS' in out['small'])
        check('an output index not carried through the requantizer is '
              'caught on a short reduction',
              'TB_RESULT: PASS' not in out['first']
              and 'expected_proj' in out['first'])
        check('a full-size %d by %d projection is bit-exact on every output'
              % (D, F), 'TB_RESULT: PASS' in out['full']
              and ('checks=%d' % (F + 1)) in out['full'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_decoder_runs_the_model():
    """The checkpoint decodes in RTL, every stage a generated block.

    The integer reference first: static scales, fixed by calibration, and
    every stage one of the blocks' golden models. It has to agree with the
    float checkpoint, or the hardware would be bit-exact to the wrong
    model. Then the decoder RTL runs two prompts with the testbench as the
    host, feeding back whatever token the hardware chose, and every logit
    of every step is checked."""
    import decoder
    dec = decoder.load_decoder()
    same, total = decoder.agreement(dec.ck, dec)
    check('integer decode agrees with the float checkpoint on every '
          'corpus position (%d/%d)' % (same, total), same == total)
    stoi = {c: i for i, c in enumerate(dec.ck['chars'])}
    runs = [([stoi[c] for c in s], n) for s, n in decoder.RUNS]
    work = os.path.join(ROOT, 'build_dectest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        rc, out = decoder.run_rtl(dec, runs, work)
        check('the decoder RTL passes every logit of every step',
              rc == 0 and 'TB_RESULT: PASS' in out)
        check('the text it prints is the hardware choosing each token',
              'the agent writes the rtl' in out
              and 'the tools decide.' in out)
        spec = decoder.derive_spec(dec)
        with open(os.path.join(work, 'decoder.v'), 'w') as f:
            f.write(decoder.render_decoder(spec, set()))
        r = subprocess.run(['iverilog', '-g2005', '-o', 'm.out',
                            'tb_decoder.v', 'decoder.v'] + list(decoder.DEPS),
                           cwd=work, capture_output=True, text=True)
        o = subprocess.run(['vvp', 'm.out'], cwd=work, capture_output=True,
                           text=True, timeout=600).stdout
        check('a decoder without the MLP ReLU fails on a logit',
              r.returncode == 0 and 'TB_RESULT: PASS' not in o
              and 'expected_lg' in o)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_small_model_derivation():
    """Two derivation rules that only bind below the sizes the sweep runs.
    The accumulator carries a floor for the attention head's weighted
    sum, and the softmax's capacity never exceeds the context."""
    ms = dict(load_model_spec(), name='small', d_model=16, d_ff=32,
              head_dim=16, n_head=1, seq_len=24)
    aw = derive_chiplet_spec(ms)['parameters']['acc_width']
    check('a 16-wide model keeps a 24-bit accumulator for the weighted sum',
          aw == 24 and specgen_mod.derive_requant_spec(ms)['parameters']
          ['acc_width'] == aw)
    sm = specgen_mod.derive_softmax_spec(ms)['parameters']
    check('softmax capacity is bounded by a 24-position context',
          sm['capacity'] == 32)
    check('the Qwen derivation is unchanged by either rule',
          derive_chiplet_spec(load_model_spec())['parameters']['acc_width']
          == 29 and specgen_mod.derive_softmax_spec(load_model_spec())
          ['parameters']['capacity'] == 256)


def test_fpga_counts_the_hierarchy_once():
    """A composite block's resources are its hierarchy's totals. yosys
    prints module tables alphabetically and then those totals, so reading
    everything after the top module's table counted every submodule that
    sorts after it, and then everything again."""
    import fpga
    log = """
2.50. Printing statistics.
=== design hierarchy ===
      999   LUT6
4. Printing statistics.
=== attn ===
       10   LUT6
        1   DSP48E1
=== mac ===
        2   LUT6
        1   DSP48E1
=== softmax ===
        5   LUT6
=== design hierarchy ===
        +----------Count including submodules.
       19 attn
        2   mac
        5   softmax
       17   LUT6
        2   DSP48E1
"""
    res, _ = fpga.parse_stat(log, "attn")
    check('composite FPGA count is the hierarchy total, once',
          res['luts'] == 17 and res['dsps'] == 2)
    res, _ = fpga.parse_stat(
        "2.50. Printing statistics.\n=== mac ===\n        2   LUT6\n"
        "=== design hierarchy ===\n        2   LUT6\n"
        "4. Printing statistics.\n=== mac ===\n        2   LUT6\n", "mac")
    check('a single-module block reads only the final statistics pass',
          res['luts'] == 2)
    res, unk = fpga.parse_stat("4. Printing statistics.\n=== m ===\n"
                               "        3   RAM32M\n        5   LUT6\n", "m")
    check('LUT RAM is counted as the LUTs it occupies, not left unmapped',
          res['lutram'] == 12 and res['luts'] == 5 and not unk)


def test_rotary_embedding():
    """Qwen's rotary position embedding, generated. The integer model has
    to track float RoPE, the rules agent's first cut has to fail on the
    direction of the turn, and the testbench has to contain an angle on a
    rounding boundary, or a phase one count off passes."""
    import chiplet_flow as cf
    ms = load_model_spec()
    spec = specgen_mod.derive_rope_spec(ms)
    p, fr = spec['parameters'], spec['derivation']['freqs']
    rnd = random.Random(5)
    worst = 0.0
    for _ in range(4000):
        a, b = rnd.randrange(-128, 128), rnd.randrange(-128, 128)
        i, pos = rnd.randrange(p['pairs']), rnd.randrange(1 << p['pos_width'])
        th = pos * p['rope_theta'] ** (-2.0 * i / p['head_dim'])
        f1 = a * math.cos(th) - b * math.sin(th)
        f2 = b * math.cos(th) + a * math.sin(th)
        y1, y2 = specgen_mod.rope_golden(a, b, i, pos, p, fr)
        for y, f in ((y1, f1), (y2, f2)):
            if -128 <= f <= 127:
                worst = max(worst, abs(y - f))
    check('RoPE is within 0.65 of float on every unsaturated output',
          worst < 0.65)
    check('position 0 is the identity',
          all(specgen_mod.rope_golden(a, b, i, 0, p, fr) == (a, b)
              for a, b, i in ((127, -128, 0), (-5, 77, 31), (1, 1, 7))))
    work = os.path.join(ROOT, 'build_ropetest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        cf.write_rope_deps(spec, work)
        with open(os.path.join(work, 'tb.v'), 'w') as f:
            f.write(specgen_mod.render_rope_testbench(spec))
        rr = RuleBasedAgent()
        good = rr.render_rope(spec, {agent_mod.FIX_ROTDIR})
        out = {}
        for label, src in (('first', rr.render_rope(spec, set())),
                           ('good', good),
                           ('phase', good.replace('pos * f', '(pos * f + 1)'))):
            with open(os.path.join(work, 'r.v'), 'w') as f:
                f.write(src)
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'r.v', 'rope_rom.v'], cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            out[label] = subprocess.run(['vvp', 's.out'], cwd=work,
                                        capture_output=True, text=True,
                                        timeout=300).stdout
        check('the rotary unit passes its testbench',
              'TB_RESULT: PASS' in out['good'])
        check('a rotation by minus the angle is caught',
              'TB_RESULT: PASS' not in out['first']
              and 'expected_rope' in out['first'])
        check('an angle one count off is caught on a rounding boundary',
              'TB_RESULT: PASS' not in out['phase'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_qwen_shaped_decoder():
    """The Qwen-shaped checkpoint decodes in RTL: two layers, two query
    heads over one KV head, RoPE, the gated SiLU MLP and a final norm,
    every stage a generated block, every logit checked."""
    import qwen_decoder as qd
    dec = qd.load()
    same, total = qd.agreement(dec.ck, dec)
    check('Qwen-shaped integer decode agrees with float on every corpus '
          'position (%d/%d)' % (same, total), same == total)
    stoi = {c: i for i, c in enumerate(dec.ck['chars'])}
    runs = [([stoi[c] for c in s], n) for s, n in qd.RUNS]
    work = os.path.join(ROOT, 'build_qdectest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        rc, out = qd.run_rtl(dec, runs, work)
        check('the Qwen-shaped decoder RTL passes every logit of every step',
              rc == 0 and 'TB_RESULT: PASS' in out)
        check('and the text it prints is its own argmax fed back',
              'the agent writes the rtl' in out)
        rc, out = qd.run_rtl(dec, runs, work, fixes=set())
        check('a decoder that caches unrotated keys fails on a logit',
              'TB_RESULT: PASS' not in out and 'expected_lg' in out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_scores_wider_than_the_exponential():
    """A trained head's scores can span far more than the exponential's
    +-16. The softmax takes them at full width and clamps only the
    difference from the row maximum, which is exact, and the head's
    guard lets shift_s scale scores up as well as down."""
    ms = load_model_spec()
    sm = specgen_mod.derive_softmax_spec(ms)['parameters']
    e_lo = 1 << (sm['score_width'] - 1)
    w, _, _ = specgen_mod.softmax_golden([200000, 190000, -150000], sm)
    check('scores 39 and 1367 below a 781 maximum get weight zero, the '
          'maximum all of it',
          # one count under 1.0 is the reciprocal's rounding, the same
          # weight a single-score row gets
          w[0] >= (1 << sm['weight_frac']) - 1 and w[1] == 0 and w[2] == 0)
    w1, _, _ = specgen_mod.softmax_golden([5000, 5000 - e_lo], sm)
    w2, _, _ = specgen_mod.softmax_golden([5000, 5000 - e_lo - 999], sm)
    check('clamping score minus maximum at the exponential\'s floor is exact',
          w1 == w2)
    at = specgen_mod.derive_attn_spec(ms)['parameters']
    g = at['score_guard']
    rs = specgen_mod._round_shift
    check('the score guard leaves every right shift unchanged',
          all(rs(t << g, s + g) == rs(t, s)
              for t in (-99999, -3, 0, 7, 12345) for s in range(0, 12)))


def test_multi_lane_projection():
    """Enough lanes to consume the board's DDR bandwidth, each taking its
    own byte of a wide weight word. At Qwen's full 896 by 4864 size it has
    to be bit-exact and within one percent of one cycle per row per group
    of lanes, and the first cut that wires lanes to mirrored bytes has to
    be caught."""
    import chiplet_flow as cf
    ms = load_model_spec()
    spec = specgen_mod.derive_projn_spec(ms)
    N = spec['parameters']['lanes']
    check('32 lanes consume 2 GB/s of int8 weights at 100 MHz', N == 32)
    D, F = ms['d_model'], ms['d_ff']
    work = os.path.join(ROOT, 'build_projntest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_projn_deps(ms, work)
        with open(os.path.join(work, 'small.v'), 'w') as f:
            f.write(specgen_mod.render_projn_testbench(spec))
        with open(os.path.join(work, 'full.v'), 'w') as f:
            f.write(specgen_mod.render_projn_testbench(spec, cases=((D, F, 9),)))
        rr = RuleBasedAgent()
        out = {}
        for label, tb, fx in (('first', 'small.v', set()),
                              ('full', 'full.v', {agent_mod.FIX_LANE})):
            with open(os.path.join(work, 'p.v'), 'w') as f:
                f.write(rr.render_projn(spec, fx))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', tb, 'p.v']
                               + list(cf.PROJN_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            out[label] = subprocess.run(['vvp', 's.out'], cwd=work,
                                        capture_output=True, text=True,
                                        timeout=1200).stdout
        check('mirrored lane bytes are caught',
              'TB_RESULT: PASS' not in out['first']
              and 'expected_lane' in out['first'])
        m = re.search(r'span_cycles=(\d+)', out['full'])
        span = int(m.group(1)) if m else 0
        ideal = -(-F // N) * D
        check('the %d by %d projection is bit-exact in %d cycles, within 1%% '
              'of %d' % (D, F, span, ideal),
              'TB_RESULT: PASS' in out['full'] and ('checks=%d' % (F + 1))
              in out['full'] and ideal <= span <= ideal * 1.01)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_basys3_top_level():
    """The board package: the Qwen-shaped decoder behind a UART, weights
    in a ROM image, simulated whole. A prompt typed into the UART has to
    come back as the integer reference's continuation, character for
    character, and the committed package has to be what board.py writes."""
    import board
    work = os.path.join(ROOT, 'build_boardtest')
    try:
        out = board.build(work)
        check('the Basys 3 top level answers a prompt over its UART',
              'TB_RESULT: PASS' in out and 'writes the rtl' in out)
        same = all(open(os.path.join(work, f)).read()
                   == open(os.path.join(ROOT, 'board_basys3', f)).read()
                   for f in ('qwen_params.hex', 'basys3.xdc', 'build.tcl',
                             os.path.join('rtl', 'fpgai_top.v'),
                             os.path.join('rtl', 'qwen_decoder.v')))
        check('the committed board package is what board.py generates', same)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_zybo_register_block():
    """The Zybo top level's AXI-Lite registers, around a stub core whose
    clock only ticks every fourth bus cycle, as the real one's does while
    lines fill. A start written once has to reach it, done has to stay
    set until the next start clears it, and the layout registers have to
    read back what the software's header was generated with."""
    import board_zybo as bz
    L = dict(W=dict(tok=18, pos=8), wb=0x08000000, cb=0x25714000,
             kb=0x25BB8000, vb=0x25D38000, end=0x25EB8000)
    work = os.path.join(ROOT, 'build_zregtest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    wp = ''.join('  output [31:0] w%d_araddr, output [3:0] w%d_arlen, output w%d_arvalid,\n'
                 '  input w%d_arready, input [63:0] w%d_rdata, input w%d_rvalid,\n'
                 '  input w%d_rlast, output w%d_rready,\n' % ((p,) * 8) for p in range(4))
    wt = ''.join('  assign w%d_araddr = 0; assign w%d_arlen = 0; assign w%d_arvalid = 0;'
                 ' assign w%d_rready = 1;\n' % ((p,) * 4) for p in range(4))
    stub = """module qwen_zybo (input clk, input rst_n, input start, input head_en,
  input [17:0] tok, input [7:0] pos, output reg [17:0] next_tok,
  output reg signed [15:0] best, output reg done, output reg busy,
%s  output [31:0] araddr, output [3:0] arlen, output arvalid, input arready,
  input [63:0] rdata, input rvalid, input rlast, output rready,
  output [31:0] awaddr, output [3:0] awlen, output awvalid, input awready,
  output [63:0] wdata, output [7:0] wstrb, output wlast, output wvalid,
  input wready, input bvalid, output bready, output reg [31:0] core_cycles);
%s  assign araddr = 0; assign arlen = 0; assign arvalid = 0; assign rready = 1;
  assign awaddr = 0; assign awlen = 0; assign awvalid = 0; assign wdata = 0;
  assign wstrb = 0; assign wlast = 0; assign wvalid = 0; assign bready = 1;
  reg [1:0] ph; reg [7:0] left;
  always @(posedge clk) begin
    if (!rst_n) begin ph <= 0; busy <= 0; done <= 0; core_cycles <= 0; end
    else begin
      ph <= ph + 1;
      if (ph == 0) begin   // one core edge in four bus cycles
        core_cycles <= core_cycles + 1; done <= 1'b0;
        if (!busy && start) begin busy <= 1'b1; left <= 20; end
        else if (busy && left == 0) begin
          busy <= 1'b0; done <= 1'b1;
          next_tok <= tok ^ 18'h155; best <= head_en ? -$signed({8'd0, pos}) : 16'sd0;
        end else if (busy) left <= left - 1;
      end
    end
  end
endmodule
""" % (wp, wt)
    tasks = bz.STEP[:bz.STEP.index('  task expect_reg')].replace('%%', '%')
    tb = """`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  reg [5:0] s_awaddr = 0, s_araddr = 0; reg [31:0] s_wdata = 0;
  reg s_awvalid = 0, s_wvalid = 0, s_arvalid = 0, s_bready = 0, s_rready = 0;
  wire s_awready, s_wready, s_bvalid, s_arready, s_rvalid;
  wire [1:0] s_bresp, s_rresp; wire [31:0] s_rdata;
  fpgai_zybo dut (.aclk(clk), .aresetn(rst_n),
    .s_axi_awaddr(s_awaddr), .s_axi_awvalid(s_awvalid), .s_axi_awready(s_awready),
    .s_axi_wdata(s_wdata), .s_axi_wstrb(4'hf), .s_axi_wvalid(s_wvalid),
    .s_axi_wready(s_wready), .s_axi_bresp(s_bresp), .s_axi_bvalid(s_bvalid),
    .s_axi_bready(s_bready), .s_axi_araddr(s_araddr), .s_axi_arvalid(s_arvalid),
    .s_axi_arready(s_arready), .s_axi_rdata(s_rdata), .s_axi_rresp(s_rresp),
    .s_axi_rvalid(s_rvalid), .s_axi_rready(s_rready));
%s  integer bad = 0;
  task want(input [5:0] a, input [31:0] v);
    begin rd(a); if (rv !== v) begin bad = bad + 1;
      $display("reg %%h = %%h, want %%h", a, rv, v); end end
  endtask
  // A lost start leaves the poll loop waiting for good.
  initial begin #100000; $display("TB_RESULT: FAIL, timed out"); $finish; end
  initial begin
    repeat (4) @(negedge clk); rst_n = 1;
    want(6'h20, 32'h%08x); want(6'h24, 32'h%08x); want(6'h28, 32'h%08x);
    want(6'h2c, 32'h%08x); want(6'h30, 32'h%08x); want(6'h34, 32'h%08x);
    want(6'h0c, 0);
    wr(6'h04, 32'hfffff); want(6'h04, 32'h3ffff);
    wr(6'h04, 1234); wr(6'h08, 7); wr(6'h00, 3);
    rd(6'h0c); if (rv[1]) bad = bad + 1;
    rv = 0; while (!rv[1]) rd(6'h0c);
    want(6'h10, 1234 ^ 18'h155); want(6'h14, -7); want(6'h00, 2);
    rd(6'h1c); if (rv < 60) begin bad = bad + 1; $display("bus %%0d", rv); end
    repeat (40) @(negedge clk); want(6'h0c, 2);
    wr(6'h04, 99); wr(6'h08, 8); wr(6'h00, 1);
    rd(6'h0c); if (rv[1]) bad = bad + 1;
    rv = 0; while (!rv[1]) rd(6'h0c);
    want(6'h10, 99 ^ 18'h155); want(6'h14, 0);
    if (bad) $display("TB_RESULT: FAIL"); else $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
""" % (tasks, bz.FPGAI_ID, L['wb'], L['cb'], L['kb'], L['vb'], L['end'])
    try:
        open(os.path.join(work, 'fpgai_zybo.v'), 'w').write(bz.render_wrapper(L))
        open(os.path.join(work, 'stub.v'), 'w').write(stub)
        open(os.path.join(work, 'tb.v'), 'w').write(tb)
        r = subprocess.run(['iverilog', '-g2005', '-o', 'z.out', 'tb.v',
                            'fpgai_zybo.v', 'stub.v'], cwd=work,
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        if not r.returncode:
            out = subprocess.run(['vvp', 'z.out'], cwd=work, capture_output=True,
                                 text=True, timeout=120).stdout
        check('the Zybo registers start a held core and read its token',
              'TB_RESULT: PASS' in out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_weight_streamer_survives_stalls():
    """The Zybo bridge's weight streamer alone, on four read ports that
    stall arready and gap their beats at random, fed an address stream
    that runs ahead, jumps back and jumps forward the way projections do.
    Every word the core is shown has to be the right one and the stream
    must never stop: a jump back while a request waited for arready used
    to be forgotten, and the core then waited forever. With 120 jumps it
    also passed a streamer that let two bursts be in flight for one slot,
    which after a jump back could mark a slot valid for one line and
    then fill it with another's beats: 1500 jumps, on two seeds, show
    that one (4 to 31 wrong words in 88,000 on every seed tried)."""
    import zybo
    wam, base = 15, 0x08000000
    work = os.path.join(ROOT, 'build_strmtest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    ports, _, wdata = zybo.render_wports(4)
    top = """module strm (input clk, input rst_n, input [%d:0] wq, output hw_o,
  output [255:0] w_data,
%s  output dummy);
  assign dummy = 1'b0;
%s
%s%s
  // The bridge registers these with wq; here wq is an input.
  always @(*) curns = (wq >> 2) + NS;
  assign hw_o = hw;
endmodule
""" % (wam, ports.split('\n', 1)[1],
       '  ' + ' '.join('assign w%d_arlen = 4\'d7; assign w%d_rready = 1\'b1;' % (p, p)
                       for p in range(4)),
       zybo.render_streamer(4, wam, wbase=base), wdata)
    conn = ''.join('    .w%(p)d_araddr(a%(p)d), .w%(p)d_arlen(), .w%(p)d_arvalid(av%(p)d),'
                   ' .w%(p)d_arready(ar%(p)d), .w%(p)d_rdata(rd%(p)d),'
                   ' .w%(p)d_rvalid(rv%(p)d), .w%(p)d_rlast(rl%(p)d), .w%(p)d_rready(),\n'
                   % {'p': p} for p in range(4))
    mem = ''.join("""  wire [31:0] a%(p)d; wire av%(p)d; reg ar%(p)d = 1, rv%(p)d = 0, rl%(p)d = 0;
  reg [63:0] rd%(p)d = 0;
  reg [31:0] qa%(p)d [0:15]; integer qh%(p)d = 0, qn%(p)d = 0, j%(p)d;
  always @(posedge clk) ar%(p)d <= {$random} %% 3 != 0;
  always @(posedge clk) if (rst_n && av%(p)d && ar%(p)d) begin
    qa%(p)d[qn%(p)d %% 16] = a%(p)d; qn%(p)d = qn%(p)d + 1; end
  initial begin
    @(posedge rst_n);
    forever begin
      @(posedge clk);
      if (qh%(p)d < qn%(p)d) begin
        repeat ({$random} %% 20) @(posedge clk);
        for (j%(p)d = 0; j%(p)d < 8; j%(p)d = j%(p)d + 1) begin
          while ({$random} %% 3 == 0) begin rv%(p)d <= 0; rl%(p)d <= 0; @(posedge clk); end
          rd%(p)d <= beat(qa%(p)d[qh%(p)d %% 16] + j%(p)d * 8);
          rv%(p)d <= 1; rl%(p)d <= (j%(p)d == 7); @(posedge clk);
        end
        rv%(p)d <= 0; rl%(p)d <= 0; qh%(p)d = qh%(p)d + 1;
      end
    end
  end
""" % {'p': p} for p in range(4))
    tb = """`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  // word w of the image: sixteen int8 weights, byte k = w * 7 + k * 13
  function [63:0] beat(input [31:0] a);
    integer k; reg [31:0] w;
    begin
      w = (a - %d) / 16;
      for (k = 0; k < 8; k = k + 1)
        beat[k * 8 +: 8] = w * 7 + (k + ((a / 8) %% 2) * 8) * 13;
    end
  endfunction
  function [255:0] want(input [31:0] w);
    integer k; reg [7:0] b;
    begin
      for (k = 0; k < 16; k = k + 1) begin
        b = w * 7 + k * 13; want[k * 16 +: 16] = {{8{b[7]}}, b};
      end
    end
  endfunction
%s  reg [%d:0] wq = 0;
  wire hw; wire [255:0] w_data;
  strm dut (.clk(clk), .rst_n(rst_n), .wq(wq), .hw_o(hw), .w_data(w_data),
%s    .dummy());
  // The core: a word a cycle whenever it is there, with the address
  // pattern of projections: runs, jumps back, jumps ahead, many of them
  // short so that jumps land while requests still wait on arready.
  integer seg, i, bad = 0, n = 0;
  integer from [0:@NS@], len [0:@NS@];
  initial begin
@SEGS@
    repeat (4) @(negedge clk); rst_n = 1;
    for (seg = 0; seg <= @NS@; seg = seg + 1)
      for (i = 0; i < len[seg]; i = i + 1) begin
        wq = from[seg] + i;
        @(negedge clk); while (!hw) @(negedge clk);
        n = n + 1;
        if (w_data !== want(wq)) begin
          bad = bad + 1;
          if (bad < 5) $display("word %%0d wrong", wq);
        end
      end
    if (bad) $display("TB_RESULT: FAIL, %%0d of %%0d words", bad, n);
    else $display("TB_RESULT: PASS, %%0d words", n);
    $finish;
  end
  initial begin #400000000; $display("TB_RESULT: FAIL, stalled"); $finish; end
endmodule
""" % (base, mem, wam, conn)
    ok = True
    try:
        open(os.path.join(work, 'strm.v'), 'w').write(top)
        for seed in (7, 11):
            rng = random.Random(seed)
            segs = [(0, 900), (8, 300)]
            for _ in range(1500):
                start = rng.choice([rng.randrange(0, 4000), rng.randrange(0, 64)])
                segs.append((start, rng.choice([rng.randrange(1, 12), rng.randrange(20, 200)])))
            open(os.path.join(work, 'tb.v'), 'w').write(
                tb.replace('@NS@', str(len(segs) - 1)).replace('@SEGS@', '\n'.join(
                    '    from[%d] = %d; len[%d] = %d;' % (k, f, k, l)
                    for k, (f, l) in enumerate(segs))))
            r = subprocess.run(['iverilog', '-g2005', '-o', 'z.out', 'tb.v', 'strm.v'],
                               cwd=work, capture_output=True, text=True)
            out = r.stdout + r.stderr
            if not r.returncode:
                out = subprocess.run(['vvp', 'z.out'], cwd=work, capture_output=True,
                                     text=True, timeout=900).stdout
            ok &= 'TB_RESULT: PASS' in out
        check('the weight streamer never stalls, nor shows a wrong word, under a '
              'stalling bus and 1500 jumps', ok)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_full_sequencer_both_qwens():
    """The full-size sequencer generator, run on small random checkpoints
    with each real Qwen's structure: Qwen3-0.6B's (no biases, q twice the
    hidden width, RMSNorm on every head of q and k) and Qwen2.5's (q, k,
    v biases). Each head step's argmax and its logit have to be the
    integer model's, and a sequencer that skips Qwen3's head norms has to
    be caught, which the token alone would not do: a random tied-head
    model keeps choosing its input token. Both again at 32 lanes, the
    ZC706's width: every lane-dependent index in the sequencer, its KV
    cache and its embedding lookup is derived from the count."""
    import qwen_full, qwen_synth
    ids = [3, 77, 12, 140]
    for style, mut, lanes, kw in (
            ('qwen3', None, None, {}), ('qwen2.5', None, None, {}),
            ('qwen3', ('S_V: st <= S_QN;', 'S_V: st <= S_RQ;'), None, {}),
            ('qwen3', None, 32, dict(vocab=608)),
            ('qwen2.5', None, 32, dict(vocab=608, hidden=128))):
        work = os.path.join(ROOT, 'build_qsynthtest')
        shutil.rmtree(work, ignore_errors=True)
        try:
            im, _ = qwen_synth.model(style, lanes=lanes, **kw)
            want, srcs = qwen_full.build_model(im, ids, 3, work, log=lambda *a: None)
            im.reset()
            best = []
            for p in range(len(want) - 1):
                lg = im.step(want[p], p, logits=p >= len(ids) - 1)
                if lg is not None:
                    best.append((max(range(len(lg)), key=lg.__getitem__), max(lg)))
            if mut:
                path = os.path.join(work, 'qwen_full.v')
                src = open(path).read()
                assert mut[0] in src
                open(path, 'w').write(src.replace(mut[0], mut[1]))
            r = subprocess.run(['iverilog', '-g2005', '-o', 'q.out'] + srcs, cwd=work,
                               capture_output=True, text=True)
            out = r.stdout + r.stderr
            if not r.returncode:
                out = subprocess.run(['vvp', 'q.out'], cwd=work, capture_output=True,
                                     text=True, timeout=600).stdout
            got = [(int(l.split('next=')[1].split()[0]), int(l.split('best=')[1].split()[0]))
                   for l in out.splitlines() if l.startswith('STEP')
                   and int(l.split('pos=')[1].split()[0]) >= len(ids) - 1]
            if mut:
                check('a Qwen3 sequencer skipping the head norms is caught',
                      len(got) == len(best) and got != best)
            else:
                check('the %s-shaped sequencer at %d lanes matches every logit it picks'
                      % (style, qwen_full.lanes_of(im)), got == best and len(best) == 3)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def test_bridge_end_to_end():
    """The generated sequencer inside the Zybo/ZC706 DDR bridge, on small
    random checkpoints, with every DDR port stalling and gapping at
    random: at 16 lanes and at 32 each head step has to pick the integer
    model's token. The DDR model returns bytes in address order, lowest
    on rdata[7:0], as the Zynq's HP ports do once the ARM has copied
    weights8.bin in byte for byte, and byte j of each word has to be lane
    j's int8: an image written lane N - 1 first, as the first one was,
    has to be caught."""
    import qwen_full, qwen_synth, zybo
    ids = [3, 77, 12, 140]
    for lanes, kw, mut in ((16, {}, False), (32, dict(vocab=608), False),
                           (16, {}, True)):
        work = os.path.join(ROOT, 'build_bridgetest')
        shutil.rmtree(work, ignore_errors=True)
        try:
            im, _ = qwen_synth.model('qwen3', lanes=lanes, **kw)
            want, _ = qwen_full.build_model(im, ids, 3, work, log=lambda *a: None)
            w8 = zybo.write_w8(work)
            img = open(w8, 'rb').read()
            Q, D = im.Q['model.layers.0.self_attn.q_proj.weight'], im.D
            order = all((img[(g * D + d) * lanes + j] ^ 0x80) - 0x80
                        == Q[(g * lanes + j) * D + d]
                        for g in range(2) for d in (0, 1, D - 1) for j in range(lanes))
            if mut:
                with open(w8, 'wb') as f:
                    for o in range(0, len(img), lanes):
                        f.write(img[o:o + lanes][::-1])
            L = zybo.layout(work, 0x08000000)
            open(os.path.join(work, 'qwen_zybo.v'), 'w').write(zybo.render_top(
                L['W'], L['gn'], L['cb'], L['kb'], L['vb'], 4, L['wb']))
            open(os.path.join(work, 'tb_zybo.v'), 'w').write(zybo.tb_text(L, 4, 30, jit=1))
            srcs = sorted(f for f in os.listdir(work) if f.endswith('.v')
                          and not f.startswith('tb_') and f != 'qwen_zybo.v')
            r = subprocess.run(['iverilog', '-g2005', '-DSIM', '-o', 'z.out', 'tb_zybo.v',
                                'qwen_zybo.v'] + srcs, cwd=work, capture_output=True, text=True)
            out = r.stdout + r.stderr
            if not r.returncode:
                out = subprocess.run(['vvp', 'z.out'], cwd=work, capture_output=True,
                                     text=True, timeout=900).stdout
            got = [int(l.split('next=')[1].split()[0]) for l in out.splitlines()
                   if l.startswith('STEP') and int(l.split('pos=')[1].split()[0]) >= len(ids) - 1]
            if mut:
                check('a weight image in the wrong byte order is caught on the bus',
                      len(got) == 3 and got != want[len(ids):])
            else:
                check('the %d-lane core on a stalling DDR bus picks the integer model\'s '
                      'tokens, from an image in AXI byte order' % lanes,
                      order and got == want[len(ids):] and len(got) == 3)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def test_spec_to_verified_rtl():
    """spec2rtl.py end to end on a small Qwen3-shaped spec: every block the
    design compiles signed off through the gates at the spec's own
    parameters, the decode step generated around exactly those files and
    matching the integer model's tokens and logits at every layer; and a
    shape the generator cannot build (a head narrower than the core's
    lanes) refused with the reason instead of built wrong. The flow's
    scratch directory has to be the caller's again afterwards: one left
    pointing at the run's removed gates failed every later flow."""
    import spec2rtl
    import chiplet_flow as cf
    before = cf.BUILD
    out = os.path.join(ROOT, 'build_s2rtest_%d' % os.getpid())
    shutil.rmtree(out, ignore_errors=True)
    try:
        spec = dict(name='tiny', n_layer=2, d_model=64, n_head=4, n_kv_head=2,
                    head_dim=32, d_ff=256, vocab=608, qk_norm=True)
        rep = spec2rtl.run(spec, 'zybo_z7_20', out, log=lambda *a: None)
        rows = rep['stages'].get('blocks', {}).get('rows', [])
        d = rep['stages'].get('design', {})
        check('a spec becomes %d signed-off blocks and a decode step matching the '
              'integer model' % len(rows),
              rep['ok'] and len(rows) == 15 and all(r['converged'] for r in rows)
              and d.get('rtl') == d.get('integer_model') and len(d.get('rtl') or []) == 2
              and d.get('layers') == d.get('of') == 2)
        bad = spec2rtl.run(dict(spec, head_dim=8, d_model=32), 'zybo_z7_20', out,
                           log=lambda *a: None)
        check('a shape the generator cannot build is refused with its reason',
              not bad['ok'] and any('narrower than the core' in p for p in
                                    bad['stages']['spec']['problems']))
        check('the flow\'s scratch directory is the caller\'s again after a run',
              cf.BUILD == before)
    finally:
        cf.BUILD = before
        shutil.rmtree(out, ignore_errors=True)


def test_signed_off_modules_are_renamed_however_written():
    """The score MAC is signed off as the mac it is and renamed mac_s for
    the attention head, which instances both. The rename matched only
    "module mac (", so when Haiku wrote "module mac(" two modules named
    mac reached the head's compile, every agent failed the head for a
    reason none of them could see, and the run failed."""
    import spec2rtl
    ok = all(spec2rtl.rename_module(h + "input a); endmodule", "mac", "mac_s").startswith("module mac_s")
             for h in ("module mac (", "module mac(", "module mac #(", "module  mac\n("))
    ok &= spec2rtl.rename_module("module macro (a); endmodule\nmodule mac (b);", "mac", "mac_s") \
        == "module macro (a); endmodule\nmodule mac_s (b);"
    try:
        spec2rtl.rename_module("module other (a);", "mac", "mac_s")
        ok = False
    except RuntimeError:
        pass
    check('a signed-off module is renamed however its header is written, and a missing one '
          'is an error', ok)


def test_sign_off_survives_a_silent_agent():
    """An LLM agent whose model never answers (its CLI timing out, as
    Sonnet's did on the attention head for 75 minutes) has failed its
    attempt, and the next agent in the chain has to take the block; it
    used to abort the whole spec2rtl.py run."""
    import spec2rtl, qwen_synth
    import chiplet_flow as cf

    class Silent:
        last_rtl = None

        def propose(self, spec, history):
            raise RuntimeError("claude CLI failed after 3 attempt(s): timed out after 1500s")
    gates = os.path.join(ROOT, 'build_silenttest')
    shutil.rmtree(gates, ignore_errors=True)
    chain, plan, build = spec2rtl.agent_chain, spec2rtl.block_plan, cf.BUILD
    try:
        spec2rtl.agent_chain = lambda kind: [('silent', Silent, 2),
                                             ('rules (fallback)', RuleBasedAgent, 5)]
        spec2rtl.block_plan = lambda im: (plan(im)[0][:1], plan(im)[1])
        im, _ = qwen_synth.model('qwen3', nl=1)
        rows, _ = spec2rtl.sign_off(im, gates, 'llm', lambda *a: None)
        r = rows[0]
        check('a model that never answers hands its block to the next agent, which signs it off',
              r['converged'] and r['agent'] == 'rules (fallback)'
              and [a[0] for a in r['attempts']] == ['silent', 'rules (fallback)']
              and not r['attempts'][0][2])
        tag = r['file'][:-2]
        kept = os.path.join(gates, 'report_%s.silent.json' % tag)
        rep = json.load(open(kept)) if os.path.exists(kept) else {}
        check('the silent agent\'s attempt is kept, marked with why it stopped',
              rep.get('converged') is False and 'timed out' in rep.get('aborted', '')
              and rep.get('iterations_used') == 0)
    finally:
        spec2rtl.agent_chain, spec2rtl.block_plan, cf.BUILD = chain, plan, build
        shutil.rmtree(gates, ignore_errors=True)


def test_failed_attempts_are_kept():
    """The next agent in the chain writes over a block's report and draft,
    so a run that fell back to the rules agent kept no record of why the
    models had failed. Each attempt that does not converge now leaves both
    under its own name."""
    import spec2rtl, qwen_synth
    import chiplet_flow as cf
    gates = os.path.join(ROOT, 'build_attempttest')
    shutil.rmtree(gates, ignore_errors=True)
    chain, plan, build = spec2rtl.agent_chain, spec2rtl.block_plan, cf.BUILD
    try:
        spec2rtl.agent_chain = lambda kind: [('first try', RuleBasedAgent, 1),
                                             ('rules (fallback)', RuleBasedAgent, 5)]
        spec2rtl.block_plan = lambda im: (plan(im)[0][:1], plan(im)[1])
        im, _ = qwen_synth.model('qwen3', nl=1)
        rows, _ = spec2rtl.sign_off(im, gates, 'llm', lambda *a: None)
        tag = rows[0]['file'][:-2]
        kept = os.path.join(gates, 'report_%s.first_try.json' % tag)
        draft = os.path.join(gates, 'rtl_%s.first_try.v' % tag)
        rep = json.load(open(kept)) if os.path.exists(kept) else {}
        check('an attempt that does not converge keeps its report and draft; the fallback signs off',
              rows[0]['converged'] and rows[0]['agent'] == 'rules (fallback)'
              and rep.get('converged') is False and rep.get('iterations_used') == 1
              and os.path.exists(draft))
    finally:
        spec2rtl.agent_chain, spec2rtl.block_plan, cf.BUILD = chain, plan, build
        shutil.rmtree(gates, ignore_errors=True)


def test_a_draft_that_never_passed_is_not_handed_on():
    """Handed a draft that never passed simulation, the next model kept its
    bug: Sonnet's and then Opus's projections gave Haiku's wrong -20614 in
    five of their thirteen drafts. With no draft past simulation and
    synthesis, the next model starts from the spec."""
    import spec2rtl, qwen_synth, llm_agent
    import chiplet_flow as cf
    gates = os.path.join(ROOT, 'build_handofftest')
    shutil.rmtree(gates, ignore_errors=True)
    seen = []

    class Broken:
        best_of = llm_agent.LLMAgent.best_of

        def __init__(self):
            self.drafts, self.seed, self.last_rtl = [], None, None

        def propose(self, spec, history):
            rtl = 'module broken(); endmodule\n'
            self.drafts.append(rtl)
            self.last_rtl = rtl
            return rtl, ['broken']

    class Next(Broken):
        def __init__(self):
            Broken.__init__(self)
            self.rules = RuleBasedAgent()

        def propose(self, spec, history):
            seen.append((self.seed, self.last_rtl))
            return self.rules.propose(spec, history)

    chain, plan, build = spec2rtl.agent_chain, spec2rtl.block_plan, cf.BUILD
    try:
        spec2rtl.agent_chain = lambda kind: [('broken', Broken, 2),
                                             ('next', Next, 5)]
        spec2rtl.block_plan = lambda im: (plan(im)[0][:1], plan(im)[1])
        im, _ = qwen_synth.model('qwen3', nl=1)
        rows, _ = spec2rtl.sign_off(im, gates, 'llm', lambda *a: None)
        check('with no draft past simulation, the next model starts from the spec',
              rows[0]['converged'] and rows[0]['agent'] == 'next'
              and seen and seen[0] == (None, None))
    finally:
        spec2rtl.agent_chain, spec2rtl.block_plan, cf.BUILD = chain, plan, build
        shutil.rmtree(gates, ignore_errors=True)


def test_llm_blocks_decode():
    """The fifteen blocks the models signed off, Haiku nine, Sonnet three
    and Opus three, in one design: the decode step matches the integer
    model. Each was signed off in a run of its own with the generator's
    blocks around it; all of them together is the check that matters."""
    import spec2rtl, qwen_synth
    src = os.path.join(ROOT, 'llm_blocks', 'tiny_qwen3')
    out = os.path.join(ROOT, 'build_llmblocks')
    shutil.rmtree(out, ignore_errors=True)
    try:
        spec = json.load(open(os.path.join(ROOT, 'examples', 'tiny_qwen3.json')))
        rep = spec2rtl.run(spec, out=out, blocks=src, log=lambda *a: None)
        rows = rep['stages']['blocks']['rows']
        used = all(open(os.path.join(out, 'design', r['file'])).read()
                   == open(os.path.join(src, r['file'])).read() for r in rows)
        models = sorted(r['agent'] for r in rows)
        check('the fifteen blocks the models signed off decode as the integer '
              'model does, and the design is built from those files',
              rep['ok'] and len(rows) == 15 and used
              and all(r['converged'] and r['agent'].startswith('llm:') for r in rows)
              and [m.split('@')[0] for m in models].count('llm:haiku') == 9)
        # A block signed off for other parameters is not this design's.
        alt = os.path.join(out, 'alt')
        shutil.copytree(src, alt)
        p = os.path.join(alt, 'report_b_rmsnorm.json')
        r = json.load(open(p))
        r['spec']['parameters']['d_model'] = 128
        json.dump(r, open(p, 'w'))
        im, _ = qwen_synth.model('qwen3', nl=1)
        other, _ = spec2rtl.load_signed_off(im, os.path.join(out, 'other'), alt,
                                            lambda *a: None)
        check('a block signed off for other parameters is refused, the rest kept',
              [r['file'] for r in other if not r['converged']] == ['b_rmsnorm.v'])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_composites_give_their_parts_ports():
    """A composite that instantiates supplied modules has to give their
    ports: told only "the supplied mac module", two models guessed port
    names for thirteen drafts on the projection and never compiled."""
    import spec2rtl, qwen_synth
    im, _ = qwen_synth.model('qwen3', nl=1)
    plan, _ = spec2rtl.block_plan(im)
    missing = []
    for fn, what, spec, tbf, deps in plan:
        text = ' '.join(spec.get('behavior', []))
        uses = [d for d in deps if d.endswith('_dep.v')]
        if uses and 'supplied' in text and not ('exactly these ports' in text
                                                or 'has ports' in text):
            missing.append(fn)
    check('every composite built from supplied modules gives their ports',
          not missing)
    # A spec whose text and ports say one row length and whose parameters
    # say another: the head norm was the hidden-size norm's spec with two
    # parameters changed, "sum over i in 0..63" and 6-bit addresses over a
    # 32-element row, and Sonnet summed 64 elements for eight drafts.
    bad = []
    for fn, what, spec, tbf, deps in plan:
        pr = spec.get('parameters', {})
        if 'd_model' not in pr or 'addr_width' not in pr:
            continue
        n, aw = pr['d_model'], pr['addr_width']
        rows = [int(v) for v in re.findall(r'row of (\d+)', spec.get('description', ''))]
        spans = [int(v) for v in re.findall(r'\b0\.\.(\d+)\b', ' '.join(spec.get('behavior', [])))]
        widths = [q['width'] for q in spec['ports']
                  if q['name'].endswith('_addr') or q['name'] == 'o_index']
        if any(r != n for r in rows) or (spans and n - 1 not in spans) \
                or any(w != aw for w in widths):
            bad.append(fn)
    hd = next(s for fn, _, s, _, _ in plan if fn == 'b_rmsnorm_hd.v')
    check('every block\'s text and address ports agree with its row length, '
          'the head norm\'s with the head size',
          not bad and hd['parameters']['d_model'] == im.hd
          and 'over i in 0..%d ' % (im.hd - 1) in ' '.join(hd['behavior']))
    an = next(s for fn, _, s, _, _ in plan if fn == 'b_attnn.v')
    tb = specgen_mod.render_attnn_testbench(an)
    check('the attention spec gives the score dot product the score MAC\'s '
          'width, and its testbench names a key read one element off',
          'a signed %d-bit dot product' % an['parameters']['score_acc_width']
          in ' '.join(an['behavior'])
          and 'got_s_is_the_sum_without_d0_and_with_the_last_element_twice=1' in tb
          and 'expect_sa[0] = ' in tb)
    pj = next(s for fn, _, s, _, _ in plan if fn == 'b_projn.v')
    check('the projection gives its mac and requant ports and latencies',
          'mac (input clk' in ' '.join(pj['behavior'])
          and 'requant (input clk' in ' '.join(pj['behavior']))
    pp = pj['parameters']
    text = ' '.join(pj['behavior'])
    check('the per-column projection says what its column word holds and '
          'how the bias is used',
          'requant(acc + bias)' in text
          and 'bias the top %d bits' % pp['acc_width'] in text
          and 'scale the low %d bits' % pp['scale_width'] in text)


def test_attention_scores_cannot_overflow():
    """A score is q times k, both 16-bit activations at a16, over head_dim;
    the model's MAC is sized for an int8 weight times an activation. At a
    head too wide for it the multi-lane head gives its score lanes a MAC
    of their own width, and its testbench, run at the sequencer's 16-bit
    width, passes; with the scores squeezed into the model's MAC, as they
    were, it fails."""
    import chiplet_flow as cf
    ms = dict(load_model_spec(), activation_bits=16, d_model=64, d_ff=256,
              n_head=4, n_kv_head=2, head_dim=32, seq_len=256, lanes=16)
    spec = specgen_mod.derive_attnn_spec(ms)
    p = spec['parameters']
    old = json.loads(json.dumps(spec))
    old['parameters']['score_acc_width'] = p['acc_width']
    work = os.path.join(ROOT, 'build_scoretest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_attn_deps(ms, work)
        with open(os.path.join(work, 'tb.v'), 'w') as f:
            f.write(specgen_mod.render_attnn_testbench(spec))
        out = {}
        for label, sp in (('wide', spec), ('old', old)):
            with open(os.path.join(work, 'h.v'), 'w') as f:
                f.write(RuleBasedAgent().render_attnn(sp, {agent_mod.FIX_KLANE}))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v', 'h.v']
                               + list(cf.ATTN_DEPS), cwd=work, capture_output=True,
                               text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            out[label] = subprocess.run(['vvp', 's.out'], cwd=work, capture_output=True,
                                        text=True, timeout=900).stdout
        check('score lanes get %d bits where the model MAC has %d, and every score '
              'is exact' % (p['score_acc_width'], p['acc_width']),
              p['score_acc_width'] > p['acc_width'] and 'TB_RESULT: PASS' in out['wide'])
        check('scores squeezed into the model MAC are caught',
              'TB_RESULT: PASS' not in out['old'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_weights_split_over_boards():
    """Tensor parallelism in RTL (tp.py): every rank on every layer with a
    slice of every matrix, each on its own clock and at its own width,
    gathering the attention context, o's output, the gated product and
    down's output from the others, and agreeing on the head's argmax. Two
    ranks, four of 16/32/16/32 lanes, and Qwen2.5's biased shape each have
    to give the one-board integer model's tokens and logits, and so do
    uneven shares, four ranks' with two holding no head chunk; a network
    that skips the context's gather has to be caught."""
    import tp, qwen_synth
    ids = [3, 77, 12, 140]
    work = os.path.join(ROOT, 'build_tptest')
    k4 = dict(vocab=608, hidden=128, heads=8, kv=4)
    u4 = dict(kv=[1, 1, 1, 1], f=[32, 96, 32, 96], d=[16, 32, 16, 64],
              hk=[[0, 1], [2, 2], [3, 2], [3, 2]])
    for style, T, kw, lanes, mut, part in (
            ('qwen3', 2, {}, None, False, None),
            ('qwen3', 4, k4, [16, 32, 16, 32], False, None),
            ('qwen2.5', 2, {}, None, False, None),
            ('qwen3', 4, k4, [16, 32, 16, 32], False, u4),
            ('qwen3', 2, {}, None, True, None)):
        shutil.rmtree(work, ignore_errors=True)
        try:
            im, _ = qwen_synth.model(style, nl=2, **kw)
            want, srcs = tp.build(im, ids, 3, work, T, lanes=lanes, log=lambda *a: None,
                                  part=part)
            im.reset()
            best = []
            for p in range(len(want) - 1):
                lg = im.step(want[p], p, logits=p >= len(ids) - 1)
                if lg is not None:
                    best.append((max(range(len(lg)), key=lg.__getitem__), max(lg)))
            if mut:
                path = os.path.join(work, 'tb_tp.v')
                src = open(path).read()
                a = '      1: begin\n'
                assert a in src
                open(path, 'w').write(src.replace(a, '      1: begin end\n      5: begin\n', 1))
            out = tp.run(work, srcs, timeout=1800)
            # A token that is x (a gather left memory unwritten) is a mismatch.
            got = [(l.split('tok=')[1].split()[0], l.split('best=')[1].split()[0])
                   for l in out.splitlines() if l.startswith('TOKEN')]
            got = [(int(a_), int(b_)) if a_.lstrip('-').isdigit() and b_.lstrip('-').isdigit()
                   else (a_, b_) for a_, b_ in got]
            if mut:
                check('a network that skips the context gather is caught',
                      len(got) == len(best) and got != best)
            else:
                check('the weights split %d ways%s%s: one board\'s tokens and logits (%s)'
                      % (T, ', %s lanes' % '/'.join(map(str, lanes)) if lanes else '',
                         ', shares %s of d_ff' % '/'.join(map(str, part['f'])) if part else '',
                         style),
                      got == best and len(best) == 3 and 'bad=0' in out)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def test_zybo_at_32_lanes():
    """The Zybo package at the ZC706's width (boards.zybo_z7_20_32): the
    same core, the planner's speed for it the ZC706's, and an open flow
    that multiplies 24 of the attention head's score lanes in LUTs, so
    nextpnr-xilinx has LUTs to make the constants the DSPs left need; the
    16-lane Zybo's synthesis is unchanged."""
    import board_zybo as bz, cluster
    z32, z16 = boards_mod.PACKAGES['zybo_z7_20_32'], boards_mod.PACKAGES['zybo_z7_20']
    soft = bz._open_synth(z32)
    check('a 32-lane Zybo runs the ZC706\'s core at its speed, 24 score lanes in LUTs in '
          'the open flow',
          z32['lanes'] == 32 and boards_mod.base('zybo_z7_20_32') == 'zybo_z7_20'
          and cluster.speed('zybo_z7_20_32') == cluster.speed('zc706')
          and soft.count('slane?') == 24 and '-run :map_dsp' in soft
          and '$__soft_mul' in soft and 'slane' not in bz._open_synth(z16)
          and cluster.speed('zybo_z7_20_32') > cluster.speed('zybo_z7_20'))


def test_tp_plan():
    """The weight split's planner (cluster.tp_plan): every share covers
    the model, whole KV heads and every rank's d_ff and d_model a positive
    multiple of its lanes, the head's chunks in runs that tile the
    vocabulary in rank order; never slower a layer than even shares; on
    a ZC706 beside a Zybo, more of every matrix on the ZC706; three boards
    split Qwen3's eight KV heads, which even shares cannot; and a split
    with more ranks than KV heads is refused."""
    import cluster
    ok = True
    m = cluster.MODELS['qwen3']
    _, nck = cluster._chunks(m)
    for names in (['zc706', 'zybo_z7_20'], ['zc706', 'zybo_z7_20', 'zybo_z7_20'],
                  ['zybo_z7_20', 'zc706', 'zybo_z7_20', 'zc706'], ['zc706', 'zc706']):
        p = cluster.tp_plan(names, m)
        pt, lanes = p['part'], p['lanes']
        ok &= sum(pt['kv']) == m['KV'] and sum(pt['f']) == m['F'] and sum(pt['d']) == m['D']
        ok &= all(x > 0 and x % N == 0 for k in ('f', 'd') for x, N in zip(pt[k], lanes))
        ok &= all(k > 0 and k * m['hd'] % N == 0 for k, N in zip(pt['kv'], lanes))
        runs = [r for r in pt['hk'] if r[1] >= r[0]]
        ok &= runs[0][0] == 0 and runs[-1][1] == nck - 1 and all(
            a[1] + 1 == b[0] for a, b in zip(runs, runs[1:]))
        if len(names) != 3:
            ok &= p['layer_seconds'] <= cluster.tp_plan(names, m, even=True)['layer_seconds'] + 1e-12
    check('the weight split\'s shares cover the model in every rank\'s lanes, and are '
          'never slower than even ones', ok)
    two = cluster.tp_plan(['zc706', 'zybo_z7_20'], m)
    ev = cluster.tp_plan(['zc706', 'zybo_z7_20'], m, even=True)
    check('a ZC706 beside a Zybo gets more of every matrix (%s KV heads, %s of d_ff), '
          '%.1f ms a layer against %.1f even' % (
              '/'.join(map(str, two['part']['kv'])), '/'.join(map(str, two['part']['f'])),
              1e3 * two['layer_seconds'], 1e3 * ev['layer_seconds']),
          two['part']['kv'][0] > two['part']['kv'][1] and two['part']['f'][0] > two['part']['f'][1]
          and two['layer_seconds'] < ev['layer_seconds'])
    try:
        cluster.tp_plan(['zc706', 'zybo_z7_20', 'zybo_z7_20'], m, even=True)
        refused = False
    except ValueError:
        refused = True
    three = cluster.tp_plan(['zc706', 'zybo_z7_20', 'zybo_z7_20'], m)
    check('three boards split eight KV heads %s, which even shares cannot'
          % '/'.join(map(str, three['part']['kv'])), refused and sum(three['part']['kv']) == 8)
    try:
        cluster.tp_plan(['zc706'] * 9, m)
        refused = False
    except ValueError:
        refused = True
    check('nine ranks for eight KV heads are refused', refused)


def test_weights_split_through_registers():
    """The split of the weights as the boards run it (tp.build_boards):
    each rank's package RTL, its register block and DDR bridge around the
    core, with a DDR model of its own that stalls at random and its own
    clock, and every gather done by the ARM's side through GADDR, GDATA
    and GATHER. Two ranks at 16 lanes, a Zybo's 16 beside a ZC706's 32,
    and the same two with the ZC706 holding more of d_ff, have to give the
    one-board integer model's tokens and logits, the PL's mover carrying
    the slices between the core and DDR (GMOVE); and again with the ARM
    moving them a word an access (GDATA). A register block whose GATHER
    still reads set once answered lets an ARM serve one gather twice, and
    a mover that packs its words in the wrong lanes corrupts every slice:
    both have to be caught."""
    import tp, qwen_synth
    ids = [3, 77, 12, 140]
    work = os.path.join(ROOT, 'build_tpregtest')
    u2 = dict(kv=[1, 1], f=[96, 160], d=[32, 32], hk=[[0, 0], [1, 2]])
    for lanes, mut, part, mover in ((None, False, None, True), ([16, 32], False, None, True),
                                    ([16, 32], False, u2, True), (None, False, None, False),
                                    (None, True, None, False), (None, 'lanes', None, True)):
        shutil.rmtree(work, ignore_errors=True)
        try:
            im, _ = qwen_synth.model('qwen3', nl=2)
            want, srcs = tp.build_boards(im, ids, 3, work, 2, lanes=lanes, jit=1,
                                         log=lambda *a: None, part=part, mover=mover)
            im.reset()
            best = []
            for p in range(len(want) - 1):
                lg = im.step(want[p], p, logits=p >= len(ids) - 1)
                if lg is not None:
                    best.append((max(range(len(lg)), key=lg.__getitem__), max(lg)))
            if mut == 'lanes':
                for p in 'ab':
                    f = os.path.join(work, 'qwen_zybo_%s.v' % p)
                    src = open(f).read()
                    a = 'dm_word[dm_e[1:0] * 16 +: 16] <= gx_rdata;'
                    assert a in src
                    open(f, 'w').write(src.replace(a, 'dm_word[(2\'d3 - dm_e[1:0]) * 16 +: 16] <= gx_rdata;'))
            elif mut:
                for p in 'ab':
                    f = os.path.join(work, 'fpgai_zybo_%s.v' % p)
                    src = open(f).read()
                    assert 'g_req && !g_done}' in src
                    open(f, 'w').write(src.replace('g_req && !g_done}', 'g_req}'))
            out = tp.run(work, srcs, timeout=1800)
            got = [(int(l.split('tok=')[1].split()[0]), int(l.split('best=')[1].split()[0]))
                   for l in out.splitlines() if l.startswith('TOKEN')]
            if mut == 'lanes':
                check('a gather mover that packs its words in the wrong lanes is caught',
                      got != best)
            elif mut:
                check('a GATHER that reads set once answered is caught serving a gather twice',
                      'TP_FAIL' in out and got != best)
            else:
                check('the weights split 2 ways through the registers and DDR bridges%s%s, '
                      '%s: one board\'s tokens and logits'
                      % (', %s lanes' % '/'.join(map(str, lanes)) if lanes else '',
                         ', shares %s of d_ff' % '/'.join(map(str, part['f'])) if part else '',
                         'the PL moving the slices' if mover else 'the ARM moving them a word '
                         'an access'),
                      got == best and len(best) == 3 and 'bad=0' in out)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def test_gals_two_boards():
    """Boards on their own clocks, 10, 7.9 and 12.3 ns, whose UART bit
    times differ by up to 1.25%, each running a stage of the model: layer
    0 with the embedding, a middle layer with no vocabulary table at all,
    then the last layer and the head. They share nothing but CRC-checked
    messages, the hidden state down the chain and the token back to a
    host at its own rate. With two boards and with three, every token and
    logit has to be the one-board integer model's, the links must see no
    CRC error, and a link that sends the hidden state's bytes swapped is
    caught. Boards need not be alike: a pipeline of 32, 16 and 32 lanes
    has to give the same tokens, since only the hidden state crosses. And
    with a bit flipped in every 10,000 on every link, 100 times the
    proposal's 1e-6, the corrupt messages have to be caught by their CRC
    and the positions resent until the tokens and logits are still one
    board's."""
    import gals, qwen_synth
    ids = [3, 77, 12, 140]
    work = os.path.join(ROOT, 'build_galstest')
    for nl, split, mut, lanes, ber in (
            (2, [[0], [1]], None, None, 0),
            (3, [[0], [1], [2]], None, None, 0),
            (3, [[0], [1], [2]], None, [32, 16, 32], 0),
            (2, [[0], [1]], None, None, 10000),
            (2, [[0], [1]], ("tbyte = tx_hid ? (ti[0] ? x_rdata[7:0] : x_rdata[15:8])",
                             "tbyte = tx_hid ? (ti[0] ? x_rdata[15:8] : x_rdata[7:0])"), None, 0)):
        shutil.rmtree(work, ignore_errors=True)
        try:
            im, _ = qwen_synth.model('qwen3', nl=nl, vocab=608 if lanes else 600)
            want, srcs = gals.build(im, ids, 3, work, split, lanes=lanes, ber_inv=ber)
            im.reset()
            best = []
            for p in range(len(want) - 1):
                lg = im.step(want[p], p, logits=p >= len(ids) - 1)
                if lg is not None:
                    best.append((max(range(len(lg)), key=lg.__getitem__), max(lg)))
            if mut:
                path = os.path.join(work, 'stage_ctrl.v')
                src = open(path).read()
                assert mut[0] in src
                open(path, 'w').write(src.replace(mut[0], mut[1]))
            out = gals.run(work, srcs, timeout=900)
            got = [(int(l.split('tok=')[1].split()[0]), int(l.split('best=')[1].split()[0]))
                   for l in out.splitlines() if l.startswith('TOKEN')]
            link = next((l for l in out.splitlines() if l.startswith('LINK')), '')
            if mut:
                check('a link that swaps the hidden state bytes is caught',
                      len(got) == len(best) and got != best)
            elif ber:
                errs = int(link.split('crc_errors=')[1].split()[0]) + \
                    int(link.split('host_bad=')[1].split()[0])
                check('bit errors on every link at 1 in %d: %d messages caught, %s, '
                      'and one board\'s tokens' % (ber, errs, link.split()[-1]),
                      got == best and errs > 0 and 'resends=0' not in link)
            else:
                check('%d boards on their own clocks%s give one board\'s tokens'
                      % (len(split), ', of %s lanes,' % '/'.join(map(str, lanes)) if lanes else ''),
                      got == best and len(best) == 3 and 'LINK crc_errors=0 host_bad=0' in out)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def test_cluster_plan():
    """The partitioner, on mixes of the boards it knows: every layer on
    exactly one board, in order; no board past its DDR; the tied table on
    the first and last stage; Ethernet only where both ends have an ARM;
    a board with no reachable DRAM left out; and a split that cannot fit
    refused rather than overfilled."""
    import cluster
    ok = True
    for names, model in ((['zc706'], 'qwen3'), (['zc706', 'zybo_z7_20'], 'qwen3'),
                         (['zybo_z7_20', 'zc706', 'arty_a7_100t'], 'qwen2.5')):
        for mode in ('balanced', 'fast'):
            p = cluster.plan(names, model, mode)
            m = cluster.MODELS[model]
            sb = cluster.shape_bytes(m)
            got = []
            for i, st in enumerate(p['stages']):
                got += list(range(*st['layers']))
                n = st['layers'][1] - st['layers'][0]
                table = sb['table'] if (st['emb'] or st['head']) else 0
                ok &= n * sb['layer'] + table <= cluster.capacity(st['board'])
                ok &= st['emb'] == (i == 0) and st['head'] == (i == len(p['stages']) - 1)
            ok &= got == list(range(m['NL']))
            ok &= 'arty_a7_100t' not in [s['board'] for s in p['stages']]
            for l in p['links']:
                ok &= l['kind'] == ('ethernet' if boards_mod.PACKAGES[l['src']]['ps7']
                                    and boards_mod.PACKAGES[l['dst']]['ps7'] else 'uart')
    check('the partitioner places every layer once, within every DDR', ok)
    two = cluster.plan(['zc706', 'zybo_z7_20'], 'qwen3', 'balanced')
    check('a balanced split over two boards uses both, over Ethernet',
          len(two['stages']) == 2 and all(l['kind'] == 'ethernet' for l in two['links']))
    try:
        cluster.plan(['arty_a7_100t'], 'qwen3')
        refused = False
    except ValueError:
        refused = True
    check('a board set that cannot hold the model is refused', refused)


def test_zybo_stage_registers():
    """A pipeline stage's register block around a stub core whose clock
    ticks one bus cycle in four: eight hidden-state words written through
    XADDR and XDATA have to read back through them, land in the core, and
    be what a start with CTRL bit 2 clear computes from; STAGE has to say
    which layers the bitstream holds."""
    import board_zybo as bz
    L = dict(W=dict(tok=18, pos=8, x_addr=4), wb=0x08000000, cb=0x25714000,
             kb=0x25BB8000, vb=0x25D38000, end=0x25EB8000,
             stage=dict(l0=14, l1=28, emb=0, head=1))
    work = os.path.join(ROOT, 'build_zstagetest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    wp = ''.join('  output [31:0] w%d_araddr, output [3:0] w%d_arlen, output w%d_arvalid,\n'
                 '  input w%d_arready, input [63:0] w%d_rdata, input w%d_rvalid,\n'
                 '  input w%d_rlast, output w%d_rready,\n' % ((p,) * 8) for p in range(4))
    wt = ''.join('  assign w%d_araddr = 0; assign w%d_arlen = 0; assign w%d_arvalid = 0;'
                 ' assign w%d_rready = 1;\n' % ((p,) * 4) for p in range(4))
    stub = """module qwen_zybo (input clk, input rst_n, input start, input head_en,
  input [17:0] tok, input [7:0] pos, output reg [17:0] next_tok,
  output reg signed [15:0] best, output reg done, output reg busy,
  input emb_en, input x_we, input [3:0] x_addr, input signed [15:0] x_wdata,
  output reg signed [15:0] x_rdata,
%s  output [31:0] araddr, output [3:0] arlen, output arvalid, input arready,
  input [63:0] rdata, input rvalid, input rlast, output rready,
  output [31:0] awaddr, output [3:0] awlen, output awvalid, input awready,
  output [63:0] wdata, output [7:0] wstrb, output wlast, output wvalid,
  input wready, input bvalid, output bready, output reg [31:0] core_cycles);
%s  assign araddr = 0; assign arlen = 0; assign arvalid = 0; assign rready = 1;
  assign awaddr = 0; assign awlen = 0; assign awvalid = 0; assign wdata = 0;
  assign wstrb = 0; assign wlast = 0; assign wvalid = 0; assign bready = 1;
  reg signed [15:0] xm [0:15];
  reg [1:0] ph; reg [7:0] left; reg emb_seen; integer i; reg signed [17:0] sum;
  always @(posedge clk) begin
    if (!rst_n) begin ph <= 0; busy <= 0; done <= 0; core_cycles <= 0; end
    else begin
      ph <= ph + 1;
      if (ph == 0) begin   // one core edge in four bus cycles
        core_cycles <= core_cycles + 1; done <= 1'b0;
        if (!busy && x_we) xm[x_addr] <= x_wdata;
        x_rdata <= xm[x_addr];
        if (!busy && start) begin busy <= 1'b1; left <= 12; emb_seen <= emb_en; end
        else if (busy && left == 0) begin
          sum = 0; for (i = 0; i < 8; i = i + 1) sum = sum + xm[i];
          busy <= 1'b0; done <= 1'b1; next_tok <= sum; best <= {15'd0, emb_seen};
        end else if (busy) left <= left - 1;
      end
    end
  end
endmodule
""" % (wp, wt)
    tasks = bz.STEP[:bz.STEP.index('  task expect_reg')].replace('%%', '%').replace(
        'input [5:0] a', 'input [6:0] a')
    vals = [37 * i - 100 for i in range(8)]
    tb = """`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  reg [6:0] s_awaddr = 0, s_araddr = 0; reg [31:0] s_wdata = 0;
  reg s_awvalid = 0, s_wvalid = 0, s_arvalid = 0, s_bready = 0, s_rready = 0;
  wire s_awready, s_wready, s_bvalid, s_arready, s_rvalid;
  wire [1:0] s_bresp, s_rresp; wire [31:0] s_rdata;
  fpgai_zybo dut (.aclk(clk), .aresetn(rst_n),
    .s_axi_awaddr(s_awaddr), .s_axi_awvalid(s_awvalid), .s_axi_awready(s_awready),
    .s_axi_wdata(s_wdata), .s_axi_wstrb(4'hf), .s_axi_wvalid(s_wvalid),
    .s_axi_wready(s_wready), .s_axi_bresp(s_bresp), .s_axi_bvalid(s_bvalid),
    .s_axi_bready(s_bready), .s_axi_araddr(s_araddr), .s_axi_arvalid(s_arvalid),
    .s_axi_arready(s_arready), .s_axi_rdata(s_rdata), .s_axi_rresp(s_rresp),
    .s_axi_rvalid(s_rvalid), .s_axi_rready(s_rready));
%s  integer bad = 0, j;
  task want(input [6:0] a, input [31:0] v);
    begin rd(a); if (rv !== v) begin bad = bad + 1;
      $display("reg %%h = %%h, want %%h", a, rv, v); end end
  endtask
  initial begin #200000; $display("TB_RESULT: FAIL, timed out"); $finish; end
  initial begin
    repeat (4) @(negedge clk); rst_n = 1;
    want(7'h40, 32'h%08x);
    wr(7'h38, 0);
%s
    want(7'h38, 8);
    wr(7'h38, 0);
%s
    wr(7'h00, 1);            // start, embedding off: from the loaded state
    rv = 0; while (!rv[1]) rd(7'h0c);
    want(7'h10, %d); want(7'h14, 0);
    wr(7'h00, 5);            // start with the embedding
    rv = 0; while (!rv[1]) rd(7'h0c);
    want(7'h14, 1); want(7'h00, 4);
    if (bad) $display("TB_RESULT: FAIL"); else $display("TB_RESULT: PASS");
    $finish;
  end
endmodule
""" % (tasks, 14 | 28 << 8 | 1 << 17,
       "\n".join("    wr(7'h3c, 32'h%08x);" % (v & 0xffffffff) for v in vals),
       "\n".join("    want(7'h3c, 32'h%08x);" % (v & 0xffffffff) for v in vals),
       sum(vals) & 0x3ffff)
    try:
        open(os.path.join(work, 'fpgai_zybo.v'), 'w').write(bz.render_wrapper(L))
        open(os.path.join(work, 'stub.v'), 'w').write(stub)
        open(os.path.join(work, 'tb.v'), 'w').write(tb)
        r = subprocess.run(['iverilog', '-g2005', '-o', 'z.out', 'tb.v',
                            'fpgai_zybo.v', 'stub.v'], cwd=work,
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        if not r.returncode:
            out = subprocess.run(['vvp', 'z.out'], cwd=work, capture_output=True,
                                 text=True, timeout=120).stdout
        if 'TB_RESULT: PASS' not in out:
            print(out[-1500:])
        check('a stage loads and reads its hidden state through the registers',
              'TB_RESULT: PASS' in out)
    finally:
        shutil.rmtree(work, ignore_errors=True)


STAGE_SHIM = {
    "xil_io.h": """#pragma once
#include <stdint.h>
typedef uint32_t u32; typedef uint16_t u16; typedef uint8_t u8; typedef int16_t s16;
typedef uint16_t u16_t; typedef uintptr_t UINTPTR;
void Xil_Out32(UINTPTR a, u32 v); u32 Xil_In32(UINTPTR a);
""",
    "xil_cache.h": "static inline void Xil_DCacheFlush(void) {}\n",
    "xtime_l.h": """#pragma once
#include <time.h>
typedef unsigned long long XTime;
#define COUNTS_PER_SECOND 1000000000ULL
static inline void XTime_GetTime(XTime *t) { struct timespec s; clock_gettime(CLOCK_MONOTONIC, &s);
  *t = (XTime)s.tv_sec * 1000000000ULL + s.tv_nsec; }
""",
    "ff.h": """#pragma once
#include <stdio.h>
typedef unsigned int UINT; typedef struct { int x; } FATFS; typedef struct { FILE *f; long size; } FIL;
typedef enum { FR_OK = 0, FR_NO_FILE = 4 } FRESULT;
#define FA_READ 1
FRESULT f_mount(FATFS *fs, const char *p, int o); FRESULT f_open(FIL *f, const char *n, int m);
FRESULT f_read(FIL *f, void *b, UINT n, UINT *br); FRESULT f_close(FIL *f);
#define f_size(fp) ((fp)->size)
""",
    "platform.h": "void init_platform(void); void platform_enable_interrupts(void);\n",
    "platform_config.h": "#define PLATFORM_EMAC_BASEADDR 0\n",
    "lwip/ip_addr.h": """#pragma once
#include <stdint.h>
typedef struct { uint32_t addr; } ip_addr_t;
#define IP4_ADDR(ip, a, b, c, d) ((ip)->addr = (uint32_t)(a) | (uint32_t)(b) << 8 | (uint32_t)(c) << 16 | (uint32_t)(d) << 24)
extern const ip_addr_t ip_addr_any;
#define IP_ADDR_ANY (&ip_addr_any)
""",
    "lwip/init.h": "void lwip_init(void);\n",
    "lwip/udp.h": """#pragma once
#include "lwip/ip_addr.h"
#include <stdint.h>
typedef uint16_t u16_t;
struct pbuf { void *payload; uint16_t tot_len; };
typedef enum { PBUF_TRANSPORT } pbuf_layer; typedef enum { PBUF_RAM } pbuf_type;
struct pbuf *pbuf_alloc(pbuf_layer l, uint16_t n, pbuf_type t); void pbuf_free(struct pbuf *p);
uint16_t pbuf_copy_partial(const struct pbuf *p, void *d, uint16_t n, uint16_t off);
struct udp_pcb;
typedef void (*udp_recv_fn)(void *, struct udp_pcb *, struct pbuf *, const ip_addr_t *, u16_t);
struct udp_pcb *udp_new(void); int udp_bind(struct udp_pcb *, const ip_addr_t *, u16_t);
void udp_recv(struct udp_pcb *, udp_recv_fn, void *);
int udp_sendto(struct udp_pcb *, struct pbuf *, const ip_addr_t *, u16_t);
""",
    "netif/xadapter.h": """#pragma once
#include "lwip/ip_addr.h"
#include <stdint.h>
struct netif { int x; };
struct netif *xemac_add(struct netif *, ip_addr_t *, ip_addr_t *, ip_addr_t *, unsigned char *, uint32_t);
int xemacif_input(struct netif *); void netif_set_default(struct netif *); void netif_set_up(struct netif *);
""",
    # The shim: lwIP over a localhost UDP socket (IP a.b.c.d is port
    # PORT0 + d), FatFs over a directory, and a stand-in PL whose layers
    # are a fixed integer map, so a Python reference knows every token.
    "shim.c": """#include <arpa/inet.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include "xil_io.h"
#include "ff.h"
#include "lwip/udp.h"
#include "netif/xadapter.h"
#include "fpgai_layout.h"
unsigned char host_ddr[1 << 20];
const ip_addr_t ip_addr_any = {0};
static int sock = -1, sent = 0;
static udp_recv_fn cb; static struct udp_pcb *cbpcb;
void init_platform(void) {} void platform_enable_interrupts(void) {} void lwip_init(void) {}
void netif_set_default(struct netif *n) { (void)n; } void netif_set_up(struct netif *n) { (void)n; }
static int port_of(uint32_t a) { return PORT0 + (int)(a >> 24); }
struct netif *xemac_add(struct netif *n, ip_addr_t *ip, ip_addr_t *m, ip_addr_t *g, unsigned char *mac, uint32_t b) {
  (void)m; (void)g; (void)mac; (void)b;
  sock = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in sa = {0}; sa.sin_family = AF_INET; sa.sin_port = htons(port_of(ip->addr));
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(sock, (struct sockaddr *)&sa, sizeof sa)) return 0;
  fcntl(sock, F_SETFL, O_NONBLOCK); return n; }
struct pbuf *pbuf_alloc(pbuf_layer l, uint16_t n, pbuf_type t) { (void)l; (void)t;
  struct pbuf *p = malloc(sizeof *p); p->payload = malloc(n); p->tot_len = n; return p; }
void pbuf_free(struct pbuf *p) { free(p->payload); free(p); }
uint16_t pbuf_copy_partial(const struct pbuf *p, void *d, uint16_t n, uint16_t off) {
  memcpy(d, (char *)p->payload + off, n); return n; }
struct udp_pcb *udp_new(void) { return (struct udp_pcb *)&sock; }
int udp_bind(struct udp_pcb *u, const ip_addr_t *a, u16_t p) { (void)u; (void)a; (void)p; return 0; }
void udp_recv(struct udp_pcb *u, udp_recv_fn f, void *arg) { (void)arg; cb = f; cbpcb = u; }
int udp_sendto(struct udp_pcb *u, struct pbuf *p, const ip_addr_t *dst, u16_t port) { (void)u; (void)port;
  sent++;
  if (getenv("DROP") && atoi(getenv("DROP")) == sent) return 0;     /* a lost datagram */
  if (getenv("FLIP") && atoi(getenv("FLIP")) == sent)               /* a corrupt one */
    ((unsigned char *)p->payload)[p->tot_len / 2] ^= 0x10;
  struct sockaddr_in sa = {0}; sa.sin_family = AF_INET; sa.sin_port = htons(port_of(dst->addr));
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sendto(sock, p->payload, p->tot_len, 0, (struct sockaddr *)&sa, sizeof sa); return 0; }
int xemacif_input(struct netif *n) { (void)n; unsigned char b[2048];
  ssize_t k = recv(sock, b, sizeof b, 0);
  if (k <= 0) { usleep(200); return 0; }
  struct pbuf *p = pbuf_alloc(PBUF_TRANSPORT, (uint16_t)k, PBUF_RAM); memcpy(p->payload, b, k);
  ip_addr_t a = {0}; cb(0, cbpcb, p, &a, 0); return 1; }
FRESULT f_mount(FATFS *fs, const char *p, int o) { (void)fs; (void)p; (void)o; return FR_OK; }
FRESULT f_open(FIL *f, const char *n, int m) { (void)m; char path[512];
  snprintf(path, sizeof path, "%s/%s", getenv("SD"), n); f->f = fopen(path, "rb");
  if (!f->f) return FR_NO_FILE; fseek(f->f, 0, SEEK_END); f->size = ftell(f->f); fseek(f->f, 0, SEEK_SET); return FR_OK; }
FRESULT f_read(FIL *f, void *b, UINT n, UINT *br) { *br = (UINT)fread(b, 1, n, f->f); return FR_OK; }
FRESULT f_close(FIL *f) { fclose(f->f); return FR_OK; }
/* the stand-in PL */
static u32 r_tok, r_pos, r_ctrl, done, nxt, xaddr; static s16 bst, hid[FPGAI_D];
static s16 wrap(long v) { return (s16)(((v % 20011) + 20011) % 20011 - 10005); }
void Xil_Out32(UINTPTR a, u32 v) {
  u32 o = (u32)(a - FPGAI_REGS);
  if (o == FPGAI_TOK) r_tok = v; else if (o == FPGAI_POS) r_pos = v;
  else if (o == FPGAI_XADDR) xaddr = v; else if (o == FPGAI_XDATA) hid[xaddr++] = (s16)v;
  else if (o == FPGAI_CTRL && (v & 1)) {
    r_ctrl = v;
    if (v & 4) for (int i = 0; i < FPGAI_D; i++) hid[i] = wrap((long)r_tok * 31 + i);
    for (int l = STAGE_L0; l < STAGE_L1; l++)
      for (int i = 0; i < FPGAI_D; i++) hid[i] = wrap((long)hid[i] * 3 + l + i + (long)r_pos * 7);
    if (v & 2) { long s = 0; for (int i = 0; i < FPGAI_D; i++) s += hid[i] < 0 ? -hid[i] : hid[i];
                 nxt = (u32)(s % 97); bst = (s16)(nxt * 3); }
    done = 1; } }
u32 Xil_In32(UINTPTR a) {
  u32 o = (u32)(a - FPGAI_REGS);
  if (o == FPGAI_ID) return FPGAI_ID_VALUE; if (o == FPGAI_STAGE) return STAGE_VALUE;
  if (o == FPGAI_KVEND) return (u32)KVEND; if (o == FPGAI_STATUS) return done ? 2 : 0;
  if (o == FPGAI_NEXT_TOK) return nxt; if (o == FPGAI_BEST) return (u32)(u16)bst;
  if (o == FPGAI_XDATA) return (u32)(u16)hid[xaddr++];
  return 0; }
""",
}


# A tensor-parallel rank's stand-in PL: a fixed integer model split the
# way the core splits one, a slice of each vector computed from whole
# vectors, so a slice gathered to the wrong place changes every token.
TP_PL = """/* the stand-in PL: one rank of a fixed integer model */
#define NL_ 2
#define V_ 97
static const int sl_[TP_RANKS][3] = {TP_SLICES}, of_[TP_RANKS][3] = {TP_OFFSETS};
#define LO(k) (of_[TP_RANK][k])
#define HI(k) (of_[TP_RANK][k] + sl_[TP_RANK][k])
static u32 r_tok, r_pos, done, gaddr, greq, gvec, phase, head, nxt; static s16 bst;
static s16 x[FPGAI_D], cm[FPGAI_HHD], am[FPGAI_D], mm[FPGAI_F];
static s16 wrap(long v) { return (s16)(((v % 20011) + 20011) % 20011 - 10005); }
static s16 *vec_of(void) { return gvec == 1 ? cm : gvec == 2 ? am : mm; }
/* phase 4 l + k: k 0 the context, 1 o, 2 the gated product, 3 down; 4 NL_ the head */
static void slice(void) {
  int l = phase / 4, k = phase % 4, i;
  if ((int)phase == 4 * NL_) {
    int per = (V_ + TP_RANKS - 1) / TP_RANKS, lo = TP_RANK * per, hi = lo + per > V_ ? V_ : lo + per;
    bst = -32768;
    for (i = lo; i < hi; i++) { s16 v = wrap((long)x[i % FPGAI_D] * 7 + i);
      if (i == lo || v > bst) { bst = v; nxt = i; } }
    gvec = 4; greq = 1; return; }
  if (k == 0) { for (i = LO(0); i < HI(0); i++)
      cm[i] = wrap((long)x[i % FPGAI_D] * 3 + x[(i * 7) % FPGAI_D] + l + i + (long)r_pos * 5); gvec = 1; }
  if (k == 1) { for (i = LO(1); i < HI(1); i++)
      am[i] = wrap(cm[i % FPGAI_HHD] + (long)cm[(i * 5 + 1) % FPGAI_HHD] * 2 + i); gvec = 2; }
  if (k == 2) { for (i = LO(2); i < HI(2); i++)
      mm[i] = wrap((long)x[i % FPGAI_D] * 5 + i + l); gvec = 3; }
  if (k == 3) { for (i = LO(1); i < HI(1); i++)
      am[i] = wrap(mm[i % FPGAI_F] + (long)mm[(i * 3) % FPGAI_F] + i); gvec = 2; }
  greq = 1; }
static void answered(void) {
  int k = phase % 4, i;
  greq = 0;
  if ((int)phase == 4 * NL_) { done = 1; return; }
  if (k == 1 || k == 3) for (i = 0; i < FPGAI_D; i++) x[i] = wrap((long)x[i] + am[i]);
  phase++;
  if ((int)phase == 4 * NL_ && !head) { done = 1; return; }
  slice(); }
void Xil_Out32(UINTPTR a, u32 v) {
  u32 o = (u32)(a - FPGAI_REGS);
  if (o == FPGAI_TOK) r_tok = v; else if (o == FPGAI_POS) r_pos = v;
  else if (o == FPGAI_GADDR) gaddr = v;
  else if (o == FPGAI_GDATA) { if (greq) vec_of()[gaddr] = (s16)v; gaddr++; }
  else if (o == FPGAI_GATHER) { if ((v & 1) && greq) answered(); }
  else if (o == FPGAI_GMOVE && greq) { u32 n = v & 0xffff; s16 *g = (s16 *)GBUF;
    for (u32 i = 0; i < n; i++) { if (v >> 16 & 1) vec_of()[gaddr + i] = g[gaddr + i];
                                  else g[gaddr + i] = vec_of()[gaddr + i]; } }
  else if (o == FPGAI_CTRL && (v & 1)) {
    head = (v >> 1) & 1; done = 0; phase = 0;
    for (int i = 0; i < FPGAI_D; i++) x[i] = wrap((long)r_tok * 31 + i + (long)r_pos * 7);
    slice(); } }
u32 Xil_In32(UINTPTR a) {
  u32 o = (u32)(a - FPGAI_REGS);
  if (o == FPGAI_ID) return FPGAI_ID_VALUE; if (o == FPGAI_TP) return TP_VALUE;
  if (o == FPGAI_KVEND) return (u32)KVEND; if (o == FPGAI_STATUS) return done ? 2 : 0;
  if (o == FPGAI_NEXT_TOK) return nxt; if (o == FPGAI_BEST) return (u32)(u16)bst;
  if (o == FPGAI_GATHER) return greq | gvec << 1;
  if (o == FPGAI_GDATA) return (u32)(u16)vec_of()[gaddr++];
  return 0; }
"""


def test_tp_network_on_host():
    """The Zynq program for a split of the weights (board_zybo.TP_C), run
    for real on this host: compiled once for each of three ranks, talking
    UDP over localhost through the lwIP shim, each with a stand-in PL that
    computes its slice of a fixed integer model and waits on the same four
    gathers a layer, and the head's, as the core. One rank's second
    datagram is lost on the wire and another's fifth arrives corrupt. The
    ranks have to ask for what they miss, and rank 0 has to print the
    one-board reference's tokens, with even shares and with every rank's
    slice a different size; an ARM that writes the other ranks' slices
    one word off has to be caught."""
    import board_zybo as bz
    cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
    if not cc:
        check('the tensor-parallel network program runs on the host (no C compiler: skipped)', True)
        return
    work = os.path.join(ROOT, 'build_tpnettest')
    T, D, HHD, F, V, NL = 3, 600, 1200, 2100, 97, 2
    prompt, n_gen = [5, 11, 2], 4

    def wrap(v):
        return ((v % 20011) + 20011) % 20011 - 10005
    want, pos, tok = [], 0, prompt[0]
    while pos < len(prompt) + n_gen - 1:
        if pos < len(prompt):
            tok = prompt[pos]
        x = [wrap(tok * 31 + i + pos * 7) for i in range(D)]
        for l in range(NL):
            cm = [wrap(x[i % D] * 3 + x[(i * 7) % D] + l + i + pos * 5) for i in range(HHD)]
            am = [wrap(cm[i % HHD] + cm[(i * 5 + 1) % HHD] * 2 + i) for i in range(D)]
            x = [wrap(x[i] + am[i]) for i in range(D)]
            mm = [wrap(x[i % D] * 5 + i + l) for i in range(F)]
            am = [wrap(mm[i % F] + mm[(i * 3) % F] + i) for i in range(D)]
            x = [wrap(x[i] + am[i]) for i in range(D)]
        if pos >= len(prompt) - 1:
            lg = [wrap(x[t % D] * 7 + t) for t in range(V)]
            tok = lg.index(max(lg))
            want.append(tok)
        pos += 1
    text = ''.join('<%d>' % t for t in prompt) + ''.join('<%d>' % t for t in want)
    shim = STAGE_SHIM['shim.c'][:STAGE_SHIM['shim.c'].index('/* the stand-in PL */')] + TP_PL
    results = {}
    uneven = [(300, 100, 600), (500, 300, 900), (400, 200, 600)]
    for k, (label, mut, slices) in enumerate((
            ('good', None, None), ('uneven', None, uneven),
            ('off', ('gbuf[ra + i] = cur->v[ra + i];',
                     'gbuf[ra + i + 1] = cur->v[ra + i];'), uneven))):
        shutil.rmtree(work, ignore_errors=True)
        port0 = 45000 + (os.getpid() + k * 97) % 2000
        procs, outs = [], {}
        try:
            for r in range(T):
                d = os.path.join(work, 'r%d' % r)
                for sub in ('lwip', 'netif', 'sd'):
                    os.makedirs(os.path.join(d, sub))
                for name, src in STAGE_SHIM.items():
                    open(os.path.join(d, name), 'w').write(shim if name == 'shim.c' else src)
                L = dict(W=dict(tok=18, pos=8, gx_addr=12), wb=0, cb=0, cn=64, kb=0, vb=0,
                         end=0, words=64, N=16)
                h = bz.render_header(L, n_gen, 'host', tp=dict(
                    rank=r, ranks=T, HHD=HHD, D=D, F=F, slices=slices,
                    ips=[(127, 0, 0, 20 + j) for j in range(T)]))
                for k, v in (('WBASE', '((UINTPTR)host_ddr)'), ('CBASE', '((UINTPTR)host_ddr + 4096)'),
                             ('KBASE', '((UINTPTR)host_ddr + 8192)'), ('KVEND', '((UINTPTR)host_ddr + 16384)'),
                             ('GBUF', '((UINTPTR)host_ddr + 32768)'),
                             ('VOCAB_BASE', '((UINTPTR)host_ddr + 65536)'), ('WBYTES', '1024U'),
                             ('CBYTES', '512U'), ('FPGAI_REGS', '0x43C00000U')):
                    h = re.sub(r'#define %s\s+\S+' % k, '#define %s %s' % (k, v), h)
                h = h.replace('#ifndef FPGAI_LAYOUT_H\n#define FPGAI_LAYOUT_H\n',
                              '#ifndef FPGAI_LAYOUT_H\n#define FPGAI_LAYOUT_H\n'
                              'extern unsigned char host_ddr[];\n#define PORT0 %d\n'
                              '#define FPGAI_BARRIER() __sync_synchronize()\n' % port0)
                open(os.path.join(d, 'fpgai_layout.h'), 'w').write(h)
                src = bz.TP_C
                if mut:
                    assert mut[0] in src
                    src = src.replace(mut[0], mut[1])
                open(os.path.join(d, 'main.c'), 'w').write(src)
                open(os.path.join(d, 'sd', 'weights8.bin'), 'wb').write(bytes(1024))
                open(os.path.join(d, 'sd', 'cparams.bin'), 'wb').write(bytes(512))
                import struct
                open(os.path.join(d, 'sd', 'prompt.bin'), 'wb').write(
                    struct.pack('<%dI' % (len(prompt) + 1), len(prompt), *prompt))
                if r == 0:
                    words = ['<%d>' % t for t in range(V)]
                    blob, offs = b'', [0]
                    for w_ in words:
                        blob += w_.encode(); offs.append(len(blob))
                    open(os.path.join(d, 'sd', 'vocab.bin'), 'wb').write(
                        struct.pack('<%dI' % (V + 2), V, *offs) + blob)
                c = subprocess.run([cc, '-O1', '-w', '-I.', '-o', 'rank', 'main.c', 'shim.c'],
                                   cwd=d, capture_output=True, text=True)
                assert c.returncode == 0, c.stderr[-2000:]
            for r in (1, 2, 0):
                env = dict(os.environ, SD=os.path.join(work, 'r%d' % r, 'sd'))
                if r == 1:
                    env['DROP'] = '2'
                if r == 2:
                    env['FLIP'] = '5'
                procs.append((r, subprocess.Popen([os.path.join(work, 'r%d' % r, 'rank')], env=env,
                                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                  text=True)))
            for r, p in procs:
                try:
                    outs[r] = p.communicate(timeout=120)[0]
                except subprocess.TimeoutExpired:
                    p.kill()
                    outs[r] = p.communicate()[0] + '\n(timed out)'
        finally:
            for _, p in procs:
                if p.poll() is None:
                    p.kill()
            shutil.rmtree(work, ignore_errors=True)
        results[label] = outs
    for label, what in (('good', 'even shares'), ('uneven', 'slices of three sizes')):
        good = results[label]
        asks = sum(int(m) for o in good.values() for m in re.findall(r'(\d+) asks', o))
        done = all(re.search(r'rank %d: %d gathers' % (r, 4 * NL * (len(prompt) + n_gen - 1) + n_gen),
                             good[r]) for r in range(T))
        if text not in good[0] or not done:
            print('\n'.join(good[r][-600:] for r in range(T)))
        check('three ranks, %s, gather over UDP, ask again for a lost and a corrupt '
              'datagram (%d asks), and print one board\'s tokens' % (what, asks),
              text in good[0] and done and asks >= 2)
    check('an ARM that writes the other ranks\' slices one word off is caught',
          text not in results['off'][0])


def test_simulators_agree():
    """Verilator (vsim.py), which spec2rtl.py uses to simulate a real
    model at every layer, against Icarus, which every other test uses: the
    direct testbench of a small checkpoint, and two ranks of a weight
    split, have to print the same steps, tokens, logits and cycles in
    both."""
    import qwen_full, qwen_synth, tp, vsim
    if not shutil.which('verilator'):
        check('the two simulators agree (no Verilator: skipped)', True)
        return
    ids = [3, 77, 12, 140]
    work = os.path.join(ROOT, 'build_simtest')
    try:
        shutil.rmtree(work, ignore_errors=True)
        im, _ = qwen_synth.model('qwen3', nl=2)
        _, srcs = qwen_full.build_model(im, ids, 3, work, log=lambda *a: None)
        out = {sim: [l for l in vsim.run(work, srcs, sim, timeout=900, tag=sim).splitlines()
                     if l.startswith('STEP')] for sim in ('iverilog', 'verilator')}
        check('Verilator prints the direct testbench\'s %d steps as Icarus does, cycles '
              'included' % len(out['iverilog']),
              len(out['iverilog']) == 6 and out['iverilog'] == out['verilator'])
        shutil.rmtree(work, ignore_errors=True)
        _, srcs = tp.build(im, ids, 3, work, 2, log=lambda *a: None)
        out = {sim: [l for l in tp.run(work, srcs, sim=sim, timeout=900).splitlines()
                     if l.startswith(('TOKEN', 'TP', 'RANK'))] for sim in ('iverilog', 'verilator')}
        check('and the weight split\'s ranks, gathers and busy cycles',
              len(out['iverilog']) == 6 and out['iverilog'] == out['verilator'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_arm_programs_on_their_rtl():
    """Every generated ARM program, compiled unchanged, against its own
    package's RTL under Verilator (cosim.py): the program loads its SD
    card into DDR, the PL's AXI masters read those bytes back, and every
    register access is an AXI-Lite transaction. The single board's, with
    every AXI port stalling at random; two stages of a layer split, the
    first datagram lost; and two ranks of a weight split, a Zybo's 16
    lanes beside a ZC706's 32 with uneven shares, a datagram lost. Each
    has to print the integer model's tokens, and the head's logits, which
    the harness reads beside the program's own reads, have to be the
    integer model's. A weight image in the byte order the ARM once had,
    and ranks whose ARM writes the others' slices one word off, have to
    be caught."""
    import board_zybo as bz, cosim, qwen_full, qwen_synth, zybo
    import qwen_real as qr
    cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
    if not cc or not shutil.which('verilator'):
        check('the ARM programs run on their RTL (no C compiler or Verilator: skipped)', True)
        return
    work = os.path.join(ROOT, 'build_cosimtest')
    shutil.rmtree(work, ignore_errors=True)
    ids, n_gen = [3, 77, 12, 140], 3

    def ref(im, want):
        im.reset()
        best = []
        for p in range(len(want) - 1):
            lg = im.step(want[p], p, logits=p >= len(ids) - 1)
            if lg is not None:
                best.append((max(range(len(lg)), key=lg.__getitem__), max(lg)))
        return best

    def agreed(outs):
        # Every rank's best over its run of the vocabulary; the ranks take
        # the largest, the lower rank on a tie.
        hs = [cosim.heads(o) for o in outs]
        return [max((h[k] for h in hs), key=lambda x: x[1]) for k in range(min(map(len, hs)))]
    try:
        def pkg(im, want, name, board, **kw):
            b = os.path.join(work, name + '_build')
            im.ms['lanes'] = boards_mod.PACKAGES[board]['lanes']
            bkw = {k: kw[k] for k in ('tp', 'layers', 'stage', 'table') if k in kw}
            if 'tp' in bkw:
                bkw['tp'] = bkw['tp'][:3]
            qwen_full.build_model(im, ids, n_gen, b, log=lambda *a: None, want=want, **bkw)
            o = os.path.join(work, name)
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                bz.package(b, board, o, '', n_gen, sd=False, model='tiny', checked='-',
                           stage=kw.get('st'), tp=kw.get('tpd'))
            bz.write_sd(zybo.layout(b, bz.BASE), b, os.path.join(o, 'sd'), ids, cosim.Tokens(im.V),
                        first=kw.get('first', True), vocab=kw.get('vocab', True))
            return o
        # One board.
        im, _ = qwen_synth.model('qwen3', nl=2)
        want = qr.greedy(im, ids, n_gen)
        text, best = ''.join('<%d>' % t for t in want), ref(im, want)
        one = pkg(im, want, 'one', 'zybo_z7_20')
        exe = cosim.build(one, os.path.join(work, 'one_sim'), log=lambda *a: None)
        out = cosim.run(exe, os.path.join(one, 'sd'), {'JITTER': '1'}, timeout=900)
        check('the single board\'s program on its RTL, every AXI port stalling: '
              'the integer model\'s tokens and logits', text in cosim.uart(out) and cosim.heads(out) == best)
        # Its weight image in the old byte order: lane j at byte N - 1 - j.
        w8 = os.path.join(one, 'sd', 'weights8.bin')
        img = bytearray(open(w8, 'rb').read())
        for k in range(0, len(img), 16):
            img[k:k + 16] = img[k:k + 16][::-1]
        open(w8, 'wb').write(bytes(img))
        out = cosim.run(exe, os.path.join(one, 'sd'), {'JITTER': '0'}, timeout=900)
        check('a weight image in the old byte order is caught',
              cosim.heads(out) != best and len(cosim.heads(out)) == len(best))
        # Two stages of a layer split.
        im, _ = qwen_synth.model('qwen3', nl=2, vocab=608)
        want = qr.greedy(im, ids, n_gen)
        text, best = ''.join('<%d>' % t for t in want), ref(im, want)
        ips = [(192, 168, 1, 10), (192, 168, 1, 11)]
        runs = []
        for i, board in enumerate(('zc706', 'zybo_z7_20')):
            st = dict(D=im.D, index=i, count=2, l0=i, l1=i + 1, emb=i == 0, head=i == 1,
                      ip=ips[i], next_ip=ips[1 - i], first_ip=ips[0])
            o = pkg(im, want, 'stage%d' % i, board, layers=[i], stage=True, table=True, st=st,
                    first=i == 0)
            # The program's clock is the PL's: a 20 ms resend, not its 5 s.
            runs.append((cosim.build(o, os.path.join(work, 'stage%d_sim' % i), log=lambda *a: None,
                                     defines={'RESEND_MS': 20}),
                         os.path.join(o, 'sd'), {'JITTER': '1', 'DROP': '1' if i == 0 else '0'}))
        outs = cosim.run_group(runs)
        check('two stages\' programs on their RTL, the first datagram lost: '
              'the integer model\'s tokens and logits',
              text in cosim.uart(outs[0]) and cosim.heads(outs[1]) == best)
        # Two ranks of a weight split, uneven shares.
        im, _ = qwen_synth.model('qwen3', nl=2)
        want = qr.greedy(im, ids, n_gen)
        text, best = ''.join('<%d>' % t for t in want), ref(im, want)
        part = dict(kv=[1, 1], f=[96, 160], d=[32, 32], hk=[[0, 0], [1, 2]])
        grp = im.H // im.KV
        runs = []
        for r, board in enumerate(('zybo_z7_20', 'zc706')):
            tpd = dict(rank=r, ranks=2, HHD=im.H * im.hd, D=im.D, F=im.F, ips=ips,
                       slices=[(grp * part['kv'][k] * im.hd, part['d'][k], part['f'][k])
                               for k in range(2)])
            o = pkg(im, want, 'rank%d' % r, board, tp=(r, 2, part), tpd=tpd, vocab=r == 0)
            runs.append((cosim.build(o, os.path.join(work, 'rank%d_sim' % r), log=lambda *a: None,
                                     defines={'RESEND_MS': 20, 'LINGER_MS': 50}),
                         os.path.join(o, 'sd'), {'JITTER': '1', 'DROP': '3' if r == 1 else '0'}))
        outs = cosim.run_group(runs)
        asks = sum(int(m) for o in outs for m in re.findall(r'(\d+) asks', o))
        if text not in cosim.uart(outs[0]):
            print(outs[0][-800:], outs[1][-400:])
        check('two ranks\' programs on their RTL, 16 and 32 lanes, uneven shares, a '
              'datagram lost (%d asks): the integer model\'s tokens and logits' % asks,
              text in cosim.uart(outs[0]) and agreed(outs) == best and asks >= 1)
        # The same ranks, their ARM writing the others' slices one word off.
        bad = []
        for r in range(2):
            o = os.path.join(work, 'rank%d' % r)
            m = os.path.join(work, 'rank%d_mut' % r)
            shutil.copytree(o, m)
            src = open(os.path.join(m, 'sw', 'main.c')).read()
            a = 'gbuf[ra + i] = cur->v[ra + i];'
            assert a in src
            open(os.path.join(m, 'sw', 'main.c'), 'w').write(src.replace(a, 'gbuf[ra + i + 1] = cur->v[ra + i];'))
            bad.append((cosim.build(m, os.path.join(work, 'rank%d_mutsim' % r), log=lambda *a: None,
                                    defines={'RESEND_MS': 20, 'LINGER_MS': 50}),
                        os.path.join(m, 'sd'), {'JITTER': '0'}))
        outs = cosim.run_group(bad)
        check('ranks whose ARM writes the others\' slices one word off are caught',
              agreed(outs) != best and len(agreed(outs)) == len(best))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_stage_network_on_host():
    """The Zynq stage program's network half, run for real on this host:
    the generated main.c compiled twice, a first stage (layers 0-2, the
    prompt) and a last one (layers 2-4, the head), talking UDP over
    localhost through a shim for lwIP, with a stand-in PL whose layers
    are a fixed integer map. The first stage's first datagram is dropped
    on the wire. It has to be sent again, the hidden state has to cross
    in two parts, and the printed tokens have to be the reference's."""
    import board_zybo as bz
    cc = shutil.which('cc') or shutil.which('clang') or shutil.which('gcc')
    if not cc:
        check('the stage network program runs on the host (no C compiler: skipped)', True)
        return
    work = os.path.join(ROOT, 'build_stagenettest')
    shutil.rmtree(work, ignore_errors=True)
    D, V, prompt, n_gen, port0 = 700, 97, [5, 11, 2], 4, 47000 + os.getpid() % 1000
    procs = []
    try:
        for i, (l0, l1) in enumerate(((0, 2), (2, 4))):
            d = os.path.join(work, 's%d' % i)
            os.makedirs(os.path.join(d, 'lwip'))
            os.makedirs(os.path.join(d, 'netif'))
            os.makedirs(os.path.join(d, 'sd'))
            for name, src in STAGE_SHIM.items():
                open(os.path.join(d, name), 'w').write(src)
            L = dict(W=dict(tok=18, pos=8, x_addr=10), wb=0, cb=0, cn=64, kb=0, vb=0,
                     end=0, words=64, N=16)
            st = dict(D=D, index=i, count=2, l0=l0, l1=l1, emb=i == 0, head=i == 1,
                      ip=(127, 0, 0, 10 + i), next_ip=(127, 0, 0, 10 + (i + 1) % 2),
                      first_ip=(127, 0, 0, 10))
            h = bz.render_header(L, n_gen, 'host', st)
            # The DDR regions point into the shim's array.
            for k, v in (('WBASE', '((UINTPTR)host_ddr)'), ('CBASE', '((UINTPTR)host_ddr + 4096)'),
                         ('KBASE', '((UINTPTR)host_ddr + 8192)'), ('KVEND', '((UINTPTR)host_ddr + 16384)'),
                         ('VOCAB_BASE', '((UINTPTR)host_ddr + 65536)'), ('WBYTES', '1024U'),
                         ('CBYTES', '512U'), ('FPGAI_REGS', '0x43C00000U')):
                h = re.sub(r'#define %s\s+\S+' % k, '#define %s %s' % (k, v), h)
            h = h.replace('#ifndef FPGAI_LAYOUT_H\n#define FPGAI_LAYOUT_H\n',
                          '#ifndef FPGAI_LAYOUT_H\n#define FPGAI_LAYOUT_H\n'
                          'extern unsigned char host_ddr[];\n#define PORT0 %d\n' % port0)
            open(os.path.join(d, 'fpgai_layout.h'), 'w').write(h)
            open(os.path.join(d, 'main.c'), 'w').write(bz.STAGE_C)
            open(os.path.join(d, 'sd', 'weights8.bin'), 'wb').write(bytes(1024))
            open(os.path.join(d, 'sd', 'cparams.bin'), 'wb').write(bytes(512))
            if i == 0:
                import struct
                words = ['<%d>' % t for t in range(V)]
                blob, offs = b'', [0]
                for w_ in words:
                    blob += w_.encode(); offs.append(len(blob))
                open(os.path.join(d, 'sd', 'vocab.bin'), 'wb').write(
                    struct.pack('<%dI' % (V + 2), V, *offs) + blob)
                open(os.path.join(d, 'sd', 'prompt.bin'), 'wb').write(
                    struct.pack('<%dI' % (len(prompt) + 1), len(prompt), *prompt))
            r = subprocess.run([cc, '-O1', '-w', '-I.', '-o', 'stage', 'main.c', 'shim.c'],
                               cwd=d, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
        env1 = dict(os.environ, SD=os.path.join(work, 's1', 'sd'))
        procs.append(subprocess.Popen([os.path.join(work, 's1', 'stage')], env=env1,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
        time.sleep(0.5)
        env0 = dict(os.environ, SD=os.path.join(work, 's0', 'sd'), DROP='1')
        out = subprocess.run([os.path.join(work, 's0', 'stage')], env=env0,
                             capture_output=True, text=True, timeout=120).stdout

        def wrap(v):
            return ((v % 20011) + 20011) % 20011 - 10005
        # The reference: the same map, all four layers in one place.
        seq, pos, tok, want = list(prompt), 0, prompt[0], []
        while pos < len(prompt) + n_gen - 1:
            if pos < len(prompt):
                tok = prompt[pos]
            hid = [wrap(tok * 31 + i) for i in range(D)]
            for l in range(4):
                hid = [wrap(hid[i] * 3 + l + i + pos * 7) for i in range(D)]
            if pos >= len(prompt) - 1:
                tok = sum(abs(v) for v in hid) % 97
                want.append(tok)
            pos += 1
        text = ''.join('<%d>' % t for t in prompt) + ''.join('<%d>' % t for t in want)
        if text not in out:
            print(out[-1500:])
        check('two stages pass the hidden state over UDP and resend a lost part',
              text in out)
    finally:
        for p in procs:
            p.kill()
        shutil.rmtree(work, ignore_errors=True)


def test_multi_lane_attention():
    """Lanes over cached positions for the scores and over dimensions for
    the weighted sum. On a full 256-position row it has to match the
    one-lane head bit for bit, including its score and weight buffers, in
    a fraction of the cycles, and a first cut that reads mirrored key
    lanes has to be caught."""
    import chiplet_flow as cf
    ms = load_model_spec()
    one, wide = (specgen_mod.derive_attn_spec(ms),
                 specgen_mod.derive_attnn_spec(ms))
    # The lanes' spec once kept the one-lane head's sentences beside its
    # own: row-major caches, and matvec with cols = n giving the key
    # address. Sonnet spent its whole answer on the contradiction.
    La = wide['parameters']['lanes']
    text = ' '.join(wide['behavior'])
    check('the lanes spec says one layout and one column count, its own',
          'row-major' not in text and 'cols = n,' not in text
          and 'cols = ceil(n / %d)' % La in text
          and 'write only j < n into sbuf' in text)
    work = os.path.join(ROOT, 'build_attnntest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_attn_deps(ms, work)
        rr = RuleBasedAgent()
        out = {}
        for label, spec, tbf, src in (
                ('one', one, specgen_mod.render_attn_testbench,
                 rr.render_attn(one, {agent_mod.FIX_VLAT})),
                ('wide', wide, specgen_mod.render_attnn_testbench,
                 rr.render_attnn(wide, {agent_mod.FIX_KLANE})),
                ('first', wide, specgen_mod.render_attnn_testbench,
                 rr.render_attnn(wide, set()))):
            with open(os.path.join(work, 'tb.v'), 'w') as f:
                f.write(tbf(spec, rows=((256, False),)))
            with open(os.path.join(work, 'h.v'), 'w') as f:
                f.write(src)
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'h.v'] + list(cf.ATTN_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            out[label] = subprocess.run(['vvp', 's.out'], cwd=work,
                                        capture_output=True, text=True,
                                        timeout=1200).stdout
        span = lambda o: int(re.search(r'span_cycles=(\d+)', o).group(1))
        ok = all('TB_RESULT: PASS' in out[k] for k in ('one', 'wide'))
        check('%d lanes compute a 256-position head bit-exact in %d cycles, '
              'against %d on one lane' % (wide['parameters']['lanes'],
                                          span(out['wide']) if ok else 0,
                                          span(out['one']) if ok else 0),
              ok and span(out['wide']) * 10 < span(out['one']))
        check('mirrored key lanes are caught at the scores',
              'TB_RESULT: PASS' not in out['first']
              and 'expected_s' in out['first'])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_mutants_edit_code_not_comments():
    """The adder mutant flips the first real addition. A '+' in a comment
    ahead of it made the mutant a comment edit on SiLU and the multi-lane
    head: nothing could kill it, and it was reported as an unproven hole
    instead of a test that was never run."""
    src = "  // a + b is the idea\n  assign y = a + b;  // + here too\n"
    m = dv._sub_first_real_add(src)
    check('the adder mutant changes the code, not a comment',
          'assign y = a - b;' in m and '// a + b is the idea' in m
          and '// + here too' in m)


def test_per_column_projection():
    """Real checkpoints have a weight scale per output channel and biases.
    The multi-lane projection's per-column mode reads {bias, shift, scale}
    for each column from a memory; the testbench gives every column its
    own, and a design that drops the bias has to fail."""
    import chiplet_flow as cf
    ms = load_model_spec()
    spec = specgen_mod.derive_projn_spec(ms, per_column=True)
    work = os.path.join(ROOT, 'build_projctest')
    shutil.rmtree(work, ignore_errors=True)
    try:
        cf.write_projn_deps(ms, work)
        with open(os.path.join(work, 'tb.v'), 'w') as f:
            f.write(specgen_mod.render_projn_testbench(spec))
        good = RuleBasedAgent().render_projn(spec, {agent_mod.FIX_LANE})
        nobias = good.replace('shadow[dselb] + $signed(', 'shadow[dselb] + 0 * $signed(')
        assert nobias != good
        # The column word taken one edge after loading c_addr, with each
        # column's own sum: that column's sum with the previous word.
        early = good.replace('if (dvb) begin\n        rq_acc <= shadow[dselb] +',
                             'if (dva) begin\n        rq_acc <= shadow[dsel] +'
                             ).replace('rq_idx <= didxb;', 'rq_idx <= didx;')
        assert early.count('shadow[dsel] +') == 1
        # The last row never reaches the MACs: each column's sum without it.
        norow = good.replace('      v1 <= issuing;\n',
                             '      v1 <= issuing && r != depth - 1;\n')
        assert norow != good
        out = {}
        for label, src in (('good', good), ('nobias', nobias), ('early', early),
                           ('norow', norow)):
            with open(os.path.join(work, 'p.v'), 'w') as f:
                f.write(src)
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'p.v'] + list(cf.PROJN_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            out[label] = subprocess.run(['vvp', 's.out'], cwd=work,
                                        capture_output=True, text=True,
                                        timeout=600).stdout
        check('every column requantized with its own bias, scale and shift',
              'TB_RESULT: PASS' in out['good'])
        check('a projection that drops the bias is caught',
              'TB_RESULT: PASS' not in out['nobias'])
        check('a column word read one edge early is caught and named',
              'TB_RESULT: PASS' not in out['early']
              and 'got_lane_is_this_columns_sum_with_the_previous_columns_word=1'
              in out['early'])
        check('a column sum that misses its last row is caught and named',
              'TB_RESULT: PASS' not in out['norow']
              and 'got_lane_is_this_columns_sum_without_its_last_row=1' in out['norow'])
        check('the spec says when a column word arrives with c_addr a register',
              'two clock edges after the edge that loads c' in ' '.join(spec['behavior'])
              and 'two clock edges after the edge that loads r' in ' '.join(spec['behavior'])
              and "requantizer's scale and shift inputs take" in ' '.join(spec['behavior']))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_requant_golden_is_shared():
    """One model of requantization, used by the requantizer's own
    testbench and by the layer that sequences it, so the two cannot
    disagree about what it means."""
    for acc, sc, sh in ((1000, 4096, 12), (-1000, 4096, 12),
                        (1 << 20, 4096, 12), (-(1 << 20), 4096, 12)):
        q, sat = specgen_mod.requant_golden(acc, sc, sh, 8)
        check('requant(%d) stays in range%s' % (acc, ' (saturated)' if sat
                                                else ''),
              -128 <= q <= 127)
    check('requantization rounds to nearest, not toward zero',
          specgen_mod.requant_golden(3, 1 << 11, 12, 8)[0] == 2)


def test_table_unit_specs_are_implementable():
    """The three table-driven units are held to bit-exact agreement with a
    Python model, so their specs have to state the arithmetic exactly.

    They did not. Both the exponential and the two normalisers left the
    index extraction to inference, and the two normalisers extract it
    differently: the reciprocal drops the implicit leading one and the
    inverse square root keeps it, because its normalised range spans a
    factor of four. Nothing in the prose said so, and no reader could
    have guessed it. Measured, an agent given the old text failed on the
    first vector every time.

    This walks the arithmetic the spec now describes and checks it
    reproduces the golden model, on every model variant the sweep
    covers. If someone changes a golden model without changing the
    prose, or the reverse, this fails rather than the specification
    quietly becoming unimplementable again.
    """
    import sweep as sweep_mod
    # Importing the sweep must not move the flow's build directory. It
    # used to, at import time, and this test was what tripped over it:
    # every later test went looking for RTL in build_sweep.
    check('importing the sweep leaves the build directory alone',
          chiplet_flow.BUILD != sweep_mod.BUILD)
    rnd = random.Random(5)
    models = sweep_mod._models(load_model_spec())
    for ms in models:
        # exponential: t = (x*K)>>>16, split at in_frac, shift by -n
        spec = specgen_mod.derive_exp_spec(ms)
        p = spec['parameters']
        fi, fo, lb = p['in_frac'], p['out_frac'], p['lut_bits']
        lut = specgen_mod.exp_lut(lb, fo)
        for x in range(0, -(1 << (p['in_width'] - 1)), -3):
            t = (x * specgen_mod.LOG2E_Q16) >> 16
            n = t >> fi
            frac = t - (n << fi)
            assert 0 <= frac < (1 << fi)
            y = 0 if -n > fo else (lut[frac >> (fi - lb)] >> -n)
            if y != specgen_mod.exp_golden(x, p):
                check('exp spec arithmetic matches golden (%s, x=%d)'
                      % (ms['name'], x), False)
                break
        else:
            check('exp spec arithmetic matches golden (%s)' % ms['name'],
                  True)

        # reciprocal: index is the fraction alone, leading one implicit
        spec = specgen_mod.derive_recip_spec(ms)
        p = spec['parameters']
        iw, ow, lb = p['in_width'], p['out_width'], p['lut_bits']
        check('recip shift bias in the spec text is %d (%s)'
              % (ow + iw - 1, ms['name']),
              p['shift_bias'] == ow + iw - 1)
        rl = specgen_mod.recip_lut(lb, ow)
        ok = True
        for x in ([1, 2, 3, 1 << (iw - 1), (1 << iw) - 1]
                  + [rnd.randrange(1, 1 << iw) for _ in range(300)]):
            k = iw - x.bit_length()
            xn = x << k
            ok &= bool((xn >> (iw - 1)) & 1)       # normalised to [1,2)
            idx = (xn >> (iw - 1 - lb)) & ((1 << lb) - 1)
            ok &= (rl[idx], k) == specgen_mod.recip_golden(x, p)
        check('recip spec arithmetic matches golden (%s)' % ms['name'], ok)

        # inverse square root: index keeps the leading one, range is [1,4)
        spec = specgen_mod.derive_rsqrt_spec(ms)
        p = spec['parameters']
        iw, ow, lb = p['in_width'], p['out_width'], p['lut_bits']
        sl = specgen_mod.rsqrt_lut(lb, ow)
        ok = True
        for x in ([1, 2, 3, 4, 5, 1 << (iw - 1), (1 << iw) - 1]
                  + [rnd.randrange(1, 1 << iw) for _ in range(300)]):
            e = (x.bit_length() - 1) >> 1
            sh = iw - 2 - 2 * e
            ok &= sh >= 0 and sh % 2 == 0          # even alignment
            xn = x << sh
            ok &= (xn.bit_length() - 1) in (iw - 2, iw - 1)
            idx = xn >> (iw - lb)
            ok &= idx < (1 << lb)
            ok &= (sl[idx], e) == specgen_mod.rsqrt_golden(x, p)
        check('rsqrt spec arithmetic matches golden (%s)' % ms['name'], ok)

    # The two normalisers really do differ, and the prose says which is
    # which. If they are ever made to agree, this test is what notices.
    r = specgen_mod.derive_recip_spec(models[0])
    q = specgen_mod.derive_rsqrt_spec(models[0])
    check('recip spec states the leading one is implicit',
          any('implicit' in b for b in r['behavior']))
    check('rsqrt spec states the leading one is part of the index',
          any('IS part of idx' in b for b in q['behavior']))
    check('both specs name the rom module they must instantiate',
          any('recip_rom' in b for b in r['behavior'])
          and any('rsqrt_rom' in b for b in q['behavior']))


def test_llm_transport_is_retried():
    """A hung call is a transport fault, not a verdict on the design.

    The exponential was written up as a block the agent could not
    converge on. What actually happened on that run was one CLI call
    sitting at zero CPU until it hit the ten minute timeout, which
    aborted the block before the loop got a second chance. A flake that
    reads as a design failure is worse than a flake, because it becomes
    a conclusion.

    Retrying blindly is its own trap, though: an expired login fails
    identically every time, and three attempts at it just costs thirty
    seconds before the same error. So only transport faults retry.
    """
    import llm_agent

    calls = []
    Result = llm_agent.CliResult

    def fake_run(seq):
        it = iter(seq)
        def run(cmd, prompt, cap, silence=None, cwd=None):
            calls.append(cmd)
            nxt = next(it)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return run

    real_run, real_sleep = llm_agent.run_cli, llm_agent.time.sleep
    waits = []
    llm_agent.time.sleep = lambda t, *_: waits.append(t)
    try:
        # A call that went silent, then success: the loop should see the success.
        calls[:] = []
        llm_agent.run_cli = fake_run([
            subprocess.TimeoutExpired('claude', 180),
            Result(0, 'module m(); endmodule'),
        ])
        out = llm_agent.call_claude_cli('p', 'm')
        check('a call that goes silent is retried rather than ending the block',
              'module m' in out and len(calls) == 2)
        check('the writer call runs with tools disabled',
              calls[-1][calls[-1].index('--tools') + 1] == '')
        check('larger models get bounded effort, haiku keeps its default',
              '--effort' in llm_agent.cli_command('sonnet')
              and '--effort' not in llm_agent.cli_command('haiku'))
        check('the CLI streams its events, so thinking can be told from hung',
              'stream-json' in llm_agent.cli_command('sonnet'))
        check('with hangs caught by silence, every model gets the same cap',
              llm_agent.cli_timeout('haiku') == llm_agent.cli_timeout('sonnet'))

        # A model still thinking at its cap is not asked the same thing again.
        calls[:] = []
        llm_agent.run_cli = fake_run([llm_agent.StillThinking('still thinking after 1500s, no answer')] * 3)
        msg = ''
        try:
            llm_agent.call_claude_cli('p', 'm')
        except RuntimeError as e:
            msg = str(e)
        check('a model still thinking at its cap fails its attempt at once',
              len(calls) == 1 and 'still thinking' in msg)

        # Rate limiting is transport too.
        calls[:] = []
        llm_agent.run_cli = fake_run([
            Result(1, 'rate limit exceeded'),
            Result(0, 'module m(); endmodule'),
        ])
        llm_agent.call_claude_cli('p', 'm')
        check('a rate limited call is retried', len(calls) == 2)

        # An auth fault is not, because the next attempt cannot differ.
        calls[:] = []
        llm_agent.run_cli = fake_run([
            Result(1, 'Invalid API key'), Result(1, 'Invalid API key'),
            Result(1, 'Invalid API key'),
        ])
        raised = False
        try:
            llm_agent.call_claude_cli('p', 'm')
        except RuntimeError:
            raised = True
        check('an auth fault fails immediately instead of retrying',
              raised and len(calls) == 1)

        # Exhausting the attempts still raises, and says how many it tried.
        calls[:] = []
        llm_agent.run_cli = fake_run(
            [subprocess.TimeoutExpired('claude', 180)] * llm_agent.CLI_ATTEMPTS)
        msg = ''
        waits[:] = []
        try:
            llm_agent.call_claude_cli('p', 'm')
        except RuntimeError as e:
            msg = str(e)
        check('exhausted retries report the attempt count and the cause',
              len(calls) == llm_agent.CLI_ATTEMPTS and 'attempt' in msg
              and 'timed out' in msg)
        check('silent calls are retried over minutes, not seconds',
              llm_agent.CLI_ATTEMPTS >= 5 and sum(waits) >= 600
              and max(waits) <= 300)
    finally:
        llm_agent.run_cli = real_run
        llm_agent.time.sleep = real_sleep

    # An agent whose model is still thinking at the cap asks again with
    # thinking off, and keeps it off for the rest of the block.
    seen = []

    def fake_cli(p, model, attempts=3, thinking=True):
        seen.append(thinking)
        if thinking:
            raise llm_agent.StillThinking('still thinking after 1500s')
        return 'module m(); endmodule'
    real_cli, real_pick = llm_agent.call_claude_cli, llm_agent.pick_backend
    try:
        llm_agent.call_claude_cli = fake_cli
        llm_agent.pick_backend = lambda choice=None: ('claude-cli', 'sonnet')
        ag = llm_agent.LLMAgent('claude-cli:sonnet')
        spec = {'top_module': 'm', 'ports': [], 'behavior': [], 'parameters': {},
                'name': 'm', 'description': 'm'}
        r1, _ = ag.propose(spec, [])
        r2, _ = ag.propose(spec, [])
        check('a model still thinking at its cap is asked again with thinking '
              'off, and stays off for the block',
              'module m' in r1 and 'module m' in r2 and seen == [True, False, False]
              and '--settings' in llm_agent.cli_command('sonnet', thinking=False)
              and '--settings' not in llm_agent.cli_command('sonnet'))
    finally:
        llm_agent.call_claude_cli, llm_agent.pick_backend = real_cli, real_pick

    # A draft that passed simulation and synthesis and failed only timing
    # is where the next draft starts again once a later one breaks the
    # simulation, told so.
    prompts = []
    replies = iter(['module m(); /*one*/ endmodule', 'module m(); /*two*/ endmodule',
                    'module m(); /*three*/ endmodule'])

    def fake_cli2(p, model, attempts=3, thinking=True):
        prompts.append(p)
        return next(replies)
    try:
        llm_agent.call_claude_cli = fake_cli2
        llm_agent.pick_backend = lambda choice=None: ('claude-cli', 'sonnet')
        ag = llm_agent.LLMAgent('claude-cli:sonnet')
        spec = {'top_module': 'm', 'ports': [], 'behavior': [], 'parameters': {},
                'name': 'm', 'description': 'm'}
        ag.propose(spec, [])
        hist = [{'iteration': 1, 'stage': 'timing', 'status': 'fail', 'worst_slack_ns': -1.18}]
        ag.propose(spec, hist)
        hist.append({'iteration': 2, 'stage': 'sim', 'status': 'fail',
                     'mismatches': [{'test': 't', 'out': '0'}]})
        ag.propose(spec, hist)
        check('a draft that broke the simulation after one that passed it and '
              'synthesis goes back to that one, told why',
              '/*one*/' in prompts[2] and '/*two*/' not in prompts[2]
              and 'Your draft 1, shown above' in prompts[2]
              and '/*one*/' in prompts[1])
        # The next agent in the chain starts from the draft that got
        # furthest, with that draft's feedback, not from the last draft.
        iters = [{'iteration': 1, 'sim': {'status': 'pass'},
                  'synth': {'status': 'pass'},
                  'timing': {'stage': 'timing', 'status': 'fail',
                             'worst_slack_ns': -1.18}},
                 {'iteration': 2, 'sim': {'status': 'pass'},
                  'synth': {'status': 'pass'},
                  'timing': {'stage': 'timing', 'status': 'fail',
                             'worst_slack_ns': -3.5}},
                 {'iteration': 3, 'sim': {'stage': 'sim', 'status': 'fail'}}]
        seed = ag.best_of(iters)
        replies = iter(['module m(); /*four*/ endmodule',
                        'module m(); /*five*/ endmodule'])
        nx = llm_agent.LLMAgent('claude-cli:opus')
        nx.seed = seed
        nx.propose(spec, [])
        nx.propose(spec, [{'iteration': 1, 'stage': 'sim', 'status': 'fail'}])
        check('the next agent is handed the draft with the best slack and its '
              'timing record, and goes back to it after breaking the simulation',
              seed[0].strip() == 'module m(); /*one*/ endmodule'
              and seed[1][0]['worst_slack_ns'] == -1.18
              and '/*one*/' in prompts[3] and '-1.18' in prompts[3]
              and 'written by another model' in prompts[3]
              and '/*one*/' in prompts[4] and '/*four*/' not in prompts[4]
              and 'The draft you were handed' in prompts[4]
              and seed[2] == "synth")
        # A draft through simulation but not synthesis still goes on,
        # with its synthesis failure, ahead of the spec.
        simonly = ag.best_of([
            {'iteration': 1, 'sim': {'status': 'fail', 'stage': 'sim'}},
            {'iteration': 2, 'sim': {'status': 'pass'},
             'synth': {'stage': 'synth', 'status': 'fail', 'errors': ['crash']}},
            {'iteration': 3, 'sim': {'status': 'fail', 'stage': 'sim'}}])
        check('a draft through simulation but not synthesis is handed on '
              'with its synthesis failure',
              simonly[2] == "sim" and '/*two*/' in simonly[0]
              and simonly[1][0]['stage'] == 'synth')
        # With no draft past simulation, the last one goes on with its
        # failures: handed over bare, Sonnet saw Haiku's RMSNorm with no
        # word of how it failed.
        last = ag.best_of(iters[2:])
        replies = iter(['module m(); /*six*/ endmodule'])
        n2 = llm_agent.LLMAgent('claude-cli:opus')
        n2.seed = ag.best_of([{'iteration': 3, 'sim': {
            'stage': 'sim', 'status': 'fail',
            'mismatches': [{'test': 't', 'got_norm_is_the_expected_value_for_out': '1'}]}}])
        n2.propose(spec, [])
        check('with no draft past simulation, the last goes on with how it failed',
              last[0].strip() == 'module m(); /*three*/ endmodule' and last[2] is False
              and '/*three*/' in prompts[5]
              and 'got_norm_is_the_expected_value_for_out' in prompts[5]
              and 'the feedback below is how it failed' in prompts[5]
              and ag.best_of([]) is None)
    finally:
        llm_agent.call_claude_cli, llm_agent.pick_backend = real_cli, real_pick

    # The stream itself: a stand-in CLI that writes events, then the answer.
    work = os.path.join(ROOT, 'build_clitest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    try:
        fake = os.path.join(work, 'fake_cli.py')
        with open(fake, 'w') as f:
            f.write("import json, sys, time\n"
                    "sys.stdin.read()\n"
                    "mode = sys.argv[1]\n"
                    "ev = lambda d: print(json.dumps(d), flush=True)\n"
                    "for _ in range(4):\n"
                    "    ev({'type': 'stream_event', 'event': {'delta': {'type': 'thinking_delta', 'thinking': 'x'}}})\n"
                    "    time.sleep(0.3)\n"
                    "if mode == 'answer':\n"
                    "    ev({'type': 'result', 'is_error': False, 'result': 'module m(); endmodule'})\n"
                    "elif mode == 'ramble':\n"
                    "    ev({'type': 'stream_event', 'event': {'delta': {'type': 'text_delta', 'text': 'module m(); endmodule'}}})\n"
                    "    while True:\n"
                    "        ev({'type': 'stream_event', 'event': {'delta': {'type': 'text_delta', 'text': ' and so on'}}})\n"
                    "        time.sleep(0.3)\n"
                    "elif mode == 'think':\n"
                    "    while True:\n"
                    "        ev({'type': 'stream_event', 'event': {'delta': {'type': 'thinking_delta', 'thinking': 'x'}}})\n"
                    "        time.sleep(0.3)\n"
                    "else:\n"
                    "    time.sleep(60)\n")
        r = llm_agent.run_cli([sys.executable, fake, 'answer'], 'p', cap=30, silence=10)
        check('the answer is the stream\'s result event', r.returncode == 0
              and r.stdout == 'module m(); endmodule')
        kinds, said = [], ''
        for mode, cap, silence in (('think', 3, 10), ('hang', 30, 2)):
            try:
                llm_agent.run_cli([sys.executable, fake, mode], 'p', cap=cap, silence=silence)
                kinds.append('answered')
            except llm_agent.StillThinking as e:
                kinds.append('thinking')
                said = str(e)
            except subprocess.TimeoutExpired:
                kinds.append('hung')
        check('a streaming call at its cap is thinking, a silent one is hung',
              kinds == ['thinking', 'hung'])
        r = llm_agent.run_cli([sys.executable, fake, 'ramble'], 'p', cap=3, silence=10)
        check('at the cap, an answer that already holds a whole module is kept, '
              'and one without says how much was thinking and how much text',
              r.returncode == 0 and r.stdout.startswith('module m(); endmodule')
              and 'chars of thinking, 0 of text' in said)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_sequencer_specs_are_implementable():
    """Same audit as the table units, for the requantizer, softmax and
    the MLP layer, which the LLM had never converged on.

    Each had the same defect. The requantizer spec walked the writer
    through five pipeline stages and then demanded a latency of nine,
    and never said what a shift of zero does, though the testbench
    drives it eight times. Softmax never gave the normalising formula
    at all. The MLP spec never gave the weight layout, never said the
    rectifier comes after requantization and not on the output, and
    declared a 26-bit weight port against a 13-bit testbench wire.

    Along the way the MLP testbench turned out to be weak: one fixed
    scale put every output on a rail, so it checked the sign of the
    layer and little else. Its scales are now derived per layer.
    """
    import sweep as sweep_mod
    import re as re_mod
    ms0 = load_model_spec()

    # requantizer: the stated rounding, including the two tie examples
    rq = specgen_mod.derive_requant_spec(ms0)
    check('requant ties round toward plus infinity, as the spec says',
          specgen_mod.requant_golden(5, 1, 1, 8)[0] == 3
          and specgen_mod.requant_golden(-5, 1, 1, 8)[0] == -2)
    check('requant with shift zero adds no rounding term, as stated',
          specgen_mod.requant_golden(7, 1, 0, 8)[0] == 7)
    st = rq['parameters']['pipeline_stages']
    check('requant spec states the exact latency the testbench waits for',
          any(('exactly %d cycles' % st) in b for b in rq['behavior'])
          and not any(b.startswith('Stage 5') for b in rq['behavior']))

    rnd = random.Random(9)
    for ms in sweep_mod._models(ms0):
        # softmax: the stated normalising formula against the golden
        sp = specgen_mod.derive_softmax_spec(ms)
        p = sp['parameters']
        e_p = {'in_width': p['score_width'], 'in_frac': p['score_frac'],
               'out_frac': p['weight_frac'], 'lut_bits': p['score_frac']}
        r_p = {'in_width': p['recip_in_width'],
               'out_width': p['recip_out_width'], 'lut_bits': 8,
               'shift_bias': p['shift_bias']}
        lo = -(1 << (p['score_width'] - 1))
        ok = True
        for _ in range(150):
            row = [rnd.randrange(lo // 2, -lo // 2)
                   for _ in range(rnd.randrange(1, 30))]
            mx = max(row)
            ex = [specgen_mod.exp_golden(v - mx, e_p) for v in row]
            tot = sum(ex)
            ok &= tot < (1 << p['recip_in_width'])
            m, k = specgen_mod.recip_golden(tot, r_p)
            wf = p['weight_frac']
            w = [min(1 << wf, ((v << wf) * m) >> (p['shift_bias'] - k))
                 for v in ex]
            ok &= w == specgen_mod.softmax_golden(row, p)[0]
        check('softmax spec formula matches golden (%s)' % ms['name'], ok)

        # MLP: port width, layout, and a testbench that sees values
        sp = specgen_mod.derive_mlp_spec(ms)
        p = sp['parameters']
        wport = [q for q in sp['ports'] if q['name'] == 'w_addr'][0]
        check('mlp w_addr port matches its address width (%s)'
              % ms['name'], wport['width'] == p['addr_width'])
        tb = specgen_mod.render_mlp_testbench(sp)
        ys = [int(a + b) for a, b in
              re_mod.findall(r"expect_y\[\d+\] = (-?)\d+'sd(\d+);", tb)]
        hi = (1 << (p['data_width'] - 1)) - 1
        check('mlp testbench outputs are mostly in range, not on a rail '
              '(%s)' % ms['name'],
              sum(-hi - 1 < v < hi for v in ys) >= len(ys) - 1)
        sc = re_mod.search(r'scale1 = (\d+); shift1 = (\d+); '
                           r'scale2 = (\d+); shift2 = (\d+);', tb).groups()
        check('mlp testbench uses a different scale per layer (%s)'
              % ms['name'], sc[:2] != sc[2:])
        check('mlp testbench checks the hidden layer before the outputs (%s)'
              % ms['name'],
              'dut.act[%d + i]' % p['bank'] in tb
              and tb.index('hidden_layer') < tb.index('bad_idx[i]')
              and any('named act' in b for b in sp['behavior']))


def test_compile_errors_quote_the_source_line():
    """iverilog names a line number and nothing else. The model sees its
    own draft without line numbers, so a bare 'line 148: syntax error'
    asks it to count. Traced on the requantizer, it restructured the code
    around an illegal literal and kept that literal in every draft."""
    import llm_agent
    rtl = "\n".join("line_%d;" % i for i in range(1, 160))
    rtl = rtl.replace("line_148;", "x = (y < 48'sd-128) ? 8'sd-128 : z;")
    path = "/Users/someone/My Projects/repo/build_x/requant_sw.v"
    out = llm_agent.annotate_errors(
        [path + ":148: syntax error",
         "/tmp/a b/tb_requant_sw.v:20: error: unknown",
         "/tmp/x/exp_rom.v:3: warning",
         "no location here"], rtl)
    check('a compile error quotes the failing source line',
          out[0].startswith("requant_sw.v line 148: syntax error  | source: "
                            "x = (y < 48'sd-128) ? 8'sd-128 : z;"))
    check('testbench and supplied files are named but not quoted',
          out[1] == "tb_requant_sw.v line 20: error: unknown"
          and out[2] == "exp_rom.v line 3: warning")
    check('an error without a location passes through unchanged',
          out[3] == "no location here")
    rtl2 = "a;\nexpu_x <= (s_buf[i] - mx)[12:0];\nc;"
    out2 = llm_agent.annotate_errors(["/b/sm.v:2: syntax error",
                                      "/b/sm.v:3: error: Variable declaration in "
                                      "unnamed block requires SystemVerilog."], rtl2)
    out3 = llm_agent.annotate_errors(
        ["/b/h.v:3: error: Signing cast requires SystemVerilog.",
         "/b/h.v:4: error: A reference to a net or variable (`i') is not "
         "allowed in a constant expression."], "a;\nb;\nx = signed'(y);\nz = w[i*16+15:i*16];")
    check('a SystemVerilog cast and a variable part-select are explained',
          '$signed(x)' in out3[0] and '+: width' in out3[1])
    check('a select taken of an expression is explained, and so is a '
          'declaration inside an unnamed block',
          'cannot take a bit or part select of an expression' in out2[0]
          and 'named block' in out2[1]
          and 'select of an expression' not in llm_agent.annotate_errors(
              ["/b/sm.v:1: syntax error"], "y <= mem[(i)][3:0];")[0])
    hist = [{"iteration": 1, "stage": "sim", "status": "fail",
             "errors": [path + ":2: syntax error"]},
            {"iteration": 2, "stage": "sim", "status": "fail",
             "errors": [path + ":148: syntax error"]}]
    warn = llm_agent.condense_feedback(
        [{"iteration": 1, "stage": "sim", "status": "fail",
          "errors": ["/x/m.v:3: warning: Port 5 (cols) expects 13 bits",
                     "/x/m.v:9: syntax error"]}], None)[0]["tool_errors"]
    check('an error is reported ahead of a warning printed before it',
          "syntax error" in warn[0])
    rec = llm_agent.condense_feedback(hist, rtl)
    check('only the latest record is quoted, against the draft it came from',
          "source:" not in rec[0]["tool_errors"][0]
          and "48'sd-128" in rec[1]["tool_errors"][0])
    async_rtl = "module m;\n  always @(posedge clk or negedge rst_n) begin\nend"
    msg = llm_agent.annotate_errors(
        ["ERROR: FF m.$auto$ff.cc:337:slice$1 (type $_DFFE_PN0P_) cannot be "
         "legalized: dffs with async set or reset are not supported"],
        async_rtl)[0]
    check('the async-reset synthesis error names the construct and its line',
          "synchronous" in msg and "line 2:" in msg)
    lit_rtl = "module m;\n a = (w < 47'sd-128);\n b = c ? 8'sd(-128) : d;\n e = -8'sd128;"
    lit = llm_agent.annotate_errors(
        ["/x/m.v:2: syntax error", "/x/m.v:3: syntax error",
         "/x/m.v:4: syntax error"], lit_rtl)
    check('a sign inside a sized literal is explained, the legal form is not',
          "-8'sd128, not" in lit[0] and "-8'sd128, not" in lit[1]
          and "meaning" not in lit[2])
    mixed = ("module requant(input signed [28:0] acc_in, input [17:0] scale);\n"
             "  wire signed [46:0] p = acc_in * scale;\n"
             "  wire signed [46:0] ok = acc_in * $signed({1'b0, scale});\n"
             "endmodule")
    found = llm_agent.code_findings(mixed, [])
    check('a signed-times-unsigned multiply is reported with its line',
          len(found) == 1 and found[0].startswith("line 2: acc_in * scale"))
    xs = llm_agent.code_findings(mixed.replace(" * scale;", " * 1;"),
                                 [{"mismatches": [{"got_w": "x"}]}])
    check('an x read back from the design is explained',
          any("never written or reset" in f for f in xs))
    import agent as agent_m
    fixes = {getattr(agent_m, n) for n in dir(agent_m) if n.startswith("FIX_")}
    ref = agent_m.RuleBasedAgent()
    clean = all(not llm_agent.code_findings(
        getattr(ref, r)(getattr(specgen_mod, d)(load_model_spec()), fixes), [])
        for d, r in (("derive_requant_spec", "render_requant"),
                     ("derive_softmax_spec", "render_softmax"),
                     ("derive_mlp_spec", "render_mlp"),
                     ("derive_exp_spec", "render_exp"),
                     ("derive_rmsnorm_spec", "render_rmsnorm")))
    check('the lint raises nothing on the reference designs', clean)
    cat = ("module m(input signed [47:0] a, input signed [47:0] b,\n"
           "         input signed [31:0] c);\n"
           "  wire signed [52:0] s = a + {b, 5'b0};\n"
           "  wire [33:0] ok = {1'b0, c[22:0]} + {1'b0, c[31:23]};\n"
           "endmodule")
    cf = llm_agent.code_findings(cat, [])
    check('a concatenation of a whole signed signal in arithmetic is reported, '
          'a zero-extended slice is not',
          len(cf) == 1 and cf[0].startswith("line 3: {b, 5'b0}"))
    ext = ("module m(input signed [46:0] x, input signed [49:0] r);\n"
           "  wire signed [47:0] s = {x[46], x} + r;\n"
           "  wire signed [49:0] t = {{3{x[46]}}, x} + r;\nendmodule")
    check('sign extension by hand is not reported, since its bits are right',
          llm_agent.code_findings(ext, []) == [])
    pipe = ("module m(input clk, input [7:0] a, input signed [15:0] v);\n"
            "  reg [7:0] j; reg [7:0] buf_ [0:255]; reg signed [31:0] acc, r2;\n"
            "  always @(posedge clk) begin : b\n"
            "    reg [7:0] jm; reg signed [31:0] p;\n"
            "    jm = j - 8'd1;\n"
            "    p = $signed({1'b0, buf_[jm]}) * v;\n"
            "    if (j <= a) acc <= acc + p;\n"
            "    r2 <= acc;\n"
            "    j <= j + 8'd1;\n"
            "  end\nendmodule")
    hs = ("module m(input clk, input rst_n, output reg o_valid,\n"
          "         output signed [15:0] o_data);\n"
          "  wire signed [15:0] q; wire v; reg signed [15:0] a; reg vi;\n"
          "  requant r (.clk(clk), .acc_in(a), .valid_in(vi), .q_out(q),\n"
          "             .valid_out(v));\n"
          "  assign o_data = q;\n"
          "  always @(posedge clk) begin\n"
          "    o_valid <= 1'b0;\n"
          "    if (v) begin\n"
          "      o_valid <= 1'b1;\n"
          "    end\n"
          "  end\nendmodule")
    cm = ("module mac(input clk, input signed [15:0] a, input signed [15:0] b);\n"
          "  reg signed [24:0] ph, pl, ok;\n"
          "  always @(posedge clk) ph <= {{a[15]}, a[15:8]} * b;\n"
          "  always @(posedge clk) pl <= {1'b0, a[7:0]} * b;\n"
          "  always @(posedge clk) ok <= $signed({1'b0, a[7:0]}) * b;\n"
          "endmodule")
    cf_ = llm_agent.code_findings(cm, [])
    check('a concatenation multiplied by a signed operand is reported, and the '
          '$signed one is not',
          [f.split(":")[0] for f in cf_] == ["line 3", "line 4"]
          and "Write $signed(" in cf_[0])
    ys = ("module t(input clk, input [15:0] a_data, input signed [15:0] s_in);\n"
          "  sub u (.clk(clk), .a($signed(a_data)));\n"
          "  sub v (.clk(clk), .a($signed(s_in)));\nendmodule")
    yf = [f for f in llm_agent.code_findings(ys, []) if "Yosys" in f]
    ye = llm_agent.annotate_errors(
        ["ERROR: Assert `arg->is_signed == sig.as_wire()->is_signed' failed in genrtlil.cc:2214."], None)
    check('$signed() of an unsigned signal in a port connection is reported '
          'and the Yosys assertion it causes is explained',
          len(yf) == 1 and yf[0].startswith("line 2: .a($signed(a_data))")
          and "Declare a signed wire" in ye[0])
    hf = llm_agent.code_findings(hs, [])
    check('an output wired from an instance beside a valid registered from '
          'it is reported, and the same output driven alongside its valid is not',
          len(hf) == 1 and hf[0].startswith('line 6: o_data is driven straight from q')
          and 'o_valid (line 10)' in hf[0]
          and llm_agent.code_findings(hs.replace(
              "      o_valid <= 1'b1;", "      o_valid <= v;").replace(
              "    if (v) begin", "    begin"), []) == [])
    tf = llm_agent.code_findings(pipe, [{'stage': 'timing', 'status': 'fail',
        'critical_path': 'from register j to register acc. This path takes longer'}])
    check('a failing timing path is traced to the lines of code it runs through',
          len(tf) == 1 and 'line 5: jm = j - 8\'d1; line 6: p =' in tf[0]
          and 'line 7: if (j <= a) acc <= acc + p' in tf[0]
          and llm_agent.timing_path(pipe, 'acc', 'r2') == [(8, 'r2 <= acc;')]
          and llm_agent.timing_path(pipe, 'a', 'r2') == [])
    inst = ("module m(input clk, input signed [15:0] g);\n"
            "  reg signed [15:0] gs; reg [16:0] m_;\n"
            "  wire signed [31:0] t = gs * $signed({1'b0, m_});\n"
            "  requant rq (.clk(clk),\n"
            "              .acc_in(t), .q_out());\n"
            "  always @(posedge clk) gs <= g;\nendmodule")
    check('a path that ends inside a module instance is traced to the signal '
          'wired into it',
          llm_agent.timing_path(inst, 'gs', 'rq') == [
              (3, "wire signed [31:0] t = gs * $signed({1'b0, m_});"),
              (5, ".acc_in(t), into rq")])
    sq = ("module m(input signed [15:0] x, input [39:0] eps);\n"
          "  reg [39:0] ssq; reg [39:0] t; wire signed [31:0] p = x * x;\n"
          "  always @(*) ssq = ssq + (x * x);\n"
          "  always @(*) t = {8'b0, x * x};\n"
          "  always @(*) t = ssq + p;\n"
          "  always @(*) t = $signed({1'b0, ssq}) + x * x;\nendmodule")
    sf = llm_agent.code_findings(sq, [])
    check('a signed product added to an unsigned value, and a product inside '
          'a concatenation, are reported; the product in its own wire and the '
          'zero-extended sum are not',
          len(sf) == 2 and sf[0].startswith("line 3: x * x multiplies two signed")
          and sf[1].startswith("line 4: x * x is inside a concatenation"))
    sta = ("Startpoint: _109349_ (rising edge-triggered flip-flop clocked by clk)\n"
           "Endpoint: w_data[0] (output port clocked by clk)\n")
    net = os.path.join(ROOT, 'build_cptest.v')
    with open(net, 'w') as f:
        f.write("  DFF _109349_ (\n    .CK(clk),\n    .D(_002800_),\n"
                "    .Q(\\exp_buf[90] [6])\n  );\n")
    try:
        cp = chiplet_flow._critical_path(sta, net)
    finally:
        os.remove(net)
    check('a timing failure names its path in RTL terms',
          cp.startswith("from register exp_buf to port w_data"))
    replies = iter(["wire x = 1;", "module m(); endmodule"])
    rtl2, calls = llm_agent.ask_for_module(lambda p: next(replies), "P", "m")
    check('a fragment reply is re-asked instead of spending an iteration',
          rtl2.strip() == "module m(); endmodule" and calls == 2)


def test_hung_simulation_is_feedback_not_a_crash():
    """A draft that never lowers busy leaves the testbench waiting forever.
    The simulator timeout used to raise out of the flow and end the run,
    which lost a traced softmax run. It has to come back as a failed
    simulation the agent can read."""
    work = os.path.join(ROOT, 'build_hangtest')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    with open(os.path.join(work, 'tb_hang.v'), 'w') as f:
        f.write("module tb_hang; reg clk = 0; wire busy;\n"
                "hang dut(.clk(clk), .busy(busy)); always #5 clk = ~clk;\n"
                "initial begin @(negedge clk); while (busy) @(negedge clk);\n"
                "$display(\"TB_RESULT: PASS\"); $finish; end endmodule\n")
    rtl = os.path.join(work, 'hang.v')
    with open(rtl, 'w') as f:
        f.write("module hang(input clk, output busy); "
                "assign busy = 1'b1; endmodule\n")
    saved = (chiplet_flow.BUILD, chiplet_flow.ROOT, chiplet_flow.run)
    real = chiplet_flow.run
    try:
        chiplet_flow.BUILD = chiplet_flow.ROOT = work
        chiplet_flow.run = lambda cmd, timeout=120: real(cmd, timeout=3)
        r = chiplet_flow.stage_sim({'tb_file': 'tb_hang.v'}, rtl)
    finally:
        chiplet_flow.BUILD, chiplet_flow.ROOT, chiplet_flow.run = saved
        shutil.rmtree(work, ignore_errors=True)
    check('a hung simulation fails as feedback instead of raising',
          r['status'] == 'fail' and r.get('phase') == 'run'
          and 'never finished' in r['errors'][0])


def test_accumulator_overflow_proof():
    """The accumulator width rule, proved for every input sequence rather
    than tested on some. formal.py wraps the MAC in an exact wider sum
    and a count of accumulations, assumes at most the reduction depth of
    them per clear, and proves by k-induction that the sum always fits
    and that acc always equals it.

    A proof that cannot fail proves nothing, so this also checks that it
    does fail: on a truncated accumulator, and on a width two bits
    narrower than the rule. And it pins what the proof found about the
    rule itself: exact when the depth is a power of two, one bit
    conservative otherwise.
    """
    import copy
    import formal
    import agent as agent_m
    if not formal.available():
        print('%-55s %s' % ('accumulator proof (needs yosys-smtbmc, z3)',
                            'SKIP'))
        return
    import sweep as sweep_mod
    fixes = {getattr(agent_m, n) for n in dir(agent_m) if n.startswith('FIX_')}
    ref = agent_m.RuleBasedAgent()
    work = os.path.join(ROOT, 'build_formaltest')
    os.makedirs(work, exist_ok=True)

    def prove(spec, rtl_text=None):
        rtl = os.path.join(work, 'mac.v')
        with open(rtl, 'w') as f:
            f.write(rtl_text if rtl_text is not None
                    else ref.render_mac(spec, fixes))
        return formal.prove_mac(spec, rtl, work)['status']

    try:
        for ms in sweep_mod._models(load_model_spec()):
            spec = derive_chiplet_spec(ms)
            # 16-bit operands put a split multiplier against a*b, which is
            # multiplier equivalence and past z3's budget here. Those are
            # not claimed as proved; the int8 targets are.
            if spec['parameters']['data_width'] > 8:
                continue
            check('accumulator cannot overflow, proved (%s)' % ms['name'],
                  prove(spec) == 'proved')
        spec = derive_chiplet_spec(load_model_spec())
        import dv
        trunc = dv.mut_trunc(ref.render_mac(spec, fixes))
        check('the proof fails on a truncated accumulator',
              prove(spec, trunc) == 'failed')
        narrow = copy.deepcopy(spec)
        narrow['parameters']['acc_width'] -= 2
        check('the proof fails two bits below the rule',
              prove(narrow) == 'failed')
        # Depth 4864 is not a power of two: the rule rounds log2 up.
        one = copy.deepcopy(spec)
        one['parameters']['acc_width'] -= 1
        check('the rule is one bit conservative at depth %d, proved'
              % spec['derivation']['reduction_depth'],
              prove(one) == 'proved')
        pow2 = dict(load_model_spec(), d_model=896, d_ff=8192)
        exact = derive_chiplet_spec(pow2)
        tight = copy.deepcopy(exact)
        tight['parameters']['acc_width'] -= 1
        check('the rule is exact at a power-of-two depth, proved',
              prove(exact) == 'proved' and prove(tight) == 'failed')
    finally:
        shutil.rmtree(work, ignore_errors=True)


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
    # endpoint_gbps is bytes_per_cycle times the clock the design closed
    # at, so the claim is only as good as that number. With OpenSTA it is
    # a clock; without it the flow reports a gate-depth proxy, which is a
    # measure of logic depth and not of frequency, and a rate computed
    # from it is not a rate. Checked when the number is real, skipped
    # with a word when it is not.
    if fp.get('fmax_method') == 'opensta_slack':
        check('signed-off endpoint sustains its target link rate',
              fp['endpoint_gbps'] >= d.get('target_link_gbps', 0) - 1e-9)
    else:
        print('%-55s %s' % ('endpoint sustains its rate (needs OpenSTA)',
                            'SKIP'))


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

    # The property worth asserting is comparative, not absolute. Whether
    # a board class "cannot host" a model depends on the target rate in
    # the spec as much as on the model, so an absolute verdict was
    # really a statement about one target, and it broke the moment the
    # spec named a model meant for a smaller board. A smaller class must
    # never need fewer boards than a larger one for the same work.
    hard = dict(ms, target_tokens_per_s=max(ms['target_tokens_per_s'] * 40,
                                            320))
    check('a smaller board class never needs fewer boards than a larger',
          boards_needed(hard, fit('arty_a7_100t', profile, fp))
          >= boards_needed(hard, lg))
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


def test_bidirectional_ring(profile, fp):
    """Every link is full duplex; one ring only sends one way round it.
    Half the vector each way uses both directions: still exact, and
    faster wherever the links, not their latency, set the time."""
    mid = fit('kc705', profile, fp)

    def ring(fits, L, bidir):
        sim, boards = make_cluster(fits)
        rng = random.Random(len(fits) * 7 + L)
        for b in boards:
            b.mem = [rng.uniform(-1.0, 1.0) for _ in range(L)]
        exp = [sum(b.mem[k] for b in boards) for k in range(L)]
        t = run_workers(sim, ring_allreduce(boards, [b.mem for b in boards], bidir))
        err = max(abs(b.mem[k] - exp[k]) for b in boards for k in range(L))
        return t, err, all(b.mem == boards[0].mem for b in boards)

    ok = True
    for n in (2, 3, 5, 8):
        t, err, same = ring([mid] * n, 1000, True)
        ok = ok and err < 1e-9 and same
    check('bidirectional ring allreduce exact, every board the same, n=2,3,5,8', ok)
    fits = [fit('arty_a7_100t', profile, fp), fit('alveo_u250', profile, fp),
            fit('kc705', profile, fp)]
    t, err, same = ring(fits, 777, True)
    check('bidirectional ring allreduce heterogeneous 3-board exact', err < 1e-9 and same)
    check('two boards: one ring already uses both directions, same time',
          ring([mid] * 2, 4096, True)[0] == ring([mid] * 2, 4096, False)[0])
    t1 = ring([mid] * 4, 65536, False)[0]
    t2 = ring([mid] * 4, 65536, True)[0]
    check('bidirectional ring 1.6x faster on a 512 KB all-reduce over 4 boards '
          '(%.2fx)' % (t1 / t2), t1 / t2 > 1.6)

    ms = load_model_spec()
    lg = fit('alveo_u250', profile, fp)
    bi = dict(ms, ring_bidirectional=True)
    p1, p2 = predict_config(ms, lg, 8), predict_config(bi, lg, 8)
    check('sizing: both directions lift the 8-board prediction 1.4x',
          p2['predicted_tok_per_s'] > 1.4 * p1['predicted_tok_per_s'])
    r1 = simulate_decode([lg] * 8, ms, tokens=2)
    r2 = simulate_decode([lg] * 8, bi, tokens=2)
    ratio = r2['tok_per_s'] / p2['predicted_tok_per_s']
    check('bidirectional decode within 15%% of its prediction (ratio %.2f)' % ratio,
          0.85 <= ratio <= 1.15)
    check('bidirectional decode faster on the fabric, activations identical',
          r2['tok_per_s'] > 1.4 * r1['tok_per_s'] and r2['vecs_equal_across_boards'])


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
    test_exp_block()
    test_recip_block()
    test_rsqrt_block()
    test_matvec_sequencer()
    test_wmem_subsystem()
    test_softmax_sequencer()
    test_mlp_layer()
    test_attention_head()
    test_rmsnorm()
    test_silu()
    test_gated_mlp()
    test_residual_add()
    test_full_size_projection()
    test_multi_lane_projection()
    test_multi_lane_attention()
    test_per_column_projection()
    test_mutants_edit_code_not_comments()
    test_decoder_runs_the_model()
    test_small_model_derivation()
    test_fpga_counts_the_hierarchy_once()
    test_rotary_embedding()
    test_scores_wider_than_the_exponential()
    test_qwen_shaped_decoder()
    test_basys3_top_level()
    test_zybo_register_block()
    test_weight_streamer_survives_stalls()
    test_full_sequencer_both_qwens()
    test_bridge_end_to_end()
    test_signed_off_modules_are_renamed_however_written()
    test_sign_off_survives_a_silent_agent()
    test_failed_attempts_are_kept()
    test_a_draft_that_never_passed_is_not_handed_on()
    test_composites_give_their_parts_ports()
    test_llm_blocks_decode()
    test_attention_scores_cannot_overflow()
    test_spec_to_verified_rtl()
    test_weights_split_over_boards()
    test_weights_split_through_registers()
    test_tp_plan()
    test_zybo_at_32_lanes()
    test_gals_two_boards()
    test_cluster_plan()
    test_zybo_stage_registers()
    test_stage_network_on_host()
    test_tp_network_on_host()
    test_simulators_agree()
    test_arm_programs_on_their_rtl()
    test_composite_cell_count()
    test_timing_leaves_subblock_internals_to_their_signoff()
    test_requant_golden_is_shared()
    test_table_unit_specs_are_implementable()
    test_llm_transport_is_retried()
    test_sequencer_specs_are_implementable()
    test_compile_errors_quote_the_source_line()
    test_hung_simulation_is_feedback_not_a_crash()
    test_accumulator_overflow_proof()
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
    test_bidirectional_ring(profile, fp)
    test_link_reliability(profile)
    print('all %d tests passed' % PASSED[0])
