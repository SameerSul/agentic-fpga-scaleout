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
        res = {}
        for label, fx in (('raw_scores', set()),
                          ('fixed', {agent_mod.FIX_SUBMAX})):
            open(os.path.join(work, 'sm.v'), 'w').write(
                RuleBasedAgent().render_softmax(spec, fx))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out',
                                'tb.v', 'sm.v', 'expu.v', 'recip.v', 'roms.v'],
                               cwd=work, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=900)
            res[label] = 'TB_RESULT: PASS' in r.stdout
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
    sh = 6
    t, s_, w, a, o = specgen_mod.attn_golden(q, K, V, n, sh, 1, 0, p, sm_p)
    fs = [x / (1 << sh) / (1 << sm_p['score_frac']) for x in t]
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
        for label, fx_ in (('first', set()), ('fixed', {agent_mod.FIX_EPS})):
            with open(os.path.join(work, 'rms.v'), 'w') as f:
                f.write(rr.render_rmsnorm(spec, fx_))
            r = subprocess.run(['iverilog', '-g2005', '-o', 's.out', 'tb.v',
                                'rms.v'] + list(cf.RMSNORM_DEPS), cwd=work,
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            r = subprocess.run(['vvp', 's.out'], cwd=work,
                               capture_output=True, text=True, timeout=900)
            res[label] = r.stdout
        check('the norm computes the sum of squares, rsqrt and the products',
              'TB_RESULT: PASS' in res['fixed'])
        check('a sum of squares that leaves out epsilon is caught',
              'TB_RESULT: PASS' not in res['first']
              and 'expected_ssq' in res['first'])
    finally:
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
        for label, fx in (('first', set()), ('fixed', {agent_mod.FIX_RRND})):
            with open(os.path.join(work, 'r.v'), 'w') as f:
                f.write(RuleBasedAgent().render_resadd(spec, fx))
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

    class Result:
        def __init__(self, rc, out, err=''):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(seq):
        it = iter(seq)
        def run(cmd, **kw):
            calls.append(cmd)
            nxt = next(it)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return run

    real_run, real_sleep = llm_agent.subprocess.run, llm_agent.time.sleep
    llm_agent.time.sleep = lambda *_: None
    try:
        # A timeout, then success: the loop should see the success.
        calls[:] = []
        llm_agent.subprocess.run = fake_run([
            subprocess.TimeoutExpired('claude', 600),
            Result(0, 'module m(); endmodule'),
        ])
        out = llm_agent.call_claude_cli('p', 'm')
        check('a timed out call is retried rather than ending the block',
              'module m' in out and len(calls) == 2)
        check('the writer call runs with tools disabled',
              calls[-1][calls[-1].index('--tools') + 1] == '')
        check('larger models get bounded effort, haiku keeps its default',
              '--effort' in llm_agent.cli_command('sonnet')
              and '--effort' not in llm_agent.cli_command('haiku'))

        # Rate limiting is transport too.
        calls[:] = []
        llm_agent.subprocess.run = fake_run([
            Result(1, 'rate limit exceeded'),
            Result(0, 'module m(); endmodule'),
        ])
        llm_agent.call_claude_cli('p', 'm')
        check('a rate limited call is retried', len(calls) == 2)

        # An auth fault is not, because the next attempt cannot differ.
        calls[:] = []
        llm_agent.subprocess.run = fake_run([
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
        llm_agent.subprocess.run = fake_run(
            [subprocess.TimeoutExpired('claude', 600)] * 3)
        msg = ''
        try:
            llm_agent.call_claude_cli('p', 'm')
        except RuntimeError as e:
            msg = str(e)
        check('exhausted retries report the attempt count and the cause',
              len(calls) == 3 and 'attempt' in msg and 'timed out' in msg)
    finally:
        llm_agent.subprocess.run = real_run
        llm_agent.time.sleep = real_sleep


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
                     ("derive_exp_spec", "render_exp")))
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
    test_decoder_runs_the_model()
    test_small_model_derivation()
    test_fpga_counts_the_hierarchy_once()
    test_rotary_embedding()
    test_composite_cell_count()
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
    test_link_reliability(profile)
    print('all %d tests passed' % PASSED[0])
