"""The 'right fabric' calculator and the LLM decode hosting simulation.

Given the model spec (the LLM is the input workload), the measured chiplet
profile, and the measured fabric endpoint profile, sizing answers: how many
boards of a given class does it take to host this model at the target token
rate? It tries N in CANDIDATES with an analytic compute + communication
model, then the chosen configuration is validated by simulate_decode(), which
runs decode on the discrete-event fabric with every all-reduce as real
packetized, CRC-checked, credit-gated traffic.

Pure Python 3 stdlib."""
import json
import os
import random

from fabric import make_cluster, fabric_stats, HDR_BYTES
from collectives import ring_allreduce, run_workers

ROOT = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = (1, 2, 4, 8, 16)


def load_model_spec(path=None):
    return json.load(open(path or os.path.join(ROOT, "model_spec.json")))


def model_summary(ms):
    """Derived per-token work: MAC counts and all-reduce traffic."""
    d, f, L = ms["d_model"], ms["d_ff"], ms["n_layer"]
    attn_macs = 4 * d * d          # QKV projections + output projection
    mlp_macs = 2 * d * f           # c_fc + c_proj
    per_layer_macs = attn_macs + mlp_macs
    per_token_macs = L * per_layer_macs
    ars_per_token = L * ms["allreduces_per_token_per_layer"]
    ar_bytes = d * ms["dtype_bytes"]
    return {
        "attn_macs_per_layer": 4 * d * d,
        "mlp_macs_per_layer": 2 * d * f,
        "per_layer_macs": per_layer_macs,
        "per_token_macs": per_token_macs,
        "allreduces_per_token": ars_per_token,
        "allreduce_bytes": ar_bytes,
        "comm_bytes_per_token": ars_per_token * ar_bytes,
    }


def predict_config(ms, fit_result, n):
    """Analytic time per token on n boards of one class: the tensor-parallel
    compute share plus n_layer*2 ring all-reduces of a [1, d_model] activation
    at the synthesized link rate. Ring step time = propagation + serialization
    of the chunk; 2*(n-1) steps per all-reduce."""
    s = model_summary(ms)
    compute_ns = s["per_token_macs"] / n / fit_result["macs_per_s"] * 1e9
    if n == 1:
        comm_ns = 0.0
    else:
        chunk = s["allreduce_bytes"] / n
        Bpns = fit_result["link_gbps"] / 8.0
        step_ns = fit_result["link_prop_ns"] + (chunk + HDR_BYTES) / Bpns
        comm_ns = s["allreduces_per_token"] * 2 * (n - 1) * step_ns
    token_ns = compute_ns + comm_ns
    return {
        "boards": n,
        "compute_ns_per_token": compute_ns,
        "comm_ns_per_token": comm_ns,
        "token_ns": token_ns,
        "predicted_tok_per_s": 1e9 / token_ns,
        "comm_fraction": comm_ns / token_ns,
    }


def size_fabric(ms, fit_result, candidates=CANDIDATES):
    """Smallest board count that meets target_tokens_per_s, with the full
    candidate sweep for the report. chosen is None if the cap cannot reach
    the target on this board class."""
    target = ms["target_tokens_per_s"]
    sweep = [predict_config(ms, fit_result, n) for n in candidates]
    chosen = next((p for p in sweep if p["predicted_tok_per_s"] >= target), None)
    return {"board": fit_result["board"], "target_tok_per_s": target,
            "sweep": sweep, "chosen": chosen}


def simulate_decode(fits, ms, tokens=8, ber=0.0, seed=11):
    """Host the model: run decode on the discrete-event fabric. Compute time
    is charged from the measured chiplet profile at full model dimensions
    (weights are not materialized; the sharded numerics are validated at
    reduced dimensions elsewhere). Every all-reduce is real fabric traffic:
    a [1, d_model] activation vector, packetized with headers and CRC32,
    credit-gated, go-back-N retransmitted on corruption.

    Returns dict with tok_per_s, total time, collective count, fabric stats,
    and the final reduced activation vector (for bit-exactness checks)."""
    n = len(fits)
    s = model_summary(ms)
    L = ms["n_layer"]
    d = ms["d_model"]
    sim, boards = make_cluster(fits, ber=ber, seed=seed)
    rng = random.Random(seed)
    # One persistent activation vector per board; values are arbitrary floats,
    # what matters is that they travel and reduce through the real link layer.
    vecs = [[rng.uniform(-1.0, 1.0) for _ in range(d)] for _ in range(n)]

    # Pre-build every collective so all boards agree on op ids: for each of
    # tokens * n_layer * 2 all-reduces there is one generator per board.
    ars = [ring_allreduce(boards, vecs)
           for _ in range(tokens * L * ms["allreduces_per_token_per_layer"])]

    def worker(i):
        b = boards[i]
        k = 0
        for _ in range(tokens):
            for _ in range(L):
                yield from b.compute(s["attn_macs_per_layer"] / n)
                yield from ars[k][i]
                k += 1
                yield from b.compute(s["mlp_macs_per_layer"] / n)
                yield from ars[k][i]
                k += 1

    t = run_workers(sim, [worker(i) for i in range(n)])
    st = fabric_stats(boards)
    assert st["overflow_drops"] == 0
    return {
        "boards": n,
        "tokens": tokens,
        "total_ns": t,
        "token_ns": t / tokens,
        "tok_per_s": tokens / (t / 1e9),
        "collectives": len(ars),
        "collectives_per_token": len(ars) / tokens,
        "fabric": st,
        "final_vec": vecs[0][:8],
        "vecs_equal_across_boards": all(v == vecs[0] for v in vecs),
    }
