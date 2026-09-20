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
    """Derived per-token work: MAC counts, all-reduce traffic, weight bytes,
    and KV cache traffic.

    Batch-1 decode identity: each block weight is used exactly once per
    token, so weight traffic per token = per_token_macs * weight_bits / 8.
    Attention against the KV cache adds 2 * seq_len * d_model MACs per layer
    (QK^T scores plus the AV combine) and reads the whole cache each token.
    The LM head and embeddings are excluded (modeled as host-side)."""
    d, f, L = ms["d_model"], ms["d_ff"], ms["n_layer"]
    seq = ms.get("seq_len", 0)
    wbytes = ms["weight_bits"] / 8.0
    abytes = ms["activation_bits"] / 8.0
    attn_macs = 4 * d * d          # QKV projections + output projection
    mlp_macs = 2 * d * f           # c_fc + c_proj
    per_layer_macs = attn_macs + mlp_macs
    per_token_macs = L * per_layer_macs
    kv_macs_layer = 2 * seq * d    # QK^T + AV against the cache
    ars_per_token = L * ms["allreduces_per_token_per_layer"]
    ar_bytes = d * ms["dtype_bytes"]
    return {
        "attn_macs_per_layer": attn_macs,
        "mlp_macs_per_layer": mlp_macs,
        "per_layer_macs": per_layer_macs,
        "per_token_macs": per_token_macs,
        "kv_macs_per_layer": kv_macs_layer,
        "kv_macs_per_token": L * kv_macs_layer,
        "macs_per_token_total": per_token_macs + L * kv_macs_layer,
        "weight_bytes": per_token_macs * wbytes,
        "attn_weight_bytes_per_layer": attn_macs * wbytes,
        "mlp_weight_bytes_per_layer": mlp_macs * wbytes,
        "kv_cache_bytes": L * 2 * d * seq * abytes,
        "kv_read_bytes_per_layer": 2 * d * seq * abytes,
        "kv_read_bytes_per_token": L * 2 * d * seq * abytes,
        "allreduces_per_token": ars_per_token,
        "allreduce_bytes": ar_bytes,
        "comm_bytes_per_token": ars_per_token * ar_bytes,
    }


def predict_config(ms, fit_result, n):
    """Analytic time per token on n boards of one class:

        token_ns = max(compute_ns, mem_ns) + comm_ns

    compute is the tensor-parallel MAC share (projections, MLP, and attention
    against the KV cache); mem is the per-token DDR traffic (the weight shard,
    unless it fits in on-chip SRAM and is modeled resident, plus the KV cache
    read, which always lives in DDR), overlapped with compute by double
    buffering, hence the max; comm is n_layer*2 ring all-reduces of a
    [1, d_model] activation at the synthesized link rate, with ring step =
    propagation + serialization of the chunk and 2*(n-1) steps per all-reduce."""
    s = model_summary(ms)
    compute_ns = s["macs_per_token_total"] / n / fit_result["macs_per_s"] * 1e9
    weights_per_board = s["weight_bytes"] / n
    sram = fit_result.get("sram_bytes")
    resident = sram is not None and weights_per_board <= sram
    mem_bytes = (0.0 if resident else weights_per_board) \
        + s["kv_read_bytes_per_token"] / n
    bpns = fit_result.get("mem_bytes_per_ns")
    mem_ns = mem_bytes / bpns if bpns else 0.0
    if n == 1:
        comm_ns = 0.0
    else:
        chunk = s["allreduce_bytes"] / n
        Bpns = fit_result["link_gbps"] / 8.0
        step_ns = fit_result["link_prop_ns"] + (chunk + HDR_BYTES) / Bpns
        comm_ns = s["allreduces_per_token"] * 2 * (n - 1) * step_ns
    # Block-granular overlap, matching the simulator: within each attention
    # and MLP block, compute overlaps that block's own memory traffic, so the
    # token pays L * (max per attn block + max per mlp block). Summing maxes
    # per block is slower than max of sums; the simulator does the former, so
    # the analytic model must too or it predicts optimistically.
    rate = fit_result["macs_per_s"] / 1e9  # MACs per ns
    attn_c = (s["attn_macs_per_layer"] + s["kv_macs_per_layer"]) / n / rate
    mlp_c = s["mlp_macs_per_layer"] / n / rate
    if bpns:
        attn_m = ((0.0 if resident else s["attn_weight_bytes_per_layer"] / n)
                  + s["kv_read_bytes_per_layer"] / n) / bpns
        mlp_m = (0.0 if resident else
                 s["mlp_weight_bytes_per_layer"] / n) / bpns
    else:
        attn_m = mlp_m = 0.0
    blocks_ns = ms["n_layer"] * (max(attn_c, attn_m) + max(mlp_c, mlp_m))
    token_ns = blocks_ns + comm_ns
    bound = "comm" if comm_ns >= blocks_ns else (
        "memory" if mem_ns > compute_ns else "compute")
    return {
        "boards": n,
        "compute_ns_per_token": compute_ns,
        "mem_ns_per_token": mem_ns,
        "comm_ns_per_token": comm_ns,
        "weights_per_board_bytes": weights_per_board,
        "sram_resident": resident,
        "bound": bound,
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
    """Host the model: run decode on the discrete-event fabric. Compute and
    memory time are charged from the measured chiplet profile and the board's
    memory system at full model dimensions (weights are not materialized; the
    sharded numerics are validated at reduced dimensions elsewhere). Weight
    shards that fit the board's SRAM budget are resident (no per-token DDR
    traffic); otherwise each layer streams its shard, and the KV cache read
    is always DDR traffic, overlapped with compute as in predict_config.
    Every all-reduce is real fabric traffic: a [1, d_model] activation
    vector, packetized with headers and CRC32, credit-gated, go-back-N
    retransmitted on corruption.

    Returns dict with tok_per_s, total time, collective count, fabric stats,
    and the final reduced activation vector (for bit-exactness checks)."""
    n = len(fits)
    s = model_summary(ms)
    L = ms["n_layer"]
    d = ms["d_model"]
    sram = min((f.get("sram_bytes") or float("inf")) for f in fits)
    resident = s["weight_bytes"] / n <= sram
    attn_w = 0.0 if resident else s["attn_weight_bytes_per_layer"] / n
    mlp_w = 0.0 if resident else s["mlp_weight_bytes_per_layer"] / n
    kv_rd = s["kv_read_bytes_per_layer"] / n
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
                yield from b.compute_mem(
                    (s["attn_macs_per_layer"] + s["kv_macs_per_layer"]) / n,
                    attn_w + kv_rd)
                yield from ars[k][i]
                k += 1
                yield from b.compute_mem(s["mlp_macs_per_layer"] / n, mlp_w)
                yield from ars[k][i]
                k += 1

    t = run_workers(sim, [worker(i) for i in range(n)])
    st = fabric_stats(boards)
    assert st["overflow_drops"] == 0
    return {
        "boards": n,
        "tokens": tokens,
        "sram_resident": resident,
        "total_ns": t,
        "token_ns": t / tokens,
        "tok_per_s": tokens / (t / 1e9),
        "collectives": len(ars),
        "collectives_per_token": len(ars) / tokens,
        "fabric": st,
        "final_vec": vecs[0][:8],
        "vecs_equal_across_boards": all(v == vecs[0] for v in vecs),
    }
