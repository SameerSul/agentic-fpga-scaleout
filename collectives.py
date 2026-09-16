"""Collective operations over the simulated fabric.

ring_allreduce: reduce-scatter then all-gather around the ring, the standard
bandwidth-optimal algorithm (each board moves 2*(n-1)/n of the vector). Works
for arbitrary N and for heterogeneous boards: the discrete-event simulation
charges each hop its own link rate, so the slowest ring segment gates each
step, which is the honest model.
naive_allreduce: gather everything to board 0, reduce, broadcast back.
Both return one worker generator per board; run them with run_workers."""
from fabric import start

_op = [0]


def _next_op():
    _op[0] += 1
    return _op[0]


def _noop():
    return
    yield


def run_workers(sim, gens):
    """Start all workers, run the sim, return the wall time (ns) until the
    last worker finishes (measured inside the sim, stale timers excluded)."""
    t0 = sim.now
    fin = []

    def wrap(g):
        yield from g
        fin.append(sim.now)

    for g in gens:
        start(sim, wrap(g))
    sim.run()
    return max(fin) - t0


def ring_allreduce(boards, data):
    """data: list of equal-length float lists, one per board, reduced in
    place so every board ends with the elementwise sum."""
    n = len(boards)
    if n == 1:
        return [_noop()]
    L = len(data[0])
    assert all(len(d) == L for d in data)
    op = _next_op()
    bnd = [(k * L) // n for k in range(n + 1)]

    def worker(i):
        b = boards[i]
        nxt = boards[(i + 1) % n].id
        prv = boards[(i - 1) % n].id
        vec = data[i]
        for s in range(n - 1):
            si, ri = (i - s) % n, (i - s - 1) % n
            start(b.sim, b.send(nxt, vec[bnd[si]:bnd[si + 1]], ('rs', op, s)))
            got = yield from b.recv(prv, ('rs', op, s))
            lo, hi = bnd[ri], bnd[ri + 1]
            vec[lo:hi] = [a + c for a, c in zip(vec[lo:hi], got)]
        for s in range(n - 1):
            si, ri = (i + 1 - s) % n, (i - s) % n
            start(b.sim, b.send(nxt, vec[bnd[si]:bnd[si + 1]], ('ag', op, s)))
            got = yield from b.recv(prv, ('ag', op, s))
            vec[bnd[ri]:bnd[ri + 1]] = got

    return [worker(i) for i in range(n)]


def naive_allreduce(boards, data):
    """All boards send their full vector to board 0, board 0 reduces and
    broadcasts the result back over its direct links."""
    n = len(boards)
    if n == 1:
        return [_noop()]
    op = _next_op()

    def root():
        b = boards[0]
        vec = data[0]
        for j in range(1, n):
            got = yield from b.recv(boards[j].id, ('g', op, j))
            vec[:] = [a + c for a, c in zip(vec, got)]
        for j in range(1, n):
            start(b.sim, b.send(boards[j].id, vec, ('b', op, j)))

    def leaf(j):
        b = boards[j]
        yield from b.send(boards[0].id, data[j], ('g', op, j))
        got = yield from b.recv(boards[0].id, ('b', op, j))
        data[j][:] = got

    return [root()] + [leaf(j) for j in range(1, n)]
