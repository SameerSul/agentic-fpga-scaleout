"""Collective operations over the simulated fabric.

ring_allreduce: reduce-scatter then all-gather around the ring, the standard
bandwidth-optimal algorithm (each board moves 2*(n-1)/n of the vector). Works
for arbitrary N and for heterogeneous boards: the discrete-event simulation
charges each hop its own link rate, so the slowest ring segment gates each
step, which is the honest model. With bidir it sends half the vector each
way round, using both directions of every full-duplex link.
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


def ring_allreduce(boards, data, bidir=False):
    """data: list of equal-length float lists, one per board, reduced in
    place so every board ends with the elementwise sum.

    Every link is full duplex, and one ring only sends one way round it:
    on three boards or more, each link's other direction carries nothing
    but acknowledgements. bidir sends half the vector each way at once, so
    each direction of each link carries half the bytes: the same 2*(n-1)
    steps, each serializing half as much. Two boards gain nothing, since
    one ring there already uses both directions of the only link, so it
    stays one ring."""
    n = len(boards)
    if n == 1:
        return [_noop()]
    L = len(data[0])
    assert all(len(d) == L for d in data)
    op = _next_op()
    if bidir and n > 2 and L > 1:
        rings = [(0, L // 2, 1), (L // 2, L, -1)]
    else:
        rings = [(0, L, 1)]

    def one_way(i, lo, hi, d):
        """The standard ring over vec[lo:hi], in direction d: board i's
        position round that ring is p, and it sends to position p+1."""
        b = boards[i]
        nxt = boards[(i + d) % n].id
        prv = boards[(i - d) % n].id
        p = i if d == 1 else (-i) % n
        bnd = [lo + (k * (hi - lo)) // n for k in range(n + 1)]
        vec = data[i]
        for s in range(n - 1):
            si, ri = (p - s) % n, (p - s - 1) % n
            start(b.sim, b.send(nxt, vec[bnd[si]:bnd[si + 1]], ('rs', op, d, s)))
            got = yield from b.recv(prv, ('rs', op, d, s))
            a, z = bnd[ri], bnd[ri + 1]
            vec[a:z] = [x + c for x, c in zip(vec[a:z], got)]
        for s in range(n - 1):
            si, ri = (p + 1 - s) % n, (p - s) % n
            start(b.sim, b.send(nxt, vec[bnd[si]:bnd[si + 1]], ('ag', op, d, s)))
            got = yield from b.recv(prv, ('ag', op, d, s))
            vec[bnd[ri]:bnd[ri + 1]] = got

    def worker(i):
        others = [start(boards[i].sim, one_way(i, *r)) for r in rings[1:]]
        yield from one_way(i, *rings[0])
        for ev in others:
            yield ev

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
