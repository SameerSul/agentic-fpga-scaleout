"""Discrete-event simulator for a multi-FPGA chip-to-chip interconnect fabric.

Time unit: nanoseconds. Pure Python 3 stdlib.
Models: serial links (latency, serialization, bit errors), a NIC link layer
(CRC32, go-back-N retransmission, credit-based flow control), and boards
(memory, DMA engine, MAC-array compute engine with a cycle model).

Scaleout adaptation: a Board's compute throughput is no longer a fixed
systolic constant. It comes from boards.fit(board_profile, chiplet_profile),
so the measured RTL profile (cycles_per_mac, fmax) and the board class
(clock cap, capacity) set the cycle model. Heterogeneous clusters are
supported: each point-to-point link runs at the min of the two endpoints'
transceiver rates and the max of their propagation delays.
"""
import heapq
import math
import random
import struct
import zlib
from collections import deque

PAYLOAD = 1024      # data payload bytes per packet (128 doubles)
HDR_BYTES = 20      # data packet header on the wire
CTRL_BYTES = 12     # ACK / credit packet on the wire
WINDOW = 32         # go-back-N sender window, packets
RX_CAP = 64         # receiver buffer, packets (credit pool size)
DMA_SLICE = 16384   # bytes moved per DMA grant, keeps TX and RX drain fair

DATA, ACK, CRED = 0, 1, 2


class Sim:
    """Event queue with simulated time in ns."""

    def __init__(self, seed=1):
        self.now = 0.0
        self.rng = random.Random(seed)
        self._q = []
        self._n = 0

    def at(self, delay, fn, *args):
        self._n += 1
        heapq.heappush(self._q, (self.now + delay, self._n, fn, args))

    def run(self):
        while self._q:
            t, _, fn, args = heapq.heappop(self._q)
            self.now = t
            fn(*args)


class Event:
    __slots__ = ('sim', 'done', 'value', '_waiters')

    def __init__(self, sim):
        self.sim = sim
        self.done = False
        self.value = None
        self._waiters = []

    def succeed(self, value=None):
        if self.done:
            return
        self.done = True
        self.value = value
        for w in self._waiters:
            self.sim.at(0.0, w, value)
        self._waiters = None

    def _wait(self, cb):
        if self.done:
            self.sim.at(0.0, cb, self.value)
        else:
            self._waiters.append(cb)


def start(sim, gen):
    """Run a generator as a process. Yield a float (delay in ns) or an Event.
    Returns an Event that fires with the generator's return value."""
    done = Event(sim)

    def step(val):
        try:
            y = gen.send(val)
        except StopIteration as e:
            done.succeed(e.value)
            return
        if isinstance(y, Event):
            y._wait(step)
        else:
            sim.at(y, step, None)

    sim.at(0.0, step, None)
    return done


class Resource:
    """One-at-a-time resource with a FIFO wait queue (models a DMA engine)."""

    def __init__(self, sim):
        self.sim = sim
        self.busy = False
        self.q = deque()

    def acquire(self):
        ev = Event(self.sim)
        if not self.busy:
            self.busy = True
            ev.succeed()
        else:
            self.q.append(ev)
        return ev

    def release(self):
        if self.q:
            self.q.popleft().succeed()
        else:
            self.busy = False

    def use(self, delay_ns):
        yield self.acquire()
        yield delay_ns
        self.release()


class Pkt:
    __slots__ = ('kind', 'size', 'seq', 'tag', 'idx', 'tot',
                 'payload', 'crc', 'ack', 'credits')

    def __init__(self, kind, size, seq=-1, tag=None, idx=0, tot=0,
                 payload=b'', crc=0, ack=-1, credits=0):
        self.kind = kind
        self.size = size
        self.seq = seq
        self.tag = tag
        self.idx = idx
        self.tot = tot
        self.payload = payload
        self.crc = crc
        self.ack = ack
        self.credits = credits


class Link:
    """Unidirectional point-to-point link: serialization at the line rate,
    fixed propagation delay, optional per-packet software overhead, and
    random corruption from a configured bit-error rate (data packets)."""

    def __init__(self, sim, gbps, prop_ns, ber=0.0, pkt_overhead_ns=0.0):
        self.sim = sim
        self.Bpns = gbps / 8.0
        self.prop = prop_ns
        self.ber = ber
        self.ovh = pkt_overhead_ns
        self.free_at = 0.0
        self.bytes_tx = 0

    def transmit(self, pkt, deliver):
        s = self.sim
        t0 = max(s.now, self.free_at)
        self.free_at = t0 + self.ovh + pkt.size / self.Bpns
        self.bytes_tx += pkt.size
        corrupt = False
        if self.ber > 0.0 and pkt.kind == DATA:
            p = 1.0 - (1.0 - self.ber) ** (pkt.size * 8)
            corrupt = s.rng.random() < p
        s.at(self.free_at + self.prop - s.now, deliver, pkt, corrupt)


class _Tx:
    """Go-back-N sender with credit-based flow control toward one peer."""

    def __init__(self, nic, peer_id, peer_nic, link):
        self.nic = nic
        self.sim = nic.sim
        self.me = nic.board.id
        self.peer_nic = peer_nic
        self.link = link
        self.q = deque()
        self.inflight = deque()
        self.next_seq = 0
        self.credits = RX_CAP
        self.retransmits = 0
        self.stalls = 0
        self.stall_ns = 0.0
        self._stall_t0 = None
        self._timer = 0
        pkt_ns = (HDR_BYTES + PAYLOAD) / link.Bpns + link.ovh
        self.rto = 2.0 * link.prop + (WINDOW + RX_CAP) * pkt_ns + 2000.0

    def enqueue(self, pkts):
        self.q.extend(pkts)
        self.pump()

    def pump(self):
        sent = False
        while self.q and len(self.inflight) < WINDOW and self.credits > 0:
            p = self.q.popleft()
            p.seq = self.next_seq
            self.next_seq += 1
            self.credits -= 1
            self.inflight.append(p)
            self.link.transmit(p, self._deliver)
            sent = True
        if sent:
            self._arm()
        if self.q and self.credits == 0 and self._stall_t0 is None:
            self._stall_t0 = self.sim.now
            self.stalls += 1

    def _deliver(self, pkt, corrupt):
        self.peer_nic._on_pkt(self.me, pkt, corrupt)

    def _arm(self):
        self._timer += 1
        self.sim.at(self.rto, self._timeout, self._timer)

    def _timeout(self, gen):
        if gen != self._timer or not self.inflight:
            return
        for p in self.inflight:
            self.link.transmit(p, self._deliver)
            self.retransmits += 1
        self._arm()

    def on_ack(self, a):
        moved = False
        while self.inflight and self.inflight[0].seq <= a:
            self.inflight.popleft()
            moved = True
        if moved:
            if self.inflight:
                self._arm()
            else:
                self._timer += 1
            self.pump()

    def on_cred(self, k):
        self.credits += k
        if self._stall_t0 is not None:
            self.stall_ns += self.sim.now - self._stall_t0
            self._stall_t0 = None
        self.pump()

    def ctrl(self, pkt):
        self.link.transmit(pkt, self._deliver)


class _Rx:
    """In-order receiver: CRC check, cumulative ACKs, bounded RX buffer
    drained through the board DMA, credits returned as slots free up."""

    def __init__(self, nic, peer_id):
        self.nic = nic
        self.sim = nic.sim
        self.peer = peer_id
        self.rx_next = 0
        self.buf = deque()
        self.max_occ = 0
        self.overflow_drops = 0
        self.crc_drops = 0
        self.seq_drops = 0
        self.reasm = {}
        self.waiting = {}
        self.mailbox = {}
        self._kick = None
        start(self.sim, self._drain())

    def on_data(self, pkt, corrupt):
        payload = pkt.payload
        if corrupt:
            b = bytearray(payload)
            b[self.sim.rng.randrange(len(b))] ^= 1 << self.sim.rng.randrange(8)
            payload = bytes(b)
        if zlib.crc32(payload) != pkt.crc:
            self.crc_drops += 1
            self._ack(self.rx_next - 1)
            return
        if pkt.seq != self.rx_next:
            self.seq_drops += 1
            self._ack(self.rx_next - 1)
            return
        if len(self.buf) >= RX_CAP:
            self.overflow_drops += 1
            return
        self.rx_next += 1
        self.buf.append((pkt, payload))
        self.max_occ = max(self.max_occ, len(self.buf))
        self._ack(pkt.seq)
        if self._kick is not None:
            k, self._kick = self._kick, None
            k.succeed()

    def _ack(self, seq):
        self.nic.tx[self.peer].ctrl(Pkt(ACK, CTRL_BYTES, ack=seq))

    def _drain(self):
        board = self.nic.board
        while True:
            if not self.buf:
                self._kick = Event(self.sim)
                yield self._kick
                continue
            pkt, payload = self.buf.popleft()
            yield from board.dma.use(len(payload) / board.drain_Bpns)
            self.nic.tx[self.peer].ctrl(Pkt(CRED, CTRL_BYTES, credits=1))
            self._reassemble(pkt, payload)

    def _reassemble(self, pkt, payload):
        e = self.reasm.get(pkt.tag)
        if e is None:
            e = self.reasm[pkt.tag] = [[None] * pkt.tot, 0]
        e[0][pkt.idx] = payload
        e[1] += 1
        if e[1] == pkt.tot:
            blob = b''.join(e[0])
            vals = list(struct.unpack('<%dd' % (len(blob) // 8), blob))
            del self.reasm[pkt.tag]
            w = self.waiting.pop(pkt.tag, None)
            if w is not None:
                w.succeed(vals)
            else:
                self.mailbox[pkt.tag] = vals


class NIC:
    def __init__(self, sim, board):
        self.sim = sim
        self.board = board
        self.tx = {}
        self.rx = {}

    def _peer(self, peer_id, peer_nic, link_out):
        self.tx[peer_id] = _Tx(self, peer_id, peer_nic, link_out)
        self.rx[peer_id] = _Rx(self, peer_id)

    def _on_pkt(self, src, pkt, corrupt):
        if pkt.kind == ACK:
            self.tx[src].on_ack(pkt.ack)
        elif pkt.kind == CRED:
            self.tx[src].on_cred(pkt.credits)
        else:
            self.rx[src].on_data(pkt, corrupt)

    def recv_event(self, src, tag):
        rx = self.rx[src]
        ev = Event(self.sim)
        if tag in rx.mailbox:
            ev.succeed(rx.mailbox.pop(tag))
        else:
            rx.waiting[tag] = ev
        return ev


def connect(a, b, ber=0.0, pkt_overhead_ns=0.0, gbps=None, prop_ns=None):
    """Full-duplex connection between two boards (two unidirectional links).
    Heterogeneous rule: the link runs at the min of the two endpoints'
    transceiver rates and the max of their propagation delays, unless the
    caller overrides both explicitly."""
    if gbps is None:
        gbps = min(a.link_gbps, b.link_gbps)
    if prop_ns is None:
        prop_ns = max(a.link_prop_ns, b.link_prop_ns)
    ab = Link(a.sim, gbps, prop_ns, ber, pkt_overhead_ns)
    ba = Link(a.sim, gbps, prop_ns, ber, pkt_overhead_ns)
    a.nic._peer(b.id, b.nic, ab)
    b.nic._peer(a.id, a.nic, ba)


class Board:
    """FPGA board: memory (list of floats), DMA engine, MAC-array compute.

    The compute model is set by a fit() result (boards.py): `instances`
    chiplet MAC units run in parallel at `clock_mhz`, each retiring one MAC
    every `cycles_per_mac` cycles as measured by the RTL testbench. matmul
    stays numerically real: the cycle model only charges time."""

    def __init__(self, sim, bid, fit_result, dma_gbps=64.0, drain_gbps=None):
        self.sim = sim
        self.id = bid
        self.mem = []
        self.dma = Resource(sim)
        self.dma_Bpns = dma_gbps / 8.0
        self.drain_Bpns = (drain_gbps if drain_gbps else dma_gbps) / 8.0
        self.fit = fit_result
        self.instances = max(1, fit_result["instances"])
        self.cpm = fit_result["cycles_per_mac"]
        self.clk_ns = 1000.0 / fit_result["clock_mhz"]
        self.link_gbps = fit_result["link_gbps"]
        self.link_prop_ns = fit_result["link_prop_ns"]
        self.busy_ns = 0.0
        self.nic = NIC(sim, self)

    def send(self, dst, values, tag):
        """Process: DMA data from memory to the NIC in slices, packetize,
        hand to the link layer. Returns when the last slice is queued."""
        blob = struct.pack('<%dd' % len(values), *values)
        tot = (len(blob) + PAYLOAD - 1) // PAYLOAD
        tx = self.nic.tx[dst]
        idx = 0
        for off in range(0, len(blob), DMA_SLICE):
            sl = blob[off:off + DMA_SLICE]
            yield from self.dma.use(len(sl) / self.dma_Bpns)
            pkts = []
            for j in range(0, len(sl), PAYLOAD):
                pl = sl[j:j + PAYLOAD]
                pkts.append(Pkt(DATA, HDR_BYTES + len(pl), tag=tag, idx=idx,
                                tot=tot, payload=pl, crc=zlib.crc32(pl)))
                idx += 1
            tx.enqueue(pkts)

    def recv(self, src, tag):
        """Process: wait for a complete message from src with this tag."""
        vals = yield self.nic.recv_event(src, tag)
        return vals

    def matmul(self, A, B):
        """Process: MAC-array cycle model, then the real product.
        cycles = M*N*K*cycles_per_mac / instances + fill/drain."""
        M, K, N = len(A), len(B), len(B[0])
        side = math.ceil(math.sqrt(self.instances))
        cycles = (M * N * K * self.cpm) / self.instances + (M + N + 2 * side)
        dur = cycles * self.clk_ns
        self.busy_ns += dur
        yield dur
        return mm(A, B)

    def elementwise(self, rows, fn, lanes=256):
        """Process: elementwise op at `lanes` results per cycle."""
        n = sum(len(r) for r in rows)
        dur = (n / lanes) * self.clk_ns
        self.busy_ns += dur
        yield dur
        return [[fn(v) for v in r] for r in rows]


def mm(A, B):
    Bt = list(zip(*B))
    return [[sum(x * y for x, y in zip(r, c)) for c in Bt] for r in A]


def relu(v):
    return v if v > 0.0 else 0.0


def make_cluster(fits, ber=0.0, pkt_overhead_ns=0.0,
                 dma_gbps=64.0, drain_gbps=None, seed=1):
    """Build a cluster from a list of fit() results, one per board (mixes
    allowed). Full mesh of links; the ring collective uses neighbor links.
    Each pairwise link rate follows the min-of-endpoints rule in connect()."""
    sim = Sim(seed)
    boards = [Board(sim, i, f, dma_gbps, drain_gbps)
              for i, f in enumerate(fits)]
    n = len(boards)
    for i in range(n):
        for j in range(i + 1, n):
            connect(boards[i], boards[j], ber, pkt_overhead_ns)
    return sim, boards


def fabric_stats(boards):
    st = {'retransmits': 0, 'crc_drops': 0, 'seq_drops': 0,
          'overflow_drops': 0, 'max_rx_occupancy': 0,
          'credit_stalls': 0, 'stall_ns': 0.0, 'wire_bytes': 0}
    for b in boards:
        for tx in b.nic.tx.values():
            st['retransmits'] += tx.retransmits
            st['credit_stalls'] += tx.stalls
            st['stall_ns'] += tx.stall_ns
            st['wire_bytes'] += tx.link.bytes_tx
        for rx in b.nic.rx.values():
            st['crc_drops'] += rx.crc_drops
            st['seq_drops'] += rx.seq_drops
            st['overflow_drops'] += rx.overflow_drops
            st['max_rx_occupancy'] = max(st['max_rx_occupancy'], rx.max_occ)
    return st
