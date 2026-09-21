"""FPGA portability layer: real device budgets, transports, and fit().

Two things make this model honest rather than a single made-up capacity
number. First, resources are multi-dimensional and the generated blocks do
not compete for the same one: the MAC chiplet infers a DSP slice (one per
MAC), while the CRC fabric endpoint is pure LUT logic. A device's DSP count
therefore sets the compute ceiling and its LUT count sets the link ceiling,
which a single proxy cannot express. Those per-block numbers come from real
yosys synth_xilinx runs recorded in the profiles (see fpga.py).

Second, the board-to-board transport is a choice with measurable cost.
Ethernet buys commodity cabling, switching, and vendor neutrality; Aurora
buys latency and a smaller footprint. Both are modeled, so sizing can show
what the choice actually costs instead of asserting one is better.

Device figures are public datasheet values for the named parts; effective
DDR bandwidth and MAC footprints are representative engineering estimates,
not measured on hardware.
"""
import math

FIT_FRACTION = 0.7   # device fraction available after control/routing/PS
LINE_CODE = 64.0 / 66.0   # 64b/66b encoding, both Aurora and 10/25GBASE-R
CABLE_NS_PER_M = 5.0
CABLE_M = 3.0        # a rack-local direct-attach copper cable

BOARDS = {
    "arty_a7_100t": {
        "name": "arty_a7_100t",
        "class": "small (Artix-7 XC7A100T)",
        "family": "xc7",
        "luts": 63400, "ffs": 126800, "dsps": 240,
        "sram_bytes": 622080,          # 135 BRAM36
        "max_chiplet_clock_mhz": 150.0,
        "mem_gbytes_per_s": 1.0,       # 16-bit DDR3L-1333, effective
        "serdes_line_gbps": 6.25,      # GTP
        "eth_line_gbps": 1.25,         # 1000BASE-X on the available cage
        "num_links": 2,
    },
    "zcu102": {
        "name": "zcu102",
        "class": "mid (Zynq UltraScale+ XCZU9EG)",
        "family": "xcup",
        "luts": 274080, "ffs": 548160, "dsps": 2520,
        "sram_bytes": 4202496,         # 912 BRAM36, no URAM on this part
        "max_chiplet_clock_mhz": 300.0,
        "mem_gbytes_per_s": 12.0,      # 64-bit DDR4-2133, effective
        "serdes_line_gbps": 16.375,    # GTH
        "eth_line_gbps": 10.3125,      # 10GBASE-R over the SFP+ cages
        "num_links": 4,
    },
    "alveo_u250": {
        "name": "alveo_u250",
        "class": "large (Alveo U250, XCU250)",
        "family": "xcup",
        "luts": 1728000, "ffs": 3456000, "dsps": 12288,
        "sram_bytes": 56401920,        # 2000 BRAM36 + 1280 URAM288
        "max_chiplet_clock_mhz": 500.0,
        "mem_gbytes_per_s": 64.0,      # 4 channels of DDR4-2400, effective
        "serdes_line_gbps": 25.78125,  # GTY
        "eth_line_gbps": 25.78125,     # 25GBASE-R lanes of the QSFP28 cages
        "num_links": 8,
    },
}

# Board-to-board transports. frame_overhead_bytes is everything on the wire
# that is not our packet; endpoint_latency_ns is transmit plus receive
# through the core; switch_latency_ns is one cut-through hop. luts/ffs per
# port are the transport core itself, charged on top of the generated CRC
# endpoint, because an Ethernet MAC is real logic the design has to fit.
TRANSPORTS = {
    "aurora": {
        "name": "Aurora 64B/66B, direct attach",
        "rate_key": "serdes_line_gbps",
        # Aurora framing is a few control words per frame.
        "frame_overhead_bytes": 8,
        "endpoint_latency_ns": 200.0,
        "switch_latency_ns": 0.0,
        "mtu_bytes": 16384,
        "luts_per_port": 1500, "ffs_per_port": 2000,
        "switchable": False,
    },
    "ethernet_direct": {
        "name": "Ethernet, direct attach (no switch)",
        "rate_key": "eth_line_gbps",
        # 7 preamble + 1 SFD + 14 header + 4 FCS + 12 interframe gap.
        "frame_overhead_bytes": 38,
        "endpoint_latency_ns": 550.0,
        "switch_latency_ns": 0.0,
        "mtu_bytes": 9000,             # jumbo frames
        "luts_per_port": 5000, "ffs_per_port": 6000,
        "switchable": True,
    },
    "ethernet_switched": {
        "name": "Ethernet through a cut-through switch",
        "rate_key": "eth_line_gbps",
        "frame_overhead_bytes": 38,
        "endpoint_latency_ns": 550.0,
        "switch_latency_ns": 450.0,    # cut-through; store-and-forward is ~4x
        "mtu_bytes": 9000,
        "luts_per_port": 5000, "ffs_per_port": 6000,
        "switchable": True,
    },
}
DEFAULT_TRANSPORT = "ethernet_direct"


def _cap(budget, need):
    """How many instances fit, limited by whichever resource runs out first.
    Returns (count, binding_resource_name)."""
    best, who = None, None
    for k in ("dsps", "luts", "ffs"):
        if need.get(k, 0) > 0:
            n = int(budget.get(k, 0) // need[k])
            if best is None or n < best:
                best, who = n, k
    return (max(0, best), who) if best is not None else (0, None)


def fit(board, chiplet_profile, fabric_profile=None,
        transport=DEFAULT_TRANSPORT):
    """Deploy the measured blocks onto one board class.

    Compute: instances is bounded by the scarcest of DSP, LUT, and FF, after
    the transport cores and CRC endpoints for every link are reserved. With
    one DSP per MAC this is normally DSP-bound, which is the finding that
    generic-cell synthesis hides.

    Link: the usable rate is min(the transport's payload rate after 64b/66b
    coding, the synthesized endpoint's own measured throughput). Latency is
    transport endpoint latency plus cable flight plus any switch hop, and
    per-frame wire overhead becomes a fixed serialization cost per packet."""
    if isinstance(board, str):
        board = BOARDS[board]
    tr = TRANSPORTS[transport] if isinstance(transport, str) else transport

    budget = {k: board[k] * FIT_FRACTION for k in ("luts", "ffs", "dsps")}
    per = dict(chiplet_profile.get("fpga") or {})

    endpoint_gbps = None
    ep = (fabric_profile or {}).get("fpga") or {}
    if fabric_profile is not None:
        # One transport core plus one generated endpoint per link.
        for k, port_key in (("luts", "luts_per_port"), ("ffs", "ffs_per_port")):
            budget[k] -= board["num_links"] * (tr[port_key] + ep.get(k, 0))
            budget[k] = max(0.0, budget[k])
        ep_clock = min(fabric_profile["fmax_estimate_mhz"],
                       board["max_chiplet_clock_mhz"])
        endpoint_gbps = fabric_profile["bytes_per_cycle"] * 8 * ep_clock / 1000.0

    instances, bound_by = _cap(budget, per)
    clock_mhz = min(chiplet_profile["fmax_estimate_mhz"],
                    board["max_chiplet_clock_mhz"])
    cpm = chiplet_profile["cycles_per_mac"]
    macs_per_s = instances * clock_mhz * 1e6 / cpm

    wire_gbps = board[tr["rate_key"]] * LINE_CODE
    link_gbps = min(wire_gbps, endpoint_gbps) if endpoint_gbps else wire_gbps
    prop_ns = (tr["endpoint_latency_ns"] + tr["switch_latency_ns"]
               + CABLE_M * CABLE_NS_PER_M)
    # Per-frame wire overhead, charged as serialization time per packet.
    pkt_overhead_ns = tr["frame_overhead_bytes"] / (link_gbps / 8.0)

    return {
        "board": board["name"],
        "board_class": board["class"],
        "family": board["family"],
        "instances": instances,
        "bound_by": bound_by,
        "clock_mhz": clock_mhz,
        "cycles_per_mac": cpm,
        "macs_per_s": macs_per_s,
        "dsp_utilization": (instances * per.get("dsps", 0)
                            / board["dsps"] if board["dsps"] else 0.0),
        "lut_utilization": (instances * per.get("luts", 0)
                            + board["num_links"] * (tr["luts_per_port"]
                                                    + ep.get("luts", 0)))
                           / board["luts"],
        "transport": tr["name"],
        "transport_key": transport if isinstance(transport, str) else "custom",
        "link_gbps": link_gbps,
        "wire_gbps": wire_gbps,
        "endpoint_gbps": endpoint_gbps,
        "link_prop_ns": prop_ns,
        "pkt_overhead_ns": pkt_overhead_ns,
        "frame_overhead_bytes": tr["frame_overhead_bytes"],
        "mtu_bytes": tr["mtu_bytes"],
        "num_links": board["num_links"],
        "mem_bytes_per_ns": board["mem_gbytes_per_s"],
        "sram_bytes": board["sram_bytes"],
    }
