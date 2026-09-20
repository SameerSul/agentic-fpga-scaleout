"""FPGA portability layer: board classes as data, plus fit().

The same chiplet_profile.json (written by chiplet_flow.py) deploys onto any
board profile here with no changes upstream, and clusters may mix profiles.

Simplification, stated: lut_capacity_proxy is expressed in generic-cell units
so the yosys generic-liberty cell count maps onto it directly. Generic cells
are not FPGA LUTs; a real port would use a LUT-level utilization report from
place-and-route. FIT_FRACTION reserves fabric for the link layer, NIC, and
routing so chiplet instances never claim the whole device.
"""
import math

FIT_FRACTION = 0.7  # fraction of the capacity proxy available to chiplets

# Three board classes spanning small to large. link_gbps / link_prop_ns /
# num_links describe the transceiver class; max_chiplet_clock_mhz caps the
# chiplet clock by board fabric speed grade.
# mem_gbytes_per_s is effective DDR bandwidth available to the fabric-side
# datapath (bytes/ns numerically equals GB/s); sram_bytes is the on-chip
# BRAM+URAM budget. Weights whose per-board shard fits in sram_bytes are
# modeled as SRAM-resident (no DDR traffic per token); otherwise every token
# streams the shard from DDR. Both are representative class parameters.
BOARDS = {
    "artix7_small": {
        "name": "artix7_small",
        "class": "small (Artix-7 class)",
        "lut_capacity_proxy": 20000,
        "max_chiplet_clock_mhz": 150.0,
        "link_gbps": 6.6,
        "link_prop_ns": 500.0,
        "num_links": 4,
        "mem_gbytes_per_s": 1.6,
        "sram_bytes": 600e3,
    },
    "zynq_us_mid": {
        "name": "zynq_us_mid",
        "class": "mid (Kintex/Zynq UltraScale class)",
        "lut_capacity_proxy": 120000,
        "max_chiplet_clock_mhz": 300.0,
        "link_gbps": 16.3,
        "link_prop_ns": 400.0,
        "num_links": 8,
        "mem_gbytes_per_s": 3.2,
        "sram_bytes": 4.5e6,
    },
    "versal_large": {
        "name": "versal_large",
        "class": "large (Versal/Alveo class)",
        "lut_capacity_proxy": 450000,
        "max_chiplet_clock_mhz": 500.0,
        "link_gbps": 32.0,
        "link_prop_ns": 300.0,
        "num_links": 16,
        "mem_gbytes_per_s": 12.0,
        "sram_bytes": 24e6,
    },
}


def fit(board, chiplet_profile, fabric_profile=None):
    """Compute how the measured chiplet deploys on one board class.

    instances = floor(usable capacity / chiplet cell count)
    clock     = min(measured fmax, board clock cap)
    MACs/s    = instances * clock_mhz * 1e6 / cycles_per_mac

    When a fabric_profile (the synthesized endpoint, fabric_profile.json) is
    given, the usable link rate becomes min(board transceiver rate, endpoint
    rate): the endpoint clock caps its own datapath at min(fmax, board clock
    cap), and endpoint cells are reserved off the capacity proxy per link."""
    if isinstance(board, str):
        board = BOARDS[board]
    cells = chiplet_profile["cell_count"]
    cpm = chiplet_profile["cycles_per_mac"]
    usable = board["lut_capacity_proxy"] * FIT_FRACTION
    link_gbps = board["link_gbps"]
    endpoint_gbps = None
    if fabric_profile is not None:
        usable -= fabric_profile["cell_count"] * board["num_links"]
        ep_clock = min(fabric_profile["fmax_estimate_mhz"],
                       board["max_chiplet_clock_mhz"])
        endpoint_gbps = fabric_profile["bytes_per_cycle"] * 8 * ep_clock / 1000.0
        link_gbps = min(link_gbps, endpoint_gbps)
    instances = max(0, math.floor(usable / cells))
    clock_mhz = min(chiplet_profile["fmax_estimate_mhz"],
                    board["max_chiplet_clock_mhz"])
    macs_per_s = instances * clock_mhz * 1e6 / cpm
    return {
        "board": board["name"],
        "board_class": board["class"],
        "instances": instances,
        "clock_mhz": clock_mhz,
        "cycles_per_mac": cpm,
        "macs_per_s": macs_per_s,
        "utilization": instances * cells / board["lut_capacity_proxy"],
        "link_gbps": link_gbps,
        "transceiver_gbps": board["link_gbps"],
        "endpoint_gbps": endpoint_gbps,
        "link_prop_ns": board["link_prop_ns"],
        "num_links": board["num_links"],
        "mem_bytes_per_ns": board["mem_gbytes_per_s"],
        "sram_bytes": board["sram_bytes"],
    }
