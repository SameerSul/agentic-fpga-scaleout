"""A board package's own ARM program against its own RTL, on this host.

The package's sw/main.c, compiled unchanged, runs as a host program whose
register accesses are AXI-Lite transactions on the package's RTL, the
register block, the DDR bridge and the core, built with Verilator. DDR
is one array: the program's loads from the SD card write into it at the
bus addresses the header gives, and the PL's five AXI masters read and
write the same bytes, so the weight image, the constants, the KV cache
and the vocabulary cross exactly the path they would on a Zynq. The
masters are answered after a fixed latency, and with jitter every port
also stalls its handshakes and gaps its beats at random. The ARM is
infinitely fast between accesses; the PL advances only while the program
talks to it, which a polling program always is.

Only the Zynq's surroundings are stand-ins: FatFs reads a directory,
lwIP is a UDP socket on localhost (address a.b.c.d is port PORT0 + d),
and the caches are coherent. For a multi-board run each board is its own
process, as on the boards.

Run: python3 cosim.py board_zybo --jitter
     python3 cosim.py build_x/rank0_zc706 build_x/rank1_zybo_z7_20
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# The Zynq's libraries, for a host: lwIP over a UDP socket, FatFs over a
# directory, the timer over the monotonic clock. Xil_In32 and Xil_Out32
# are the harness's.
SHIM = {
    "xil_io.h": """#pragma once
#include <stdint.h>
#include <stddef.h>
typedef uint32_t u32; typedef uint16_t u16; typedef uint8_t u8; typedef int16_t s16;
typedef uint16_t u16_t; typedef uintptr_t UINTPTR;
#ifdef __cplusplus
extern "C" {
#endif
void Xil_Out32(UINTPTR a, u32 v); u32 Xil_In32(UINTPTR a);
void *cosim_bus(UINTPTR a);
void *cosim_memset(void *p, int v, size_t n);
void cosim_idle(void);
#ifdef __cplusplus
}
#endif
""",
    "xil_cache.h": "static inline void Xil_DCacheFlush(void) {}\n",
    # The timer is the PL's: simulated nanoseconds, the bus clock's cycles,
    # so a program's timeouts mean what they mean on the board, and the
    # step times it prints are the board's, less the ARM's own work.
    "xtime_l.h": """#pragma once
typedef unsigned long long XTime;
#define COUNTS_PER_SECOND 1000000000ULL
unsigned long long cosim_ns(void);
static inline void XTime_GetTime(XTime *t) { *t = cosim_ns(); }
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
    # memset on a bus address (the KV cache's clearing) lands in DDR.
    "cosim.h": """#pragma once
#include <string.h>
#include "xil_io.h"
#define memset(p, v, n) cosim_memset((p), (v), (n))
#define FPGAI_BARRIER() __sync_synchronize()
""",
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
const ip_addr_t ip_addr_any = {0};
static int sock = -1, sent = 0;
static udp_recv_fn cb; static struct udp_pcb *cbpcb;
void init_platform(void) {} void platform_enable_interrupts(void) {} void lwip_init(void) {}
void netif_set_default(struct netif *n) { (void)n; } void netif_set_up(struct netif *n) { (void)n; }
static int port_of(uint32_t a) { return atoi(getenv("PORT0")) + (int)(a >> 24); }
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
  struct sockaddr_in sa = {0}; sa.sin_family = AF_INET; sa.sin_port = htons(port_of(dst->addr));
  sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  sendto(sock, p->payload, p->tot_len, 0, (struct sockaddr *)&sa, sizeof sa); return 0; }
int xemacif_input(struct netif *n) { (void)n; unsigned char b[2048];
  ssize_t k = recv(sock, b, sizeof b, 0);
  if (k <= 0) { cosim_idle(); return 0; }   /* the PL runs while the ARM waits */
  struct pbuf *p = pbuf_alloc(PBUF_TRANSPORT, (uint16_t)k, PBUF_RAM); memcpy(p->payload, b, k);
  ip_addr_t a = {0}; cb(0, cbpcb, p, &a, 0); return 1; }
FRESULT f_mount(FATFS *fs, const char *p, int o) { (void)fs; (void)p; (void)o; return FR_OK; }
FRESULT f_open(FIL *f, const char *n, int m) { (void)m; char path[1024];
  snprintf(path, sizeof path, "%s/%s", getenv("SD"), n); f->f = fopen(path, "rb");
  if (!f->f) return FR_NO_FILE; fseek(f->f, 0, SEEK_END); f->size = ftell(f->f); fseek(f->f, 0, SEEK_SET); return FR_OK; }
/* A read into a bus address lands in DDR. */
FRESULT f_read(FIL *f, void *b, UINT n, UINT *br) {
  *br = (UINT)fread(cosim_bus((UINTPTR)b), 1, n, f->f); return FR_OK; }
FRESULT f_close(FIL *f) { fclose(f->f); return FR_OK; }
""",
}

# The PL: the package's RTL under Verilator, its AXI-Lite slave driven by
# Xil_In32 and Xil_Out32, its masters answered from DDR.
HARNESS = r"""#include "Vfpgai_zybo.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include "xil_io.h"

static const uint64_t BASE = %(base)dULL, SIZE = %(size)dULL, REGS = %(regs)dULL;
static const int LAT = %(lat)d;
static uint8_t *ddr;
static Vfpgai_zybo *pl;
static uint64_t cyc = 0, limit = 0;
static int jit = 0;
static uint32_t rng = 12345;
static bool stall() { if (!jit) return false; rng = rng * 1103515245u + 12345u; return (rng >> 16) %% 4 == 0; }

extern "C" void *cosim_bus(UINTPTR a) {
  if (a >= BASE && a < BASE + SIZE) return ddr + (a - BASE);
  return (void *)a;
}
extern "C" void *cosim_memset(void *p, int v, size_t n) { return memset(cosim_bus((UINTPTR)p), v, n); }
static void tick();
static void up();
// The ARM polling the network with nothing to read: on a board the PL runs
// on meanwhile, so here it takes a thousand bus cycles.
extern "C" void cosim_idle(void) { up(); for (int i = 0; i < 1000; i++) tick(); }
extern "C" unsigned long long cosim_ns(void) { up(); return cyc * %(ns)dULL; }

static uint64_t rd64(uint64_t a) {
  if (a < BASE || a + 8 > BASE + SIZE) { fprintf(stderr, "PL read outside DDR: %%llx\n", (unsigned long long)a); exit(3); }
  uint64_t v; memcpy(&v, ddr + (a - BASE), 8); return v;      // little-endian, as AXI
}

struct Burst { uint64_t addr; int beats, beat; uint64_t ready; };
struct RPort { std::deque<Burst> q; bool arready, rvalid, rlast; uint64_t rdata; };
static RPort wp[4], kvr;
// KV writes: one address, then its beats, then the response.
static bool aw_have = false, bvalid = false, awready = false, wready = false;
static uint64_t aw_addr = 0; static int w_beat = 0;

#define PORT(p, f) (p == 0 ? pl->m_axi_w0_##f : p == 1 ? pl->m_axi_w1_##f : p == 2 ? pl->m_axi_w2_##f : pl->m_axi_w3_##f)
static void set_w(int p, bool arr, bool rv, bool rl, uint64_t rd) {
  switch (p) {
  case 0: pl->m_axi_w0_arready = arr; pl->m_axi_w0_rvalid = rv; pl->m_axi_w0_rlast = rl; pl->m_axi_w0_rdata = rd; pl->m_axi_w0_rresp = 0; break;
  case 1: pl->m_axi_w1_arready = arr; pl->m_axi_w1_rvalid = rv; pl->m_axi_w1_rlast = rl; pl->m_axi_w1_rdata = rd; pl->m_axi_w1_rresp = 0; break;
  case 2: pl->m_axi_w2_arready = arr; pl->m_axi_w2_rvalid = rv; pl->m_axi_w2_rlast = rl; pl->m_axi_w2_rdata = rd; pl->m_axi_w2_rresp = 0; break;
  default: pl->m_axi_w3_arready = arr; pl->m_axi_w3_rvalid = rv; pl->m_axi_w3_rlast = rl; pl->m_axi_w3_rdata = rd; pl->m_axi_w3_rresp = 0; break;
  }
}

static void drive(RPort &r) {
  r.arready = !stall();
  r.rvalid = false; r.rlast = false; r.rdata = 0;
  if (!r.q.empty() && cyc >= r.q.front().ready && !stall()) {
    Burst &b = r.q.front();
    r.rvalid = true; r.rlast = b.beat == b.beats - 1; r.rdata = rd64(b.addr + 8 * b.beat);
  }
}
static void take(RPort &r, bool arvalid, uint32_t araddr, int arlen, bool rready) {
  if (arvalid && r.arready) r.q.push_back(Burst{araddr, arlen + 1, 0, cyc + LAT});
  if (r.rvalid && rready) {
    Burst &b = r.q.front();
    if (++b.beat == b.beats) r.q.pop_front();
  }
}

// One bus clock: the slaves' outputs are registered, so they are set
// before the rising edge and every handshake is judged on the values the
// edge sees.
static void tick() {
  pl->aclk = 0; pl->eval();
  for (int p = 0; p < 4; p++) { drive(wp[p]); set_w(p, wp[p].arready, wp[p].rvalid, wp[p].rlast, wp[p].rdata); }
  drive(kvr);
  pl->m_axi_kv_arready = kvr.arready; pl->m_axi_kv_rvalid = kvr.rvalid; pl->m_axi_kv_rlast = kvr.rlast;
  pl->m_axi_kv_rdata = kvr.rdata; pl->m_axi_kv_rresp = 0;
  awready = !aw_have && !bvalid && !stall(); wready = aw_have && !stall();
  pl->m_axi_kv_awready = awready; pl->m_axi_kv_wready = wready; pl->m_axi_kv_bvalid = bvalid;
  pl->m_axi_kv_bresp = 0;
  pl->eval();
  bool av[4] = {(bool)pl->m_axi_w0_arvalid, (bool)pl->m_axi_w1_arvalid, (bool)pl->m_axi_w2_arvalid, (bool)pl->m_axi_w3_arvalid};
  uint32_t aa[4] = {pl->m_axi_w0_araddr, pl->m_axi_w1_araddr, pl->m_axi_w2_araddr, pl->m_axi_w3_araddr};
  int al[4] = {pl->m_axi_w0_arlen, pl->m_axi_w1_arlen, pl->m_axi_w2_arlen, pl->m_axi_w3_arlen};
  bool rr[4] = {(bool)pl->m_axi_w0_rready, (bool)pl->m_axi_w1_rready, (bool)pl->m_axi_w2_rready, (bool)pl->m_axi_w3_rready};
  bool kav = pl->m_axi_kv_arvalid, krr = pl->m_axi_kv_rready; uint32_t kaa = pl->m_axi_kv_araddr; int kal = pl->m_axi_kv_arlen;
  bool awv = pl->m_axi_kv_awvalid, wv = pl->m_axi_kv_wvalid, br = pl->m_axi_kv_bready;
  uint32_t awa = pl->m_axi_kv_awaddr; uint64_t wd = pl->m_axi_kv_wdata; int ws = pl->m_axi_kv_wstrb;
  bool wl = pl->m_axi_kv_wlast;
  pl->aclk = 1; pl->eval();
  for (int p = 0; p < 4; p++) take(wp[p], av[p], aa[p], al[p], rr[p]);
  take(kvr, kav, kaa, kal, krr);
  if (awv && awready) { aw_have = true; aw_addr = awa; w_beat = 0; }
  if (wv && wready) {
    uint64_t a = aw_addr + 8 * w_beat++;
    if (a < BASE || a + 8 > BASE + SIZE) { fprintf(stderr, "PL write outside DDR: %%llx\n", (unsigned long long)a); exit(3); }
    for (int k = 0; k < 8; k++) if (ws >> k & 1) ddr[a - BASE + k] = wd >> (8 * k);
    if (wl) { aw_have = false; bvalid = true; }
  }
  if (bvalid && br && pl->m_axi_kv_bvalid) bvalid = false;
  if (++cyc > limit) { fprintf(stderr, "COSIM_FAIL: %%llu bus cycles and no end\n", (unsigned long long)cyc); exit(4); }
}

static void up() {
  if (pl) return;
  ddr = (uint8_t *)calloc(SIZE, 1);
  const char *j = getenv("JITTER"); jit = j && atoi(j);
  const char *l = getenv("LIMIT"); limit = l ? strtoull(l, 0, 10) : 20000000000ULL;
  pl = new Vfpgai_zybo;
  pl->aresetn = 0; pl->s_axi_awvalid = 0; pl->s_axi_wvalid = 0; pl->s_axi_arvalid = 0;
  pl->s_axi_bready = 0; pl->s_axi_rready = 0; pl->s_axi_wstrb = 0xf;
  for (int i = 0; i < 16; i++) tick();
  pl->aresetn = 1;
  for (int i = 0; i < 4; i++) tick();
}

extern "C" void Xil_Out32(UINTPTR a, u32 v) {
  up();
  pl->s_axi_awaddr = (uint32_t)(a - REGS); pl->s_axi_wdata = v; pl->s_axi_wstrb = 0xf;
  pl->s_axi_awvalid = 1; pl->s_axi_wvalid = 1; pl->s_axi_bready = 1;
  while (pl->s_axi_awvalid || pl->s_axi_wvalid) {
    pl->aclk = 0; pl->eval();
    bool ah = pl->s_axi_awvalid && pl->s_axi_awready, wh = pl->s_axi_wvalid && pl->s_axi_wready;
    tick();
    if (ah) pl->s_axi_awvalid = 0;
    if (wh) pl->s_axi_wvalid = 0;
  }
  for (;;) {
    pl->aclk = 0; pl->eval();
    bool bh = pl->s_axi_bvalid;
    tick();
    if (bh) break;
  }
  pl->s_axi_bready = 0;
}

static u32 lite_read(uint32_t off) {
  pl->s_axi_araddr = off; pl->s_axi_arvalid = 1; pl->s_axi_rready = 1;
  for (;;) {
    pl->aclk = 0; pl->eval();
    bool ah = pl->s_axi_arready;
    tick();
    if (ah) break;
  }
  pl->s_axi_arvalid = 0;
  u32 v;
  for (;;) {
    pl->aclk = 0; pl->eval();
    bool rh = pl->s_axi_rvalid; v = pl->s_axi_rdata;
    tick();
    if (rh) break;
  }
  pl->s_axi_rready = 0;
  return v;
}

extern "C" u32 Xil_In32(UINTPTR a) {
  up();
  u32 v = lite_read((uint32_t)(a - REGS));
  // The program has read a step's token: after a head step the harness
  // reads the logit as well, which the program does not print, so a run
  // is checked to the logit as the testbenches are. Reads have no effect.
  if (a - REGS == 0x10 && (lite_read(0x00) & 2)) {
    int32_t best = (int32_t)lite_read(0x14);
    fprintf(stderr, "COSIM head tok=%%u best=%%d\n", v, best);
  }
  return v;
}

// At exit, the PL's own count of its bus cycles.
struct Report { ~Report() { if (pl) fprintf(stderr, "COSIM bus_cycles=%%llu\n", (unsigned long long)cyc); } } report_;
"""


def _header_value(h, name):
    m = re.search(r"#define %s\s+0x([0-9A-Fa-f]+)U" % name, h)
    return int(m.group(1), 16)


def build(pkg, work, lat=30, log=print, defines=None):
    """Compile pkg's RTL (Verilator) with its sw/main.c, unchanged, and the
    shims into work/cosim. Returns the executable's path. defines: macros
    for the program, such as a shorter RESEND_MS for a quick test."""
    import board_zybo
    import vsim
    pkg, work = os.path.abspath(pkg), vsim.spaceless(os.path.abspath(work))
    rtl, sw = os.path.join(pkg, "rtl"), os.path.join(pkg, "sw")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "lwip"))
    os.makedirs(os.path.join(work, "netif"))
    for name, src in SHIM.items():
        with open(os.path.join(work, name), "w") as f:
            f.write(src)
    h = open(os.path.join(sw, "fpgai_layout.h")).read()
    wb, end = _header_value(h, "WBASE"), _header_value(h, "VOCAB_BASE")
    base = board_zybo.BASE
    size = (end - base) + (48 << 20)        # the vocabulary's text after the KV cache
    # The vocabulary and a rank's gather buffer are read through pointers,
    # not loaded by address only: point them into DDR's host array, at the
    # same offsets.
    h = h.replace("#define FPGAI_LAYOUT_H\n", "#define FPGAI_LAYOUT_H\n#include \"cosim.h\"\n", 1)
    for name in ("VOCAB_BASE", "GBUF"):
        if re.search(r"#define %s\s" % name, h):
            h = re.sub(r"#define %s\s+\S+" % name, "#define %-15s ((UINTPTR)cosim_bus(0x%08XU))"
                       % (name, _header_value(h, name)), h)
    with open(os.path.join(work, "fpgai_layout.h"), "w") as f:
        f.write(h)
    shutil.copyfile(os.path.join(sw, "main.c"), os.path.join(work, "main.c"))
    with open(os.path.join(work, "harness.cpp"), "w") as f:
        f.write(HARNESS % dict(base=base, size=size, regs=board_zybo.REG_BASE, lat=lat,
                               ns=1000 // board_zybo.MHZ))
    # The RTL beside the harness, named relative to it: make cannot take
    # a source path with a space in it.
    os.makedirs(os.path.join(work, "rtl"))
    for f in os.listdir(rtl):
        if f.endswith(".v"):
            shutil.copyfile(os.path.join(rtl, f), os.path.join(work, "rtl", f))
    srcs = sorted(os.path.join("rtl", f) for f in os.listdir(rtl) if f.endswith(".v"))
    cc = shutil.which("cc") or "cc"
    dflags = ["-D%s=%s" % kv for kv in sorted((defines or {}).items())]
    for c in ("main.c", "shim.c"):
        r = subprocess.run([cc, "-O1", "-w", "-I."] + dflags + ["-c", c, "-o", c[:-2] + ".o"],
                           cwd=work, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
    r = subprocess.run(["verilator", "--cc", "--exe", "--build", "-j", "8", "-DSIM",
                        "--top-module", "fpgai_zybo", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                        "--x-assign", "0", "--x-initial", "0", "-O2", "-CFLAGS", "-I..",
                        "-LDFLAGS", "../main.o ../shim.o", "-o", "pl"] + srcs + ["harness.cpp"],
                       cwd=work, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    exe = os.path.join(work, "obj_dir", "pl")
    # $readmemh reads the gains from the working directory.
    shutil.copyfile(os.path.join(rtl, "gains.hex"), os.path.join(work, "gains.hex"))
    log("  built %s" % exe)
    return exe


def run(exe, sd, env=None, timeout=3600, wait=True):
    """Run a built program with its SD card; its UART is stdout."""
    e = dict(os.environ, SD=os.path.abspath(sd), PORT0=str(47000 + os.getpid() % 1000))
    e.update(env or {})
    p = subprocess.Popen([exe], cwd=os.path.dirname(os.path.dirname(exe)), env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not wait:
        return p
    try:
        return p.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        p.kill()
        return p.communicate()[0] + "\n(timed out)"


def uart(out):
    """The program's own output, without the harness's lines."""
    return re.sub(r"COSIM [^\n]*\n", "", out)


def heads(out):
    """(token, logit) of every head step the PL ran, as the harness saw."""
    return [(int(t), int(b)) for t, b in re.findall(r"COSIM head tok=(\d+) best=(-?\d+)", out)]


def run_group(runs, timeout=1800):
    """Several boards' programs at once, as the boards run them: runs is
    [(exe, sd, env)] in board order; all but the first start first, since
    board 0 drives the others. Returns every board's output."""
    import time
    procs = [(i, run(e, sd, env, wait=False)) for i, (e, sd, env) in list(enumerate(runs))[1:]]
    time.sleep(0.3)
    out = {0: run(runs[0][0], runs[0][1], runs[0][2], timeout=timeout)}
    for i, p in procs:
        try:
            out[i] = p.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            p.kill()
            out[i] = p.communicate()[0]
    return [out[i] for i in range(len(runs))]


class Tokens:
    """A stand-in tokenizer for a synthetic checkpoint: token t is "<t>"."""
    def __init__(self, V):
        self.inv = {t: "<%d>" % t for t in range(V)}
        self.u2b = {chr(c): c for c in range(128)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("packages", nargs="+",
                    help="a board package (rtl/, sw/, sd/), or several in board order: the "
                    "stages of a layer split or the ranks of a weight split, each its own "
                    "process on localhost UDP")
    ap.add_argument("--work", default=os.path.join(ROOT, "build_cosim"))
    ap.add_argument("--jitter", action="store_true",
                    help="every AXI port stalls and gaps its beats at random")
    a = ap.parse_args()
    env = {"JITTER": "1" if a.jitter else "0"}
    runs = [(build(p, os.path.join(a.work, "%d_%s" % (i, os.path.basename(os.path.abspath(p))))),
             os.path.join(p, "sd"), env) for i, p in enumerate(a.packages)]
    outs = run_group(runs, timeout=24 * 3600) if len(runs) > 1 else \
        [run(runs[0][0], runs[0][1], env, timeout=24 * 3600)]
    for p, o in zip(a.packages, outs):
        print("==== %s" % p)
        print(uart(o).rstrip())
        print("head steps (token, logit): %s" % heads(o))


if __name__ == "__main__":
    main()
