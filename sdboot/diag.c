/* Halo diagnostics: a halo.bin for a board run whose answer is wrong. It
   runs in place of the package's sw/main.c, from the same card, and
   prints what it finds:

     A  the clocks, resets and caches U-Boot left
     B  the PL's registers: every bit of TOK, POS and CTRL written and
        read back over GP0
     C  the model files in DDR, sampled, against the files the program
        was built with
     D  one step of the PL, token DX_TOK at position 0 with the head:
        its token and logit, and every K and V value it wrote to the KV
        cache, layer by layer, against the same RTL in simulation
        (cosim.py, which this file also runs under, as DIAG_RECORD, to
        record them: sdboot.py --diag)
     E  the card's own prompt, as sw/main.c runs it, with each token's id
        and logit
     G  D again with what software can change: the SCU on, the HP ports
        at 64 bits, if U-Boot left them otherwise
     F  D again at a slower PL clock (25 and 10 MHz), and E at the first
        clock where D matches

   Everything the PL computes is data-independent in time, so a step that
   takes its usual cycles and still answers wrong points at data: the
   registers (B), the files (C), or what the PL does with them (D, F). */
#include <stdio.h>
#include <string.h>
#include "xil_io.h"
#include "xtime_l.h"
#include "ff.h"
#include "fpgai_layout.h"

#ifdef DIAG_RECORD
/* Under cosim.py: DDR is the harness's array. */
#define AT(a) ((volatile u32 *)cosim_bus((UINTPTR)(a)))
#else
#define AT(a) ((volatile u32 *)(UINTPTR)(a))
#include "diag_expect.h"
#endif

#define REG(o) (FPGAI_REGS + (o))
#define DX_PAT 0xA5A5U                  /* the KV cache's fill before a step */
#define DX_PAT32 (DX_PAT * 0x10001U)
#ifndef DX_TOK
#define DX_TOK 785                      /* Qwen's "The" */
#endif
#define KV_BYTES (KVEND - KBASE)

static FATFS fs;
static u32 prompt[MAX_POS + 1];
static const u32 *vocab;

static int load(const char *name, UINTPTR addr, u32 want)
{
    FIL f;
    UINT n;
    u32 got = 0;
    if (f_open(&f, name, FA_READ) != FR_OK) {
        printf("missing %s\r\n", name);
        return -1;
    }
    if (want && f_size(&f) != want) {
        printf("%s is %lu bytes, this bitstream wants %lu\r\n", name,
               (unsigned long)f_size(&f), (unsigned long)want);
        return -1;
    }
    while (f_read(&f, (void *)(addr + got), 1 << 20, &n) == FR_OK && n)
        got += n;
    f_close(&f);
    return (int)got;
}

static void print_tok(u32 t)
{
    u32 n = vocab[0];
    const char *s = (const char *)(vocab + n + 2);
    if (t < n)
        printf("%.*s", (int)(vocab[t + 2] - vocab[t + 1]), s + vocab[t + 1]);
    fflush(stdout);
}

static double secs(XTime a, XTime b) { return (double)(b - a) / COUNTS_PER_SECOND; }

static void wait_us(u32 us)
{
    XTime a, b;
    XTime_GetTime(&a);
    do
        XTime_GetTime(&b);
    while (b - a < (XTime)us * (COUNTS_PER_SECOND / 1000000));
}

/* A step, as sw/main.c runs it, with a time limit: a PL that never
   finishes is reported, not waited on for good. */
static int step(u32 tok, u32 pos, int head, u32 *next)
{
    XTime a, b;
    Xil_Out32(REG(FPGAI_TOK), tok);
    Xil_Out32(REG(FPGAI_POS), pos);
    Xil_Out32(REG(FPGAI_CTRL), 1 | (head ? 2 : 0));
    XTime_GetTime(&a);
    while (!(Xil_In32(REG(FPGAI_STATUS)) & 2)) {
        XTime_GetTime(&b);
        if (secs(a, b) > 60.0) {
            printf("  the step did not finish in 60 s (STATUS 0x%lx)\r\n",
                   (unsigned long)Xil_In32(REG(FPGAI_STATUS)));
            return -1;
        }
    }
    *next = Xil_In32(REG(FPGAI_NEXT_TOK));
    return 0;
}

/* ---- A: what U-Boot left -------------------------------------------- */
#ifndef DIAG_RECORD
#define SLCR_UNLOCK 0xF8000008U
#define ARM_PLL_CTRL 0xF8000100U
#define DDR_PLL_CTRL 0xF8000104U
#define IO_PLL_CTRL 0xF8000108U
#define FPGA0_CLK_CTRL 0xF8000170U
#define FPGA_RST_CTRL 0xF8000240U
#define LVL_SHFTR_EN 0xF8000900U
#define PS_CLK_HZ 33333333U             /* ZC706 and Zybo Z7 */
#define SCU_CTRL 0xF8F00000U
#define AFI_RDCHAN_CTRL(i) (0xF8008000U + 0x1000U * (i))

static u32 pll_hz(u32 ctrl)
{
    if (ctrl & 0x10)                    /* PLL_BYPASS_FORCE */
        return PS_CLK_HZ;
    return PS_CLK_HZ * ((ctrl >> 12) & 0x7F);
}

static u32 fclk0_src_hz(void)
{
    u32 src = (Xil_In32(FPGA0_CLK_CTRL) >> 4) & 3;
    return pll_hz(Xil_In32(src == 2 ? ARM_PLL_CTRL : src == 3 ? DDR_PLL_CTRL : IO_PLL_CTRL));
}

static u32 fclk0_hz(void)
{
    u32 c = Xil_In32(FPGA0_CLK_CTRL), d0 = (c >> 8) & 0x3F, d1 = (c >> 20) & 0x3F;
    return d0 && d1 ? fclk0_src_hz() / (d0 * d1) : 0;
}

static void env(void)
{
    u32 sctlr;
    __asm__ volatile("mrc p15, 0, %0, c1, c0, 0" : "=r"(sctlr));
    printf("A  CPU: MMU %s, D-cache %s, I-cache %s (SCTLR 0x%08lx); L2 0x%lx, SCU 0x%lx\r\n",
           sctlr & 1 ? "on" : "off", sctlr & 4 ? "on" : "off", sctlr & 0x1000 ? "on" : "off",
           (unsigned long)sctlr, (unsigned long)Xil_In32(0xF8F02100U),
           (unsigned long)Xil_In32(SCU_CTRL));
    printf("   IO PLL %lu MHz (0x%08lx), FCLK0 %lu.%02lu MHz (FPGA0_CLK_CTRL 0x%08lx)\r\n",
           (unsigned long)(pll_hz(Xil_In32(IO_PLL_CTRL)) / 1000000),
           (unsigned long)Xil_In32(IO_PLL_CTRL), (unsigned long)(fclk0_hz() / 1000000),
           (unsigned long)(fclk0_hz() / 10000 % 100), (unsigned long)Xil_In32(FPGA0_CLK_CTRL));
    printf("   FPGA_RST_CTRL 0x%lx, LVL_SHFTR_EN 0x%lx, AFI read width 0x%lx 0x%lx 0x%lx 0x%lx\r\n",
           (unsigned long)Xil_In32(FPGA_RST_CTRL), (unsigned long)Xil_In32(LVL_SHFTR_EN),
           (unsigned long)Xil_In32(AFI_RDCHAN_CTRL(0)), (unsigned long)Xil_In32(AFI_RDCHAN_CTRL(1)),
           (unsigned long)Xil_In32(AFI_RDCHAN_CTRL(2)), (unsigned long)Xil_In32(AFI_RDCHAN_CTRL(3)));
}

/* FCLK0 at hz, the PL held in reset while it changes. */
static void set_fclk0(u32 hz)
{
    u32 c = Xil_In32(FPGA0_CLK_CTRL), src = fclk0_src_hz();
    u32 total = (src + hz / 2) / hz, d0 = 0, d1;
    for (d1 = 1; d1 < 64; d1++)
        if (total % d1 == 0 && total / d1 < 64) {
            d0 = total / d1;
            break;
        }
    if (!d0)
        return;
    Xil_Out32(SLCR_UNLOCK, 0xDF0D);
    Xil_Out32(FPGA_RST_CTRL, Xil_In32(FPGA_RST_CTRL) | 1);
    wait_us(1000);
    Xil_Out32(FPGA0_CLK_CTRL, (c & ~((0x3FU << 20) | (0x3FU << 8))) | (d1 << 20) | (d0 << 8));
    wait_us(10000);
    Xil_Out32(FPGA_RST_CTRL, Xil_In32(FPGA_RST_CTRL) & ~1U);
    wait_us(10000);
}
#endif

/* ---- B: GP0 writes, read back --------------------------------------- */
/* A lost address bit sends a TOK write to CTRL, where an odd value starts
   a step: any write that leaves the PL busy is waited out and reported. */
static int wrote_started(const char *what)
{
    if (!(Xil_In32(REG(FPGAI_STATUS)) & 1))
        return 0;
    printf("   writing %s started a step: the write went to CTRL\r\n", what);
    for (int i = 0; i < 6000 && !(Xil_In32(REG(FPGAI_STATUS)) & 2); i++)
        wait_us(10000);
    return 1;
}

static int regs(void)
{
    u32 tok_bad = 0, pos_bad = 0, ctrl_bad = 0, stray = 0;
    for (int b = 17; b >= 0; b--) {
        u32 v = 1U << b;
        Xil_Out32(REG(FPGAI_TOK), v);
        stray |= (u32)wrote_started("TOK");
        tok_bad |= (Xil_In32(REG(FPGAI_TOK)) ^ v) & 0x3FFFF;
    }
    Xil_Out32(REG(FPGAI_TOK), 0x3FFFE);
    tok_bad |= (Xil_In32(REG(FPGAI_TOK)) ^ 0x3FFFE) & 0x3FFFF;
    for (int b = 7; b >= 0; b--) {
        u32 v = 1U << b;
        Xil_Out32(REG(FPGAI_POS), v);
        stray |= (u32)wrote_started("POS");
        pos_bad |= (Xil_In32(REG(FPGAI_POS)) ^ v) & 0xFF;
    }
    Xil_Out32(REG(FPGAI_CTRL), 2);
    ctrl_bad |= (Xil_In32(REG(FPGAI_CTRL)) ^ 2) & 2;
    Xil_Out32(REG(FPGAI_CTRL), 0);
    ctrl_bad |= Xil_In32(REG(FPGAI_CTRL)) & 2;
    Xil_Out32(REG(FPGAI_TOK), DX_TOK);
    if (!tok_bad && !pos_bad && !ctrl_bad && !stray) {
        printf("B  registers: every bit of TOK, POS and head_en reads back as written\r\n");
        return 0;
    }
    printf("B  registers: bits that do not read back as written: TOK 0x%05lx, POS 0x%02lx, "
           "head_en %lu\r\n", (unsigned long)tok_bad, (unsigned long)pos_bad,
           (unsigned long)(ctrl_bad >> 1));
    return 1;
}

/* ---- C: the files in DDR -------------------------------------------- */
/* FNV-1a over a file's first 4 KB, 64 bytes at every MB and its last 64:
   sdboot.py computes the same over the package's sd/ files. */
static u32 fnv(u32 h, UINTPTR a, u32 n)
{
    for (u32 i = 0; i < n; i++)
        h = (h ^ *(volatile const u8 *)(a + i)) * 16777619U;
    return h;
}

static u32 sample(UINTPTR a, u32 size)
{
    u32 h = fnv(2166136261U, a, size < 4096 ? size : 4096);
    for (u32 o = 1 << 20; o + 64 <= size; o += 1 << 20)
        h = fnv(h, a + o, 64);
    return size >= 64 ? fnv(h, a + size - 64, 64) : h;
}

#ifndef DIAG_RECORD
static int files(u32 wsize, u32 csize, u32 vsize)
{
    u32 h[3] = {sample(WBASE, wsize), sample(CBASE, csize), sample(VOCAB_BASE, vsize)};
    static const char *const name[3] = {"weights8.bin", "cparams.bin", "vocab.bin"};
    static const u32 want[3] = {DX_HASH_W, DX_HASH_C, DX_HASH_V};
    int bad = 0;
    for (int i = 0; i < 3; i++)
        bad |= h[i] != want[i];
    if (!bad) {
        printf("C  files: weights8.bin, cparams.bin and vocab.bin in DDR are this "
               "program's, sampled\r\n");
        return 0;
    }
    for (int i = 0; i < 3; i++)
        printf("C  %s in DDR: %s (0x%08lx, the program's 0x%08lx)\r\n", name[i],
               h[i] == want[i] ? "the program's" : "NOT the program's",
               (unsigned long)h[i], (unsigned long)want[i]);
    return 1;
}
#endif

/* ---- D: one step, its KV cache against the simulation's -------------- */
static void fill(void)
{
    volatile u32 *p = AT(KBASE);
    for (u32 i = 0; i < KV_BYTES / 4; i++)
        p[i] = DX_PAT32;
}

#ifdef DIAG_RECORD
static void record(void)
{
    u32 next, c0, c1;
    fill();
    c0 = Xil_In32(REG(FPGAI_CORE_CYCLES));
    if (step(DX_TOK, 0, 1, &next) < 0)
        return;
    c1 = Xil_In32(REG(FPGAI_CORE_CYCLES));
    printf("DX step %u %u %d %u %u\n", (unsigned)DX_TOK, (unsigned)next,
           (int)Xil_In32(REG(FPGAI_BEST)), (unsigned)(c1 - c0),
           (unsigned)Xil_In32(REG(FPGAI_BUS_CYCLES)));
    printf("DX vbase %u\n", (unsigned)(Xil_In32(REG(FPGAI_VBASE)) - KBASE));
    volatile const u32 *p = AT(KBASE);
    for (u32 i = 0; i < KV_BYTES / 4; i++) {
        u32 w = p[i];
        if ((w & 0xFFFF) != DX_PAT)
            printf("DX w %u %u\n", (unsigned)(4 * i), (unsigned)(w & 0xFFFF));
        if ((w >> 16) != DX_PAT)
            printf("DX w %u %u\n", (unsigned)(4 * i + 2), (unsigned)(w >> 16));
    }
    fflush(stdout);                     /* before the harness's own last line */
}
#else
#define NG (2 * DX_NL)
static u32 g_tot[NG], g_ok[NG], g_pat[NG], g_zero[NG], g_bits[NG], g_stray[NG];
static u32 g_off[NG], g_got[NG], g_want[NG];

static int group(u32 off, u32 vhalf)
{
    u32 per = vhalf / DX_NL;
    return off < vhalf ? (int)(off / per) : DX_NL + (int)((off - vhalf) / per);
}

/* Returns 0 when the step matches the simulation everywhere; *hash is
   over every value it wrote where the simulation wrote, so two runs that
   both differ can still be told apart or alike. */
static int step_test(int again, int verbose, u32 *hash)
{
    u32 next, c0, c1, e = 0, ok = 0, stray = 0, first_stray = 0, first_val = 0;
    XTime a, b;
    u32 vhalf = Xil_In32(REG(FPGAI_VBASE)) - KBASE;
    fill();
    c0 = Xil_In32(REG(FPGAI_CORE_CYCLES));
    XTime_GetTime(&a);
    if (step(DX_TOK, 0, 1, &next) < 0)
        return -1;
    XTime_GetTime(&b);
    c1 = Xil_In32(REG(FPGAI_CORE_CYCLES));
    int best = (int)Xil_In32(REG(FPGAI_BEST));
    u32 bus = Xil_In32(REG(FPGAI_BUS_CYCLES));
    u32 hz = fclk0_hz();
    if (again)
        printf("D  again");
    else
        printf("D  at %lu.%02lu MHz", (unsigned long)(hz / 1000000),
               (unsigned long)(hz / 10000 % 100));
    printf(": token %lu, logit %d (simulation: %lu, %d); %lu core cycles "
           "(simulation %lu), %lu bus cycles, %.3f s\r\n", (unsigned long)next, best,
           (unsigned long)DX_NEXT, DX_BEST, (unsigned long)(c1 - c0),
           (unsigned long)DX_CORE, (unsigned long)bus, secs(a, b));
    memset(g_tot, 0, sizeof g_tot);
    memset(g_ok, 0, sizeof g_ok);
    memset(g_pat, 0, sizeof g_pat);
    memset(g_zero, 0, sizeof g_zero);
    memset(g_bits, 0, sizeof g_bits);
    memset(g_stray, 0, sizeof g_stray);
    *hash = 2166136261U;
    volatile const u32 *p = AT(KBASE);
    for (u32 i = 0; i < KV_BYTES / 4; i++) {
        u32 w = p[i];
        for (u32 h = 0; h < 2; h++) {
            u32 off = 4 * i + 2 * h, v = h ? w >> 16 : w & 0xFFFF;
            while (e < DX_N && dx_off[e] < off)
                e++;
            if (e < DX_N && dx_off[e] == off) {
                int g = group(off, vhalf);
                u32 want = dx_val[e];
                *hash = (*hash ^ v) * 16777619U;
                g_tot[g]++;
                if (v == want) {
                    g_ok[g]++;
                    ok++;
                    continue;
                }
                if (!g_bits[g]) {
                    g_off[g] = off;
                    g_got[g] = v;
                    g_want[g] = want;
                }
                g_pat[g] += v == DX_PAT;
                g_zero[g] += v == 0;
                g_bits[g] |= v ^ want;
            } else if (v != DX_PAT) {
                if (!stray++) {
                    first_stray = off;
                    first_val = v;
                }
                g_stray[group(off, vhalf)]++;
            }
        }
    }
    int match = ok == DX_N && !stray && next == DX_NEXT && best == DX_BEST;
    if (match) {
        printf("   all %u K and V values of the %d layers are the simulation's\r\n",
               (unsigned)DX_N, DX_NL);
        return 0;
    }
    printf("   %lu of the %u K and V values are the simulation's (values 0x%08lx); %lu "
           "writes where it wrote none%s", (unsigned long)ok, (unsigned)DX_N,
           (unsigned long)*hash, (unsigned long)stray, stray ? "" : "\r\n");
    if (stray)
        printf(" (first at KBASE+0x%lx: 0x%04lx)\r\n", (unsigned long)first_stray,
               (unsigned long)first_val);
    int shown = 0;
    for (int g = 0; g < NG; g++) {
        if (g_ok[g] == g_tot[g] && !g_stray[g])
            continue;
        if (!verbose && shown == 6) {
            printf("   ...\r\n");
            break;
        }
        shown++;
        printf("   layer %2d %c: %4lu of %4lu (%lu untouched, %lu zero, bits 0x%04lx", g % DX_NL,
               g < DX_NL ? 'K' : 'V', (unsigned long)g_ok[g], (unsigned long)g_tot[g],
               (unsigned long)g_pat[g], (unsigned long)g_zero[g], (unsigned long)g_bits[g]);
        if (g_ok[g] != g_tot[g])
            printf("; +0x%lx: 0x%04lx, simulation 0x%04lx", (unsigned long)g_off[g],
                   (unsigned long)g_got[g], (unsigned long)g_want[g]);
        printf(")%s\r\n", g_stray[g] ? ", and writes elsewhere" : "");
    }
    return 1;
}
#endif

/* ---- E: the card's prompt -------------------------------------------- */
static void run_prompt(void)
{
    static u32 ids[N_GEN];
    static int bests[N_GEN];
    u32 n = prompt[0], tok = 0, pos, ng = 0;
    XTime a, b;
    memset((void *)KBASE, 0, KV_BYTES);
    printf("E  the card's prompt, %lu tokens:\r\n", (unsigned long)n);
    XTime_GetTime(&a);
    for (pos = 0; pos < n && pos < MAX_POS; pos++) {
        print_tok(prompt[pos + 1]);
        if (step(prompt[pos + 1], pos, pos == n - 1, &tok) < 0)
            return;
    }
    for (int g = 0; g < N_GEN && pos < MAX_POS; g++, pos++) {
        ids[ng] = tok;
        bests[ng++] = (int)Xil_In32(REG(FPGAI_BEST));
        print_tok(tok);
        if (g + 1 == N_GEN)
            break;
        if (step(tok, pos, 1, &tok) < 0)
            return;
    }
    XTime_GetTime(&b);
    printf("\r\n   generated (token logit):");
    for (u32 i = 0; i < ng; i++)
        printf(" %lu %d", (unsigned long)ids[i], bests[i]);
    printf("\r\n   %.1f s\r\n", secs(a, b));
}

int main(void)
{
    u32 wsize, csize, vsize;
    printf("\r\nHalo diagnostics on the " FPGAI_BOARD " PL\r\n");
#ifndef DIAG_RECORD
    env();
#endif
    if (Xil_In32(REG(FPGAI_ID)) != FPGAI_ID_VALUE || Xil_In32(REG(FPGAI_WBASE)) != WBASE ||
        Xil_In32(REG(FPGAI_KVEND)) != KVEND) {
        printf("the bitstream's layout is not this program's (ID 0x%08lx)\r\n",
               (unsigned long)Xil_In32(REG(FPGAI_ID)));
        return 1;
    }
    if (f_mount(&fs, "0:/", 1) != FR_OK)
        return 1;
    int w = load("weights8.bin", WBASE, WBYTES), c = load("cparams.bin", CBASE, CBYTES),
        v = load("vocab.bin", VOCAB_BASE, 0);
    if (w < 0 || c < 0 || v < 0 || load("prompt.bin", (UINTPTR)prompt, 0) < 0)
        return 1;
    wsize = (u32)w, csize = (u32)c, vsize = (u32)v;
    vocab = (const u32 *)VOCAB_BASE;
    regs();
#ifdef DIAG_RECORD
    printf("DX hash %u %u %u\n", (unsigned)sample((UINTPTR)cosim_bus(WBASE), wsize),
           (unsigned)sample((UINTPTR)cosim_bus(CBASE), csize),
           (unsigned)sample((UINTPTR)VOCAB_BASE, vsize));
    record();
    return 0;
#else
    u32 h1, h2;
    files(wsize, csize, vsize);
    printf("D  token %u (\"", (unsigned)DX_TOK);
    print_tok(DX_TOK);
    printf("\") at position 0, the head on; the KV cache filled with 0x%04x first\r\n",
           (unsigned)DX_PAT);
    int r1 = step_test(0, 1, &h1);
    int r2 = step_test(1, 0, &h2);
    if (r1 > 0 && r2 > 0)
        printf("   the two runs %s\r\n", h1 == h2 ? "wrote the same values: the "
               "difference is systematic" : "wrote different values: it varies run to run");
    run_prompt();
    if (r1 == 0 && r2 == 0) {
        printf("Halo diagnostics done: the PL matches the simulation\r\n");
        return 0;
    }
    /* G: what software can change. U-Boot leaves the SCU, which the ACP
       (the KV cache's port) goes through, as it found it; a Vitis boot
       enables it. The AFIs reset to 64-bit reads, the width the PL uses. */
    if (!(Xil_In32(SCU_CTRL) & 1)) {
        Xil_Out32(SCU_CTRL, Xil_In32(SCU_CTRL) | 1);
        printf("G  SCU enabled (0x%lx)\r\n", (unsigned long)Xil_In32(SCU_CTRL));
        if (step_test(0, 0, &h1) == 0) {
            run_prompt();
            printf("Halo diagnostics done: the PL matches the simulation with the SCU on\r\n");
            return 0;
        }
    }
    u32 afi32 = 0;
    for (u32 i = 0; i < 4; i++)
        afi32 |= Xil_In32(AFI_RDCHAN_CTRL(i)) & 1;
    if (afi32) {
        Xil_Out32(SLCR_UNLOCK, 0xDF0D);
        for (u32 i = 0; i < 4; i++)
            Xil_Out32(AFI_RDCHAN_CTRL(i), Xil_In32(AFI_RDCHAN_CTRL(i)) & ~1U);
        printf("G  HP read ports set to 64 bits\r\n");
        if (step_test(0, 0, &h1) == 0) {
            run_prompt();
            printf("Halo diagnostics done: the PL matches the simulation with 64-bit HP ports\r\n");
            return 0;
        }
    }
    static const u32 slow[] = {25000000, 10000000};
    for (unsigned i = 0; i < sizeof slow / sizeof slow[0]; i++) {
        set_fclk0(slow[i]);
        printf("F  FCLK0 now %lu.%02lu MHz\r\n", (unsigned long)(fclk0_hz() / 1000000),
               (unsigned long)(fclk0_hz() / 10000 % 100));
        if (Xil_In32(REG(FPGAI_ID)) != FPGAI_ID_VALUE) {
            printf("   the PL does not answer at this clock\r\n");
            break;
        }
        if (step_test(0, 0, &h1) == 0) {
            run_prompt();
            break;
        }
    }
    printf("Halo diagnostics done\r\n");
    return 0;
#endif
}
