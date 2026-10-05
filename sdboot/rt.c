/* sdboot runtime: what a package's sw/main.c needs from the Xilinx
   standalone BSP and from a C library, for the program to run bare-metal
   under U-Boot, from a card that also holds the bitstream.

   The card's boot script (sdboot.py) has U-Boot program the PL, read each
   model file into the address the program loads it to, write the file's
   size into the table below, turn the caches off and start the program.
   FatFs is then a view of DDR: f_open finds the file's address and size
   in the table, and f_read copies nothing when the program asks for the
   bytes where they already are. The program, unchanged, makes the same
   checks it makes on a Vitis build: the layout registers against its
   header, each file's size against the bitstream's. */
#include <stdarg.h>
#include <stddef.h>
#include "xil_io.h"
#include "xtime_l.h"
#include "ff.h"
#include "fpgai_layout.h"

#ifndef HALO_TABLE
#define HALO_TABLE 0x07F00000U          /* below WBASE, above the program */
#endif
#define HALO_MAGIC 0x48414C4FU          /* "HALO" */
#define HALO_PROMPT (HALO_TABLE + 0x1000U)
#ifndef HALO_UART
#define HALO_UART 0xE0001000U           /* UART1: the USB-UART, ZC706 and Zybo */
#endif

/* ---- the UART U-Boot left configured -------------------------------- */
#define UART_SR 0x2C
#define UART_FIFO 0x30
#define UART_TXFULL (1U << 4)

static void putch(char c)
{
    while (Xil_In32(HALO_UART + UART_SR) & UART_TXFULL)
        ;
    Xil_Out32(HALO_UART + UART_FIFO, (u32)(unsigned char)c);
}

typedef struct { int unused; } FILE;
static FILE out_file;
FILE *stdout = &out_file;
int fflush(FILE *f) { (void)f; return 0; }

/* ---- printf: what the programs print, nothing more ------------------ */
static int emit_num(unsigned long long v, int base, int upper, int width, int zero,
                    int neg)
{
    char buf[24];
    const char *dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
    int n = 0, out = 0;
    do {
        buf[n++] = dig[v % (unsigned)base];
        v /= (unsigned)base;
    } while (v);
    if (neg)
        width--;
    if (neg && zero) {
        putch('-');
        out++;
    }
    for (int i = n; i < width; i++, out++)
        putch(zero ? '0' : ' ');
    if (neg && !zero) {
        putch('-');
        out++;
    }
    while (n)
        putch(buf[--n]), out++;
    return out;
}

int printf(const char *fmt, ...)
{
    va_list ap;
    int out = 0;
    va_start(ap, fmt);
    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            putch(*fmt);
            out++;
            continue;
        }
        fmt++;
        int zero = 0, width = 0, prec = -1, lng = 0;
        if (*fmt == '0') {
            zero = 1;
            fmt++;
        }
        while (*fmt >= '0' && *fmt <= '9')
            width = width * 10 + (*fmt++ - '0');
        if (*fmt == '.') {
            fmt++;
            if (*fmt == '*') {
                prec = va_arg(ap, int);
                fmt++;
            } else {
                prec = 0;
                while (*fmt >= '0' && *fmt <= '9')
                    prec = prec * 10 + (*fmt++ - '0');
            }
        }
        while (*fmt == 'l' || *fmt == 'z') {
            lng += *fmt == 'l';
            fmt++;
        }
        switch (*fmt) {
        case 'c':
            putch((char)va_arg(ap, int));
            out++;
            break;
        case 's': {
            const char *s = va_arg(ap, const char *);
            for (int i = 0; s[i] && (prec < 0 || i < prec); i++, out++)
                putch(s[i]);
            break;
        }
        case 'd':
        case 'i': {
            long long v = lng > 1 ? va_arg(ap, long long) : va_arg(ap, long);
            out += emit_num(v < 0 ? -(unsigned long long)v : (unsigned long long)v, 10, 0,
                            width, zero, v < 0);
            break;
        }
        case 'u':
            out += emit_num(lng > 1 ? va_arg(ap, unsigned long long)
                                    : va_arg(ap, unsigned long), 10, 0, width, zero, 0);
            break;
        case 'x':
        case 'X':
            out += emit_num(lng > 1 ? va_arg(ap, unsigned long long)
                                    : va_arg(ap, unsigned long), 16, *fmt == 'X',
                            width, zero, 0);
            break;
        case 'p':
            out += emit_num((unsigned long)va_arg(ap, void *), 16, 0, 8, 1, 0);
            break;
        case 'f': {
            double v = va_arg(ap, double);
            int p = prec < 0 ? 6 : prec, neg = v < 0;
            unsigned long long scale = 1;
            for (int i = 0; i < p; i++)
                scale *= 10;
            if (neg)
                v = -v;
            unsigned long long all = (unsigned long long)(v * (double)scale + 0.5);
            if (neg) {
                putch('-');
                out++;
            }
            out += emit_num(all / scale, 10, 0, 0, 0, 0);
            if (p) {
                putch('.');
                out++;
                out += emit_num(all % scale, 10, 0, p, 1, 0);
            }
            break;
        }
        case '%':
            putch('%');
            out++;
            break;
        default:
            putch('%');
            putch(*fmt);
            out += 2;
        }
    }
    va_end(ap);
    return out;
}

/* ---- the C library the program uses ---------------------------------- */
void *memset(void *d, int c, size_t n)
{
    unsigned char *p = d;
    u32 w = (unsigned char)c * 0x01010101U;
    while (n && ((UINTPTR)p & 3)) {
        *p++ = (unsigned char)c;
        n--;
    }
    for (; n >= 4; n -= 4, p += 4)
        *(volatile u32 *)p = w;
    while (n--)
        *p++ = (unsigned char)c;
    return d;
}

void *memcpy(void *d, const void *s, size_t n)
{
    unsigned char *a = d;
    const unsigned char *b = s;
    while (n--)
        *a++ = *b++;
    return d;
}

int strcmp(const char *a, const char *b)
{
    while (*a && *a == *b)
        a++, b++;
    return (unsigned char)*a - (unsigned char)*b;
}

/* ---- the global timer ------------------------------------------------ */
#define GT_LO 0xF8F00200U
#define GT_HI 0xF8F00204U
#define GT_CTRL 0xF8F00208U

void XTime_GetTime(XTime *t)
{
    u32 hi, lo;
    do {
        hi = Xil_In32(GT_HI);
        lo = Xil_In32(GT_LO);
    } while (Xil_In32(GT_HI) != hi);
    *t = ((XTime)hi << 32) | lo;
}

void rt_init(void)
{
    Xil_Out32(GT_CTRL, Xil_In32(GT_CTRL) | 1U);        /* timer running */
}

/* ---- FatFs, as a view of what U-Boot already put in DDR ------------- */
static const struct { const char *name; UINTPTR addr; int slot; } FILES[] = {
    {"weights8.bin", WBASE, 1},
    {"cparams.bin", CBASE, 2},
    {"vocab.bin", 0, 3},                /* VOCAB_BASE, set below */
    {"prompt.bin", HALO_PROMPT, 4},
};

static volatile const u32 *table(void) { return (volatile const u32 *)HALO_TABLE; }

FRESULT f_mount(FATFS *fs, const char *path, unsigned char opt)
{
    (void)fs; (void)path; (void)opt;
    if (table()[0] != HALO_MAGIC) {
        printf("sdboot: no file table at 0x%08lx; the boot script did not run\r\n",
               (unsigned long)HALO_TABLE);
        return FR_DISK_ERR;
    }
    return FR_OK;
}

FRESULT f_open(FIL *fp, const char *path, unsigned char mode)
{
    (void)mode;
    for (unsigned i = 0; i < sizeof FILES / sizeof FILES[0]; i++)
        if (strcmp(path, FILES[i].name) == 0) {
            u32 size = table()[FILES[i].slot];
            if (!size)
                return FR_NO_FILE;
            fp->src = (const u8 *)(FILES[i].slot == 3 ? (UINTPTR)VOCAB_BASE : FILES[i].addr);
            fp->size = size;
            fp->pos = 0;
            return FR_OK;
        }
    return FR_NO_FILE;
}

FRESULT f_read(FIL *fp, void *buf, UINT btr, UINT *br)
{
    u32 n = fp->size - fp->pos;
    if (n > btr)
        n = btr;
    if ((const u8 *)buf != fp->src + fp->pos)
        memcpy(buf, fp->src + fp->pos, n);
    fp->pos += n;
    *br = n;
    return FR_OK;
}

FRESULT f_close(FIL *fp)
{
    (void)fp;
    return FR_OK;
}

#ifdef HALO_FAKE_PL
/* ---- QEMU only: a stand-in for the PL's registers --------------------
   It answers the layout check and gives, at each head step, the next of
   the tokens the package's program printed in the cosim. It checks the
   boot chain, the files and the program, not the PL. */
static const u32 fake_tokens[] = {HALO_FAKE_TOKENS};
static u32 fake_next, fake_head;

u32 halo_fake_in(UINTPTR a)
{
    switch (a - FPGAI_REGS) {
    case FPGAI_ID: return FPGAI_ID_VALUE;
    case FPGAI_WBASE: return WBASE;
    case FPGAI_KVEND: return KVEND;
    case FPGAI_STATUS: return 2;
    case FPGAI_NEXT_TOK: return fake_next;
    default: return 0;
    }
}

void halo_fake_out(UINTPTR a, u32 v)
{
    if (a - FPGAI_REGS == FPGAI_CTRL && (v & 1) && (v & 2)) {
        u32 n = sizeof fake_tokens / sizeof fake_tokens[0];
        fake_next = fake_tokens[fake_head < n ? fake_head : n - 1];
        fake_head++;
    }
}
#endif
