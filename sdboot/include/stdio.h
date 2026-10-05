/* sdboot: no C library on the card, so printf is sdboot/rt.c's, on the
   PS UART that U-Boot left at 115200 8N1. */
#ifndef SDBOOT_STDIO_H
#define SDBOOT_STDIO_H
#include <stdarg.h>
typedef struct { int unused; } FILE;
extern FILE *stdout;
int printf(const char *fmt, ...);
int fflush(FILE *f);
#endif
