/* sdboot: the boot script turns the caches off before it starts the
   program, so every write is already in DDR where the PL reads it. */
#ifndef XIL_CACHE_H
#define XIL_CACHE_H
#include "xil_io.h"
static inline void Xil_DCacheFlush(void) {}
static inline void Xil_DCacheFlushRange(UINTPTR a, u32 n) { (void)a; (void)n; }
#endif
