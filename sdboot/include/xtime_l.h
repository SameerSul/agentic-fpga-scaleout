/* sdboot: the Cortex-A9 global timer, as the standalone BSP reads it. It
   counts at half the CPU clock: 333.33 MHz on the ZC706 and the Zybo. */
#ifndef XTIME_L_H
#define XTIME_L_H
#include "xil_io.h"
typedef u64 XTime;
#define COUNTS_PER_SECOND 333333333ULL
void XTime_GetTime(XTime *t);
#endif
