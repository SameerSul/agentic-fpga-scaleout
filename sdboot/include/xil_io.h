/* sdboot: the Xilinx standalone BSP's register access, for a package's
   sw/main.c run bare-metal under U-Boot instead of under Vitis. */
#ifndef XIL_IO_H
#define XIL_IO_H
#include <stdint.h>
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef uint64_t u64;
typedef uintptr_t UINTPTR;
typedef intptr_t INTPTR;
#ifdef HALO_FAKE_PL
/* For QEMU, which has no PL: the register window at 0x43C00000 answered
   by sdboot/rt.c's stand-in, everything else real. */
u32 halo_fake_in(UINTPTR a);
void halo_fake_out(UINTPTR a, u32 v);
static inline u32 Xil_In32(UINTPTR a)
{
    return (a >> 16) == 0x43C0 ? halo_fake_in(a) : *(volatile u32 *)a;
}
static inline void Xil_Out32(UINTPTR a, u32 v)
{
    if ((a >> 16) == 0x43C0)
        halo_fake_out(a, v);
    else
        *(volatile u32 *)a = v;
}
#else
static inline u32 Xil_In32(UINTPTR a) { return *(volatile u32 *)a; }
static inline void Xil_Out32(UINTPTR a, u32 v) { *(volatile u32 *)a = v; }
#endif
#endif
