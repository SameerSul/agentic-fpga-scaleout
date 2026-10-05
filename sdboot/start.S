/* sdboot entry: U-Boot's "go" calls this as a function, on U-Boot's own
   stack, with the caches and the MMU off. Clear .bss, start the timer,
   run the program, and return its exit code to U-Boot. */
    .section .text.start, "ax"
    .arm
    .global _start
_start:
    push    {r4, lr}
    ldr     r0, =__bss_start
    ldr     r1, =__bss_end
    mov     r2, #0
1:  cmp     r0, r1
    strlo   r2, [r0], #4
    blo     1b
    bl      rt_init
    bl      main
    pop     {r4, pc}
