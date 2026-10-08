/* sdboot entry for a JTAG load (sdboot.py --jtag): XSCT has run ps7_init,
   downloaded this program and the model files, and set the PC here, with
   no U-Boot underneath. Set a stack below the file table, clear .bss,
   set up the UART and the timer, run the program, report its exit code,
   and park the core. */
    .section .text.start, "ax"
    .arm
    .global _start
_start:
    ldr     sp, =0x07E00000
    ldr     r0, =__bss_start
    ldr     r1, =__bss_end
    mov     r2, #0
1:  cmp     r0, r1
    strlo   r2, [r0], #4
    blo     1b
    bl      rt_init
    bl      main
    bl      rt_exit
2:  wfe
    b       2b
