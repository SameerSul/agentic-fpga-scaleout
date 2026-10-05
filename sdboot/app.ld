/* sdboot: the program at 0x00100000, where a Vitis build also puts it,
   below the file table (0x07F00000) and the weights (0x08000000). */
ENTRY(_start)
SECTIONS
{
    . = 0x00100000;
    .text : { *(.text.start) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    . = ALIGN(4);
    __bss_start = .;
    .bss : { *(.bss*) *(COMMON) }
    . = ALIGN(4);
    __bss_end = .;
    /DISCARD/ : { *(.ARM.exidx*) *(.comment) }
}
