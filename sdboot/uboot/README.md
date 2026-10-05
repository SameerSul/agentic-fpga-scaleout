# U-Boot for the self-booting card

`<board>/boot.bin` (U-Boot's SPL, with the Zynq boot header),
`<board>/u-boot.img` and `<board>/u-boot-dtb.bin` (for QEMU) are U-Boot
v2026.07, built unmodified apart from the boot command, which runs the
card's `halo.scr`; `zc706/` with `DEVICE_TREE=zynq-zc706` and
`zybo_z7_20/` with `DEVICE_TREE=zynq-zybo-z7`:

```bash
git clone --depth 1 --branch v2026.07 https://github.com/u-boot/u-boot.git
cd u-boot
make CROSS_COMPILE=arm-none-eabi- xilinx_zynq_virt_defconfig
# CONFIG_BOOTCOMMAND="fatload mmc 0 0x3000000 halo.scr && source 0x3000000" in .config
make CROSS_COMPILE=arm-none-eabi- DEVICE_TREE=zynq-zc706 -j8
```

On macOS this needs Homebrew's `arm-none-eabi-gcc`, `make` (as `make` on
PATH), `bash` 4.2 or later (`CONFIG_SHELL=/opt/homebrew/bin/bash`), and
OpenSSL's headers (`HOSTCFLAGS=-I$(brew --prefix openssl@3)/include
HOSTLDFLAGS=-L$(brew --prefix openssl@3)/lib`).

The SPL runs the board's own `ps7_init` from
`board/xilinx/zynq/<device tree>/ps7_init_gpl.c`: DDR, MIO, and FCLK0 at
50 MHz, the clock the design is timed at (the IO PLL's 1000 MHz over 20
on the ZC706, over 5 x 4 on the Zybo Z7).
After `fpga loadb`, U-Boot enables the PS-PL level shifters and releases
the fabric resets, which Vitis's `ps7_post_config` would otherwise do.

U-Boot is GPL-2.0; its source is the tag above.
