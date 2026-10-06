"""A board package as one SD card that boots by itself: no Vivado, no
Vitis, no JTAG. Put the card in, set the boot mode to SD, power on, and
read the UART.

The card:

  boot.bin      U-Boot's SPL, with the board's own ps7_init (DDR, clocks,
                MIO; FCLK0 at 50 MHz, which halo.scr lowers to 40)
  u-boot.img    U-Boot, whose boot command runs halo.scr
  halo.scr      sets the PL's clock (PL_MHZ), programs the PL from
                fpgai.bit, reads each model file into
                the address the program loads it to and records its size,
                turns the caches off and starts the program
  halo.bin      the package's sw/main.c, unchanged, built bare-metal with
                sdboot/ (the BSP's register access, timer and FatFs, served
                from what U-Boot put in DDR, and a printf on the UART)
  fpgai.bit     the open-flow bitstream (open/build_open.sh), which has no
                PS configuration in it: the SPL's ps7_init is that
  weights8.bin, cparams.bin, vocab.bin, prompt.bin   the package's sd/

U-Boot's SPL and U-Boot are built from U-Boot v2026.07 with
xilinx_zynq_virt_defconfig and the board's device tree, zynq-zc706 or
zynq-zybo-z7 (sdboot/uboot/).

Everything but the model files is committed in the package's boot/
folder, so making a card needs Python and the package's sd/ files only:

  python3 sdboot.py board_zc706_qwen3 --out /Volumes/HALO      onto the card
  python3 sdboot.py board_zc706_qwen3 --image halo_sd.img      a FAT32 image (mtools)
  python3 sdboot.py board_zc706_qwen3 --build --bit <fpgai.bit>   remake boot/
  python3 sdboot.py board_zc706_qwen3 --qemu --tokens ...      boot it in QEMU

--build needs arm-none-eabi-gcc and mkimage; --qemu also qemu-system-arm.

--qemu boots the card's U-Boot in QEMU's xilinx-zynq-a9 machine with the
image as its SD card. QEMU has no PL, so that build answers the program's
register reads from a stand-in (the cosim's tokens, in order): it checks
the boot chain, every file's load, the table, the program and its UART
output, not the PL, which only a board can.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(ROOT, "sdboot")
TABLE = 0x07F00000
MAGIC = 0x48414C4F
BIT_STAGE = 0x01000000
SCRIPT_STAGE = 0x03000000
APP = 0x00100000
DATA = ("weights8.bin", "cparams.bin", "vocab.bin", "prompt.bin")

# The PL's clock, FCLK0, set by the script before the PL is programmed:
# the IO PLL's 1000 MHz (ZC706 and Zybo Z7 alike) over 25. The open-flow
# bitstream meets 50 MHz within a core's clock domain by 2%, and nextpnr
# leaves the paths from the bus clock into the core's gated clock
# untimed; the longest of those is 20.0 ns. At 40 MHz each has 25 ns.
PL_MHZ = 40
FPGA0_CLK_CTRL = 0xF8000170
SLCR_UNLOCK, SLCR_LOCK = 0xF8000008, 0xF8000004


def fclk_ctrl(mhz, pll_mhz=1000):
    """FPGA0_CLK_CTRL for mhz from the IO PLL: DIVISOR1 << 20 | DIVISOR0 << 8."""
    total = pll_mhz // mhz
    assert total * mhz == pll_mhz
    d1 = next(d for d in range(1, 64) if total % d == 0 and total // d < 64)
    return (d1 << 20) | ((total // d1) << 8)


SCRIPT = """echo "Halo: FCLK0 at {mhz} MHz"
mw.l {unlock:#010x} 0xdf0d
mw.l {fclk:#010x} {fclk_val:#010x}
mw.l {lock:#010x} 0x767b
echo "Halo: programming the PL from fpgai.bit"
if fatload mmc 0 {bit:#010x} fpgai.bit && fpga loadb 0 {bit:#010x} ${{filesize}}; then
  echo "Halo: PL programmed"
else
  echo "Halo: could not program the PL; stopping"
  exit
fi
echo "Halo: reading the model files"
mw.l {table:#010x} 0 8
if fatload mmc 0 {wb:#010x} weights8.bin; then mw.l {t1:#010x} ${{filesize}}; fi
if fatload mmc 0 {cb:#010x} cparams.bin; then mw.l {t2:#010x} ${{filesize}}; fi
if fatload mmc 0 {vb:#010x} vocab.bin; then mw.l {t3:#010x} ${{filesize}}; fi
if fatload mmc 0 {pb:#010x} prompt.bin; then mw.l {t4:#010x} ${{filesize}}; fi
mw.l {table:#010x} {magic:#010x}
if fatload mmc 0 {app:#010x} halo.bin; then
  dcache off
  icache off
  go {app:#010x}
else
  echo "Halo: no halo.bin on the card"
fi
"""


def _layout(pkg):
    h = open(os.path.join(pkg, "sw", "fpgai_layout.h")).read()
    get = lambda n: int(re.search(r"#define %s\s+0x([0-9A-Fa-f]+)U" % n, h).group(1), 16)
    return {n: get(n) for n in ("WBASE", "CBASE", "VOCAB_BASE")}, \
        re.search(r'#define FPGAI_BOARD\s+"([^"]+)"', h).group(1)


def _slug(board):
    """sdboot/uboot/'s folder for a board: "Zybo Z7-20" is zybo_z7_20."""
    return re.sub(r"[^a-z0-9]+", "_", board.lower()).strip("_")


def _run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("%s failed:\n%s%s" % (cmd[0], r.stdout[-2000:], r.stderr[-2000:]))
    return r.stdout


def build_program(pkg, out, fake_tokens=None, main=None, include=None, name="halo.bin"):
    """The package's sw/main.c, unchanged, as halo.bin; or main (the
    diagnostics), with include on the path for its expectations."""
    elf = os.path.join(out, name[:-4] + ".elf")
    defs = []
    if fake_tokens:
        defs = ["-DHALO_FAKE_PL", "-DHALO_FAKE_TOKENS=%s" % ",".join(map(str, fake_tokens))]
    inc = ["-I" + include] if include else []
    _run(["arm-none-eabi-gcc", "-mcpu=cortex-a9", "-marm", "-mfloat-abi=soft",
          "-mno-unaligned-access", "-ffreestanding", "-fno-builtin", "-nostdlib",
          "-nostartfiles", "-O2", "-Wall", "-Wno-unused-function",
          "-I" + os.path.join(SD, "include"), "-I" + os.path.join(pkg, "sw")] + inc + defs +
         [os.path.join(SD, "start.S"), os.path.join(SD, "rt.c"),
          main or os.path.join(pkg, "sw", "main.c"), "-T", os.path.join(SD, "app.ld"),
          "-Wl,--no-warn-rwx-segments", "-lgcc", "-o", elf])
    _run(["arm-none-eabi-objcopy", "-O", "binary", elf, os.path.join(out, name)])
    os.remove(elf)
    return os.path.join(out, name)


def diag_record(pkg, work, log=print, tok=785, sd=None, layers=None):
    """sdboot/diag.c under cosim.py, on the package's RTL and sd/ files:
    the step it makes on the board, as the RTL makes it. Returns the
    token, logit, core cycles, the sampled files' hashes and every K and
    V value the step wrote, by offset from KBASE."""
    import cosim
    tmp = os.path.join(work, "pkg")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "sw"))
    os.symlink(os.path.join(pkg, "rtl"), os.path.join(tmp, "rtl"))
    shutil.copyfile(os.path.join(pkg, "sw", "fpgai_layout.h"),
                    os.path.join(tmp, "sw", "fpgai_layout.h"))
    shutil.copyfile(os.path.join(SD, "diag.c"), os.path.join(tmp, "sw", "main.c"))
    exe = cosim.build(tmp, os.path.join(work, "cosim"), log=log,
                      defines={"DIAG_RECORD": "1", "DX_TOK": str(tok)})
    out = cosim.run(exe, sd or os.path.join(pkg, "sd"), env={"JITTER": "0"})
    rec = {"w": []}
    for line in out.splitlines():
        f = line.split()
        if not f or f[0] != "DX":
            continue
        if f[1] == "step":
            rec.update(tok=int(f[2]), next=int(f[3]), best=int(f[4]), core=int(f[5]),
                       bus=int(f[6]))
        elif f[1] == "hash":
            rec["hash"] = [int(x) for x in f[2:5]]
        elif f[1] == "vbase":
            rec["vbase"] = int(f[2])
        elif f[1] == "w":
            rec["w"].append((int(f[2]), int(f[3])))
    if "next" not in rec or not rec["w"]:
        raise RuntimeError("the diagnostics did not run under cosim:\n" + out[-3000:])
    if layers is None:
        rtl = open(os.path.join(pkg, "rtl", "qwen_full.v")).read()
        layers = int(re.search(r"lyr \+ 1 < (\d+)", rtl).group(1))
    rec["layers"] = layers
    return rec


def write_expect(rec, path):
    """diag_expect.h: what diag.c compares the board with."""
    w = sorted(rec["w"])
    with open(path, "w") as f:
        f.write("/* GENERATED by sdboot.py --diag from cosim.py's run of diag.c. */\n")
        f.write("#define DX_TOK %d\n#define DX_NEXT %du\n#define DX_BEST %d\n"
                "#define DX_CORE %du\n" % (rec["tok"], rec["next"], rec["best"], rec["core"]))
        f.write("#define DX_HASH_W 0x%08Xu\n#define DX_HASH_C 0x%08Xu\n#define DX_HASH_V 0x%08Xu\n"
                % tuple(rec["hash"]))
        f.write("#define DX_NL %d\n#define DX_N %du\n" % (rec["layers"], len(w)))
        f.write("static const u32 dx_off[%d] = {\n" % len(w))
        for i in range(0, len(w), 12):
            f.write(",".join(str(o) for o, _ in w[i:i + 12]) + ",\n")
        f.write("};\nstatic const u16 dx_val[%d] = {\n" % len(w))
        for i in range(0, len(w), 16):
            f.write(",".join(str(v) for _, v in w[i:i + 16]) + ",\n")
        f.write("};\n")
    return path


def build_diag(pkg, work=None, log=print):
    """<package>/boot/halo_diag.bin: the diagnostics, with what the RTL
    does in simulation to compare against. Put it on the card as halo.bin."""
    work = work or tempfile.mkdtemp(prefix="halo_diag_")
    rec = diag_record(pkg, work, log)
    log("  simulation: token %d, logit %d, %d core cycles, %d K and V values"
        % (rec["next"], rec["best"], rec["core"], len(rec["w"])))
    write_expect(rec, os.path.join(work, "diag_expect.h"))
    return build_program(pkg, os.path.join(pkg, "boot"), main=os.path.join(SD, "diag.c"),
                         include=work, name="halo_diag.bin")


def build_script(pkg, out, fpga=True):
    lay, _ = _layout(pkg)
    text = SCRIPT.format(mhz=PL_MHZ, unlock=SLCR_UNLOCK, lock=SLCR_LOCK, fclk=FPGA0_CLK_CTRL,
                         fclk_val=fclk_ctrl(PL_MHZ), bit=BIT_STAGE, table=TABLE, magic=MAGIC, app=APP,
                         wb=lay["WBASE"], cb=lay["CBASE"], vb=lay["VOCAB_BASE"],
                         pb=TABLE + 0x1000, t1=TABLE + 4, t2=TABLE + 8, t3=TABLE + 12,
                         t4=TABLE + 16)
    if not fpga:
        text = text.replace('if fatload mmc 0 0x01000000 fpgai.bit && fpga loadb 0 '
                            '0x01000000 ${filesize}; then',
                            'if fatload mmc 0 0x01000000 fpgai.bit; then')
    cmd = os.path.join(out, "halo.cmd")
    with open(cmd, "w") as f:
        f.write(text)
    _run(["mkimage", "-A", "arm", "-O", "u-boot", "-T", "script", "-C", "none",
          "-n", "Halo", "-d", cmd, os.path.join(out, "halo.scr")])
    return os.path.join(out, "halo.scr")


BOOT_FILES = ("boot.bin", "u-boot.img", "halo.scr", "halo.bin", "fpgai.bit.gz")


def build_boot(pkg, bit, uboot=None):
    """The package's boot/ folder: U-Boot, the script, the program and the
    bitstream, compressed (the 7Z045's 13 MB is 1.2 MB gzipped)."""
    import gzip
    _, board = _layout(pkg)
    uboot = uboot or os.path.join(SD, "uboot", _slug(board))
    out = os.path.join(pkg, "boot")
    os.makedirs(out, exist_ok=True)
    for f in ("boot.bin", "u-boot.img"):
        shutil.copyfile(os.path.join(uboot, f), os.path.join(out, f))
    build_program(pkg, out)
    build_script(pkg, out)
    os.remove(os.path.join(out, "halo.cmd"))
    with open(bit, "rb") as f, gzip.GzipFile(os.path.join(out, "fpgai.bit.gz"), "wb",
                                             mtime=0) as g:
        shutil.copyfileobj(f, g)
    return out


def copy_card(pkg, out):
    """boot/ and sd/ onto the card (or a folder): Python only."""
    import gzip
    boot, sd = os.path.join(pkg, "boot"), os.path.join(pkg, "sd")
    missing = [f for f in BOOT_FILES if not os.path.exists(os.path.join(boot, f))]
    if missing:
        raise RuntimeError("%s/boot/ lacks %s" % (os.path.basename(pkg), ", ".join(missing)))
    missing = [f for f in DATA if not os.path.exists(os.path.join(sd, f))]
    if missing:
        raise RuntimeError("%s/sd/ lacks %s: make them first (HANDOFF.md step 1)"
                           % (os.path.basename(pkg), ", ".join(missing)))
    os.makedirs(out, exist_ok=True)
    for f in BOOT_FILES[:-1]:
        shutil.copyfile(os.path.join(boot, f), os.path.join(out, f))
    with gzip.open(os.path.join(boot, "fpgai.bit.gz"), "rb") as g, \
            open(os.path.join(out, "fpgai.bit"), "wb") as f:
        shutil.copyfileobj(g, f)
    for f in DATA:
        shutil.copyfile(os.path.join(sd, f), os.path.join(out, f))
    return out


def card(pkg, out, bit=None, uboot=None, fake_tokens=None, fpga=True):
    """Every file the card holds, in out, built here (for --qemu)."""
    _, board = _layout(pkg)
    uboot = uboot or os.path.join(SD, "uboot", _slug(board))
    bit = bit or os.path.join(pkg, "open", "fpgai.bit")
    os.makedirs(out, exist_ok=True)
    for f in ("boot.bin", "u-boot.img"):
        shutil.copyfile(os.path.join(uboot, f), os.path.join(out, f))
    if not os.path.exists(bit):
        raise RuntimeError("no bitstream at %s: build it with %s/open/build_open.sh, "
                           "or pass --bit" % (bit, os.path.basename(pkg)))
    shutil.copyfile(bit, os.path.join(out, "fpgai.bit"))
    build_program(pkg, out, fake_tokens)
    build_script(pkg, out, fpga)
    for f in DATA:
        src = os.path.join(pkg, "sd", f)
        if not os.path.exists(src):
            raise RuntimeError("no %s: make the package's sd/ files first (HANDOFF.md step 1)" % src)
        dst = os.path.join(out, f)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    os.remove(os.path.join(out, "halo.cmd"))
    return out


def image(files, path, size_mb=1024):
    """A raw disk image: one FAT32 partition, MBR, holding files."""
    part_off = 1 << 20
    with open(path, "wb") as f:
        f.truncate(size_mb << 20)
    sectors = (size_mb << 20) // 512
    start = part_off // 512
    mbr = bytearray(512)
    entry = bytes([0x00, 0, 0, 0, 0x0C, 0, 0, 0]) + start.to_bytes(4, "little") + \
        (sectors - start).to_bytes(4, "little")
    mbr[446:462] = entry
    mbr[510:512] = b"\x55\xaa"
    with open(path, "r+b") as f:
        f.write(mbr)
    target = "%s@@%d" % (path, part_off)
    _run(["mformat", "-i", target, "-F", "-v", "HALO", "::"])
    for name in sorted(os.listdir(files)):
        _run(["mcopy", "-i", target, os.path.join(files, name), "::/" + name])
    return path


def qemu(img, uboot, timeout=900):
    """Boot U-Boot in QEMU's Zynq with img as the SD card; return the UART."""
    dtb = os.path.join(uboot, "u-boot-dtb.bin")
    cmd = ["qemu-system-arm", "-M", "xilinx-zynq-a9", "-m", "1024", "-nographic",
           "-serial", "null", "-serial", "stdio", "-monitor", "none",
           "-drive", "file=%s,if=sd,format=raw,index=0" % img,
           "-device", "loader,file=%s,addr=0x04000000" % dtb,
           "-device", "loader,addr=0x04000000,cpu-num=0"]
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    # Raw reads: U-Boot prints progress without a newline, and a line read
    # would wait on it for good.
    import select
    import time
    out, t0 = b"", time.time()
    def done():
        i = out.find(b"Application terminated")
        return i >= 0 and b"\n" in out[i:]
    while time.time() - t0 < timeout and not done():
        r, _, _ = select.select([p.stdout], [], [], 2)
        if r:
            chunk = os.read(p.stdout.fileno(), 65536)
            if not chunk:
                break
            out += chunk
    p.kill()
    p.wait()
    return out.decode("utf-8", "replace")


def cosim_tokens(pkg):
    """The 16 tokens the package's program printed in the cosim, for the
    QEMU stand-in. From cosim.py's head lines if a run left them, else the
    integer model's, as qwen_full.py recorded them."""
    p = os.path.join(pkg, "sd", "tokens.txt")
    if os.path.exists(p):
        return [int(t) for t in open(p).read().split()]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("package", help="a board package: board_zc706_qwen3, board_zc706")
    ap.add_argument("--out", default=None, help="write the card's files here")
    ap.add_argument("--image", default=None, help="write a raw FAT32 disk image here")
    ap.add_argument("--bit", default=None, help="the bitstream (default: <package>/open/fpgai.bit)")
    ap.add_argument("--uboot", default=None, help="boot.bin and u-boot.img (default: sdboot/uboot/<board>)")
    ap.add_argument("--build", action="store_true",
                    help="remake <package>/boot/ (needs arm-none-eabi-gcc, mkimage and --bit)")
    ap.add_argument("--qemu", action="store_true", help="boot the card in QEMU, PL stood in for")
    ap.add_argument("--diag", action="store_true",
                    help="make boot/halo_diag.bin: the diagnostics, checked against the RTL "
                    "in simulation (needs Verilator and arm-none-eabi-gcc)")
    ap.add_argument("--tokens", default=None,
                    help="for --qemu: the tokens the stand-in PL gives, comma-separated")
    a = ap.parse_args()
    pkg = os.path.abspath(a.package)
    if a.qemu:
        toks = [int(t) for t in a.tokens.split(",")] if a.tokens else cosim_tokens(pkg)
        if not toks:
            ap.error("--qemu needs --tokens (the cosim's tokens, from cosim.py's head steps)")
        work = tempfile.mkdtemp(prefix="halo_qemu_")
        files = card(pkg, os.path.join(work, "files"), a.bit, a.uboot, toks, fpga=False)
        img = image(files, os.path.join(work, "sd.img"))
        _, board = _layout(pkg)
        print(qemu(img, a.uboot or os.path.join(SD, "uboot", _slug(board))))
        shutil.rmtree(work, ignore_errors=True)
        return
    if a.diag:
        print("diagnostics in", build_diag(pkg))
        return
    if a.build:
        if not a.bit:
            ap.error("--build needs --bit, the open-flow bitstream for this package")
        print("boot files in", build_boot(pkg, os.path.abspath(a.bit), a.uboot))
        return
    if a.image:
        work = tempfile.mkdtemp(prefix="halo_card_")
        image(copy_card(pkg, os.path.join(work, "files")), os.path.abspath(a.image))
        shutil.rmtree(work, ignore_errors=True)
        print("FAT32 image in", a.image)
        return
    out = os.path.abspath(a.out or os.path.join(pkg, "card"))
    copy_card(pkg, out)
    print("card files in", out)


if __name__ == "__main__":
    main()
