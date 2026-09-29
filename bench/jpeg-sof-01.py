#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jpeg-sof-01.py
# version 2026-09-26.1 of 26.09.2026
#
# Reads the headers of JPEG files and prints what ITU-T T.81 says follows from
# them: width and height, the sampling factors Hi x Vi of every component
# (SOF marker), the size of the MCU in pixels (8*Hmax x 8*Vmax, T.81 A.2.3),
# how many data units (8x8 blocks) one MCU holds (the sum of Hi*Vi), the
# restart interval (DRI marker, in MCUs), and how many entropy-coded segments
# the frame falls into (MCUs divided by the restart interval).
#
# Nothing is decoded and nothing is written. Why: the MCU of a 4:2:2 file is
# 16x8 when the luma factors are 2x1 and 16x16 when they are 2x2 - the file
# itself says which, so it is read, not assumed.
#
# USAGE
#     python jpeg-sof-01.py <file.jpg> [<file.jpg> ...]
#
# For the Fastvideo encoder, make the files first, for example:
#     JpegSample.exe -i 2k_wild.ppm -o fv_444.jpg -q 90 -s 444
#     JpegSample.exe -i 2k_wild.ppm -o fv_422.jpg -q 90 -s 422
#     JpegSample.exe -i 2k_wild.ppm -o fv_420.jpg -q 90 -s 420
#     (and a greyscale one from a greyscale .pgm, if JpegSample takes it)

import sys

VERSION = "jpeg-sof-01 of 26.09.2026"


def headers(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file (no SOI)")
    out = {"dri": 0}
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        length = (data[i + 2] << 8) | data[i + 3]
        seg = data[i + 4:i + 2 + length]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            out["sof"] = m
            out["bits"] = seg[0]
            out["h"] = (seg[1] << 8) | seg[2]
            out["w"] = (seg[3] << 8) | seg[4]
            n = seg[5]
            out["comp"] = [(seg[6 + 3 * c], seg[7 + 3 * c] >> 4,
                            seg[7 + 3 * c] & 15) for c in range(n)]
        elif m == 0xDD:
            out["dri"] = (seg[0] << 8) | seg[1]
        elif m == 0xDA:
            break
        i += 2 + length
    if "sof" not in out:
        raise ValueError("no SOF marker before the first scan")
    return out


def describe(path):
    hd = headers(path)
    comp = hd["comp"]
    hmax = max(c[1] for c in comp)
    vmax = max(c[2] for c in comp)
    if len(comp) == 1:
        # one component: not interleaved, the MCU is one data unit (T.81 A.2.2)
        mcu_w, mcu_h, units = 8, 8, 1
    else:
        mcu_w, mcu_h = 8 * hmax, 8 * vmax
        units = sum(c[1] * c[2] for c in comp)
    mcus = (-(-hd["w"] // mcu_w)) * (-(-hd["h"] // mcu_h))
    ri = hd["dri"]
    segs = -(-mcus // ri) if ri else 1
    kind = {0xC0: "baseline", 0xC1: "extended", 0xC2: "progressive",
            0xC3: "lossless"}.get(hd["sof"], "SOF%X" % hd["sof"])
    print("%s" % path)
    print("   %d x %d, %d bit, %s, %d component(s)"
          % (hd["w"], hd["h"], hd["bits"], kind, len(comp)))
    print("   sampling factors H x V: %s"
          % ", ".join("C%d %dx%d" % c for c in comp))
    print("   MCU: %d x %d pixels, %d data units (8x8 blocks)"
          % (mcu_w, mcu_h, units))
    print("   MCUs in the frame: %d" % mcus)
    if ri:
        print("   restart interval: %d MCUs = %d data units per segment"
              % (ri, ri * units))
    else:
        print("   restart interval: none (no DRI marker)")
    print("   entropy-coded segments: %d" % segs)


def main():
    print(VERSION)
    files = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not files:
        print(__doc__ or "usage: python jpeg-sof-01.py <file.jpg> ...")
        print("usage: python jpeg-sof-01.py <file.jpg> [<file.jpg> ...]")
        return 1
    rc = 0
    for f in files:
        try:
            describe(f)
        except (OSError, ValueError) as e:
            print("%s\n   cannot read: %s" % (f, e))
            rc = 1
    return rc


if __name__ == "__main__":
    rc = main()
    if rc and not hasattr(sys, "ps1"):
        sys.exit(rc)
