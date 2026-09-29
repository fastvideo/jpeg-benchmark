#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make-big-frames-02.py
# version 2026-09-26.2 of 26.09.2026, cancels version 01
#
# NEW IN THIS VERSION: any size, not only 2x2, 3x3 and 4x4. A size is given as
# WIDTHxHEIGHT, for example 11608x8708 (Phase One iXM-RS100F, 100 MPix),
# 14204x10652 (iXM-RS150F, 150 MPix) or 19580x12600 (Canon LI8020SA, 250 MPix).
# Such a frame is made the same way - copies of the 4K frame side by side - and
# then cut to the size from the top left corner. 8k, 12k and 16k are exactly as
# in version 01: whole copies, nothing cut.
#
# Makes the big test frames of the JPEG benchmark out of the 4K frame:
#
#     8K   7680 x 4320    2 x 2 copies of the 4K frame
#     12K  11520 x 6480   3 x 3 copies
#     16K  15360 x 8640   4 x 4 copies
#
# The copies are laid side by side, row by row, with nothing between them and
# nothing changed in the pixels. This is how the 8K frame of the article was
# made, and it is kept in the repository instead of the frames themselves:
# 8K, 12K and 16K are 100, 224 and 398 MB, and anyone can make them from the
# 4K frame of the Fastvideo SDK in a few seconds.
#
# The 4K frame (4k_wild.ppm) is part of the Fastvideo SDK and is not in the
# repository. Input and output: binary PPM (P6), 8 bits, three channels.
#
# USAGE
#
#     python make-big-frames-02.py --src <4k_wild.ppm>
#         show what would be written, write nothing
#     python make-big-frames-02.py --src <4k_wild.ppm> --do
#         write 8k_wild.ppm, 12k_wild.ppm and 16k_wild.ppm next to the 4K frame
#     ... --do --sizes 12k,16k --out <folder>
#         only these sizes, into another folder
#     ... --check <8k_wild.ppm>
#         compare the pixels of an existing 8K frame with what this script
#         makes from the 4K frame (the header of the file is not compared:
#         another program may write it differently)
#
# Safe to run again: a frame that is already there with the right content
# (same MD5 as a freshly made one) is left as it is.

import argparse
import hashlib
import os
import sys

VERSION = "make-big-frames-02 of 26.09.2026"
TILES = {"8k": 2, "12k": 3, "16k": 4}


def say(*a):
    print(*a)
    sys.stdout.flush()


def read_ppm(path):
    """(width, height, pixel bytes) of a binary 8-bit PPM."""
    with open(path, "rb") as fh:
        data = fh.read()
    fields, pos = [], 0
    while len(fields) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] not in (b"\n", b""):
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(data[start:pos])
    pos += 1
    if fields[0] != b"P6" or int(fields[3]) != 255:
        raise ValueError("%s is not an 8-bit binary PPM (P6, 255)" % path)
    w, h = int(fields[1]), int(fields[2])
    pix = data[pos:pos + w * h * 3]
    if len(pix) != w * h * 3:
        raise ValueError("%s is shorter than its header says" % path)
    return w, h, pix


def frame_bytes(w, h, pix, W, H):
    """A W x H frame of copies of the w x h frame, cut from the top left:
    a sequence of byte chunks, the header first, then the rows."""
    row = w * 3
    across = -(-W // w)
    yield b"P6\n%d %d\n255\n" % (W, H)
    for Y in range(H):
        y = Y % h
        yield (pix[y * row:(y + 1) * row] * across)[:W * 3]


def big_bytes(w, h, pix, n):
    """The n x n frame: whole copies, nothing cut."""
    return frame_bytes(w, h, pix, w * n, h * n)


def size_of(name, w, h):
    """(W, H) for 8k/12k/16k or WIDTHxHEIGHT; None if the name is not one."""
    name = name.strip().lower()
    if name in TILES:
        return w * TILES[name], h * TILES[name]
    if "x" in name:
        a, _, b = name.partition("x")
        if a.isdigit() and b.isdigit() and 0 < int(a) <= 65535 \
                and 0 < int(b) <= 65535:
            return int(a), int(b)
    return None


def make_frame(src, W, H, dst):
    """Write the W x H frame made of copies of src to dst. Returns (W, H)."""
    w, h, pix = read_ppm(src)
    with open(dst + ".part", "wb") as fh:
        for c in frame_bytes(w, h, pix, W, H):
            fh.write(c)
    os.replace(dst + ".part", dst)
    return W, H


def md5_of_chunks(chunks):
    m = hashlib.md5()
    size = 0
    for c in chunks:
        m.update(c)
        size += len(c)
    return m.hexdigest(), size


def md5_of_file(path):
    m = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            m.update(block)
    return m.hexdigest()


def make_big(src, n, dst):
    """Write the n x n frame made of src to dst. Returns (width, height)."""
    w, h, pix = read_ppm(src)
    with open(dst + ".part", "wb") as fh:
        for c in big_bytes(w, h, pix, n):
            fh.write(c)
    os.replace(dst + ".part", dst)
    return w * n, h * n


def main():
    ap = argparse.ArgumentParser(description=VERSION)
    ap.add_argument("--src", required=True, help="the 4K frame, 4k_wild.ppm")
    ap.add_argument("--out", help="folder for the big frames;"
                                  " default: the folder of --src")
    ap.add_argument("--sizes", default="8k,12k,16k",
                    help="which frames: 8k, 12k, 16k or WIDTHxHEIGHT, comma"
                         " separated; default 8k,12k,16k")
    ap.add_argument("--check", help="an existing 8K frame to compare")
    ap.add_argument("--do", action="store_true", help="write the frames")
    a = ap.parse_args()

    say(VERSION)
    sizes = [s.strip().lower() for s in a.sizes.split(",") if s.strip()]
    bad = [s for s in sizes if size_of(s, 3840, 2160) is None]
    if bad or not sizes:
        say("--sizes takes 8k, 12k, 16k or WIDTHxHEIGHT (up to 65535 each);"
            " got: %s" % a.sizes)
        return 1
    if not os.path.isfile(a.src):
        say("No 4K frame at %s" % a.src)
        return 1
    w, h, pix = read_ppm(a.src)
    say("4K frame: %s, %d x %d, MD5 %s" % (a.src, w, h, md5_of_file(a.src)))
    if (w, h) != (3840, 2160):
        say("   note: this is not 3840 x 2160; the frames will be %d x %d"
            " times this size" % (w, h))
    out = a.out or os.path.dirname(os.path.abspath(a.src))
    say("frames go to: %s" % out)

    if a.check:
        if not os.path.isfile(a.check):
            say("No frame to check at %s" % a.check)
            return 1
        chunks = big_bytes(w, h, pix, 2)
        next(chunks)                                # the header
        made, size = md5_of_chunks(chunks)
        cw, ch, cpix = read_ppm(a.check)
        have = hashlib.md5(cpix).hexdigest()
        say("\nCheck of %s (pixels only):" % a.check)
        say("   made from the 4K frame: %d x %d, MD5 %s" % (w * 2, h * 2, made))
        say("   the file on disk:       %d x %d, MD5 %s" % (cw, ch, have))
        say("   %s" % ("THE SAME - the 8K frame is 2 x 2 copies of the 4K"
                       " frame" if made == have else
                       "DIFFERENT - the 8K frame on disk was made some other"
                       " way"))

    say("")
    for s in sizes:
        W, H = size_of(s, w, h)
        dst = os.path.join(out, "%s_wild.ppm" % s)
        md5, size = md5_of_chunks(frame_bytes(w, h, pix, W, H))
        state = "not there"
        if os.path.isfile(dst):
            state = ("there, the same" if md5_of_file(dst) == md5
                     else "there, DIFFERENT - will be replaced")
        say("%-11s -> %d x %d, %.1f MPix, %.0f MB, MD5 %s   [%s]"
            % (s, W, H, W * H / 1e6, size / 1e6, md5, state))
        if not a.do or state == "there, the same":
            continue
        os.makedirs(out, exist_ok=True)
        make_frame(a.src, W, H, dst)
        say("     written: %s" % dst)
    if not a.do:
        say("\nNothing written. Add --do to write the frames.")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        say("\nStopped.")
        rc = 1
    if rc and not hasattr(sys, "ps1"):
        sys.exit(rc)
