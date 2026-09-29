#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jpeg-tidy-02.py
# version 2026-09-24.2 of 24.09.2026, cancels version 01
#
# Clears the JPEG project folder before a full test, WITHOUT deleting
# anything: whatever is not needed for the test is MOVED into a folder next
# to this one, <folder>-old-<date>. When the new run is accepted, that folder
# is deleted by hand, in one go.
#
# WHAT STAYS
#
#   - the newest version of each of the six measuring scripts:
#     jpeg-run-all, jpeg-bench, jpeg-spread-check, nsys-jpeg, jpeg-420,
#     jpeg-markers (the one with the highest number; older ones are moved);
#   - this script, newest version;
#   - nvjpeg_bench\        our nvJPEG harness: its source and its programs;
#   - nvidia-jpeg-sample\  NVIDIA's programs, which nothing of ours makes;
#   - jpegtran*.exe and every .dll next to it. Version 01 moved them away, and
#     the markers test of 24.09 found no jpegtran and did not run.
#
# Everything else goes: runs, profiler results and traces, 4:2:0 and markers
# results and images, all-out, reference .jpg and .ppm working files, logs,
# old script versions. The bench makes its reference files again itself.
# The frames, JpegSample and jpegtran are in the SDK folder, which is not
# touched.
#
# USAGE
#
#     python jpeg-tidy-01.py          show what stays and what would move
#     python jpeg-tidy-01.py --do     move it
#
# Safe to run again: a second run finds nothing left to move and says so.

import argparse
import datetime
import glob
import os
import shutil
import sys

VERSION = "jpeg-tidy-02 of 24.09.2026"
HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = ["jpeg-run-all", "jpeg-bench", "jpeg-spread-check", "nsys-jpeg",
           "jpeg-420", "jpeg-markers", "jpeg-tidy"]
FOLDERS = ["nvjpeg_bench", "nvidia-jpeg-sample"]


def say(*a):
    print(*a)
    sys.stdout.flush()


def latest(prefix):
    best, best_n = None, -1
    for p in glob.glob(os.path.join(HERE, prefix + "-*.py")):
        tail = os.path.basename(p)[len(prefix) + 1:-3]
        if tail.isdigit() and int(tail) > best_n:
            best, best_n = os.path.basename(p), int(tail)
    return best


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def mb(n):
    return "%.1f MB" % (n / 1048576.0)


def finish(rc):
    """Leave without Python printing SystemExit in VS Code or Spyder."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    hosted = (hasattr(sys, "ps1") or sys.flags.interactive
              or any(m in sys.modules for m in
                     ("IPython", "ipykernel", "debugpy", "pydevd",
                      "spyder_kernels")))
    if rc and not hosted:
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(description=VERSION)
    ap.add_argument("--do", action="store_true",
                    help="move; without this nothing is changed")
    a = ap.parse_args()

    say(VERSION)
    say("no arguments: show what would move.  --do: move it."
        "  Nothing is ever deleted.")
    say("Working in: %s" % HERE)

    keep, missing = set(), []
    for prefix in SCRIPTS:
        name = latest(prefix)
        if name:
            keep.add(name)
        elif prefix != "jpeg-tidy":
            missing.append(prefix + "-NN.py")
    for d in FOLDERS:
        if os.path.isdir(os.path.join(HERE, d)):
            keep.add(d)
        else:
            missing.append(d + "\\")

    names = sorted(os.listdir(HERE), key=str.lower)
    for n in names:
        low = n.lower()
        if (low.startswith("jpegtran") and low.endswith(".exe")) \
                or low.endswith(".dll"):
            keep.add(n)
    stay = [n for n in names if n in keep]
    move = [n for n in names if n not in keep]

    say("\nSTAYS (%d):" % len(stay))
    for n in stay:
        say("   %-34s %s" % (n + ("\\" if os.path.isdir(os.path.join(HERE, n))
                                  else ""), mb(size_of(os.path.join(HERE, n)))))
    if missing:
        say("\nNOT FOUND, and the full test needs it:")
        for n in missing:
            say("   %s" % n)

    total = 0
    say("\nMOVES (%d):" % len(move))
    for n in move:
        s = size_of(os.path.join(HERE, n))
        total += s
        say("   %-34s %s" % (n + ("\\" if os.path.isdir(os.path.join(HERE, n))
                                  else ""), mb(s)))
    if not move:
        say("   nothing - the folder is already clean.")
        return 0

    stamp = datetime.date.today().strftime("%Y%m%d")
    dest = HERE.rstrip("\\/") + "-old-" + stamp
    if os.path.exists(dest):
        dest += datetime.datetime.now().strftime("-%H%M%S")
    say("\n%d item(s), %s, go to:\n   %s" % (len(move), mb(total), dest))

    if not a.do:
        say("\nNothing was moved. To move it:")
        say("   python %s --do" % os.path.basename(__file__))
        if missing:
            say("Before the full test, put the missing items above in place.")
        return 0

    os.makedirs(dest)
    failed = []
    for n in move:
        try:
            shutil.move(os.path.join(HERE, n), os.path.join(dest, n))
        except (OSError, shutil.Error) as e:
            failed.append((n, str(e)))
    say("\nMoved: %d of %d." % (len(move) - len(failed), len(move)))
    if failed:
        say("Not moved (open in another program?):")
        for n, e in failed:
            say("   %s - %s" % (n, e))
        say("Close whatever holds them and run this again with --do.")
    say("\nThe old things are in %s." % dest)
    say("Delete that folder by hand once the new run is accepted.")
    if missing:
        say("\nBefore the full test, put the missing items above in place.")
        return 1
    say("\nReady for the full test:")
    say("   python %s" % (latest("jpeg-run-all") or "jpeg-run-all-NN.py"))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        finish(main())
    except KeyboardInterrupt:
        say("\nStopped.")
        finish(1)
