#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jpeg-bench-13.py
# version 2026-09-25.13 of 25.09.2026, cancels version 12
#
# NEW IN VERSION 13 (25.09.2026)
#
#   1. FRAME 8K. Fyodor made 8k_wild.ppm (7680x4320) out of four 4k_wild.ppm.
#      It is measured whenever the file lies next to the other two frames
#      (the --img folder, which is the SDK folder); without it the run is the
#      same as version 12 and says so. Grid points that do not fit into card
#      memory at 8K are left out for both codecs, as before (plan_memory).
#   2. ENERGY AT ONE THREAD TOO. Version 12 measured energy only at the best
#      point of each codec - the fastest multithread point, the whole path
#      with the transfers. The article's speed tables are one thread without
#      the transfers, and Fyodor asked (25.09) whether energy is the same in
#      both modes. Now both are measured: "best" as before, and "single" in
#      exactly the window of the one-thread speed rows. Every energy row says
#      which of the two it is (field "point").
#   3. "Who is on the card" asks nvidia-smi --query-compute-apps and prints
#      every process, not the first 12 lines of the desktop table: on 25.09
#      the list was cut before it reached the process that loaded the card.
#
# NEW: before every measured repeat the card is checked for someone else's load, and the repeat waits while it is busy.
#
# Baseline 8-bit JPEG on the GPU: Fastvideo JpegSample against NVIDIA nvJPEG,
# on the CUDA path and on the hardware JPEG engine. Built on bench-07.py (the
# JPEG2000 comparison on the RTX 4090) and keeps its rules:
#   - nothing long starts by itself: a bare start prints how to run it;
#   - every measurement is on disk the moment it is made (results.jsonl);
#   - a sign of life at least every HEARTBEAT_S seconds;
#   - Ctrl-C stops cleanly and says what was measured and where;
#   - the nvJPEG executables are asked for -version before the first
#     measurement and the run stops if they were not built from the source
#     next to them;
#   - three repeats, a point whose repeats disagree is measured again, a grid
#     winner ahead by less than the noise is called a tie;
#   - both codecs are measured between the same two points (boundary audit);
#   - GPU energy by two meters, N and 2N frames, the difference.
#
# WHAT IS NEW COMPARED WITH THE JPEG2000 RUN, BECAUSE THIS IS JPEG
#
#   1. The quality scales are compared in two layers. JPEG writes its
#      quantisation tables into the file (DQT marker), so whether "-q 90" means
#      the same tables on both sides can be read directly and compared value by
#      value. How a table is APPLIED - rounding of the quotient, the precision
#      of the DCT - is not in the file and is not published; it shows only in
#      the result. So: tables identical but size or PSNR different -> the
#      difference is in how they are applied; tables different -> the scales do
#      not correspond, and the size search gives the pair "our q <-> their q".
#      The speed comparison itself is made at equal file size, as for JPEG2000.
#
#   2. nvJPEG has three decode paths and all three are measured together with
#      ours: the CUDA path (GPU hybrid backend) through single decodes and
#      through the native batch nvjpegDecodeBatched, and the dedicated
#      hardware JPEG engine (hardware backend). Whether the RTX 4090 has that
#      engine is NOT assumed: the documentation lists architectures, not
#      products. It is checked in three independent ways before anything is
#      measured on it (see phase_hw), and during every hardware run the
#      engine's own utilisation counter is read through NVML - a working engine
#      shows a non-zero figure, a silent fallback to CUDA shows zero.
#
#   3. nvjpeg.h (12.4) states that the GPU hybrid backend decodes the Huffman
#      stage on the GPU only when the batch is bigger than 50. The batch grid
#      therefore goes past 50 (64, 128), and every row says where its Huffman
#      stage ran. A grid that stopped at 32 would measure the CPU path of
#      nvJPEG without saying so.
#
#   4. The measurement window is host memory to host memory in every mode, on
#      both sides. The JpegSample single-thread figure and the async figure
#      both include the transfers (its exact window is to be confirmed from its
#      -info output - see FV_* below); the nvJPEG harness is started with
#      -window host2host everywhere, so the label and the measurement come
#      from the same flag.
#
# WHAT IS NEW IN VERSION 02 (21.09.2026)
#
#   Every program and every frame is called by its own full path, and each
#   side keeps its own folder: the Fastvideo SDK as delivered, NVIDIA's own
#   sample programs, and our harness. Nothing is copied between them and
#   nothing of ours is ever written into NVIDIA's folder or into the SDK -
#   Fyodor, 21.09.2026. The four folders are the four keys --fv, --img, --nv
#   and --nvidia; they are printed by a bare start with found / NOT FOUND
#   against each one.
#
#   ONE PROJECT FOLDER holds all of it, e.g. D:\_Test\jpeg:
#
#       jpeg-bench-02.py         this file; the run is started from here
#       nvjpeg_bench\            ours: nvjpeg_bench-NN.cpp and its two exes
#       nvidia-jpeg-sample\      NVIDIA's programs, written by nothing of ours
#       runs\                    the run folders, where --out points by default
#
#   The SDK stays where it was installed and is never written into; the frames
#   ship with it, so --img follows --fv. Working files (reference .jpg, decoded
#   .ppm, the one-file folders NVIDIA's programs read) are made in the CURRENT
#   folder, that is in the project folder.
#
#   Version 01 looked for everything in the folder it was started from; it is
#   replaced and deleted.
#
# WHAT IS NEW IN VERSION 03 (21.09.2026, the same day)
#
#   The project folder is THE FOLDER THIS FILE LIES IN, not the folder the
#   shell happens to be in. Version 02 resolved its subfolders against the
#   current directory, so a run started from a code editor looked for
#   nvjpeg_bench\\ next to the editor and would have written its working files
#   there. Now: paths given on the command line are read relative to where the
#   shell stands, everything else relative to this file, and the run then
#   steps into the project folder itself. Starting it as
#   "python d:\\_Test\\jpeg\\jpeg-bench-04.py" from anywhere is enough.
#
# WHAT IS NEW IN VERSION 04 (21.09.2026)
#
#   A fix to the above: --build named its log file after the target, and the
#   target is now a full path, so open() failed with "Invalid argument" and
#   the build never ran. The log is named after the file, not the path, and a
#   log name is stripped of path characters in one place, for good.
#
# WHAT IS NEW IN VERSION 05 (21.09.2026, after the first probe on the card)
#
#   The Fastvideo sample's own summary lines are now read, and with them the
#   boundary of each one. Three of its four shapes were missed by the pattern
#   of version 04 - it wanted a semicolon after FPS and nothing between
#   "images" and "=" - so those runs fell through to "FPS only" and, worse,
#   the single-image run was read as 80 FPS from the line that EXCLUDES the
#   host-to-device transfer, while the line right above it says 74 FPS for
#   the whole path. The four shapes, as the program printed them on 21.09:
#
#     for 1 images including all transfers = 13.47 ms; 441 MB/s;  74 FPS
#     for 1 images excluding host-to-device transfer: average = 12.506 ms (...
#     for 50 images excluding host-to-device transfer = 11.11 ms; ...; 4499 FPS
#     for 50 images without HDD I/O and excluding device-to-host transfer
#                                              = 35.05 ms; 1427 FPS
#     - GPU pipeline including all transfers for 50 images per 2 threads = ...
#
#   The last matching line wins, so a run that prints both windows is read by
#   the one that covers the whole path. The "average = ... (...)" shape is not
#   a summary line and is deliberately left unmatched. The sample's own
#   "Host-to-device transfer = X ms" and "Effective encoding performance
#   (includes device-to-host transfer) = ... (X ms)" are kept as their own
#   figures, so a window can be reconstructed rather than guessed.
#
# WHAT IS NEW IN VERSION 06 (21.09.2026)
#
#   nvJPEG is now measured in the SAME window as the Fastvideo sample in the
#   same mode, instead of host2host everywhere. The sample has no choice of
#   window: with -async it reports the whole path, and in plain -repeat it
#   drops one leg - the upload when encoding, the download when decoding. Our
#   harness's -window codec drops exactly those same two legs (its own words:
#   "excluding the host-to-device transfer" for the encoder, "excluding the
#   device-to-host transfer" for the decoder), so the pair matches mode by
#   mode:
#
#     mode                         Fastvideo             nvJPEG
#     -async / native batch        whole path            -window host2host
#     plain -repeat, one thread    one leg dropped       -window codec
#
#   Version 05 compared the sample's one-leg-dropped figure against nvJPEG's
#   whole path and would have credited us with a difference that is the
#   window, not the codec: on 2K q90 that is 4547 against 1991 frames a second
#   instead of a draw. The boundary audit at the end of the run now has
#   something to confirm rather than something to catch.
#
# WHAT IS NEW IN VERSION 07 (21.09.2026, after the first trial on the card)
#
#   1. A POINT IS NEVER MEASURED UNFILLED. The trial gave every point the
#      same 20 frames whatever the grid point was, so 32 threads x 2 frames
#      in flight got 0.6 frames per slot: what such a point measures is the
#      start of the pipeline, not the codec. On 2K encoding that read as a
#      fall from 1296 to 511 frames a second with more threads - a fall that
#      is the filling, not the codec. Every point now gets at least
#      FRAMES_PER_SLOT frames for each frame in flight (threads x batch), in
#      the trial as well as in the final run.
#
#   2. THE "single" COLUMN IS NAMED FOR WHAT IT IS. It is one thread in the
#      codec-only window - the sample reports no other in that mode - while
#      the grid beside it is host to host. Two windows in one table read as
#      one, so the column is now headed "1 thr" and the table says in words
#      that it is a different window and is to be read down, not across.
#
# WHAT IS NEW IN VERSION 08 (21.09.2026, after the filled trial)
#
#   1. THE GRID STARTS AT ONE THREAD. The filled trial showed both codecs
#      already saturated at eight threads - every point from 8 to 32 lands
#      within a third of the best - so the grid measured the plateau and not
#      the climb to it. 1, 2 and 4 threads are added: where each codec
#      saturates is the question the article answers, and it cannot be read
#      off a grid whose first point is past the knee.
#
#   2. ENERGY IS REFUSED WHEN IT CANNOT BE MEASURED. The trial printed eight
#      figures per frame and every one was noise: the N and 2N runs took THE
#      SAME wall time (0.37 s and 0.37 s, 0.35 and 0.36, 0.31 and 0.32),
#      because the frames were too few for the frame count to matter at all.
#      The difference of two readings then divides start-up jitter by the
#      frames, and out comes 0.0275 J for a 2K frame against 1.6061 J for a
#      4K one - 58 times, where the pixels differ by four. Now the energy
#      runs are timed rather than counted, in the trial as well, and a figure
#      is printed only when the 2N run really took ENERGY_GROWTH times the N
#      run and the N run itself lasted ENERGY_MIN_S. Otherwise a dash and the
#      reason: a measurement that did not happen is reported as one.
#
# THE FASTVIDEO SIDE: JpegSample.exe, one program for both directions (the
# direction is taken from the file extensions), keys as in the SDK .cmd files:
#   JpegSample -i x.ppm -o x.jpg -q 90 -s 444 -repeat N
#   JpegSample -i x.ppm -o x.jpg -q 90 -s 444 -repeat N -async -thread T
#              -threadR R -threadW W -b B
#   JpegSample -i x.jpg -o x.ppm -repeat N [-async ...]
# Two things about it are taken on trust until its -info output is seen and
# are kept in ONE place, FV_* below, so they can be changed without touching
# the rest: that it accepts -discard (skip writing the output file, as the
# JPEG2000 sample does), and how many reader / writer threads to give it.
# Run --probe first: it runs every program once and prints what was read from
# its output, so a wrong guess shows in a minute, not after an hour.
"""
Baseline JPEG codec comparison on the GPU, one file.

    python jpeg-bench-04.py                prints this and measures nothing
    python jpeg-bench-04.py --selftest     checks the file itself, no card
    python jpeg-bench-04.py --build        builds the nvJPEG harness, stops
    python jpeg-bench-04.py --probe        runs every program once, shows
                                           what was read from its output
    python jpeg-bench-04.py --hw           the hardware engine check only
    python jpeg-bench-04.py --trial        every branch of the code, short
    python jpeg-bench-04.py --final        the run the article is written from

Where the programs are (nothing is copied, each side keeps its folder):

    --fv DIR       the Fastvideo SDK as delivered: JpegSample and its DLLs
    --img DIR      the frames; the SDK ships them, so this defaults to --fv
    --nv DIR       ours: nvjpeg_bench-NN.cpp and the two executables of it
    --nvidia DIR   NVIDIA's own sample programs, written by nothing of ours

The last two default to subfolders of the project folder the run is started
from, so one folder holds everything and each side still has its own.

Standard library only; numpy and nvidia-ml-py are used when installed
(exact PSNR; the energy counter and the JPEG engine counter).
"""

import argparse
import csv
import datetime
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import threading
import time

# ---------------------------------------------------------------------------
# what is measured
# ---------------------------------------------------------------------------

SCRIPT_NAME = os.path.basename(__file__)     # the file name, whatever it is
BENCH_VERSION = "2026-09-25.13"              # printed into every result file
HEARTBEAT_S = 300.0

EXE = ".exe" if os.name == "nt" else ""

# WHERE THE PROGRAMS AND THE FRAMES LIVE
#
# Four folders, four keys, nothing copied between them. The defaults are the
# layout on the RTX 4090 machine as of 21.09.2026; --fv, --img, --nv and
# --nvidia override them one by one.
DEF_FV = (r"D:\_Test\fvSDK-0.23.1.0-Win64-CUDA-13.3-Trial-Exp-2027-08-03"
          r"\bin\x64\Release")            # the SDK as delivered, untouched
DEF_IMG = DEF_FV                           # --img follows --fv when unset
# Ours and NVIDIA's are SUBFOLDERS of the project folder the run is started
# from - one folder to carry, to back up and to delete, and still a folder of
# their own for their code (Fyodor, 21.09.2026: one folder is enough).
DEF_NV = "nvjpeg_bench"                    # ours: the source and its exes
DEF_NVIDIA = "nvidia-jpeg-sample"          # theirs, written by nothing of ours

NV_DIR = DEF_NV

# The project folder is where THIS FILE lies. Paths typed on the command line
# are relative to where the shell stands; everything else is relative to the
# file, so it does not matter what folder the run is started from.
HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_CWD = os.getcwd()




# ------------------------------------------------ is anyone else on the card
#
# 24.09.2026: in the run of 12:52 the card drew 250-330 W during repeats that
# came out a third slower, and 130-140 W during the fast ones of the same
# point; the idle draw at the start was 76 W against 45-49 W on 21.09. Our
# own work at those points takes 130-140 W. Something else was using the
# card, in bursts, and every slow repeat sits inside a burst. So before every
# measured repeat the card is asked whether it is busy, and if it is, the
# repeat waits - and the wait is said out loud, with who is on the card.
QUIET_UTIL = 10          # per cent: above this, with nothing of ours running
QUIET_PAUSE_S = 1.0      # let our last program's own load fall off first
QUIET_POLL_S = 15.0
QUIET_MAX_WAIT_S = 900.0
_QUIET = {"ok": None, "waited_s": 0.0, "waits": 0}


def _card_now():
    """(utilization %, power W) from nvidia-smi, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=15).decode("ascii", "replace")
        first = out.strip().splitlines()[0].split(",")
        return float(first[0]), float(first[1])
    except Exception:
        return None


def _who_is_on_card():
    """Every process that computes on the card, with its memory.

    Version 13: the whole list from --query-compute-apps. Version 12 printed
    the first 12 lines of the nvidia-smi table, and on 25.09 those were all
    desktop programs - the one that loaded the card was never named."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL,
            timeout=15).decode("utf-8", "replace")
    except Exception:
        return ["(nvidia-smi does not list the processes)"]
    rows = [l.strip() for l in out.splitlines() if l.strip()]
    return rows or ["(no compute process is listed - the load may come from"
                    " graphics)"]

def wait_until_card_quiet(tell):
    """Wait while someone else is using the card. Returns the seconds waited.

    tell - the function that prints (print or say)."""
    if _QUIET["ok"] is False:
        return 0.0
    time.sleep(QUIET_PAUSE_S)
    now = _card_now()
    if now is None:
        if _QUIET["ok"] is None:
            tell("   (nvidia-smi does not answer - cannot check whether the"
                 " card is free; measuring anyway)")
        _QUIET["ok"] = False
        return 0.0
    _QUIET["ok"] = True
    if now[0] <= QUIET_UTIL:
        return 0.0
    t0 = time.time()
    _QUIET["waits"] += 1
    tell("   CARD BUSY: %.0f %% load, %.0f W, with nothing of ours running."
         " Waiting for it to be free. On the card now:" % now)
    for l in _who_is_on_card():
        tell("      " + l)
    while time.time() - t0 < QUIET_MAX_WAIT_S:
        time.sleep(QUIET_POLL_S)
        now = _card_now() or (0.0, 0.0)
        if now[0] <= QUIET_UTIL:
            w = time.time() - t0
            _QUIET["waited_s"] += w
            tell("   card free again after %.0f s (%.0f %%, %.0f W)"
                 % (w, now[0], now[1]))
            return w
    w = time.time() - t0
    _QUIET["waited_s"] += w
    tell("   the card is STILL busy after %.0f min - measuring anyway; this"
         " repeat is not to be trusted" % (w / 60.0))
    return w


def _abs(path, base):
    return path if os.path.isabs(path) else os.path.normpath(
        os.path.join(base, path))

CODECS = {
    "fv": {"enc": "JpegSample" + EXE, "dec": "JpegSample" + EXE,
           "name": "Fastvideo JPEG"},
    "nv": {"enc": "nvjpegEncoderSample" + EXE,
           "dec": "nvjpegDecoderSample" + EXE, "name": "nvJPEG"},
}

# NVIDIA's own programs for step 1, built by get-nvidia-jpeg-sample-01.py.
# Names are those of add_executable in their CMakeLists.txt.
STOCK = {"dec": "nvJPEG" + EXE, "enc": "nvJPEG_encoder" + EXE,
         "hw": "nvjpegDecoder" + EXE}
STOCK_DIRS = [DEF_NVIDIA, ".", "nvidia-jpeg-sample"]
STEP1_AGREE = 5.0            # per cent: "the harness agrees with NVIDIA"


# The same colour frames as the JPEG2000 article on the RTX 4090, so the two
# articles can be read side by side. File names only: the folder is --img.
IMAGE_NAMES = [("2k", "2k_wild.ppm"), ("4k", "4k_wild.ppm"),
               ("8k", "8k_wild.ppm")]
# Frames measured only when their file is there. Version 13: 8K is new and
# may not have been copied yet; its absence is said, not an error.
IMAGE_OPTIONAL = {"8k"}
IMAGES_SKIPPED = []
IMAGES = list(IMAGE_NAMES)                 # filled with full paths on start

# Chroma subsampling. The first run is 4:4:4 (no subsampling); 4:2:0 is added
# with --subs 444,420. Baseline only: 4:1:0 and 4:1:1 are not part of the
# comparison.
SUB_MAIN = "444"

# Quality: the speed grid is measured at QUALITIES; the scale check walks
# SCALE_Q. The Fastvideo value is the reference; nvJPEG gets the value that
# gives the same file size (see phase_scales).
QUALITY_MAIN = 90
# Version 11 (24.09.2026): the final run measures what the article asks and
# nothing more - Fyodor: "we do not need every combination; state the goals
# and answer the main questions". Quality 95 is left out: it repeated what
# 90 says, one step lower. Other qualities can still be asked for by hand.
QUALITIES_FINAL = [90]
SCALE_Q = [80, 85, 90, 95]

# Threads x frames in a thread, the same grid as the JPEG2000 run on 4090.
# Both codecs have -async -thread -b, so both are measured on all of it.
# The climb and the plateau: the first four points are where each codec
# saturates, the rest are the plateau (filled trial of 21.09).
#
# Version 11: only the plateau is measured. The climb (1x1 .. 8x1) shows how
# a codec saturates, which the article does not use; the three plateau
# points below contain the best point of both codecs in every run so far.
# The single-frame point is measured apart, as before.
POINTS = [(8, 2), (32, 1), (32, 2)]
POINTS_FULL = [(1, 1), (2, 1), (4, 1), (8, 1), (8, 2), (16, 2), (8, 4),
               (32, 1), (32, 2)]      # the grid up to version 10, for reference
PROBE_POINT = (8, 2)
PROBE_FALLBACK = [(8, 2), (4, 2), (2, 1), (1, 1)]

# Card memory. A grid point that does not fit for ONE codec is left out for
# BOTH (bench-07, rule A of 31.08: 4K 32x2 once ran for one codec into
# thrashing and gave "46 and 15 fps" - numbers that measure the memory
# manager, not a codec). A frame in flight costs its input pixels, its output
# pixels and the codec's work buffers: SLOT_FACTOR frames of RGB is the
# estimate when the program does not report its own allocation, and the
# larger of the two is used when it does. The 24, 65 and 100-150 Mpix frames
# are where this starts to decide.
SLOT_FACTOR = 3.0
MEM_SHARE = 0.85

# Native batch of nvJPEG. It goes past 50 on purpose: below that the GPU
# hybrid backend runs the Huffman stage on the CPU (nvjpeg.h, 12.4).
# Version 11: 8 is left out - it gives the same as 1 and 32 (Huffman on the
# processor all three times, 179 / 176 / 177 fps on 21.09). 32 stays: it is
# the profiler's pair with 128.
NV_BATCHES = [1, 32, 64, 128]

# --- the Fastvideo sample: everything taken on trust is here --------------
FV_DISCARD = ["-discard"]      # skip writing the output file inside -repeat
FV_THREAD_R = 2                # reader threads, as in the SDK .cmd examples
FV_THREAD_W = 1                # writer threads, as in the SDK .cmd examples

# Search for nvJPEG quality that gives a target size. JPEG quality is an
# integer, so the size moves in steps; the tolerance is what an integer step
# can reach, and the miss is reported, not hidden.
CALIB_TOL = 0.01
SCALE_INTERVALS = [("wide", 1, 100), ("narrow", 50, 99)]

# When the repeats of one point disagree by more than RESPREAD_LIMIT per
# cent, more repeats are added - up to RESPREAD_MAX_REPS in all. If they
# still disagree after that, the point is NOT thrown away and NOT silently
# averaged: it goes into spread.csv with "wide", it is named in the run's
# closing lines, and nothing from it may go into an article until it is
# understood.
#
# Why this was tightened on 23.09.2026: version 08 added at most two extra
# repeats and then returned the median whatever the spread was, saying the
# figure only in a console line that scrolls away. On 23.09 a side script
# found repeats of one Fastvideo point at 2817, 3015 and 3674 frames per
# second - a spread of 30 per cent - and the median of that is not a
# measurement. Nothing in the run files said so.
RESPREAD_LIMIT = 7.0
RESPREAD_MAX_REPS = 7
RESPREAD_EXTRA = 2          # kept for compatibility; no longer the limit

# Every point that was measured, with its repeats: filled by measure(),
# written to spread.csv and reported at the end of the run.
RESPREAD_LOG = []
ENERGY_RUN_S = 6.0
ENERGY_MIN_S = 2.0          # below this the N run is start-up, not frames
ENERGY_GROWTH = 1.7         # 2N must really take this much longer than N
RUN_S_FINAL = 4.0
RUN_S_DEFAULT = 1.5
TRIAL_FRAMES = 20
MIN_FRAMES = 20
# ... and at least this many for every frame in flight (threads x batch), so
# that a point is measured filled rather than filling (lesson of 21.09).
FRAMES_PER_SLOT = 8

# ---------------------------------------------------------------------------
# reading program output
# ---------------------------------------------------------------------------

RE_SDK = re.compile(r"SDK version:\s*(\S+)")
RE_GPU = re.compile(r"Processing unit:\s*(.+?)\s*\(device id")
RE_PCIE = re.compile(r"PCI-Express bandwidth test \(host to device\):\s*([\d.]+)")
RE_MEM = re.compile(r"Requested GPU memory size:\s*([\d.]+)\s*(GB|MB|KB)")
RE_AVAIL = re.compile(r"Available GPU memory size:\s*([\d.]+)\s*(GB|MB|KB)")
RE_SIZE = re.compile(r"size\s*=\s*(\d+)\s*KB\s*\(([\d.]+):1\)")
RE_CALIB_FULL = re.compile(
    r"Calibration:\s*q\s*=\s*([\d.]+);\s*size\s*=\s*(\d+)\s*bytes;"
    r"\s*target\s*=\s*(\d+)\s*bytes;\s*miss\s*=\s*([-+]?[\d.]+)")
RE_STAGE = re.compile(r"^\s*([\d.]+)\s*ms\s+(\d+)\)\s*(.+?)\s*$", re.M)
# "for N images [per T threads] <any words> = X ms; [Y MB/s;] F FPS[;]"
# The words in the middle are what names the boundary ("including all
# transfers", "excluding host-to-device transfer"), so they are allowed and
# then read by boundary_of. The trailing semicolon is optional: the Fastvideo
# sample prints it only in the async shape.
RE_SUMMARY = re.compile(
    r"for\s+(\d+)\s+images"
    r"(?:\s+per\s+(\d+)\s+(?:threads?|batch))?"
    r"[^=\n]*"
    r"=\s*([\d.]+)\s*ms;"
    r"(?:\s*([\d.]+)\s*MB/s;)?"
    r"\s*([\d.]+)\s*FPS;?")
# The sample's own two legs, kept so a window can be reconstructed.
RE_FV_H2D = re.compile(r"Host-to-device transfer\s*=\s*([\d.]+)\s*ms")
RE_FV_EFF = re.compile(r"Effective encoding performance \(includes "
                       r"device-to-host transfer\)\s*=\s*[\d.]+\s*GB/s\s*"
                       r"\(([\d.]+)\s*ms\)")
RE_FPS_ANY = re.compile(r"([\d.]+)\s*FPS")
RE_RESULT = re.compile(r"^RESULT\s+(\{.*\})\s*$", re.M)
RE_HW_DEC = re.compile(r"Hardware decoder:\s*(AVAILABLE|NOT available).*?"
                       r"engines\s*=\s*(\d+);\s*cores per engine\s*=\s*(\d+)")
RE_HW_ENC = re.compile(r"Hardware encoder:\s*(AVAILABLE|NOT available).*?"
                       r"engines\s*=\s*(\d+)")
RE_HW_STREAM = re.compile(r"hardware decode\s+(SUPPORTED|NOT supported)")
RE_HW_PICTURE = re.compile(r"Picture check:\s*(.+)")

RE_STOCK_DEC = re.compile(r"Avg images per sec:\s*([\d.eE+-]+)")
RE_STOCK_ENC_T = re.compile(r"Total time spent on encoding:\s*([\d.eE+-]+)")
RE_STOCK_ENC_N = re.compile(r"Total images processed:\s*(\d+)")
RE_STOCK_NOHW = re.compile(r"Hardware Decoder not supported")

UNIT_MB = {"GB": 1024.0, "MB": 1.0, "KB": 1.0 / 1024.0}


def boundary_of(text):
    """What the program itself said went into the measured time."""
    if ("excluding host-to-device" in text
            or "excluding the host-to-device" in text):
        return "no_h2d"
    if ("excluding device-to-host" in text
            or "excluding the device-to-host" in text):
        return "no_d2h"
    if "including all transfers" in text:
        return "all"
    return "default"


def parse_output(text):
    """Everything a run said about itself.

    The nvJPEG harness prints one RESULT line of JSON, and that line is taken
    first: a number read from JSON cannot be moved by a change of wording. The
    Fastvideo sample prints prose, which is read with the same patterns as the
    JPEG2000 samples; the LAST summary line wins, as in bench-07.
    """
    out = {}
    for rx, key in ((RE_SDK, "sdk"), (RE_GPU, "gpu"), (RE_PCIE, "pcie_mb_s")):
        m = rx.search(text)
        if m:
            out[key] = m.group(1)
    m = RE_MEM.search(text)
    if m:
        out["gpu_mem_mb"] = float(m.group(1)) * UNIT_MB[m.group(2)]
    m = RE_AVAIL.search(text)
    if m:
        out["gpu_avail_mb"] = float(m.group(1)) * UNIT_MB[m.group(2)]
    m = RE_SIZE.search(text)
    if m:
        out["out_kb"] = int(m.group(1))
        out["cr"] = float(m.group(2))
    m = RE_FV_H2D.search(text)
    if m:
        out["h2d_ms"] = float(m.group(1))
    m = RE_FV_EFF.search(text)
    if m:
        out["eff_ms"] = float(m.group(1))
    m = RE_CALIB_FULL.search(text)
    if m:
        out["calib_q"] = float(m.group(1))
        out["calib_bytes"] = int(m.group(2))
        out["calib_target"] = int(m.group(3))
        out["calib_miss"] = float(m.group(4))
    for line in text.splitlines():
        if line.startswith("ERROR"):
            out.setdefault("error", line.strip()[:200])

    results = RE_RESULT.findall(text)
    if results:
        try:
            j = json.loads(results[-1])
        except ValueError:
            j = None
        if j:
            out["frames"] = int(j.get("images") or 0)
            out["total_ms"] = float(j.get("ms") or 0.0)
            out["ms_per_frame"] = float(j.get("ms_per_frame") or 0.0)
            out["fps"] = float(j.get("fps") or 0.0)
            out["mb_s"] = j.get("mb_s")
            out["window"] = j.get("window")
            out["boundary"] = ("all" if j.get("window") == "host2host"
                               else ("no_d2h" if j.get("dir") == "D"
                                     else "no_h2d"))
            for k in ("path", "huffman", "backend"):
                if j.get(k) is not None:
                    out[k] = j.get(k)
            out["parsed_from"] = "RESULT"
            return out

    best = None
    for line in text.splitlines():
        sm = RE_SUMMARY.search(line)
        if sm:
            best = (line, sm)
    if best:
        line, sm = best
        frames = int(sm.group(1))
        total = float(sm.group(3))
        out["frames"] = frames
        out["total_ms"] = total
        out["ms_per_frame"] = total / frames if frames else 0.0
        out["fps"] = float(sm.group(5))
        out["mb_s"] = sm.group(4) or ""
        out["boundary"] = boundary_of(line)
        out["parsed_from"] = "summary line"
        return out
    # Last resort: some figure in FPS. Marked, so it is never mistaken for a
    # properly read line.
    fp = RE_FPS_ANY.findall(text)
    if fp:
        out["fps"] = float(fp[-1])
        out["parsed_from"] = "FPS only - CHECK THE LOG"
    return out


# ---------------------------------------------------------------------------
# JPEG file structure: quantisation tables, restart interval, sampling
# ---------------------------------------------------------------------------

ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26,
          33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57,
          50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
          39, 46, 53, 60, 61, 54, 47, 55, 62, 63]


def jpeg_info(path_or_bytes):
    """Markers of a JPEG file, up to the first scan.

    Returns the quantisation tables in natural (row-major) order, the table
    each component uses, the sampling factors, the restart interval and the
    kind of frame. Everything here is what the file itself declares; nothing
    is inferred.
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as fh:
            data = fh.read()
    info = {"dqt": {}, "dqt_bits": {}, "dri": 0, "frame": None,
            "components": [], "dht": 0, "scans": 0, "size": len(data)}
    if data[:2] != b"\xff\xd8":
        info["error"] = "no SOI marker"
        return info
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:
            break
        seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + seglen]
        if marker == 0xDB:                          # DQT
            j = 0
            while j < len(seg):
                pq, tq = seg[j] >> 4, seg[j] & 15
                j += 1
                if pq == 0:
                    vals = list(seg[j:j + 64])
                    j += 64
                else:
                    vals = [struct.unpack(">H", seg[j + 2 * k:j + 2 * k + 2])[0]
                            for k in range(64)]
                    j += 128
                natural = [0] * 64
                for k, v in enumerate(vals):
                    natural[ZIGZAG[k]] = v
                info["dqt"][tq] = natural
                info["dqt_bits"][tq] = 16 if pq else 8
        elif marker == 0xDD:                        # DRI
            info["dri"] = struct.unpack(">H", seg[:2])[0]
        elif marker == 0xC4:                        # DHT
            info["dht"] += 1
        elif marker in (0xC0, 0xC1, 0xC2, 0xC3):    # SOF
            info["frame"] = {0xC0: "baseline", 0xC1: "extended",
                             0xC2: "progressive", 0xC3: "lossless"}[marker]
            info["precision"] = seg[0]
            info["height"], info["width"] = struct.unpack(">HH", seg[1:5])
            nc = seg[5]
            for c in range(nc):
                cid, hv, tq = seg[6 + 3 * c], seg[7 + 3 * c], seg[8 + 3 * c]
                info["components"].append({"id": cid, "h": hv >> 4,
                                           "v": hv & 15, "tq": tq})
        elif marker == 0xDA:                        # SOS: stop at the scan
            info["scans"] += 1
            break
        i += 2 + seglen
    info["subsampling"] = sampling_name(info["components"])
    return info


def sampling_name(comps):
    if not comps:
        return "?"
    if len(comps) == 1:
        return "gray"
    h0, v0 = comps[0]["h"], comps[0]["v"]
    hc, vc = comps[1]["h"], comps[1]["v"]
    key = (h0 // max(hc, 1), v0 // max(vc, 1))
    return {(1, 1): "444", (2, 1): "422", (2, 2): "420",
            (1, 2): "440", (4, 1): "411"}.get(key, "%dx%d" % key)


def tables_of(info):
    """(luma table, chroma table) in natural order, by component."""
    comps = info.get("components") or []
    if not comps:
        return None, None
    luma = info["dqt"].get(comps[0]["tq"])
    chroma = info["dqt"].get(comps[1]["tq"]) if len(comps) > 1 else None
    return luma, chroma


def compare_tables(a, b):
    """Plain words for how two files' quantisation tables relate."""
    la, ca = tables_of(a)
    lb, cb = tables_of(b)
    if la is None or lb is None:
        return {"same": None, "text": "tables not found"}
    dl = max(abs(x - y) for x, y in zip(la, lb))
    dc = (max(abs(x - y) for x, y in zip(ca, cb))
          if (ca is not None and cb is not None) else 0)
    same = (dl == 0 and dc == 0)
    return {"same": same, "luma_max_diff": dl, "chroma_max_diff": dc,
            "luma_dc": (la[0], lb[0]), "text": "identical" if same else
            "differ: luma max |d| %d, chroma max |d| %d" % (dl, dc)}


# ---------------------------------------------------------------------------
# images: PPM reading, PSNR, the watermark of a demo build
# ---------------------------------------------------------------------------

def read_ppm(path):
    """(width, height, comps, pixel bytes) of a binary 8-bit PPM/PGM."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 10 or data[0:1] != b"P":
        return None
    comps = 3 if data[1:2] == b"6" else 1
    pos = 2
    fields = []
    while len(fields) < 3 and pos < len(data):
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while pos < len(data) and data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    pos += 1
    w, h, _maxval = fields
    return w, h, comps, data[pos:pos + w * h * comps]


def ppm_geometry(path):
    try:
        r = read_ppm(path)
    except (IOError, ValueError, IndexError):
        return None
    if not r:
        return None
    return {"w": r[0], "h": r[1], "comps": r[2]}


def _np():
    try:
        import numpy
        return numpy
    except ImportError:
        return None


def psnr(src_path, out_path, exclude=None, stride=7):
    """PSNR of a decoded frame against the source, optionally leaving out a
    rectangle (x0, y0, x1, y1) - the watermark of a demo build.

    Exact with numpy; without it, on a uniform sample of the frame, and the
    result says so.
    """
    a = read_ppm(src_path)
    b = read_ppm(out_path)
    if not a or not b:
        return {"kind": "unreadable"}
    if a[:3] != b[:3]:
        return {"kind": "different geometry"}
    w, h, comps = a[:3]
    np = _np()
    if np is not None:
        va = np.frombuffer(a[3], dtype=np.uint8).reshape(h, w, comps)
        vb = np.frombuffer(b[3], dtype=np.uint8).reshape(h, w, comps)
        d = va.astype(np.int32) - vb.astype(np.int32)
        mask = np.ones((h, w), dtype=bool)
        if exclude:
            x0, y0, x1, y1 = exclude
            mask[y0:y1 + 1, x0:x1 + 1] = False
        dd = d[mask]
        mse = float(np.mean(dd.astype(np.float64) ** 2)) if dd.size else 0.0
        return {"kind": "psnr", "exact": True,
                "psnr": (10.0 * math.log10(255.0 * 255.0 / mse))
                if mse > 0 else None,
                "max_diff": int(np.abs(dd).max()) if dd.size else 0}
    da, db = a[3], b[3]
    total, count, maxd = 0.0, 0, 0
    row = w * comps
    for i in range(0, min(len(da), len(db)), max(1, stride)):
        if exclude:
            y = i // row
            x = (i % row) // comps
            x0, y0, x1, y1 = exclude
            if x0 <= x <= x1 and y0 <= y <= y1:
                continue
        d = da[i] - db[i]
        total += d * d
        count += 1
        if abs(d) > maxd:
            maxd = abs(d)
    if not count:
        return {"kind": "empty"}
    mse = total / count
    return {"kind": "psnr", "exact": False, "sample_stride": stride,
            "psnr": (10.0 * math.log10(255.0 * 255.0 / mse))
            if mse > 0 else None, "max_diff": maxd}


def watermark_box(src_path, out_path, threshold=40):
    """The rectangle where a decoded near-lossless frame departs from the
    source by more than JPEG ever does at quality 100.

    A demo build draws its watermark on the frame before encoding. At q 100
    JPEG differs from the source by a few levels; the watermark differs by a
    lot, and only inside its rectangle. That rectangle is then left out of
    PSNR for BOTH codecs - the same region on both sides, so neither is
    favoured. JPEG has no lossless mode here, so the JPEG2000 trick (compare
    against the lossless round trip) is not available; this is its stand-in.
    """
    a = read_ppm(src_path)
    b = read_ppm(out_path)
    if not a or not b or a[:3] != b[:3]:
        return None
    w, h, comps = a[:3]
    np = _np()
    if np is not None:
        va = np.frombuffer(a[3], dtype=np.uint8).reshape(h, w, comps)
        vb = np.frombuffer(b[3], dtype=np.uint8).reshape(h, w, comps)
        big = (np.abs(va.astype(np.int32) - vb.astype(np.int32))
               > threshold).any(axis=2)
        ys, xs = np.nonzero(big)
        if not ys.size:
            return None
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    x0 = y0 = 10 ** 9
    x1 = y1 = -1
    da, db = a[3], b[3]
    row = w * comps
    for i in range(0, min(len(da), len(db)), 5):
        if abs(da[i] - db[i]) > threshold:
            y = i // row
            x = (i % row) // comps
            x0, y0 = min(x0, x), min(y0, y)
            x1, y1 = max(x1, x), max(y1, y)
    return (x0, y0, x1, y1) if x1 >= 0 else None


# ---------------------------------------------------------------------------
# power, energy, the JPEG engine counter
# ---------------------------------------------------------------------------

class Power(object):
    """Samples GPU power with nvidia-smi in the background."""

    def __init__(self, device=0, interval_ms=100):
        self.device = device
        self.interval_ms = interval_ms
        self.proc = None
        self.samples = []
        self._thread = None
        self.idle_w = None

    def _reader(self):
        for line in self.proc.stdout:
            try:
                self.samples.append(float(line.decode("ascii", "replace")
                                          .strip()))
            except ValueError:
                pass

    def start(self):
        self.samples = []
        if not shutil.which("nvidia-smi"):
            self.proc = None
            return
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "-i", str(self.device),
                 "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                 "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError:
            self.proc = None
            return
        self._thread = threading.Thread(target=self._reader)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        if not self.proc:
            return None
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)

    def measure_idle(self, seconds=3.0):
        self.start()
        time.sleep(seconds)
        self.idle_w = self.stop()
        return self.idle_w


class NvmlDevice(object):
    """The card through NVML: the energy counter and the JPEG engine counter.

    Two independent things from one handle. The energy counter counts the
    millijoules the card has spent since the driver loaded, so nothing is lost
    between samples. The JPEG engine counter (nvmlDeviceGetJpgUtilization) is
    the share of time the hardware JPEG engines were busy over the driver's
    sampling period; during a run on the hardware backend it must be above
    zero, and during a run on the CUDA path it must be zero. That contrast is
    the independent evidence that the engine really did the work.
    """

    def __init__(self, device=0):
        self.nv = None
        self.handle = None
        self.error = None
        self.energy_ok = False
        self.jpg_ok = False
        self.jpg_error = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception as exc:
            self.error = str(exc) or exc.__class__.__name__
            return
        try:
            self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.energy_ok = True
        except Exception as exc:
            self.error = "energy counter: %s" % exc
        try:
            self.nv.nvmlDeviceGetJpgUtilization(self.handle)
            self.jpg_ok = True
        except Exception as exc:
            self.jpg_error = str(exc) or exc.__class__.__name__

    def read_j(self):
        if not self.energy_ok:
            return None
        try:
            return self.nv.nvmlDeviceGetTotalEnergyConsumption(
                self.handle) / 1000.0
        except Exception:
            return None

    def jpg_util(self):
        if not self.jpg_ok:
            return None
        try:
            return float(self.nv.nvmlDeviceGetJpgUtilization(self.handle)[0])
        except Exception:
            return None


class JpgSampler(object):
    """Reads the JPEG engine counter every 50 ms while a program runs."""

    def __init__(self, dev):
        self.dev = dev
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not (self.dev and self.dev.jpg_ok):
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(0.05):
            v = self.dev.jpg_util()
            if v is not None:
                self.samples.append(v)

    def stop(self):
        if not self._thread:
            return None
        self._stop.set()
        self._thread.join(timeout=2)
        self._thread = None
        if not self.samples:
            return None
        return {"max": max(self.samples),
                "mean": sum(self.samples) / len(self.samples),
                "n": len(self.samples)}


# ---------------------------------------------------------------------------
# processor time of a child process
# ---------------------------------------------------------------------------

def rusage_children():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_CHILDREN)
        return r.ru_utime + r.ru_stime
    except Exception:
        return None


def child_cpu_seconds(popen, before=None):
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class FILETIME(ctypes.Structure):
                _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

            def as_s(ft):
                return ((ft.high << 32) | ft.low) / 1e7
            c, e, k, u = FILETIME(), FILETIME(), FILETIME(), FILETIME()
            ok = ctypes.windll.kernel32.GetProcessTimes(
                int(popen._handle), ctypes.byref(c), ctypes.byref(e),
                ctypes.byref(k), ctypes.byref(u))
            return (as_s(k) + as_s(u)) if ok else None
        except Exception:
            return None
    after = rusage_children()
    if after is None or before is None:
        return None
    return max(0.0, after - before)


# ---------------------------------------------------------------------------
# running one program
# ---------------------------------------------------------------------------

def human(seconds):
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))


class Heartbeat(object):
    def __init__(self, period=HEARTBEAT_S):
        self.period = period
        self.what = "starting"
        self.since = time.time()
        self._stop = threading.Event()
        self._thread = None

    def set(self, what):
        self.what = what
        self.since = time.time()

    def start(self):
        if self._thread:
            return
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.period):
            print("   %s  still working: %s, %s so far"
                  % (datetime.datetime.now().strftime("%H:%M:%S"),
                     self.what, human(time.time() - self.since)))
            sys.stdout.flush()


HEART = Heartbeat()
_BENCHES = []


class Bench(object):
    def __init__(self, outdir, dry_run=False, dev=None):
        self.outdir = outdir
        self.logdir = os.path.join(outdir, "logs")
        self.dry_run = dry_run
        self.dev = dev
        self.rows = []
        self.n_runs = 0
        os.makedirs(self.logdir, exist_ok=True)
        self.jsonl_path = os.path.join(outdir, "results.jsonl")
        self.jsonl = None if dry_run else open(self.jsonl_path, "a",
                                               encoding="utf-8")
        _BENCHES.append(self)

    def close(self):
        if self.jsonl:
            self.jsonl.close()
            self.jsonl = None

    def run(self, exe, args, log_name, power=None, energy=False, jpg=False):
        # A log name is a tag, never a path: since version 02 the programs are
        # called by full paths, and one of those reaching a file name is what
        # broke --build on 21.09. Guarded here so it cannot happen again.
        log_name = re.sub(r'[\\/:*?"<>|]', "_", str(log_name))
        if os.path.exists(exe):
            exe = os.path.abspath(exe)
        cmd = [exe] + [str(a) for a in args]
        if self.dry_run:
            print("   would run:", " ".join(cmd))
            return {}
        self.n_runs += 1
        HEART.set(log_name)
        cpu_before = rusage_children() if os.name != "nt" else None
        e_before = self.dev.read_j() if (energy and self.dev) else None
        sampler = JpgSampler(self.dev) if jpg else None
        if power:
            power.start()
        if sampler:
            sampler.start()
        t0 = time.time()
        p = None
        rc = None
        cpu_s = None
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            out, _ = p.communicate(timeout=900)
            rc = p.returncode
            text = out.decode("utf-8", "replace")
            cpu_s = child_cpu_seconds(p, cpu_before)
        except FileNotFoundError:
            text = "ERROR: %s not found\n" % exe
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            text = "ERROR: timed out\n"
        except KeyboardInterrupt:
            if p is not None:
                try:
                    p.kill()
                    p.communicate()
                except Exception:
                    pass
            if power:
                power.stop()
            if sampler:
                sampler.stop()
            raise
        wall = time.time() - t0
        jpg_stat = sampler.stop() if sampler else None
        avg_w = power.stop() if power else None
        e_after = self.dev.read_j() if (energy and self.dev) else None
        res = parse_output(text)
        with open(os.path.join(self.logdir, log_name + ".log"), "w",
                  encoding="utf-8") as fh:
            fh.write("# parsed: %s\n" % json.dumps(
                dict((k, v) for k, v in res.items()), default=str))
            fh.write("$ " + " ".join(cmd) + "\n\n" + text)
        res["rc"] = rc
        res["wall_s"] = wall
        if cpu_s is not None:
            res["cpu_s"] = cpu_s
            if wall > 0:
                res["cores"] = cpu_s / wall
        frames = res.get("frames") or 0
        if avg_w:
            res["power_w"] = avg_w
            if frames:
                res["j_per_frame"] = avg_w * wall / frames
        if e_before is not None and e_after is not None and e_after >= e_before:
            res["energy_j_counter"] = e_after - e_before
            if frames:
                res["j_per_frame_counter"] = (e_after - e_before) / frames
        if jpg_stat:
            res["jpg_util_max"] = jpg_stat["max"]
            res["jpg_util_mean"] = jpg_stat["mean"]
        res["cmd"] = " ".join(cmd)
        res["raw"] = text
        return res

    def record(self, **kw):
        kw.setdefault("bench_version", BENCH_VERSION)
        kw.setdefault("at", datetime.datetime.now().isoformat(
            timespec="seconds"))
        g = GEOM.get(kw.get("image"))
        if g and kw.get("fps"):
            kw["mp_s"] = kw["fps"] * g["w"] * g["h"] / 1e6
        if kw.get("codec") == "nv":
            role = "dec" if kw.get("direction") == "D" else "enc"
            kw.setdefault("nv_harness",
                          NV_HARNESS.get(CODECS["nv"][role]) or "")
        self.rows.append(kw)
        if self.jsonl:
            line = dict((k, v) for k, v in kw.items() if k != "raw")
            self.jsonl.write(json.dumps(line, ensure_ascii=False,
                                        default=str) + "\n")
            self.jsonl.flush()
            os.fsync(self.jsonl.fileno())


GEOM = {}

# ---------------------------------------------------------------------------
# building and checking the nvJPEG harness
# ---------------------------------------------------------------------------


def _nv_source(folder):
    """The harness source with the highest number in OUR folder."""
    best, key = None, None
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        names = []
    for name in names:
        m = re.match(r"^nvjpeg_bench-(\d+)\.cpp$", name)
        if m and (key is None or int(m.group(1)) > key):
            best, key = os.path.join(folder, name), int(m.group(1))
    return (best, True) if best else (os.path.join(folder,
                                                   "nvjpeg_bench-NN.cpp"), False)


NV_SOURCE, NV_SOURCE_FOUND = _nv_source(DEF_NV)
NV_HARNESS = {}


def resolve_paths(fv=None, img=None, nv=None, nvidia=None, images=None):
    """Turn the four folders into full paths, once, before anything runs.

    From here on every program is called by its own path: nothing is copied
    into a shared folder, and neither side is written into by the other.
    Returns a list of complaints about --images, empty when all is well.
    """
    global IMAGES, STOCK_DIRS, NV_SOURCE, NV_SOURCE_FOUND, NV_DIR
    fv = _abs(fv, ORIG_CWD) if fv else _abs(DEF_FV, HERE)
    img = _abs(img, ORIG_CWD) if img else fv    # the frames ship with the SDK
    nv = _abs(nv, ORIG_CWD) if nv else _abs(DEF_NV, HERE)
    nvidia = _abs(nvidia, ORIG_CWD) if nvidia else _abs(DEF_NVIDIA, HERE)
    NV_DIR = nv

    CODECS["fv"]["enc"] = os.path.join(fv, "JpegSample" + EXE)
    CODECS["fv"]["dec"] = CODECS["fv"]["enc"]      # one program, both ways
    CODECS["nv"]["enc"] = os.path.join(nv, "nvjpegEncoderSample" + EXE)
    CODECS["nv"]["dec"] = os.path.join(nv, "nvjpegDecoderSample" + EXE)

    # NVIDIA's programs: their own folder first, then the two places version
    # 01 looked, so a folder made by an earlier run still works.
    STOCK_DIRS = [nvidia, os.path.join(nvidia, "bin"), ".",
                  "nvidia-jpeg-sample"]
    NV_SOURCE, NV_SOURCE_FOUND = _nv_source(nv)

    bad = []
    if images:
        IMAGES = []
        for part in images.split(","):
            tag, _, path_ = part.partition("=")
            if not path_:
                bad.append(part)
                continue
            path_ = path_.strip()
            IMAGES.append((tag.strip(), path_ if os.path.isabs(path_)
                           else _abs(path_, img)))
    else:
        IMAGES = []
        del IMAGES_SKIPPED[:]
        for t, name in IMAGE_NAMES:
            path_ = os.path.join(img, name)
            if t in IMAGE_OPTIONAL and not os.path.exists(path_):
                IMAGES_SKIPPED.append((t, path_))
                continue
            IMAGES.append((t, path_))
    return bad


def source_version(path):
    try:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
    except Exception:
        return None
    m = re.search(r"nvjpeg_bench version\s+(\S+?),", first)
    return m.group(1) if m else None


def exe_version(exe):
    try:
        p = subprocess.run([os.path.abspath(exe), "-version"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=60)
    except Exception:
        return None
    m = re.search(r"nvjpeg_bench version\s+(\S+?),",
                  p.stdout.decode("utf-8", "replace"))
    return m.group(1) if m else None


def check_nv_version(dry_run=False):
    """Stops the run if the executables were not built from the source here."""
    if not NV_SOURCE_FOUND:
        print("   source: NOT FOUND - put nvjpeg_bench-01.cpp next to the "
              "executables.")
        print("   Without it there is no telling which source the "
              "executables came from. Nothing was measured.")
        return False
    want = source_version(NV_SOURCE)
    if want is None:
        print("   source: %s, but its first line names no version" % NV_SOURCE)
        return False
    print("   source: %s (version %s)" % (NV_SOURCE, want))
    bad = []
    for role in ("enc", "dec"):
        exe = CODECS["nv"][role]
        got = None if dry_run else exe_version(exe)
        NV_HARNESS[exe] = got
        print("   %-26s %s" % (exe, got or "no version reported"))
        if not dry_run and got != want:
            bad.append(exe)
    if bad:
        print("")
        print("The executables were not built from %s. Delete them and run"
              % NV_SOURCE)
        print("   python %s --build" % SCRIPT_NAME)
        print("Nothing was measured.")
        return False
    return True


def _vcvars():
    pf = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = os.path.join(pf, "Microsoft Visual Studio", "Installer",
                           "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    try:
        out = subprocess.run([vswhere, "-latest", "-products", "*",
                              "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                              "-property", "installationPath"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=60)
        root = out.stdout.decode("utf-8", "replace").strip().splitlines()
        if root:
            cand = os.path.join(root[0], "VC", "Auxiliary", "Build",
                                "vcvars64.bat")
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    return None


def build_nv(logdir):
    """Build the two nvJPEG executables if they are missing or stale.

    nvJPEG is part of the CUDA Toolkit - unlike nvJPEG2000 nothing has to be
    installed separately: nvjpeg.h and nvjpeg.lib come with CUDA.
    """
    if not NV_SOURCE_FOUND:
        return ("no nvjpeg_bench-NN.cpp in %s, nothing to build "
                "(--nv names that folder)" % NV_DIR)
    try:
        os.makedirs(NV_DIR, exist_ok=True)
    except OSError as e:
        return "cannot use %s: %s" % (NV_DIR, e)
    targets = [(CODECS["nv"]["enc"], []), (CODECS["nv"]["dec"], ["DEC"])]
    src_time = os.path.getmtime(NV_SOURCE)
    if all(os.path.exists(t) and os.path.getmtime(t) >= src_time
           for t, _ in targets):
        return "already built and up to date"
    if os.name != "nt":
        cuda = next((r for r in (os.environ.get("CUDA_HOME", ""),
                                 os.environ.get("CUDA_PATH", ""),
                                 "/usr/local/cuda")
                     if r and os.path.exists(os.path.join(r, "include",
                                                          "nvjpeg.h"))), None)
        if not cuda:
            return "nvjpeg.h not found: set CUDA_HOME to the CUDA Toolkit"
        libs = [d for d in (os.path.join(cuda, "lib64"),
                            os.path.join(cuda, "lib"),
                            os.path.join(cuda, "targets", "aarch64-linux",
                                         "lib"))
                if os.path.isdir(d)]
        for target, extra in targets:
            cmd = [os.environ.get("CXX", "g++"), "-O2", "-std=c++14",
                   "-pthread"] + (["-DBUILD_DECODER"] if extra else []) + [
                "-I" + os.path.join(cuda, "include"), NV_SOURCE, "-o", target]
            for l in libs:
                cmd += ["-L" + l, "-Wl,-rpath," + l]
            cmd += ["-lnvjpeg", "-lcudart"]
            p = subprocess.run(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=600)
            log = "build_%s.log" % os.path.basename(target)
            with open(os.path.join(logdir, log), "w",
                      encoding="utf-8") as fh:
                fh.write(" ".join(cmd) + "\n\n"
                         + p.stdout.decode("utf-8", "replace"))
            if not os.path.exists(target):
                return "build of %s failed, see the log" % target
        return "built %s and %s" % (targets[0][0], targets[1][0])

    cuda = os.environ.get("CUDA_PATH", "")
    inc = os.path.join(cuda, "include")
    lib = os.path.join(cuda, "lib", "x64")
    if not os.path.exists(os.path.join(inc, "nvjpeg.h")):
        return ("nvjpeg.h not found under CUDA_PATH (%s). nvJPEG comes with "
                "the CUDA Toolkit; check that CUDA_PATH points at it." % cuda)
    cl_ready = shutil.which("cl") is not None
    vcvars = None if cl_ready else _vcvars()
    if not cl_ready and not vcvars:
        return ("the Microsoft compiler was not found. Run this from an "
                "'x64 Native Tools Command Prompt for VS'.")
    for target, extra in targets:
        cl = ["cl", "/nologo", "/EHsc", "/O2", "/MD", "/std:c++14"] + (
            ["/DBUILD_DECODER"] if extra else []) + [
            "/I%s" % inc, NV_SOURCE, "/Fe:" + target, "/link",
            "/LIBPATH:%s" % lib, "cudart.lib", "nvjpeg.lib"]
        if cl_ready:
            cmd, shell = cl, False
        else:
            cmd = 'call "%s" >nul && %s' % (vcvars,
                                            subprocess.list2cmdline(cl))
            shell = True
        p = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=600)
        log = "build_%s.log" % os.path.basename(target)
        with open(os.path.join(logdir, log), "w",
                  encoding="utf-8") as fh:
            fh.write(p.stdout.decode("utf-8", "replace"))
        if not os.path.exists(target):
            return "build of %s failed, see the log" % target
    junk = os.path.splitext(NV_SOURCE)[0] + ".obj"
    if os.path.exists(junk):
        try:
            os.remove(junk)
        except OSError:
            pass
    return "built %s and %s" % (targets[0][0], targets[1][0])


# ---------------------------------------------------------------------------
# command lines
# ---------------------------------------------------------------------------

def window_of(extra):
    """The window the Fastvideo sample uses in this mode - and so the window
    nvJPEG is given, so that the two are measured between the same points.

    The sample has no key for it: -async (and the native batch, which only
    nvJPEG has) reports the whole path, plain -repeat drops one leg. Our
    harness's "codec" drops that same leg on each side.
    """
    keys = [str(x) for x in extra]
    return "host2host" if ("-async" in keys or "-batched" in keys) else "codec"


def enc_args(codec, src, out, q, sub, extra, discard=True, variant="gpu"):
    """One encoding run. The same keys on both sides; only what exists on
    one side is added for it."""
    a = ["-i", src, "-o", out, "-q", int(q), "-s", sub] + list(extra)
    if codec == "fv":
        if "-async" in [str(x) for x in extra]:
            a += ["-threadR", FV_THREAD_R, "-threadW", FV_THREAD_W]
        if discard:
            a += FV_DISCARD
    else:
        a += ["-window", window_of(extra)]
        if variant == "hw":
            a += ["-encbackend", "hardware"]
        if discard:
            a += ["-discard"]
    return a


def dec_args(codec, jpg, out, extra, discard=True, backend="gpuhybrid"):
    a = ["-i", jpg, "-o", out] + list(extra)
    if codec == "fv":
        if "-async" in [str(x) for x in extra]:
            a += ["-threadR", FV_THREAD_R, "-threadW", FV_THREAD_W]
        if discard:
            a += FV_DISCARD
    else:
        a += ["-window", window_of(extra), "-backend", backend]
        if discard:
            a += ["-discard"]
    return a


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def median(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def spread_pct(values):
    vals = [v for v in values if v]
    if len(vals) < 2:
        return 0.0
    m = median(vals)
    return (100.0 * (max(vals) - min(vals)) / m) if m else 0.0


def spread(values):
    return ("(spread %.1f %%)" % spread_pct(values)) if len(values) > 1 else ""


def fmt(v, f="%.0f", dash="-"):
    return (f % v) if v not in (None, "") else dash


def rm(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def environment(device=0):
    env = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
           "os": "%s %s" % (platform.system(), platform.release()),
           "python": platform.python_version(),
           "cpu": platform.processor() or platform.machine(),
           "cores": os.cpu_count()}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", str(device),
             "--query-gpu=name,driver_version,memory.total,power.limit",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL).decode("utf-8", "replace").strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            env.update({"gpu": parts[0], "driver": parts[1],
                        "gpu_mem": parts[2], "power_limit": parts[3]})
        out = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL).decode("utf-8",
                                                               "replace")
        m = re.search(r"CUDA Version:\s*([\d.]+)", out)
        if m:
            env["cuda_driver"] = m.group(1)
    except Exception:
        pass
    return env


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

class Run(object):
    """Everything one run needs, in one place, so the phases stay short."""

    def __init__(self, args, b, power, dev):
        self.args = args
        self.b = b
        self.power = power
        self.dev = dev
        self.dry = args.dry_run
        self.codecs = ["fv", "nv"] if args.codec == "both" else [args.codec]
        self.subs = args.subs
        self.qualities = args.qualities
        self.reps = args.reps
        self.run_s = args.run_s
        self.trial = args.trial
        self.speed = {}          # (codec, dir, tag, q, sub) -> probed fps
        self.ref = {}            # (codec, tag, q, sub) -> jpg path
        self.ref_info = {}       # (codec, tag, q, sub) -> jpeg_info
        self.nv_q = {}           # (tag, fv_q, sub) -> nv quality, equal size
        self.hw = {"dec": None, "enc": None, "streams": [], "evidence": []}
        self.watermark = {}      # tag -> box or None
        self.scales = []
        self.checks = []
        self.cross = []
        self.stages = {}
        self.energy = []
        self.mem_slot = {}       # (codec, dir, tag) -> MB per frame in flight
        self.avail_mb = None     # free card memory reported by the programs
        self.skip = {}           # (tag, dir, threads, batch, path) -> reason
        self.env = {}
        self.stock = []          # step 1 rows: NVIDIA's program vs ours

    # frames per measurement: a fixed length, not a share of a budget, so a
    # wider grid does not make every measurement shorter (bench-07, .3 E)
    def frames(self, key, div=1.0, slots=1):
        floor = max(MIN_FRAMES, int(slots) * FRAMES_PER_SLOT)
        if self.trial:
            return max(TRIAL_FRAMES, floor)
        fps = self.speed.get(key) or 100.0
        return max(floor, int(fps * self.run_s / div))

    def nvq(self, tag, q, sub):
        return self.nv_q.get((tag, q, sub), q)

    def q_of(self, codec, tag, q, sub):
        return q if codec == "fv" else self.nvq(tag, q, sub)


def phase_hw(r):
    """Is the hardware JPEG engine really there, and does it really work?

    Asked of the library first (engines, stream support, the picture); the
    engine's own counter is read later, while it works. Nothing is assumed
    from the product name: the documentation lists architectures, not cards.
    """
    b = r.b
    res = b.run(CODECS["nv"]["dec"], ["-hwinfo"], "nv_hwinfo")
    if r.dry:
        return
    raw = res.get("raw", "")
    m = RE_HW_DEC.search(raw)
    r.hw["dec"] = ({"available": m.group(1) == "AVAILABLE",
                    "engines": int(m.group(2)), "cores": int(m.group(3))}
                   if m else {"available": False, "engines": 0,
                              "error": res.get("error") or "no answer"})
    m = RE_HW_ENC.search(raw)
    r.hw["enc"] = ({"available": m.group(1) == "AVAILABLE",
                    "engines": int(m.group(2))}
                   if m else {"available": False, "engines": 0})
    print("   hardware decoder: %s%s"
          % ("AVAILABLE" if r.hw["dec"]["available"] else "not available",
             (", %d engine(s) x %d core(s)" % (r.hw["dec"]["engines"],
                                              r.hw["dec"].get("cores", 0)))
             if r.hw["dec"]["available"] else ""))
    print("   hardware encoder: %s"
          % ("AVAILABLE, %d engine(s)" % r.hw["enc"]["engines"]
             if r.hw["enc"]["available"] else "not available"))
    if r.dev and r.dev.jpg_ok:
        print("   JPEG engine counter (NVML): readable - every hardware run "
              "will carry it")
    else:
        print("   JPEG engine counter (NVML): NOT readable (%s)"
              % ((r.dev.jpg_error if r.dev else None) or "no NVML"))
        print("   the hardware runs will then rest on the library's word "
              "and the picture check")


def phase_hw_streams(r):
    """Per stream: does the engine accept it, and is its picture the CUDA one?"""
    if not (r.hw["dec"] and r.hw["dec"].get("available")):
        return
    for key in sorted(r.ref):
        codec, tag, q, sub = key
        if q not in r.qualities:
            continue            # only the streams that are measured
        jpg = r.ref[key]
        res = r.b.run(CODECS["nv"]["dec"], ["-hwinfo", "-i", jpg],
                      "nv_hwinfo_%s_%s_q%s_%s" % (codec, tag, q, sub),
                      jpg=True)
        if r.dry:
            continue
        raw = res.get("raw", "")
        m1 = RE_HW_STREAM.search(raw)
        m2 = RE_HW_PICTURE.search(raw)
        row = {"stream": "%s %s q%s %s" % (codec, tag, q, sub),
               "supported": (m1.group(1) == "SUPPORTED") if m1 else None,
               "picture": m2.group(1).strip() if m2 else "-",
               "rc": res.get("rc")}
        r.hw["streams"].append(row)
        print("   %-20s  engine: %-13s  picture: %s"
              % (row["stream"], {True: "accepts", False: "REFUSES",
                                 None: "?"}[row["supported"]],
                 row["picture"]))


def phase_refs(r):
    """Reference streams: every codec, frame, quality and subsampling that the
    run needs. Written to real files - these are the decode inputs, the files
    whose quantisation tables are read, and the PSNR inputs."""
    for codec in r.codecs:
        for tag, src in IMAGES:
            for q in sorted(set(r.qualities) | set(SCALE_Q)):
                for sub in r.subs:
                    if codec == "nv" and q in r.qualities and \
                            (tag, q, sub) in r.nv_q:
                        continue          # remade later at the matched q
                    make_ref(r, codec, tag, src, q, sub)


def make_ref(r, codec, tag, src, q, sub, q_used=None):
    qq = q if q_used is None else q_used
    jpg = "%s_ref_%s_q%s_%s.jpg" % (codec, tag, q, sub)
    r.b.run(CODECS[codec]["enc"],
            enc_args(codec, src, jpg, qq, sub, ["-info"], discard=False),
            "%s_ref_%s_q%s_%s" % (codec, tag, q, sub))
    if r.dry or not os.path.exists(jpg):
        return None
    r.ref[(codec, tag, q, sub)] = jpg
    inf = jpeg_info(jpg)
    inf["q_used"] = qq
    r.ref_info[(codec, tag, q, sub)] = inf
    return jpg


def decode_file(r, codec, jpg, out, log):
    r.b.run(CODECS[codec]["dec"], dec_args(codec, jpg, out, [], discard=False),
            log)
    return os.path.exists(out)


def phase_watermark(r):
    """Does the Fastvideo build draw a watermark, and where.

    Encoded at quality 100 and decoded by the nvJPEG CUDA decoder (the same
    decoder for everyone), the frame differs from the source by a few levels
    everywhere - and by a lot inside a watermark, if there is one.
    """
    if "fv" not in r.codecs:
        return
    for tag, src in IMAGES:
        jpg = "fv_wm_%s.jpg" % tag
        ppm = "fv_wm_%s.ppm" % tag
        r.b.run(CODECS["fv"]["enc"],
                enc_args("fv", src, jpg, 100, SUB_MAIN, [], discard=False),
                "fv_wm_enc_%s" % tag)
        dec = "nv" if "nv" in r.codecs else "fv"
        if r.dry or not os.path.exists(jpg):
            continue
        decode_file(r, dec, jpg, ppm, "fv_wm_dec_%s" % tag)
        box = watermark_box(src, ppm) if os.path.exists(ppm) else None
        r.watermark[tag] = box
        print("   %-3s %s" % (tag, ("watermark in the rectangle %d,%d-%d,%d, "
                                    "left out of PSNR on both sides" % box)
                              if box else "no watermark found"))
        rm(jpg, ppm)


def psnr_of(r, codec_dec, jpg, tag, src, log):
    """PSNR of a JPEG file, decoded by the nvJPEG CUDA decoder when it is in
    the run - one decoder for every file, so PSNR compares the ENCODERS."""
    out = log + ".ppm"
    dec = "nv" if "nv" in r.codecs else codec_dec
    if not decode_file(r, dec, jpg, out, log):
        return {"kind": "no output"}
    q = psnr(src, out, exclude=r.watermark.get(tag))
    rm(out)
    return q


def phase_scales(r):
    """Do "their" quality and "ours" mean the same thing?

    Two layers. The tables: both encoders at the same -q, the DQT tables
    compared value by value. The application: size and PSNR - if the tables
    are identical and those still differ, the difference is in how the tables
    are applied, which is not visible in the file. Then the size search: which
    nvJPEG quality gives the Fastvideo file size, searched from two different
    starting intervals so that a property of the scales is told apart from a
    trace of the search itself.
    """
    if r.codecs != ["fv", "nv"]:
        print("   needs both codecs, skipped")
        return
    for tag, src in IMAGES:
        for sub in r.subs:
            for q in SCALE_Q:
                fk = ("fv", tag, q, sub)
                nk = ("nv", tag, q, sub)
                if fk not in r.ref or nk not in r.ref:
                    continue
                fi, ni = r.ref_info[fk], r.ref_info[nk]
                tab = compare_tables(fi, ni)
                target = fi["size"]
                row = {"image": tag, "sub": sub, "q": q,
                       "fv_bytes": fi["size"], "nv_bytes_same_q": ni["size"],
                       "tables_same_q": tab["text"],
                       "tables_identical": tab["same"],
                       "fv_dri": fi["dri"], "nv_dri": ni["dri"]}
                for name, lo, hi in SCALE_INTERVALS:
                    res = r.b.run(CODECS["nv"]["enc"],
                                  ["-i", src, "-targetsize", target,
                                   "-q", q, "-s", sub, "-tol", CALIB_TOL,
                                   "-qlo", lo, "-qhi", hi],
                                  "nv_scale_%s_%s_q%s_%s" % (tag, sub, q,
                                                             name))
                    row["nv_q_" + name] = res.get("calib_q")
                    row["nv_miss_" + name] = res.get("calib_miss")
                row["psnr_fv"] = psnr_of(r, "fv", r.ref[fk], tag, src,
                                         "psnr_fv_%s_q%s_%s" % (tag, q, sub)
                                         ).get("psnr")
                row["psnr_nv_same_q"] = psnr_of(
                    r, "nv", r.ref[nk], tag, src,
                    "psnr_nv_%s_q%s_%s" % (tag, q, sub)).get("psnr")
                mq = row.get("nv_q_wide")
                if mq and not r.dry:
                    mjpg = "nv_match_%s_q%s_%s.jpg" % (tag, q, sub)
                    r.b.run(CODECS["nv"]["enc"],
                            enc_args("nv", src, mjpg, int(round(mq)), sub, [],
                                     discard=False),
                            "nv_match_%s_q%s_%s" % (tag, q, sub))
                    if os.path.exists(mjpg):
                        mi = jpeg_info(mjpg)
                        row["nv_bytes_matched"] = mi["size"]
                        row["tables_matched_q"] = compare_tables(fi, mi)["text"]
                        row["psnr_nv_matched"] = psnr_of(
                            r, "nv", mjpg, tag, src,
                            "psnr_nvm_%s_q%s_%s" % (tag, q, sub)).get("psnr")
                        rm(mjpg)
                r.scales.append(row)
                print("   %-3s %s q%-3d  fv %8s B  nv same q %8s B  tables %s"
                      % (tag, sub, q, row["fv_bytes"], row["nv_bytes_same_q"],
                         tab["text"]))
                print("   %18s nv q for the fv size: %s (from [1,100]), %s "
                      "(from [50,99]); PSNR fv %s, nv same q %s, nv matched %s"
                      % ("", fmt(row.get("nv_q_wide")),
                         fmt(row.get("nv_q_narrow")),
                         fmt(row.get("psnr_fv"), "%.2f"),
                         fmt(row.get("psnr_nv_same_q"), "%.2f"),
                         fmt(row.get("psnr_nv_matched"), "%.2f")))
                if q in r.qualities and row.get("nv_q_wide"):
                    r.nv_q[(tag, q, sub)] = int(round(row["nv_q_wide"]))


def phase_matched_refs(r):
    """nvJPEG references remade at the quality that matches our file size -
    the speed comparison is made at equal size, as for JPEG2000."""
    if "nv" not in r.codecs:
        return
    for tag, src in IMAGES:
        for q in r.qualities:
            for sub in r.subs:
                nq = r.nvq(tag, q, sub)
                make_ref(r, "nv", tag, src, q, sub, q_used=nq)
                fk, nk = ("fv", tag, q, sub), ("nv", tag, q, sub)
                if r.dry or nk not in r.ref_info:
                    continue
                fb = r.ref_info.get(fk, {}).get("size")
                nb = r.ref_info[nk]["size"]
                print("   %-3s %s fv q%-3d %9s B   nv q%-3d %9s B   %s"
                      % (tag, sub, q, fb or "-", nq, nb,
                         ("%+.2f %%" % (100.0 * (nb - fb) / fb)) if fb else ""))


def card_memory_mb(env):
    """Total card memory from nvidia-smi, e.g. '24564 MiB' -> 24564.0."""
    m = re.match(r"\s*([\d.]+)\s*MiB", env.get("gpu_mem") or "")
    return float(m.group(1)) if m else None


def frame_mb(tag):
    g = GEOM.get(tag)
    return (g["w"] * g["h"] * 3 / 1048576.0) if g else 0.0


def probe_point(r, tag):
    """The largest probe point that surely fits: the probe of a 150 Mpix frame
    at 8x2 would itself run out of memory before it could say so."""
    total = r.avail_mb or card_memory_mb(r.env)
    for th, ba in PROBE_FALLBACK:
        if not total or SLOT_FACTOR * frame_mb(tag) * th * ba \
                <= MEM_SHARE * total:
            return th, ba
    return 1, 1


def phase_probe(r):
    """How fast is each workload, so every measurement gets a fixed length,
    and how much card memory a frame in flight costs."""
    for codec in r.codecs:
        for tag, src in IMAGES:
            T, B = probe_point(r, tag)
            extra = ["-repeat", 100 if not r.trial else TRIAL_FRAMES, "-async",
                     "-thread", T, "-b", B]
            for q in r.qualities:
                for sub in r.subs:
                    qq = r.q_of(codec, tag, q, sub)
                    jpg = r.ref.get((codec, tag, q, sub))
                    re_ = r.b.run(CODECS[codec]["enc"],
                                  enc_args(codec, src, "tmp.jpg", qq, sub,
                                           extra),
                                  "%s_probe_E_%s_q%s_%s" % (codec, tag, q,
                                                            sub))
                    rd = (r.b.run(CODECS[codec]["dec"],
                                  dec_args(codec, jpg, "tmp.ppm", extra),
                                  "%s_probe_D_%s_q%s_%s" % (codec, tag, q,
                                                            sub))
                          if jpg else {})
                    if r.dry:
                        continue
                    for d, res in (("E", re_), ("D", rd)):
                        r.speed[(codec, d, tag, q, sub)] = res.get("fps") or 1.0
                        if res.get("gpu_mem_mb"):
                            per = res["gpu_mem_mb"] / float(T * B)
                            r.mem_slot[(codec, d, tag)] = max(
                                r.mem_slot.get((codec, d, tag), 0.0), per)
                        if res.get("gpu_avail_mb"):
                            r.avail_mb = min(r.avail_mb or 1e12,
                                             res["gpu_avail_mb"])
                    print("   %-2s %-3s q%-3d %s  encode %8.1f fps  "
                          "decode %8.1f fps"
                          % (codec, tag, q, sub,
                             r.speed[(codec, "E", tag, q, sub)],
                             r.speed[(codec, "D", tag, q, sub)]))


def merge_row(res, rec):
    """One result row: what the program said, with what the SCRIPT ran on top.

    The script decides what a row is (codec, variant, mode, grid point,
    path); the program's own words about it are kept, but under their own
    names. Found on the stub run of 20.09: the nvJPEG RESULT line carries a
    "path" of its own ("streams"), and letting it overwrite the script's
    ("async") would have taken the whole nvJPEG decode grid out of the tables
    without a single error - a silent loss, the worst kind.
    """
    row = dict(res)
    row.pop("raw", None)
    if "path" in row:
        row["prog_path"] = row.pop("path")
    row.update(rec)
    return row


def plan_memory(r):
    """Which grid points fit in card memory - decided once, for both codecs."""
    total = r.avail_mb or card_memory_mb(r.env)
    if not total:
        print("   card memory unknown (no program reported it, no nvidia-smi):"
              " no point is left out")
        return
    for tag, _ in IMAGES:
        fm = frame_mb(tag)
        for d in ("E", "D"):
            for th, ba in POINTS:
                worst = None
                for codec in r.codecs:
                    told = r.mem_slot.get((codec, d, tag), 0.0)
                    per = max(told, SLOT_FACTOR * fm)
                    need = per * th * ba
                    if need > MEM_SHARE * total and (worst is None
                                                     or need > worst[1]):
                        worst = (codec, need, told >= SLOT_FACTOR * fm)
                if worst:
                    # Say whose number it is: the program's own report, or
                    # the estimate of SLOT_FACTOR frames per slot.
                    r.skip[(tag, d, th, ba, "async")] = (
                        ("%s reports about %.0f MB of %.0f MB"
                         % (CODECS[worst[0]]["name"], worst[1], total))
                        if worst[2] else
                        ("estimated at about %.0f MB of %.0f MB (%.0f frames "
                         "per slot)" % (worst[1], total, SLOT_FACTOR)))
        if "nv" in r.codecs:
            for nb in NV_BATCHES:
                per = max(r.mem_slot.get(("nv", "D", tag), 0.0),
                          SLOT_FACTOR * fm)
                if per * nb > MEM_SHARE * total:
                    r.skip[(tag, "D", 1, nb, "native_batch")] = (
                        "estimated at about %.0f MB of %.0f MB"
                        % (per * nb, total))
    print("   card memory: %.0f MB free%s" % (
        total, "" if r.avail_mb else " (total from nvidia-smi)"))
    if r.skip:
        print("   left out, for both codecs where both have the point:")
        for (tag, d, th, ba, path), why in sorted(r.skip.items()):
            print("     %-5s %s %3dx%-3d %-12s %s" % (tag, d, th, ba, path, why))
    else:
        print("   every grid point fits")


def find_stock(name):
    for d in STOCK_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def one_file_dir(path, src):
    """A folder holding exactly one file - NVIDIA's programs read the whole
    folder, and a stray second frame would be measured along (a lesson of
    the JPEG2000 step 1, where the folder was not emptied between frames)."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    dst = os.path.join(path, os.path.basename(src))
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    assert len(os.listdir(path)) == 1
    return path


def copies_dir(path, src, n):
    """n hard links to one file (copies if links are not possible)."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for i in range(n):
        dst = os.path.join(path, "f%05d.jpg" % i)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    return path


def phase_stock(r):
    """Step 1: NVIDIA's own programs next to our harness.

    The same file, the same boundary: NVIDIA's samples leave the decoded
    pixels on the card and bring the compressed stream to the host side, and
    our harness is started with -window codec, which is exactly that. If the
    two agree within a few per cent, the harness measures what NVIDIA's
    product does - the best check of a harness there is (JPEG2000, 31.08: the
    two came out within 5 %).
    """
    have = dict((k, find_stock(v)) for k, v in STOCK.items())
    if not any(have.values()):
        print("   NVIDIA's programs not found in %s - step 1 skipped."
              % " or ".join(STOCK_DIRS))
        print("   python get-nvidia-jpeg-sample-01.py --build --dir %s"
          "   makes them." % STOCK_DIRS[0])
        return
    for k in ("dec", "enc", "hw"):
        print("   %-15s %s" % (STOCK[k], have[k] or "not found - left out"))
    q = r.qualities[0]
    sub = r.subs[0]
    for tag, src in IMAGES:
        jpg = r.ref.get(("nv", tag, q, sub))
        nq = r.nvq(tag, q, sub)
        if not jpg:
            continue
        spd_d = r.speed.get(("nv", "D", tag, q, sub)) or 100.0
        spd_e = r.speed.get(("nv", "E", tag, q, sub)) or 100.0
        total = TRIAL_FRAMES * 4 if r.trial else \
            max(64, int(spd_d * r.run_s / 64.0) * 64)
        # ---- decoding: nvJPEG single and -batched
        if have["dec"]:
            d_in = one_file_dir("stock_in_%s" % tag, jpg)
            modes = [("single", ["-b", 1], ["-repeat", total])]
            for nb in NV_BATCHES:
                if nb == 1:
                    continue
                if r.skip.get((tag, "D", 1, nb, "native_batch")):
                    continue
                t_nb = max(total, nb * 4)
                t_nb -= t_nb % nb
                modes.append(("batch %d" % nb,
                              ["-b", nb, "-batched", "-t", t_nb],
                              ["-repeat", t_nb, "-batched", "-b", nb]))
            for name, theirs, ours in modes:
                targs = ["-i", d_in, "-fmt", "rgb", "-w", 3] + theirs
                if "-t" not in [str(x) for x in theirs]:
                    targs += ["-t", total]
                rt = r.b.run(have["dec"], targs,
                             "step1_nvJPEG_%s_%s" % (tag, name.replace(" ", "")))
                ro = r.b.run(CODECS["nv"]["dec"],
                             ["-i", jpg, "-o", "tmp.ppm", "-window", "codec",
                              "-backend", "default", "-discard"] + ours,
                             "step1_ours_D_%s_%s" % (tag, name.replace(" ", "")))
                if r.dry:
                    continue
                m = RE_STOCK_DEC.search(rt.get("raw", ""))
                row = {"image": tag, "direction": "D", "mode": name,
                       "program": STOCK["dec"],
                       "theirs": float(m.group(1)) if m else None,
                       "ours": ro.get("fps"), "q_used": nq, "sub": sub}
                r.stock.append(row)
                r.b.record(**merge_row(ro, {"codec": "nv", "variant": "gpu",
                                            "direction": "D", "image": tag,
                                            "q": q, "q_used": nq, "sub": sub,
                                            "mode": "step1", "path": name,
                                            "threads": 1, "batch": 1,
                                            "stock_fps": row["theirs"]}))
                print("   %-3s D %-9s NVIDIA %9s   ours %9s   %s"
                      % (tag, name, fmt(row["theirs"], "%.1f"),
                         fmt(row["ours"], "%.1f"), ratio_text(row)))
            shutil.rmtree(d_in, ignore_errors=True)
        # ---- the hardware witness: NVIDIA's decoder that tries the engine
        if have["hw"]:
            d_in = one_file_dir("stock_hw_%s" % tag, jpg)
            rt = r.b.run(have["hw"], ["-i", d_in, "-b", 64, "-t", 256, "-w", 2],
                         "step1_nvjpegDecoder_%s" % tag, jpg=True)
            if not r.dry:
                raw = rt.get("raw", "")
                used = None if not raw else (RE_STOCK_NOHW.search(raw) is None)
                m = RE_STOCK_DEC.search(raw)
                r.hw.setdefault("witness", []).append({
                    "image": tag, "engine_used": used,
                    "fps": float(m.group(1)) if m else None,
                    "jpg_util_max": rt.get("jpg_util_max"),
                    "error": rt.get("error")})
                print("   %-3s NVIDIA's nvjpegDecoder: %s%s"
                      % (tag, {True: "uses the hardware decoder",
                               False: "says 'Hardware Decoder not supported'",
                               None: "no answer"}[used],
                         ("  (JPEG engine counter max %.0f %%)"
                          % rt["jpg_util_max"])
                         if rt.get("jpg_util_max") is not None else ""))
            shutil.rmtree(d_in, ignore_errors=True)
        # ---- encoding: nvJPEG_encoder on N and 2N copies
        if have["enc"]:
            base = "stock_src_%s.jpg" % tag
            # the input of NVIDIA's encoder is a JPEG; quality 100, 4:4:4 keeps
            # its pixels next to the source frame
            r.b.run(CODECS["nv"]["enc"],
                    enc_args("nv", src, base, 100, "444", [], discard=False),
                    "step1_src_%s" % tag)
            if not r.dry and not os.path.exists(base):
                continue
            n = TRIAL_FRAMES if r.trial else \
                max(20, min(200, int(spd_e / 4.0 * 2.0)))
            res = {}
            for mult in (1, 2):
                d_in = copies_dir("stock_enc_%s_%d" % (tag, mult), base,
                                  n * mult)
                d_out = "stock_encout_%s_%d" % (tag, mult)
                res[mult] = r.b.run(have["enc"],
                                    ["-i", d_in, "-o", d_out, "-q", nq,
                                     "-s", sub, "-fmt", "rgb"],
                                    "step1_nvJPEG_encoder_%s_%dn" % (tag, mult))
                shutil.rmtree(d_in, ignore_errors=True)
                shutil.rmtree(d_out, ignore_errors=True)
            ro = r.b.run(CODECS["nv"]["enc"],
                         enc_args("nv", src, "tmp.jpg", nq, sub,
                                  ["-repeat", n * 2]) + ["-window", "codec"],
                         "step1_ours_E_%s" % tag)
            rm(base)
            if r.dry:
                continue
            vals = []
            for mult in (1, 2):
                raw = res[mult].get("raw", "")
                mt, mn = RE_STOCK_ENC_T.search(raw), RE_STOCK_ENC_N.search(raw)
                vals.append((float(mt.group(1)) if mt else None,
                             int(mn.group(1)) if mn else None))
            (t1, k1), (t2, k2) = vals
            theirs = None
            why = None
            if None in (t1, t2, k1, k2):
                why = "their output was not read - see the step1 logs"
            elif k2 <= k1 or t2 <= t1:
                why = "the 2N run was not longer than the N run"
            else:
                theirs = 1000.0 * (k2 - k1) / (t2 - t1)
            row = {"image": tag, "direction": "E",
                   "mode": "single, %d vs %d files" % (k1 or 0, k2 or 0),
                   "program": STOCK["enc"], "theirs": theirs,
                   "ours": ro.get("fps"), "q_used": nq, "sub": sub,
                   "why": why,
                   "their_first_call_ms": (t1 - (t2 - t1) * k1 / (k2 - k1))
                   if theirs else None}
            r.stock.append(row)
            r.b.record(**merge_row(ro, {"codec": "nv", "variant": "gpu",
                                        "direction": "E", "image": tag, "q": q,
                                        "q_used": nq, "sub": sub,
                                        "mode": "step1", "path": "single",
                                        "threads": 1, "batch": 1,
                                        "stock_fps": theirs}))
            print("   %-3s E single    NVIDIA %9s   ours %9s   %s%s"
                  % (tag, fmt(theirs, "%.1f"), fmt(row["ours"], "%.1f"),
                     ratio_text(row), ("  - " + why) if why else ""))


def ratio_text(row):
    a, b = row.get("ours"), row.get("theirs")
    if not (a and b):
        return ""
    d = 100.0 * (a / b - 1.0)
    return "ours %+.1f %%%s" % (d, "" if abs(d) <= STEP1_AGREE
                                else "  <- MORE THAN %.0f %%" % STEP1_AGREE)


def measure(r, exe, mk, key, log, rec, jpg=False, repeat_extra=True):
    """One point: reps repeats, more while they disagree, median printed."""
    why = r.skip.get((rec.get("image"), rec.get("direction"),
                      rec.get("threads"), rec.get("batch"), rec.get("path")))
    if why:
        print("   %-2s %-3s %-3s %s %sx%s %s  LEFT OUT: %s"
              % (rec.get("codec"), rec.get("variant"), rec.get("image"),
                 rec.get("direction"), rec.get("threads"), rec.get("batch"),
                 rec.get("path"), why))
        return None, []
    got = []
    rep = 0
    may_add = bool(r.reps > 1 and repeat_extra)
    slots = (rec.get("threads") or 1) * (rec.get("batch") or 1)
    n = r.frames(key[0], key[1], slots)
    while True:
        rep += 1
        waited = 0.0 if r.dry else wait_until_card_quiet(print)
        res = r.b.run(exe, mk(n), "%s_r%d" % (log, rep), power=r.power,
                      jpg=jpg)
        if not r.dry:
            res["card_wait_s"] = round(waited, 1)
            r.b.record(**merge_row(res, rec))
            if res.get("fps"):
                got.append(res["fps"])
        if r.dry:
            return None, []
        if rep < r.reps:
            continue
        if (may_add and rep < RESPREAD_MAX_REPS
                and spread_pct(got) > RESPREAD_LIMIT):
            continue
        break
    sp = spread_pct(got)
    wide = bool(got) and sp > RESPREAD_LIMIT
    RESPREAD_LOG.append({
        "log": log, "codec": rec.get("codec"), "variant": rec.get("variant"),
        "image": rec.get("image"), "direction": rec.get("direction"),
        "q": rec.get("q"), "sub": rec.get("sub"),
        "threads": rec.get("threads"), "batch": rec.get("batch"),
        "path": rec.get("path"), "repeats": len(got),
        "median": median(got), "spread_pct": sp, "wide": wide,
        "values": list(got)})
    if wide:
        print("   ^ SPREAD %.1f %% over %d repeats - this point is NOT"
              " measured. Values: %s"
              % (sp, len(got), ", ".join("%.0f" % v for v in got)))
    return median(got), got


def phase_measure(r):
    """Single frame (latency) and the threads x batch grid, both codecs,
    both directions; nvJPEG decode additionally through its native batch on
    the CUDA path and on the hardware engine."""
    hw_dec = bool(r.hw["dec"] and r.hw["dec"].get("available"))
    hw_enc = bool(r.hw["enc"] and r.hw["enc"].get("available"))
    for tag, src in IMAGES:
        for q in r.qualities:
            for sub in r.subs:
                for codec in r.codecs:
                    qq = r.q_of(codec, tag, q, sub)
                    jpg = r.ref.get((codec, tag, q, sub))
                    enc_variants = ["gpu"] + (["hw"] if (codec == "nv"
                                                         and hw_enc) else [])
                    if codec == "fv":
                        enc_variants = ["fv"]
                    base = {"codec": codec, "image": tag, "q": q, "q_used": qq,
                            "sub": sub}
                    # ---- encoding
                    for var in enc_variants:
                        rec = dict(base, direction="E", variant=var,
                                   mode="latency", threads=1, batch=1,
                                   path="single")
                        med, got = measure(
                            r, CODECS[codec]["enc"],
                            lambda n, var=var: enc_args(
                                codec, src, "tmp.jpg", qq, sub,
                                ["-repeat", n], variant=var),
                            (((codec, "E", tag, q, sub)), 4.0),
                            "%s_%s_E_%s_q%s_%s_single" % (codec, var, tag, q,
                                                         sub),
                            rec, jpg=(var == "hw"))
                        if not r.dry:
                            print("   %-2s %-3s %-3s q%-3d %s E single   "
                                  "%8.1f fps %s" % (codec, var, tag, q, sub,
                                                    med or 0, spread(got)))
                        for th, ba in POINTS:
                            rec = dict(base, direction="E", variant=var,
                                       mode="throughput", threads=th,
                                       batch=ba, path="async")
                            med, got = measure(
                                r, CODECS[codec]["enc"],
                                lambda n, th=th, ba=ba, var=var: enc_args(
                                    codec, src, "tmp.jpg", qq, sub,
                                    ["-repeat", n, "-async", "-thread", th,
                                     "-b", ba], variant=var),
                                ((codec, "E", tag, q, sub), 1.0),
                                "%s_%s_E_%s_q%s_%s_t%d_b%d" % (
                                    codec, var, tag, q, sub, th, ba),
                                rec, jpg=(var == "hw"))
                            if not r.dry:
                                print("   %-2s %-3s %-3s q%-3d %s E %2dx%-2d    "
                                      "%8.1f fps %s" % (codec, var, tag, q,
                                                        sub, th, ba, med or 0,
                                                        spread(got)))
                    if not jpg:
                        continue
                    # ---- decoding: single frame and the grid
                    var = "fv" if codec == "fv" else "gpu"
                    rec = dict(base, direction="D", variant=var,
                               mode="latency", threads=1, batch=1,
                               path="single")
                    med, got = measure(
                        r, CODECS[codec]["dec"],
                        lambda n: dec_args(codec, jpg, "tmp.ppm",
                                           ["-repeat", n]),
                        ((codec, "D", tag, q, sub), 4.0),
                        "%s_%s_D_%s_q%s_%s_single" % (codec, var, tag, q, sub),
                        rec)
                    if not r.dry:
                        print("   %-2s %-3s %-3s q%-3d %s D single   "
                              "%8.1f fps %s" % (codec, var, tag, q, sub,
                                                med or 0, spread(got)))
                    for th, ba in POINTS:
                        rec = dict(base, direction="D", variant=var,
                                   mode="throughput", threads=th, batch=ba,
                                   path="async")
                        med, got = measure(
                            r, CODECS[codec]["dec"],
                            lambda n, th=th, ba=ba: dec_args(
                                codec, jpg, "tmp.ppm",
                                ["-repeat", n, "-async", "-thread", th,
                                 "-b", ba]),
                            ((codec, "D", tag, q, sub), 1.0),
                            "%s_%s_D_%s_q%s_%s_t%d_b%d" % (codec, var, tag, q,
                                                          sub, th, ba),
                            rec)
                        if not r.dry:
                            print("   %-2s %-3s %-3s q%-3d %s D %2dx%-2d    "
                                  "%8.1f fps %s" % (codec, var, tag, q, sub,
                                                    th, ba, med or 0,
                                                    spread(got)))
                    if codec != "nv":
                        continue
                    # ---- nvJPEG native batch: CUDA path and hardware engine
                    backends = [("gpu", "gpuhybrid")] + (
                        [("hw", "hardware")] if hw_dec else [])
                    for var, bk in backends:
                        if var == "hw":
                            rec = dict(base, direction="D", variant="hw",
                                       mode="latency", threads=1, batch=1,
                                       path="native_batch")
                            med, got = measure(
                                r, CODECS["nv"]["dec"],
                                lambda n, bk=bk: dec_args(
                                    "nv", jpg, "tmp.ppm", ["-repeat", n],
                                    backend=bk),
                                (("nv", "D", tag, q, sub), 4.0),
                                "nv_hw_D_%s_q%s_%s_single" % (tag, q, sub),
                                rec, jpg=True)
                            if not r.dry:
                                print("   nv hw  %-3s q%-3d %s D single   "
                                      "%8.1f fps %s" % (tag, q, sub, med or 0,
                                                        spread(got)))
                        for nb in NV_BATCHES:
                            rec = dict(base, direction="D", variant=var,
                                       mode="throughput", threads=1, batch=nb,
                                       path="native_batch")
                            med, got = measure(
                                r, CODECS["nv"]["dec"],
                                lambda n, nb=nb, bk=bk: dec_args(
                                    "nv", jpg, "tmp.ppm",
                                    ["-repeat", max(n, nb), "-batched",
                                     "-b", nb], backend=bk),
                                (("nv", "D", tag, q, sub), 1.0),
                                "nv_%s_D_%s_q%s_%s_batch%d" % (var, tag, q,
                                                              sub, nb),
                                rec, jpg=(var == "hw"))
                            if not r.dry:
                                last = [x for x in r.b.rows
                                        if x.get("path") == "native_batch"
                                        and x.get("batch") == nb
                                        and x.get("variant") == var][-1:]
                                where = (last[0].get("huffman")
                                         if last else "?")
                                util = (last[0].get("jpg_util_max")
                                        if last else None)
                                print("   nv %-3s %-3s q%-3d %s D batch %-4d "
                                      "%8.1f fps %s  Huffman: %s%s"
                                      % (var, tag, q, sub, nb, med or 0,
                                         spread(got), where,
                                         ("  JPEG engine max %.0f %%" % util)
                                         if util is not None else ""))


def best_point(rows, codec, variant, direction, tag, q, sub):
    """(threads, batch, path, fps) of the fastest throughput point."""
    acc = {}
    for x in rows:
        if (x.get("mode") != "throughput" or not x.get("fps")
                or x.get("codec") != codec or x.get("variant") != variant
                or x.get("direction") != direction or x.get("image") != tag
                or x.get("q") != q or x.get("sub") != sub):
            continue
        acc.setdefault((x["threads"], x["batch"], x.get("path")),
                       []).append(x["fps"])
    if not acc:
        return None
    (th, ba, path), v = max(acc.items(), key=lambda kv: median(kv[1]))
    return th, ba, path, median(v)


def single_point(rows, codec, variant, direction, tag, q, sub):
    """(1, 1, "single", fps) of the one-thread point, or None."""
    v = [x["fps"] for x in rows
         if x.get("mode") == "latency" and x.get("fps")
         and (x.get("path") or "single") == "single"
         and x.get("codec") == codec and x.get("variant") == variant
         and x.get("direction") == direction and x.get("image") == tag
         and x.get("q") == q and x.get("sub") == sub]
    return (1, 1, "single", median(v)) if v else None


def phase_energy(r):
    """Energy per frame by the card's own counter, N and 2N frames, the
    difference divided by N - at two points of every codec and path:

      best    the fastest multithread point, the whole path with transfers
              (what version 12 measured);
      single  one thread, in the window of the one-thread speed rows
              (new in version 13: the article's speed tables are this mode).
    """
    if not (r.dev and r.dev.energy_ok):
        print("   the NVML energy counter is not available; install "
              "nvidia-ml-py. Skipped.")
        return
    for tag, src in IMAGES:
        for q in r.qualities:
            for sub in r.subs:
                for codec in r.codecs:
                    variants = (["fv"] if codec == "fv" else
                                ["gpu"] + (["hw"] if (r.hw["dec"] or {}).get(
                                    "available") else []))
                    for var, d, which in [(v_, d_, w_) for v_ in variants
                                          for d_ in ("E", "D")
                                          for w_ in ("best", "single")]:
                        if True:        # keeps the indentation of version 12
                            pt = (best_point if which == "best"
                                  else single_point)(r.b.rows, codec, var, d,
                                                     tag, q, sub)
                            if not pt:
                                continue
                            th, ba, path, fps = pt
                            qq = r.q_of(codec, tag, q, sub)
                            jpg = r.ref.get((codec, tag, q, sub))
                            # Timed, not counted, in the trial as well:
                            # too few frames and the run measures its own
                            # start-up (lesson of 21.09).
                            n1 = max(MIN_FRAMES, int(fps * ENERGY_RUN_S))
                            if path == "native_batch":
                                n1 = max(n1, ba)

                            def mk(nn, d=d, th=th, ba=ba, path=path, qq=qq,
                                   var=var):
                                if path == "single":
                                    # the same command as the one-thread
                                    # speed row: no -async, codec window
                                    if d == "E":
                                        return enc_args(codec, src, "tmp.jpg",
                                                        qq, sub,
                                                        ["-repeat", nn],
                                                        variant=var)
                                    return dec_args(codec, jpg, "tmp.ppm",
                                                    ["-repeat", nn])
                                if d == "E":
                                    return enc_args(codec, src, "tmp.jpg", qq,
                                                    sub, ["-repeat", nn,
                                                          "-async", "-thread",
                                                          th, "-b", ba],
                                                    variant=var)
                                if path == "native_batch":
                                    return dec_args(
                                        codec, jpg, "tmp.ppm",
                                        ["-repeat", nn, "-batched", "-b", ba],
                                        backend="hardware" if var == "hw"
                                        else "gpuhybrid")
                                return dec_args(codec, jpg, "tmp.ppm",
                                                ["-repeat", nn, "-async",
                                                 "-thread", th, "-b", ba])
                            exe = CODECS[codec]["enc" if d == "E" else "dec"]
                            pair = []
                            for mult in (1, 2):
                                res = r.b.run(exe, mk(n1 * mult),
                                              "%s_%s_%s_%s_q%s_%s_energy_%s_%dn"
                                              % (codec, var, d, tag, q, sub,
                                                 which, mult),
                                              power=r.power, energy=True,
                                              jpg=(var == "hw"))
                                pair.append(res)
                            r1, r2 = pair
                            f1, f2 = r1.get("frames"), r2.get("frames")
                            e1 = r1.get("energy_j_counter")
                            e2 = r2.get("energy_j_counter")
                            diff = None
                            why = None
                            if e1 is None or e2 is None:
                                why = "no counter reading"
                            elif not (f1 and f2 and f2 > f1):
                                why = "frame counts %s and %s" % (f1, f2)
                            elif e2 <= e1:
                                why = ("the 2N run spent no more than the N run "
                                       "(%.2f J vs %.2f J): too short for the "
                                       "counter" % (e2, e1))
                            elif (r1.get("wall_s") or 0) < ENERGY_MIN_S:
                                why = ("the N run lasted %.2f s, under %.1f s: "
                                       "start-up, not frames"
                                       % (r1.get("wall_s") or 0, ENERGY_MIN_S))
                            elif ((r2.get("wall_s") or 0)
                                  < ENERGY_GROWTH * (r1.get("wall_s") or 0)):
                                why = ("twice the frames took %.2f s against "
                                       "%.2f s: the run is not made of frames"
                                       % (r2.get("wall_s") or 0,
                                          r1.get("wall_s") or 0))
                            else:
                                diff = (e2 - e1) / (f2 - f1)
                            r2["j_per_frame_diff"] = diff
                            r2["energy_note"] = why
                            for mult, res in ((1, r1), (2, r2)):
                                r.b.record(**merge_row(res, {
                                    "codec": codec, "variant": var,
                                    "direction": d, "image": tag, "q": q,
                                    "q_used": qq, "sub": sub,
                                    "mode": "energy", "threads": th,
                                    "batch": ba, "path": path,
                                    "point": which,
                                    "note": "%s %dn" % (which, mult)}))
                            r.energy.append({
                                "codec": codec, "variant": var,
                                "direction": d, "image": tag, "q": q,
                                "sub": sub, "threads": th, "batch": ba,
                                "path": path, "point": which,
                                "j_per_frame_diff": diff,
                                "why_none": why,
                                "j_per_frame_sampled": r2.get("j_per_frame"),
                                "cores": r2.get("cores"),
                                "fps": r2.get("fps")})
                            print("   %-2s %-3s %s %-3s q%-3d %s %-6s %2sx%-3s "
                                  "%s J/frame%s" % (codec, var, d, tag, q, sub,
                                                    which, th, ba,
                                                    fmt(diff, "%.4f"),
                                                    ("  - " + why) if why
                                                    else ""))


def phase_roundtrip(r):
    """Does every decoder give the picture, and do the decoders agree?

    Each reference is decoded by every decoder in the run; the results are
    compared with the source (PSNR, watermark rectangle left out) and with
    each other. Two correct decoders of one file differ by rounding of the
    inverse DCT at most - more than a couple of levels is a finding.
    """
    decs = [("fv", "fv", "gpuhybrid")] if "fv" in r.codecs else []
    if "nv" in r.codecs:
        decs.append(("nv", "gpu", "gpuhybrid"))
        if (r.hw["dec"] or {}).get("available"):
            decs.append(("nv", "hw", "hardware"))
    for key in sorted(r.ref):
        codec, tag, q, sub = key
        if q not in r.qualities:
            continue
        src = dict(IMAGES)[tag]
        outs = {}
        for dcodec, var, bk in decs:
            out = "rt_%s_%s_%s_%s_q%s_%s.ppm" % (codec, dcodec, var, tag, q,
                                                sub)
            args = ["-i", r.ref[key], "-o", out]
            if dcodec == "nv":
                args += ["-backend", bk]
            r.b.run(CODECS[dcodec]["dec"], args,
                    "rt_%s_by_%s_%s_%s_q%s_%s" % (codec, dcodec, var, tag, q,
                                                  sub))
            if not r.dry and os.path.exists(out):
                outs[(dcodec, var)] = out
        if r.dry:
            continue
        for (dcodec, var), out in sorted(outs.items()):
            p = psnr(src, out, exclude=r.watermark.get(tag))
            r.checks.append({"stream": codec, "decoder": "%s %s" % (dcodec,
                                                                    var),
                             "image": tag, "q": q, "sub": sub,
                             "psnr": p.get("psnr"), "exact": p.get("exact")})
        keys = sorted(outs)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = psnr(outs[keys[i]], outs[keys[j]])
                r.cross.append({"stream": codec, "image": tag, "q": q,
                                "sub": sub,
                                "a": "%s %s" % keys[i], "b": "%s %s" % keys[j],
                                "max_diff": a.get("max_diff"),
                                "psnr": a.get("psnr")})
        for out in outs.values():
            rm(out)
        print("   %-2s stream %-3s q%-3d %s  decoded by %s"
              % (codec, tag, q, sub, ", ".join("%s %s" % k for k in keys)))


def phase_stages(r):
    """Where the time of one frame goes: the three decoupled phases of the
    nvJPEG decoder; the Fastvideo -info lines, if it prints stages. Shares,
    not a sum: -info synchronises between the stages."""
    q = r.qualities[0]
    sub = r.subs[0]
    for codec in r.codecs:
        for tag, src in IMAGES:
            jpg = r.ref.get((codec, tag, q, sub))
            qq = r.q_of(codec, tag, q, sub)
            jobs = [("E", CODECS[codec]["enc"],
                     enc_args(codec, src, "tmp.jpg", qq, sub,
                              ["-repeat", 10, "-info"]))]
            if jpg:
                jobs.append(("D", CODECS[codec]["dec"],
                             dec_args(codec, jpg, "tmp.ppm",
                                      ["-repeat", 10, "-info"])))
            for d, exe, a in jobs:
                res = r.b.run(exe, a, "%s_stages_%s_%s" % (codec, d, tag))
                if r.dry:
                    continue
                st = [(name, float(ms)) for ms, _, name
                      in RE_STAGE.findall(res.get("raw", ""))]
                if st:
                    tot = sum(v for _, v in st) or 1.0
                    r.stages[(codec, d, tag)] = [(n, v, 100.0 * v / tot)
                                                 for n, v in st]
                    print("   %-2s %-3s %s  %s" % (
                        codec, tag, d, ", ".join("%s %.0f %%" % (n, s)
                                                 for n, _, s in
                                                 r.stages[(codec, d, tag)])))
                else:
                    print("   %-2s %-3s %s  no stage lines in the output"
                          % (codec, tag, d))


def boundary_audit(rows):
    """Were the two codecs measured between the same two points?"""
    seen = {}
    for x in rows:
        b = x.get("boundary")
        if not b or not x.get("codec") or x.get("mode") in ("energy", "step1"):
            continue
        k = (x.get("mode"), x.get("direction"), x.get("image"), x.get("q"),
             x.get("sub"))
        seen.setdefault(k, {}).setdefault(x["codec"], set()).add(b)
    bad, checked = [], 0
    for k in sorted(seen, key=str):
        per = seen[k]
        if len(per) < 2:
            continue
        checked += 1
        vals = set()
        for c in per:
            vals |= per[c]
        if len(vals) > 1:
            bad.append((k, dict((c, "/".join(sorted(v)))
                                for c, v in per.items())))
    print("\n[boundary] were both codecs measured between the same points?")
    if not checked:
        print("   nothing to compare: only one codec in this run")
        return True
    if not bad:
        print("   %d workloads measured for both codecs, boundary identical "
              "in every one" % checked)
        return True
    print("   MISMATCH in %d of %d workloads - NOT comparable:"
          % (len(bad), checked))
    for k, per in bad:
        print("     %-10s %s %-3s q%-3s %s  %s" % (k + (", ".join(
            "%s=%s" % (c, per[c]) for c in sorted(per)),)))
    print("   all = transfers counted on both sides; default = the program")
    print("   did not say (check the -info output of JpegSample)")
    return False


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

CSV_COLUMNS = ["bench_version", "nv_harness", "codec", "variant", "direction",
               "image", "q", "q_used", "sub", "mode", "path", "threads",
               "batch", "note", "prog_path", "frames", "boundary", "window",
               "huffman",
               "backend", "total_ms", "ms_per_frame", "fps", "mp_s", "mb_s",
               "out_kb", "cr", "gpu_mem_mb", "gpu_avail_mb", "power_w",
               "j_per_frame", "energy_j_counter", "j_per_frame_counter",
               "j_per_frame_diff", "energy_note", "stock_fps", "jpg_util_max", "jpg_util_mean", "cpu_s",
               "cores", "gpu", "sdk", "pcie_mb_s", "rc", "parsed_from",
               "error", "wall_s", "at", "cmd"]

VAR_NAME = {("fv", "fv"): "Fastvideo", ("nv", "gpu"): "nvJPEG CUDA",
            ("nv", "hw"): "nvJPEG hardware"}


def groups(rows):
    acc = {}
    for x in rows:
        if not x.get("fps") or x.get("mode") in ("energy", "step1"):
            continue
        key = (x["codec"], x.get("variant"), x["direction"], x["image"],
               x.get("q"), x.get("sub"), x["mode"], x.get("threads"),
               x.get("batch"), x.get("path"))
        acc.setdefault(key, []).append(x["fps"])
    out = {}
    for key, v in acc.items():
        m = median(v)
        out[key] = {"fps": m, "n": len(v),
                    "spread": spread_pct(v) if len(v) > 1 else 0.0}
    return out


def write_results(r, outdir, env):
    b = r.b
    path = os.path.join(outdir, "results.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, delimiter=";",
                           extrasaction="ignore")
        w.writeheader()
        for x in b.rows:
            w.writerow(x)
    g = groups(b.rows)
    L = []
    add = L.append

    def best(codec, var, d, tag, q, sub):
        cand = [(k, v) for k, v in g.items()
                if k[:6] == (codec, var, d, tag, q, sub)
                and k[6] == "throughput"]
        if not cand:
            return None, None
        k, v = max(cand, key=lambda kv: kv[1]["fps"])
        return v, k

    def lat(codec, var, d, tag, q, sub):
        cand = [v for k, v in g.items()
                if k[:6] == (codec, var, d, tag, q, sub)
                and k[6] == "latency"]
        return cand[0] if cand else None

    add("Baseline JPEG on the GPU: Fastvideo against nvJPEG")
    add("")
    add("Test system and measurement conditions")
    add("")
    sdk = {}
    for x in b.rows:
        if x.get("sdk") and x["codec"] not in sdk:
            sdk[x["codec"]] = x["sdk"]
    rows_env = [("GPU", env.get("gpu", "-")), ("Driver", env.get("driver", "-")),
                ("CUDA (driver)", env.get("cuda_driver", "-")),
                ("GPU memory", env.get("gpu_mem", "-")),
                ("Power limit", env.get("power_limit", "-")),
                ("CPU", env.get("cpu", "-")),
                ("Logical cores", str(env.get("cores", "-"))),
                ("Operating system", env.get("os", "-")),
                ("Measurement date", env.get("date", "-")),
                ("Repeats per point", str(r.reps)),
                ("Benchmark script", SCRIPT_NAME + ", version " + BENCH_VERSION),
                ("nvJPEG harness", ", ".join("%s %s" % (k, v)
                                             for k, v in NV_HARNESS.items())
                 or "-")]
    for c in r.codecs:
        rows_env.append((CODECS[c]["name"], sdk.get(c, "-")))
    w1 = max(len(a) for a, _ in rows_env)
    for a, v in rows_env:
        add("  %-*s  %s" % (w1, a, v))
    add("")
    add("  Frames: " + ", ".join("%s %dx%d" % (t, GEOM[t]["w"], GEOM[t]["h"])
                                 for t, _ in IMAGES if t in GEOM)
        + ", 8-bit colour.")
    add("  Measured window: host memory to host memory, both directions, disk")
    add("  excluded. Grid: " + ", ".join("%dx%d" % p for p in POINTS)
        + " (threads x frames in a thread);")
    add("  nvJPEG native batch: " + ", ".join(str(x) for x in NV_BATCHES)
        + ".")
    add("")

    # ------------------------------------------------------------- hardware
    add("-" * 78)
    add("The hardware JPEG engine on this card")
    add("")
    dec, enc = r.hw.get("dec") or {}, r.hw.get("enc") or {}
    add("  decoder engine: %s" % (("AVAILABLE, %d engine(s) x %d core(s)"
                                   % (dec.get("engines", 0),
                                      dec.get("cores", 0)))
                                  if dec.get("available") else
                                  "not available (%s)" % dec.get("error", "-")))
    add("  encoder engine: %s" % (("AVAILABLE, %d engine(s)"
                                   % enc.get("engines", 0))
                                  if enc.get("available") else "not available"))
    for s in r.hw.get("streams", []):
        add("  %-22s engine %-9s picture: %s"
            % (s["stream"], {True: "accepts", False: "REFUSES",
                             None: "?"}[s["supported"]], s["picture"]))
    hwrows = [x for x in b.rows if x.get("variant") == "hw"
              and x.get("jpg_util_max") is not None]
    gpurows = [x for x in b.rows if x.get("variant") == "gpu"
               and x.get("jpg_util_max") is not None]
    if hwrows:
        add("  JPEG engine counter during hardware runs: max %.0f %%, "
            "lowest run max %.0f %% (%d runs)"
            % (max(x["jpg_util_max"] for x in hwrows),
               min(x["jpg_util_max"] for x in hwrows), len(hwrows)))
        if gpurows:
            add("  and during CUDA runs: max %.0f %% - the contrast is the "
                "evidence" % max(x["jpg_util_max"] for x in gpurows))
    add("")

    # --------------------------------------------------------------- step 1
    if r.stock or r.hw.get("witness"):
        add("-" * 78)
        add("Step 1: NVIDIA's own programs against our harness")
        add("")
        add("  The same file, the same boundary: the decoded pixels stay on the")
        add("  card, the compressed stream is on the host side (our harness")
        add("  with -window codec). Frames per second.")
        add("")
        add("  frame dir  mode                   NVIDIA program      ours   "
            "difference")
        add("  " + "-" * 72)
        for x in r.stock:
            add("  %-5s %-4s %-22s %14s %9s   %s"
                % (x["image"], x["direction"], x["mode"],
                   fmt(x["theirs"], "%.1f"), fmt(x["ours"], "%.1f"),
                   ratio_text(x) or x.get("why") or "-"))
        agree = [x for x in r.stock if x.get("ours") and x.get("theirs")]
        if agree:
            worst = max(abs(100.0 * (x["ours"] / x["theirs"] - 1.0))
                        for x in agree)
            add("")
            add("  %s: the largest difference is %.1f %% (limit %.0f %%)."
                % ("The harness agrees with NVIDIA's programs"
                   if worst <= STEP1_AGREE else "NOT CONFIRMED", worst,
                   STEP1_AGREE))
        enc = [x for x in r.stock if x["direction"] == "E" and x.get("theirs")]
        if enc:
            add("  NVIDIA's encoder is timed once per file and has no -repeat;")
            add("  its figure is the difference of a run on 2N files and a run")
            add("  on N, so the start-up of its first call falls out (it was")
            add("  about %s ms here)." % ", ".join(
                fmt(x.get("their_first_call_ms"), "%.0f") for x in enc))
        for w in r.hw.get("witness", []):
            add("  %s: NVIDIA's nvjpegDecoder %s"
                % (w["image"], {True: "used the hardware decoder",
                                False: "said 'Hardware Decoder not supported'",
                                None: "gave no answer"}[w["engine_used"]]))
        add("")

    # ---------------------------------------------------------- references
    add("-" * 78)
    add("Reference streams")
    add("")
    add("  codec frame  q  q used sub   size, KB   ratio  bit/px restart  frame")
    add("  " + "-" * 74)
    for key in sorted(r.ref_info):
        codec, tag, q, sub = key
        inf = r.ref_info[key]
        g0 = GEOM.get(tag)
        raw = g0["w"] * g0["h"] * 3 if g0 else None
        add("  %-5s %-5s %-3s %-6s %-4s %9.1f %7s %7s %7s  %s"
            % (codec, tag, q, inf.get("q_used", q), sub, inf["size"] / 1024.0,
               ("%.1f:1" % (raw / float(inf["size"]))) if raw else "-",
               ("%.2f" % (inf["size"] * 8.0 / (g0["w"] * g0["h"])))
               if g0 else "-", inf["dri"] or "none", inf.get("frame")))
    add("")
    add("  q is the Fastvideo quality the row belongs to; q used is what the")
    add("  encoder was actually given - for nvJPEG at the measured qualities")
    add("  that is the value giving the same file size (see the scales).")
    add("  restart = the restart interval in MCUs (DRI marker), 'none' when")
    add("  the stream has no restart markers. It changes how much of the")
    add("  decoding can run in parallel, so it is shown for every stream.")
    add("")

    # ------------------------------------------------------------- scales
    if r.scales:
        add("-" * 78)
        add("Quality scales: does the same -q mean the same thing?")
        add("")
        add("  frame sub  q    fv, B     nv same q, B  tables at the same q")
        add("  " + "-" * 70)
        for s in r.scales:
            add("  %-5s %-4s %-3s %9s %13s   %s"
                % (s["image"], s["sub"], s["q"], s["fv_bytes"],
                   s["nv_bytes_same_q"], s["tables_same_q"]))
        add("")
        add("  frame sub  q   nv q for fv size   miss     nv size  tables at "
            "matched q")
        add("  " + "-" * 74)
        for s in r.scales:
            add("  %-5s %-4s %-3s %8s / %-6s %7s %11s   %s"
                % (s["image"], s["sub"], s["q"], fmt(s.get("nv_q_wide")),
                   fmt(s.get("nv_q_narrow")),
                   fmt(s.get("nv_miss_wide"), "%.2f %%"),
                   s.get("nv_bytes_matched", "-"),
                   s.get("tables_matched_q", "-")))
        add("")
        add("  frame sub  q   PSNR fv   PSNR nv same q   PSNR nv matched")
        add("  " + "-" * 60)
        for s in r.scales:
            add("  %-5s %-4s %-3s %8s %16s %17s"
                % (s["image"], s["sub"], s["q"],
                   fmt(s.get("psnr_fv"), "%.2f"),
                   fmt(s.get("psnr_nv_same_q"), "%.2f"),
                   fmt(s.get("psnr_nv_matched"), "%.2f")))
        add("")
        add("  How to read it. Tables identical at the same q: the two scales")
        add("  turn q into the same quantisation tables; a size or PSNR")
        add("  difference then comes from how the tables are applied, which")
        add("  the file does not show. Tables different: the scales do not")
        add("  correspond, and 'nv q for fv size' is the pair to compare at.")
        add("  The search runs from two starting intervals; if both land on")
        add("  the same q, that q is a property of the scales, not of the")
        add("  search. PSNR: every file decoded by the same decoder (nvJPEG")
        add("  CUDA) so it compares the encoders; a watermark rectangle, if")
        add("  found, is left out on both sides.")
        add("")

    # ----------------------------------------------------- the main tables
    for d, title in (("E", "Encoding"), ("D", "Decoding")):
        add("-" * 78)
        add("%s, frames per second (median of %d repeats), equal file size"
            % (title, r.reps))
        add("")
        for tag, _ in IMAGES:
            for q in r.qualities:
                for sub in r.subs:
                    add("  %s, q %s, %s" % (tag.upper(), q, sub))
                    head = "    %-18s %7s" % ("", "1 thr")
                    for th, ba in POINTS:
                        head += " %6s" % ("%dx%d" % (th, ba))
                    add(head)
                    for codec in r.codecs:
                        for var in (["fv"] if codec == "fv" else
                                    ["gpu", "hw"]):
                            if not any(k[:6] == (codec, var, d, tag, q, sub)
                                       for k in g):
                                continue
                            l_ = lat(codec, var, d, tag, q, sub)
                            line = "    %-18s %7s" % (VAR_NAME[(codec, var)],
                                                      fmt(l_ and l_["fps"]))
                            for th, ba in POINTS:
                                v = g.get((codec, var, d, tag, q, sub,
                                           "throughput", th, ba, "async"))
                                line += " %6s" % fmt(v and v["fps"])
                            add(line)
                    if d == "D" and "nv" in r.codecs:
                        add("    nvJPEG native batch:  " + "  ".join(
                            "%d" % nb for nb in NV_BATCHES))
                        for var in ("gpu", "hw"):
                            vals = [g.get(("nv", var, "D", tag, q, sub,
                                           "throughput", 1, nb,
                                           "native_batch"))
                                    for nb in NV_BATCHES]
                            if any(vals):
                                add("    %-18s %s" % (VAR_NAME[("nv", var)],
                                                      " ".join("%6s" % fmt(
                                                          v and v["fps"])
                                                          for v in vals)))
                    add("")
        left = sorted(k for k in r.skip if k[1] == d)
        if left:
            add("  A dash in a grid cell of a large frame may be a point left")
            add("  out because it does not fit in card memory - left out for")
            add("  both codecs, since a cell filled on one side and empty on")
            add("  the other would read as 'the other was slower here':")
            for k in left:
                add("      %-5s %sx%s %-12s %s" % (k[0], k[2], k[3], k[4],
                                                 r.skip[k]))
            add("")
        add("  The '1 thr' column is ONE THREAD IN THE CODEC-ONLY WINDOW:")
        add("  the Fastvideo sample reports no other window in that mode, so")
        add("  nvJPEG is given the same one. The grid columns beside it are")
        add("  host memory to host memory. Read the table DOWN a column, one")
        add("  codec against the other; reading ACROSS compares two windows.")
        add("")
        if d == "D":
            add("  The CUDA batch below 51 decodes the Huffman stage on the")
            add("  CPU (nvjpeg.h); the huffman column of results.csv says")
            add("  where it ran in every row.")
            add("")

    # ------------------------------------------------------------ summary
    add("-" * 78)
    # What the programs themselves measured on this machine, so that a
    # throughput can be put beside the bus it has to cross.
    pcie_mb_s = max([float(x["pcie_mb_s"]) for x in r.b.rows
                     if x.get("pcie_mb_s")] or [0.0])
    add("Summary: one thread and the best point, frames per second")
    add("")
    add("  dir frame q   sub  %-17s %8s %8s %9s %7s %6s %s"
        % ("codec", "1 thr", "best", "MPix/s", "GB/s", "of bus", "best point"))
    add("  " + "-" * 88)
    for d in ("E", "D"):
        for tag, _ in IMAGES:
            for q in r.qualities:
                for sub in r.subs:
                    for codec in r.codecs:
                        for var in (["fv"] if codec == "fv"
                                    else ["gpu", "hw"]):
                            bv, bk = best(codec, var, d, tag, q, sub)
                            l_ = lat(codec, var, d, tag, q, sub)
                            if not (bv or l_):
                                continue
                            mp = (bv["fps"] * GEOM[tag]["w"] * GEOM[tag]["h"]
                                  / 1e6) if (bv and tag in GEOM) else None
                            # The pixel side of every frame crosses the bus
                            # once in the host-to-host window, so this is
                            # what the best point asks of PCIe.
                            gbs = (mp * 3.0 / 1000.0) if mp else None
                            share = (100.0 * gbs * 1000.0 / pcie_mb_s
                                     if (gbs and pcie_mb_s) else None)
                            add("  %-3s %-5s %-3s %-4s %-17s %8s %8s %9s "
                                "%7s %5s%% %s"
                                % (d, tag, q, sub, VAR_NAME[(codec, var)],
                                   fmt(l_ and l_["fps"]),
                                   fmt(bv and bv["fps"]), fmt(mp),
                                   fmt(gbs, "%.1f"), fmt(share, "%.0f"),
                                   ("%sx%s %s" % (bk[7], bk[8], bk[9]))
                                   if bk else "-"))
    if pcie_mb_s:
        add("")
        add("  GB/s is the pixel side of the best point: in the host-to-host")
        add("  window every frame crosses PCIe once, uncompressed. 'of bus'")
        add("  is that against the %.1f GB/s the programs themselves measured"
            % (pcie_mb_s / 1000.0))
        add("  on this machine. A figure near the bus is a statement about")
        add("  the bus, not about the codec.")
    add("")

    # ------------------------------------------------------- repeatability
    loud = sorted([(v["spread"], k) for k, v in g.items()
                   if v["n"] > 1 and v["spread"] > RESPREAD_LIMIT],
                  reverse=True)
    add("-" * 78)
    add("Repeatability")
    add("")
    if loud:
        add("  Points whose repeats still disagree by more than %.0f %%:"
            % RESPREAD_LIMIT)
        for sp, k in loud:
            add("    %-2s %-3s %s %-3s q%-3s %-4s %sx%s %-12s %5.1f %% (%d runs)"
                % (k[0], k[1], k[2], k[3], k[4], k[5], k[7], k[8], k[9], sp,
                   g[k]["n"]))
    else:
        add("  Every point with repeats agrees to within %.0f %%."
            % RESPREAD_LIMIT)
    add("")

    # ------------------------------------------------------- round trip
    if r.checks:
        add("-" * 78)
        add("Round trip: PSNR against the source, and decoder agreement")
        add("")
        add("  stream frame q   sub  decoder           PSNR, dB")
        add("  " + "-" * 56)
        for c in r.checks:
            add("  %-6s %-5s %-3s %-4s %-17s %8s%s"
                % (c["stream"], c["image"], c["q"], c["sub"], c["decoder"],
                   fmt(c["psnr"], "%.2f"),
                   "" if c.get("exact") else " (sampled)"))
        add("")
        for c in r.cross:
            add("  %-2s %-3s q%-3s %-4s  %-14s vs %-14s  max diff %s%s"
                % (c["stream"], c["image"], c["q"], c["sub"], c["a"], c["b"],
                   fmt(c["max_diff"]),
                   (", PSNR %.1f dB" % c["psnr"]) if c.get("psnr") else
                   ", identical" if c.get("max_diff") == 0 else ""))
        add("")
        add("  Two correct decoders of one file differ by the rounding of")
        add("  the inverse DCT: a level or two. More is a finding.")
        add("")

    # ------------------------------------------------------------- energy
    if r.energy:
        add("-" * 78)
        add("GPU energy per frame (card counter, N and 2N): best = the fastest"
            " multithread point, whole path; single = one thread, codec window")
        add("")
        add("  dir frame q   sub  point  %-17s %10s %10s %6s"
            % ("codec", "J/frame", "sampled", "cores"))
        add("  " + "-" * 71)
        for e in r.energy:
            add("  %-3s %-5s %-3s %-4s %-6s %-17s %10s %10s %6s"
                % (e["direction"], e["image"], e["q"], e["sub"],
                   e.get("point", "best"),
                   VAR_NAME[(e["codec"], e["variant"])],
                   fmt(e["j_per_frame_diff"], "%.4f"),
                   fmt(e["j_per_frame_sampled"], "%.4f"),
                   fmt(e["cores"], "%.1f")))
        gaps = [e for e in r.energy if e.get("why_none")]
        if gaps:
            add("")
            add("  A dash is a value that could not be measured, never a zero:")
            for e in gaps:
                add("    %-3s %-5s q%-3s %-4s %-17s %s"
                    % (e["direction"], e["image"], e["q"], e["sub"],
                       VAR_NAME[(e["codec"], e["variant"])], e["why_none"]))
        add("")

    # ------------------------------------------------------------- stages
    if r.stages:
        add("-" * 78)
        add("Where the time of one frame goes (-info; shares, not a sum)")
        add("")
        for key in sorted(r.stages):
            c, d, tag = key
            add("  %s, %s, %s" % (CODECS[c]["name"],
                                  "encoding" if d == "E" else "decoding", tag))
            for name, ms, share in r.stages[key]:
                add("      %-44s %8.3f ms %6.1f %%" % (name[:44], ms, share))
        add("")

    # ------------------------------------------------------- unparsed runs
    loose = [x for x in b.rows if (x.get("parsed_from") or "").startswith(
        "FPS only") or (x.get("mode") in ("latency", "throughput")
                        and not x.get("fps"))]
    if loose:
        add("-" * 78)
        add("NOT PARSED OR PARSED ONLY PARTLY - check these logs")
        for x in loose[:40]:
            add("  %-2s %-3s %s %-3s q%-3s %s %sx%s  %s"
                % (x.get("codec"), x.get("variant"), x.get("direction"),
                   x.get("image"), x.get("q"), x.get("sub"), x.get("threads"),
                   x.get("batch"), x.get("error") or x.get("parsed_from")
                   or "nothing read"))
        add("")

    machine = {
        "bench_version": BENCH_VERSION, "bench_script": SCRIPT_NAME,
        "license": "CC BY 4.0 - attribution: Fastvideo JPEG benchmark. Keep "
                   "the measurement conditions next to the numbers.",
        "environment": env, "images": GEOM,
        "settings": {"grid": ["%dx%d" % p for p in POINTS],
                     "nv_batches": NV_BATCHES, "qualities": r.qualities,
                     "subsampling": r.subs, "repeats": r.reps,
                     "window": "host memory to host memory, both directions",
                     "fv_extra": {"discard": FV_DISCARD,
                                  "threadR": FV_THREAD_R,
                                  "threadW": FV_THREAD_W},
                     "nv_harness": NV_HARNESS,
                     "nv_quality_for_equal_size":
                         dict(("%s_q%s_%s" % k, v)
                              for k, v in r.nv_q.items())},
        "hardware": r.hw, "watermark": r.watermark,
        "reference_streams": [dict({"codec": k[0], "image": k[1], "q": k[2],
                                    "sub": k[3]},
                                   **{"bytes": v["size"], "dri": v["dri"],
                                      "frame": v.get("frame"),
                                      "subsampling": v.get("subsampling")})
                              for k, v in sorted(r.ref_info.items())],
        "quality_scales": r.scales,
        "measurements": [dict(zip(("codec", "variant", "direction", "image",
                                   "q", "sub", "mode", "threads", "batch",
                                   "path"), k),
                              fps=v["fps"], repeats=v["n"],
                              spread_percent=v["spread"])
                         for k, v in sorted(g.items(), key=lambda kv:
                                            str(kv[0]))],
        "step1_nvidia_programs": r.stock,
        "round_trip": r.checks, "decoder_agreement": r.cross,
        "energy": r.energy,
        "stage_breakdown": dict(("%s_%s_%s" % k,
                                 [{"stage": n, "ms": ms, "share": sh}
                                  for n, ms, sh in v])
                                for k, v in r.stages.items()),
    }
    with open(os.path.join(outdir, "results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(machine, fh, ensure_ascii=False, indent=1, default=str)
    text = "\n".join(L) + "\n"
    with open(os.path.join(outdir, "summary.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print("")
    print(text)
    # ---- spread of the repeats: one line per point, and a loud list of the
    # points whose repeats never agreed. A median of numbers that disagree is
    # not a measurement, and nothing that ends up here may go into an article
    # until it is understood (23.09.2026).
    sp_path = os.path.join(outdir, "spread.csv")
    try:
        with open(sp_path, "w", encoding="utf-8") as fh:
            fh.write("point,codec,variant,image,direction,q,sub,threads,"
                     "batch,path,repeats,median_fps,spread_pct,verdict,"
                     "values\n")
            for e in RESPREAD_LOG:
                fh.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,%s,%.2f,%s,%s\n"
                         % (e["log"], e["codec"] or "", e["variant"] or "",
                            e["image"] or "", e["direction"] or "",
                            e["q"] if e["q"] is not None else "",
                            e["sub"] or "",
                            e["threads"] if e["threads"] is not None else "",
                            e["batch"] if e["batch"] is not None else "",
                            e["path"] or "", e["repeats"],
                            ("%.2f" % e["median"]) if e["median"] else "",
                            e["spread_pct"],
                            "wide" if e["wide"] else "ok",
                            " ".join("%.2f" % v for v in e["values"])))
    except OSError as exc:
        print("spread.csv was NOT written: %s" % exc)
        sp_path = None

    wide = [e for e in RESPREAD_LOG if e["wide"]]
    print("")
    if not RESPREAD_LOG:
        print("Spread of the repeats: nothing measured in this run.")
    elif not wide:
        print("Spread of the repeats: every one of %d points agreed to"
              " within %.0f %%." % (len(RESPREAD_LOG), RESPREAD_LIMIT))
    else:
        print("SPREAD OF THE REPEATS: %d point(s) of %d never agreed, even"
              " after %d repeats:" % (len(wide), len(RESPREAD_LOG),
                                      RESPREAD_MAX_REPS))
        for e in wide:
            print("   %-28s %5.1f %% over %d repeats: %s"
                  % (e["log"], e["spread_pct"], e["repeats"],
                     ", ".join("%.0f" % v for v in e["values"])))
        print("   Their medians are in the result files like any other, and"
              " they must NOT be used: a median of numbers that disagree by"
              " tens of per cent is not a measurement.")

    for f in ("results.jsonl", "results.csv", "summary.txt", "results.json",
              "spread.csv"):
        print("written: %s" % os.path.join(outdir, f))


# ---------------------------------------------------------------------------
# --probe: every program once, what was read
# ---------------------------------------------------------------------------

def probe(args):
    """Runs each program once per direction and prints what was read.

    The Fastvideo sample's output has not been seen yet. This shows in a
    minute whether it is read correctly - the summary line, the window, the
    size - instead of after an hour of measuring.
    """
    tag, src = IMAGES[0]
    b = Bench("probe_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print("=" * 70)
    print(" %s - probe: every program once, what was read from it" %
          SCRIPT_NAME)
    print("=" * 70)
    jobs = []
    for codec in ("fv", "nv"):
        jpg = "probe_%s.jpg" % codec
        jobs.append((codec, "E single", CODECS[codec]["enc"],
                     enc_args(codec, src, jpg, QUALITY_MAIN, SUB_MAIN, ["-info"],
                              discard=False)))
        jobs.append((codec, "E repeat", CODECS[codec]["enc"],
                     enc_args(codec, src, "tmp.jpg", QUALITY_MAIN, SUB_MAIN,
                              ["-repeat", 50])))
        jobs.append((codec, "E async", CODECS[codec]["enc"],
                     enc_args(codec, src, "tmp.jpg", QUALITY_MAIN, SUB_MAIN,
                              ["-repeat", 50, "-async", "-thread", 2,
                               "-b", 2])))
        jobs.append((codec, "D repeat", CODECS[codec]["dec"],
                     dec_args(codec, jpg, "tmp.ppm", ["-repeat", 50])))
        jobs.append((codec, "D async", CODECS[codec]["dec"],
                     dec_args(codec, jpg, "tmp.ppm",
                              ["-repeat", 50, "-async", "-thread", 2,
                               "-b", 2])))
    jobs.append(("nv", "D batch 64", CODECS["nv"]["dec"],
                 dec_args("nv", "probe_nv.jpg", "tmp.ppm",
                          ["-repeat", 128, "-batched", "-b", 64])))
    for codec, what, exe, a in jobs:
        res = b.run(exe, a, "probe_%s_%s" % (codec, what.replace(" ", "_")))
        keys = ("fps", "frames", "total_ms", "boundary", "window", "path",
                "huffman", "out_kb", "sdk", "parsed_from", "error", "rc")
        print("\n %s %s\n   $ %s" % (codec, what, res.get("cmd", "")))
        def show(v):
            return ("%.1f" % v) if isinstance(v, float) else str(v)
        print("   read: " + ", ".join("%s=%s" % (k, show(res[k])) for k in keys
                                      if res.get(k) not in (None, "")))
        head = [l for l in res.get("raw", "").splitlines() if l.strip()][-6:]
        for l in head:
            print("   | " + l[:100])
    for codec in ("fv", "nv"):
        jpg = "probe_%s.jpg" % codec
        if os.path.exists(jpg):
            inf = jpeg_info(jpg)
            print("\n %s file: %d bytes, %s, %s, restart interval %s, "
                  "luma DC %s" % (codec, inf["size"], inf.get("frame"),
                                  inf.get("subsampling"), inf["dri"] or "none",
                                  (tables_of(inf)[0] or ["?"])[0]))
    print("\n Every run's full output is in %s." % b.logdir)
    print(" If a line above says 'FPS only' or reads nothing, send that log:")
    print(" the parser is adjusted to it before the real run.")
    b.close()
    return 0


# ---------------------------------------------------------------------------
# --selftest
# ---------------------------------------------------------------------------

def _tiny_jpeg(dri=0, sub="444", q_luma=2):
    """A marker skeleton of a JPEG, enough for jpeg_info - no image data."""
    def seg(marker, body):
        return b"\xff" + bytes([marker]) + struct.pack(">H", len(body) + 2) + body
    zz = bytes(range(1, 65))
    dqt = seg(0xDB, bytes([0x00]) + bytes([q_luma]) + zz[1:]) + \
        seg(0xDB, bytes([0x01]) + zz)
    hv = {"444": 0x11, "420": 0x22, "422": 0x21}[sub]
    sof = seg(0xC0, bytes([8]) + struct.pack(">HH", 16, 16) + bytes([3])
              + bytes([1, hv, 0, 2, 0x11, 1, 3, 0x11, 1]))
    dri_s = seg(0xDD, struct.pack(">H", dri)) if dri else b""
    sos = seg(0xDA, bytes([1, 1, 0, 0, 63, 0]))
    return b"\xff\xd8" + dqt + sof + dri_s + sos + b"\x00\xff\xd9"


def selftest():
    print("=" * 62)
    print(" %s, version %s - selftest" % (SCRIPT_NAME, BENCH_VERSION))
    print("=" * 62)
    # 1. JPEG markers
    inf = jpeg_info(_tiny_jpeg(dri=4, sub="420"))
    assert inf["frame"] == "baseline", inf
    assert inf["dri"] == 4, inf
    assert inf["subsampling"] == "420", inf
    luma, chroma = tables_of(inf)
    assert luma[0] == 2 and chroma[0] == 1, (luma[:3], chroma[:3])
    # The file stores tables in zigzag order: zigzag position 1 is natural
    # index 1 (first row, second column), zigzag position 2 is natural index 8
    # (second row, first column). The skeleton stores 2, 2, 3, ... in zigzag.
    assert luma[1] == 2 and luma[8] == 3, luma[:10]
    print(" JPEG markers: baseline, 4:2:0, restart 4, tables read - correct")
    a = jpeg_info(_tiny_jpeg(q_luma=2))
    b_ = jpeg_info(_tiny_jpeg(q_luma=9))
    assert compare_tables(a, a)["same"] is True
    t = compare_tables(a, b_)
    assert t["same"] is False and t["luma_max_diff"] == 7, t
    print(" table comparison: identical / differ by 7 - correct")
    # 2. the RESULT line of the harness wins over prose
    txt = ("Total decode time including all transfers for 100 images = 50.0 ms;"
           " 1 MB/s; 999.0 FPS;\n"
           'RESULT {"dir": "D", "images": 128, "threads": 1, "batch": 64, '
           '"ms": 64.0, "ms_per_frame": 0.5, "mb_s": 10.0, "fps": 2000.0, '
           '"window": "host2host", "path": "native_batch", "huffman": "gpu", '
           '"backend": "gpuhybrid"}\n')
    p = parse_output(txt)
    assert p["fps"] == 2000.0 and p["frames"] == 128, p
    assert p["boundary"] == "all" and p["huffman"] == "gpu", p
    print(" RESULT line read before the prose line - correct")
    # 3. prose of the Fastvideo kind, both wordings
    p = parse_output("- GPU pipeline including all transfers for 200 images "
                     "per 2 threads = 100.0 ms; 2000.0 FPS;\n")
    assert p["fps"] == 2000.0 and p["boundary"] == "all", p
    # The Fastvideo sample, its four shapes, copied from the probe of 21.09.
    p = parse_output(
        "Processing time on GPU for 1 images including all transfers = "
        "13.47 ms; 441 MB/s;  74 FPS \n"
        "Processing time on GPU for 1 images excluding host-to-device "
        "transfer: average = 12.506 ms (474 MB/s;  80 FPS), min = 12.506 ms")
    assert p["fps"] == 74.0 and p["boundary"] == "all", p
    p = parse_output("Processing time on GPU for 50 images excluding "
                     "host-to-device transfer = 11.11 ms; 26691 MB/s;  "
                     "4499 FPS")
    assert p["fps"] == 4499.0 and p["boundary"] == "no_h2d", p
    p = parse_output("Process time for 50 images without HDD I/O and "
                     "excluding device-to-host transfer = 35.05 ms; 1427 FPS")
    assert p["fps"] == 1427.0 and p["boundary"] == "no_d2h", p
    p = parse_output("Host-to-device transfer = 0.96 ms\n"
                     "Effective encoding performance (includes device-to-host "
                     "transfer) = 0.50 GB/s (12.51 ms)")
    assert p["h2d_ms"] == 0.96 and p["eff_ms"] == 12.51, p
    print(" the Fastvideo sample: all four shapes and both legs - correct")

    # The pairing of windows, mode by mode.
    assert window_of([]) == "codec", window_of([])
    assert window_of(["-repeat", 50]) == "codec"
    assert window_of(["-repeat", 50, "-async", "-thread", 8]) == "host2host"
    assert window_of(["-repeat", 128, "-batched", "-b", 64]) == "host2host"
    e = [str(x) for x in enc_args("nv", "a.ppm", "b.jpg", 90, "444",
                                  ["-repeat", 50])]
    assert "codec" in e and "host2host" not in e, e
    e = [str(x) for x in enc_args("nv", "a.ppm", "b.jpg", 90, "444",
                                  ["-repeat", 50, "-async", "-thread", 8])]
    assert "host2host" in e and "codec" not in e, e
    d = [str(x) for x in dec_args("nv", "a.jpg", "b.ppm", ["-repeat", 50])]
    assert "codec" in d, d
    print(" windows pair mode by mode: plain repeat = codec, async = "
          "host2host - correct")

    p = parse_output("Total encode time excluding host-to-device transfer "
                     "for 100 images = 50.0 ms; 300 MB/s; 2000.0 FPS;\n")
    assert p["boundary"] == "no_h2d", p
    p = parse_output("something 1234.5 FPS\n")
    assert p["parsed_from"].startswith("FPS only"), p
    print(" prose summary lines and the FPS-only fallback - correct")
    # 4. differential energy
    e1, e2, n1, n2 = 12.0, 20.0, 100, 200
    assert abs((e2 - e1) / (n2 - n1) - 0.08) < 1e-12
    print(" differential energy: (20-12)/(200-100) = 0.08 J - correct")
    # 5. zero is a value, not an absence (playbook, measurement rules 11)
    assert fmt(0) == "0" and fmt(None) == "-"
    assert median([0.0, 0.0, 1.0]) == 0.0
    print(" zero printed as zero, not as a dash - correct")
    # 6. command lines
    a = enc_args("fv", "x.ppm", "y.jpg", 90, "444", ["-async", "-thread", 8,
                                                    "-b", 2])
    assert "-threadR" in [str(v) for v in a] and "-discard" in a
    a = dec_args("nv", "y.jpg", "z.ppm", ["-repeat", 10], backend="hardware")
    assert a[a.index("-backend") + 1] == "hardware"
    assert a[a.index("-window") + 1] == "codec"      # plain repeat, since 06
    a = dec_args("nv", "y.jpg", "z.ppm", ["-repeat", 10, "-async", "-thread",
                                          8], backend="hardware")
    assert a[a.index("-window") + 1] == "host2host"
    print(" command lines: Fastvideo gets reader/writer threads, nvJPEG the "
          "window and backend - correct")
    # 7. NVIDIA's own programs: the lines step 1 reads
    assert RE_STOCK_NOHW.search("Hardware Decoder not supported. Falling back "
                                "to default backend")
    assert float(RE_STOCK_DEC.search("Avg images per sec: 1234.5").group(1)) \
        == 1234.5
    assert float(RE_STOCK_DEC.search("Avg images per sec: 1.2e+03").group(1)) \
        == 1200.0
    assert float(RE_STOCK_ENC_T.search("Total time spent on encoding: 812.3"
                                       ).group(1)) == 812.3
    assert int(RE_STOCK_ENC_N.search("Total images processed: 40").group(1)) \
        == 40
    t1, k1, t2, k2 = 50.0 + 20 * 2.0, 20, 50.0 + 40 * 2.0, 40
    assert abs(1000.0 * (k2 - k1) / (t2 - t1) - 500.0) < 1e-9
    print(" NVIDIA's programs: their lines read, including 'not supported';"
          " N/2N removes a 50 ms first call - correct")
    # 8. meters
    dev = NvmlDevice(0)
    print("\n NVML: %s" % ("available" if dev.handle else
                           "not available (%s)" % dev.error))
    print("   energy counter:      %s" % ("yes" if dev.energy_ok else "no"))
    print("   JPEG engine counter: %s" % ("yes" if dev.jpg_ok else
                                          "no (%s)" % (dev.jpg_error or
                                                       dev.error)))
    print("   numpy for exact PSNR: %s" % ("yes" if _np() else
                                           "no - PSNR on a sample"))
    print("\n selftest finished, nothing was measured.")
    return 0


# ---------------------------------------------------------------------------

def usage():
    here = HERE
    resolve_paths()
    print("=" * 70)
    print(" %s, version %s" % (SCRIPT_NAME, BENCH_VERSION))
    print(" Baseline JPEG: Fastvideo against nvJPEG (CUDA and hardware)")
    print("=" * 70)
    print("\n Nothing was measured: a run has to be asked for by name.\n")
    print(" IN THIS ORDER")
    for o, t in (("--selftest", "the file and the meters; no card needed"),
                 ("--build", "builds the nvJPEG harness and stops"),
                 ("--probe", "every program once; shows what was read"),
                 ("--hw", "the hardware JPEG engine check only"),
                 ("--trial", "every branch, 20 frames a point, ~10 min"),
                 ("--final", "the run the article is written from")):
        print("   python %s %-12s %s" % (SCRIPT_NAME, o, t))
    print("\n Options: --subs 444,420   --qualities 90,95   --codec fv|nv|both")
    print("          --images 2k=2k_wild.ppm,4k=4k_wild.ppm,8k=8k_wild.ppm")
    print("          --reps N   --no-energy   --out DIR   --label TEXT")
    print("          --dry-run  prints every command, runs nothing")
    print("\n WHERE IT LOOKS  (each side keeps its own folder)")
    for key, path in (("--fv", os.path.dirname(CODECS["fv"]["enc"])),
                      ("--img", os.path.dirname(IMAGES[0][1]) if IMAGES
                       else DEF_IMG),
                      ("--nv", NV_DIR),
                      ("--nvidia", STOCK_DIRS[0])):
        print("   %-9s %s%s" % (key, path,
                                "" if os.path.isdir(path) else "   NO SUCH FOLDER"))
    print("   project folder, where the working files go: %s" % here)

    print("\n WHAT IT NEEDS")
    need = [(CODECS["fv"]["enc"], "Fastvideo JPEG sample, from the SDK"),
            (CODECS["nv"]["enc"], "nvJPEG encoder (built by --build)"),
            (CODECS["nv"]["dec"], "nvJPEG decoder (built by --build)")]
    need += [(p, "test frame, %s" % t.upper()) for t, p in IMAGES]
    need += [(NV_SOURCE, "source of the harness")]
    for name, what in need:
        print("   %-38s %s" % (what, "found" if os.path.exists(name)
                               else "NOT FOUND: " + name))
    for k, what in (("dec", "NVIDIA decoder, step 1 (optional)"),
                    ("enc", "NVIDIA encoder, step 1 (optional)"),
                    ("hw", "NVIDIA hardware-aware decoder (optional)")):
        p = find_stock(STOCK[k])
        print("   %-38s %s" % (what, ("found: " + p) if p else "not found"))
    for t, p in IMAGES_SKIPPED:
        print("   %-38s %s" % ("test frame, %s (optional)" % t.upper(),
                               "not found: " + p))
    print("\n Reference .jpg files are made by the run itself.")
    return 1


def main():
    if len(sys.argv) == 1:
        return usage()
    if "--do" in sys.argv[1:]:
        print("This script has no --do: the run is named instead.\n"
              "  --selftest  --build  --probe  --hw  --trial  --final\n"
              "Nothing was measured. (--do is the write flag of the site "
              "scripts; here the name of the run says what to do.)")
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--hw", action="store_true")
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--codec", choices=["fv", "nv", "both"], default="both")
    ap.add_argument("--subs", default=SUB_MAIN,
                    help="chroma subsampling, e.g. 444 or 444,420")
    ap.add_argument("--qualities", default=None,
                    help="qualities of the speed grid, e.g. 90,95")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="where the run folder goes; default runs\\ of the "
                         "project folder (54.11 section 1.1)")
    ap.add_argument("--label", default="")
    ap.add_argument("--images", default=None,
                    help="frames as tag=file, comma separated; default "
                         "2k=2k_wild.ppm,4k=4k_wild.ppm. A bare name is "
                         "looked for in --img, a full path is taken as it "
                         "is. Large frames are added the same way: "
                         "24mp=..., 65mp=...")
    ap.add_argument("--fv", default=None,
                    help="folder of the Fastvideo SDK sample JpegSample")
    ap.add_argument("--img", default=None,
                    help="folder of the frames; defaults to --fv")
    ap.add_argument("--nv", default=None,
                    help="OUR folder: nvjpeg_bench-NN.cpp and its executables")
    ap.add_argument("--nvidia", default=None,
                    help="NVIDIA's own programs; nothing of ours goes there")
    args = ap.parse_args()
    bad = resolve_paths(args.fv, args.img, args.nv, args.nvidia, args.images)
    args.out = _abs(args.out, ORIG_CWD) if args.out else os.path.join(HERE,
                                                                      "runs")
    os.chdir(HERE)          # working files belong in the project folder
    for part in bad:
        print("--images wants tag=file, got %r" % part)
    if bad:
        return 2

    print("frames: %s" % ", ".join("%s (%s)" % (t, os.path.basename(p_))
                                   for t, p_ in IMAGES))
    for t, p_ in IMAGES_SKIPPED:
        print("frame %s is not measured: no %s. Put it next to the other"
              " frames to measure it." % (t.upper(), p_))
    if args.selftest:
        return selftest()
    for t, p in IMAGES:
        g = ppm_geometry(p)
        if g:
            GEOM[t] = g

    if args.build:
        os.makedirs("build-logs", exist_ok=True)
        print("[1] building from %s\n   %s"
              % (NV_SOURCE, build_nv("build-logs")))
        print("\n[2] what the executables answer to -version")
        return 0 if check_nv_version(False) else 1
    if args.probe:
        return probe(args)

    args.subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    if args.qualities:
        args.qualities = [int(x) for x in args.qualities.split(",")]
    else:
        args.qualities = QUALITIES_FINAL if args.final else [QUALITY_MAIN]
    if args.reps is None:
        args.reps = 3 if args.final else 1
    args.run_s = RUN_S_FINAL if args.final else RUN_S_DEFAULT

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.out, "jpeg_%s%s%s" % (
        stamp, "_trial" if args.trial else "",
        ("_" + args.label) if args.label else ""))
    dev = NvmlDevice(args.device)
    b = Bench(outdir, args.dry_run, dev)
    power = Power(args.device)
    r = Run(args, b, power, dev)
    if not args.dry_run:
        HEART.start()

    print("=" * 70)
    print(" Baseline JPEG: %s" % ", ".join(CODECS[c]["name"] for c in r.codecs))
    print(" output: %s" % outdir)
    print(" qualities %s, subsampling %s, repeats %d%s"
          % (args.qualities, args.subs, args.reps,
             ", TRIAL (20 frames a point)" if args.trial else ""))
    print("=" * 70)
    env = environment(args.device) if not args.dry_run else {}
    r.env = env

    if "nv" in r.codecs:
        if not args.dry_run and not args.no_build:
            print("\n[0] nvJPEG harness\n   " + build_nv(b.logdir))
        print("\n[0a] which harness the executables came from")
        if not check_nv_version(args.dry_run):
            return 1
    missing = [CODECS[c][x] for c in r.codecs for x in ("enc", "dec")
               if not (os.path.exists(CODECS[c][x]) or args.dry_run)]
    missing += [p for _, p in IMAGES if not (os.path.exists(p)
                                             or args.dry_run)]
    if missing:
        print("\nNot found in this folder: " + ", ".join(sorted(set(missing))))
        return 1

    if "nv" in r.codecs:
        print("\n[1] the hardware JPEG engine")
        phase_hw(r)
        if args.hw:
            print("\n[1a] reference streams for the stream check")
            for tag, src in IMAGES:
                for sub in r.subs:
                    for codec in r.codecs:
                        make_ref(r, codec, tag, src, QUALITY_MAIN, sub)
            phase_hw_streams(r)
            b.close()
            return 0

    if not args.dry_run:
        idle = power.measure_idle(3.0)
        if idle:
            print("\n   the card's idle draw: %.0f W" % idle)

    print("\n[2] watermark of the Fastvideo build")
    phase_watermark(r)
    print("\n[3] reference streams")
    phase_refs(r)
    print("\n[4] quality scales: tables, sizes, PSNR, the size search")
    phase_scales(r)
    print("\n[5] references at equal file size")
    phase_matched_refs(r)
    if "nv" in r.codecs:
        print("\n[6] hardware engine: stream by stream")
        phase_hw_streams(r)
    print("\n[7] probing the speed and the memory a frame costs")
    phase_probe(r)
    plan_memory(r)
    if "nv" in r.codecs:
        print("\n[7a] step 1: NVIDIA's own programs against our harness")
        phase_stock(r)

    n_pts = (len(IMAGES) * len(args.qualities) * len(args.subs)
             * len(r.codecs) * 2 * (1 + len(POINTS)))
    if "nv" in r.codecs:
        n_pts += (len(IMAGES) * len(args.qualities) * len(args.subs)
                  * len(NV_BATCHES)
                  * (2 if (r.hw["dec"] or {}).get("available") else 1))
    est = n_pts * args.reps * ((TRIAL_FRAMES / 500.0 if args.trial
                                else args.run_s) + 1.5)
    print("\n   %d points x %d repeats, about %s" % (n_pts, args.reps,
                                                   human(est)))
    print("\n[8] measuring")
    phase_measure(r)
    if not args.no_energy and not args.dry_run:
        print("\n[9] energy per frame: at the best point and at one thread")
        phase_energy(r)
    print("\n[10] round trip and decoder agreement")
    phase_roundtrip(r)
    print("\n[11] where the time of one frame goes")
    phase_stages(r)
    rm("tmp.jpg", "tmp.ppm")
    if args.dry_run:
        print("\ndry run, nothing measured")
        return 0
    boundary_audit(b.rows)
    write_results(r, outdir, env)
    HEART.stop()
    b.close()
    if _QUIET["waits"]:
        print("\nThe card was busy with someone else's work %d time(s);"
              " %.0f s were spent waiting for it. Every repeat carries its"
              " wait in the column card_wait_s." % (_QUIET["waits"],
                                                    _QUIET["waited_s"]))
    elif _QUIET["ok"]:
        print("\nThe card was free before every repeat.")
    print("\n%d program runs, total time %s" % (b.n_runs,
                                                 human(time.time() - START)))
    return 0


START = time.time()


def _run():
    try:
        return main()
    except KeyboardInterrupt:
        HEART.stop()
        for b in list(_BENCHES):
            try:
                b.close()
            except Exception:
                pass
            print("\nStopped by Ctrl-C after %s." % human(time.time() - START))
            print("Measured so far: %d rows, all in %s" % (len(b.rows),
                                                            b.jsonl_path))
            print("Logs of every run: %s" % b.logdir)
            print("summary.txt was NOT written: it is made at the end.")
        return 130


def finish(rc):
    """Leave without Python printing an error on the way out.

    sys.exit() works by raising SystemExit. From a plain terminal that is
    silent, but VS Code's debugger, its interactive window and Spyder show it
    as an error with a traceback - even when everything went well. So the
    exit code is handed only to those who can use it: a plain
    "python <script>" (jpeg-run-all starts every step exactly that way).
    Anywhere else the script just ends, and what it printed is the verdict.
    """
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


if __name__ == "__main__":
    finish(_run())
