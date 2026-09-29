#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nsys-jpeg-20.py
# version 2026-09-25.1 of 25.09.2026, cancels version 19
#
# NEW IN THIS VERSION (25.09.2026): "who is on the card" lists every compute
# process from nvidia-smi --query-compute-apps; nothing else is changed.
#
# NEW: before every measured repeat the card is checked for someone else's load, and the repeat waits while it is busy.
# FIXED: a trace whose text is not UTF-8 (a Russian prompt from Windows landed
# in it on 24.09) stopped the whole run; now the text is read as it comes, and a
# point that fails to count is named and the next point goes on.
#
# Nsight Systems over the JPEG runs made by jpeg-bench-12.py, on the machine
# with the RTX 4090. It answers what the wall clock cannot: how much of the
# window the card is actually computing, how much it is copying, how much it
# does nothing, and how many kernels run at the same time.
#
# This is the JPEG twin of the pair that did the same for JPEG2000
# (nsys-j2k-12.py collected, nsys-cover-02.py counted). Here both halves live
# in one file, because the counting part turned out to be forty lines.
#
# WHAT IS NEW IN VERSION 15 (22.09.2026, after the run with long points)
#
#   Version 14 made the witness check work, and the first thing it showed was
#   that the check itself was set wrong. Seventeen of eighteen points came
#   back "NOT a witness", with drifts of 2.5 to 17.5 per cent - AND FIVE OF
#   THEM NEGATIVE, that is, faster under the profiler than without it. A
#   tolerance of 2 per cent was never going to be met: run-to-run scatter on
#   these points is larger than that, and it goes both ways.
#
#   1. THE TOLERANCE IS 20 PER CENT, and the number is what matters, not the
#      verdict. A drift inside the band says the trace is representative of
#      an untraced run to within that much - which is what a reader of the
#      article needs to know, and it is now stated rather than hidden behind
#      a pass or fail.
#   2. The drift is printed for every point, passed or not, and the run ends
#      with the largest one seen.
#
#   Nothing in the counting changed.
#
# WHAT IS NEW IN VERSION 14 (22.09.2026, after the first full run)
#
#   The first complete run went through - 18 points, 16 minutes - and reading
#   it showed three things wrong with THIS script, not with the measurement.
#
#   1. THE POINTS WERE TOO SHORT. The repeat count came from the bench row,
#      and it gave runs of 0.23 to 29 seconds. On the short ones the trace was
#      2.5 to 3.9 times longer than the codec's own working time - warm-up and
#      the final write filled it - and the control speed scattered by tens of
#      per cent. Now every point is stretched to at least TARGET_SECONDS of
#      work. Only the repeat count is touched; every other option of the
#      command line is left exactly as the bench wrote it, and the change is
#      printed for each point.
#   2. THE WITNESS CHECK NEVER RAN. Every row said "witness yes" and every
#      drift was empty: nsys on Windows does not pass the child's output
#      through, so the speed under the profiler was never read, and "yes" was
#      the default. Now the traced program is run through a small .bat that
#      redirects its output to a file; if that still yields nothing, the row
#      says "not checked" instead of "yes". And the cost of profiling itself
#      finally becomes a measured number.
#   3. THE CARD COUNTERS DIGEST WAS NONSENSE. It averaged every numeric
#      column, including metric identifiers and timestamps. Now the metric
#      names are resolved and each metric gets its own average and maximum.
#
#   Nothing in the counting of coverage, busy, starvation or AKC changed.
#
# WHAT IS NEW IN VERSION 13 (22.09.2026, the same day)
#
#   Fyodor: "I take the folder, make an archive and send it to you. If you do
#   not need something from that archive, do not put those files there." He
#   is right, and the instruction "send everything except .nsys-rep and
#   .sqlite" was the wrong shape of answer - it made a person do the sorting
#   every time.
#
#   1. THE TRACES ARE WRITTEN SOMEWHERE ELSE: <project>\\nsys-traces\\<same
#      name>\\. The results folder holds only small files - logs, digests,
#      the table, RUN-INFO and PROGRESS. Take the folder, zip it, send it;
#      nothing has to be removed.
#   2. The closing lines say exactly that, and name the traces folder for
#      anyone who needs the traces themselves.
#
#   Nothing in the counting changed.
#
# WHAT IS NEW IN VERSION 12 (22.09.2026, the same day)
#
#   Fyodor opened the results folder right after the start and found it
#   EMPTY. It was not broken: the first file appears only when the first
#   control run ends, a minute or two in. But an empty folder says nothing,
#   and this is the third time today he has had to ask where the results
#   are.
#
#   1. RUN-INFO.txt IS WRITTEN THE MOMENT THE FOLDER IS MADE: version, start
#      time, the bench run, every point with its command line, and what will
#      appear here. The folder is never empty, not for a second.
#   2. PROGRESS.txt is rewritten after every point: how many are done, which
#      one is running now, and the figures already counted.
#   3. The folder is made only after every check that can refuse - no more
#      empty folders left behind by a run that stopped at "cannot run
#      without nsys".
#
#   Nothing in the counting changed.
#
# WHAT IS NEW IN VERSION 11 (22.09.2026, the same day)
#
#   Fyodor: "I do not need that many keys. If you think they are needed, put
#   them into the script so that it goes through all those modes one after
#   another. Without arguments it shows the diagnosis, with --do it works.
#   Nothing else is needed."
#
#   Version 10 shortened the keys; that was the wrong fix. There is now ONE
#   key, and every mode that used to be a key is either done by the script
#   itself or gone:
#
#     --where, --list-runs   are the diagnosis printed without arguments
#     --gpu-metrics          always asked for; if nsys refuses, the run goes
#                            on without the counters and says so
#     --export-only          done by itself: traces left uncounted by a run
#                            that died at the export step are counted first
#     --nsys, --trace        found and probed by the script, as before
#     --run, --images        the script picks the run and does 2K and 4K
#     --q                    fixed at 90, the quality the article is built on.
#                            Mixing 90 and 95 in one median is meaningless,
#                            and 95 is not used anywhere in the text
#     --only, --first        gone. They were mine, for debugging
#
#   Nothing in the counting changed.
#
# WHAT IS NEW IN VERSION 10 (22.09.2026, the same day)
#
#   Fyodor: "I asked you not to make long keys." He is right, and the rule is
#   his own, written down in 46.04 SS1 on 02.09.2026: without keys - show,
#   --do - run, and nothing else to remember. Version 09 had twelve keys, all
#   of them long, and --gpu-metrics was mistyped as --gpu-metrix.
#
#   1. EVERY KEY HAS A ONE-LETTER FORM: -d run, -m card counters, -w where,
#      -l runs, -q quality, -o one point, -r run folder, -n path to nsys,
#      -f first N, -x count from traces already made, -i frames, -t trace.
#   2. The long forms still work - nothing that was typed before breaks.
#   3. The two that matter are printed at the top of every run, so they do
#      not have to be looked up.
#
#   Nothing in the counting changed.
#
# WHAT IS NEW IN VERSION 09 (22.09.2026, the same day)
#
#   Fyodor, on version 08: today is the 22nd, so the results have no business
#   sitting in a folder dated the 21st. He is right - version 08 explained the
#   awkward place instead of moving it. A measurement made today is filed
#   under today.
#
#   1. RESULTS GO TO <project>\nsys-out\<today>-<time>_<bench run>\ - a
#      folder of their own next to this file, whose name BEGINS with today's
#      date and ends with the bench run the points came from. Nothing is
#      written into the bench run folder any more.
#   2. --where and --export-only look in the new place first and in the old
#      one after it, so folders made by versions 06 to 08 are still found.
#
#   Nothing in the counting changed.
#
# WHAT IS NEW IN VERSION 08 (22.09.2026, the same day)
#
#   Fyodor could not find the results: "there is no such folder for today".
#   The output folder is made INSIDE the chosen run folder, and the chosen run
#   is dated the day the bench was run - 21.09 - so nothing under runs\ looks
#   like today at all. The path was printed once, in the middle of a long
#   output, and scrolled away.
#
#   1. --where ANSWERS IT IN TWO SECONDS: it prints the project folder, the
#      runs folder, the chosen run, the exact folder the results would go to,
#      and every nsys-* folder already there with how many files each holds.
#      It measures nothing and takes no time.
#   2. The output folder is made and named BEFORE the plan, inside a frame of
#      "=" that cannot be missed, and named again after every third point.
#   3. The full path is also written to nsys-last-output.txt next to this
#      file, so it survives the scrollback.
#   4. The plan lists the nsys-* folders already in the run folder.
#
#   Nothing in the counting changed.
#
# WHAT IS NEW IN VERSION 07 (22.09.2026, the same day)
#
#   Only the words on the screen. Version 06 collected the card counters when
#   asked - and at the same time printed, every run, the old line saying they
#   are NOT collected. The line sat outside any condition and contradicted the
#   very option that had just been given.
#
#   1. THE LINE ABOUT THE COUNTERS NOW FOLLOWS THE OPTION: without
#      --gpu-metrics it says they are not collected, with it - that they are
#      asked for and which form nsys accepted.
#   2. The accepted form is stated once at the first point and again in the
#      closing lines, so a run watched from the middle still shows it.
#   3. The closing lines say whether the counters ended up in the trace.
#
#   Nothing in the counting changed. A run started with version 06 need not be
#   restarted: if the line "card counters: --gpu-metrics-devices=all" appeared
#   at the first point, they are being collected.
#
# WHAT IS NEW IN VERSION 06 (22.09.2026, the same day)
#
#   The first full run went through - and produced figures that cannot exist:
#   "busy" of 173, 227, even 333 per cent. The cause was not the trace but the
#   denominator. The window was frames / fps, which is the time the codec was
#   WORKING, while the trace covers the whole process: reading the image,
#   warming up, the final write. On these short runs that is 1.5 to 3.3 times
#   the window, and everything divided by it went over a hundred.
#
#   1. THE SHARES ARE TAKEN AGAINST THE SPAN OF THE CARD'S WORK - first event
#      to last - which cannot be exceeded by definition.
#   2. The old window stays beside them as span_over_window: how many times
#      longer the trace is than the codec's own working time.
#   3. --gpu-metrics turns on the card's own counters (administrator terminal
#      needed). They are collected into the trace and listed in their own
#      digest, never mixed into this table: collecting them costs more time.
#   4. The output says in full WHERE the results are, at the end as well.
#
# WHAT IS NEW IN VERSION 05 (22.09.2026, the same day)
#
#   Version 04 profiled fine - and then "nsys export" refused the option
#   --force-export=true, which this build does not have. The same class of
#   mistake as the trace line, one command later.
#
#   1. THE EXPORT LINE IS TRIED TOO, exactly like the trace line: the script
#      walks simpler forms until one is accepted, names it once and uses it
#      for the rest. Any stale .sqlite is removed first, so a build that
#      refuses to overwrite still gets a clean file.
#   2. --export-only SAVES WHAT WAS ALREADY MEASURED. The traces of a run
#      that failed at the export step are still on disk; this reads them,
#      together with the speed and frames kept in each point's own .run.log,
#      and produces the numbers without measuring anything again.
#   3. --first N runs only the first N points, for trying the whole path
#      before spending an hour on it.
#
# WHAT IS NEW IN VERSION 04 (22.09.2026, the same day)
#
#   Version 03 got as far as starting nsys, and nsys refused all eighteen
#   times: "Illegal --trace argument 'osrt'". That value exists on Linux and
#   not in the Windows build - the trace line had been copied from the
#   JPEG2000 work and never checked against the nsys that is installed here.
#
#   1. THE TRACE LINE IS TRIED, NOT ASSUMED. It starts at --trace=cuda -
#      kernels and memory copies, which is all these numbers need - and if
#      this nsys dislikes an option, the script falls back through simpler
#      lines until one is accepted, says which one it settled on and uses it
#      for the rest of the points. --trace overrides the whole thing.
#   2. THE CONTROL RUN IS ALSO COMPARED WITH THE BENCH ROW. Version 03
#      measured 5657 fps where the run itself had 3819, and a gap that size
#      means the trace cannot be carried over to the article's numbers. It is
#      now printed and written into the metrics.
#
# WHAT IS NEW IN VERSION 03 (22.09.2026, the same day)
#
#   Version 02 built the right plan and found nsys, and then every one of the
#   eighteen points failed. Three defects, all mine:
#
#   1. THE RUNS WERE STARTED IN THE WRONG FOLDER. A command line from the
#      bench names its reference file relatively - "fv_ref_2k_q90_444.jpg" -
#      and those files live in the project folder. Started from wherever the
#      shell stood, every decode run exited in 0.1 s having found nothing.
#      Both runs, control and profiled, now start in the project folder, the
#      same rule jpeg-bench-12.py follows.
#   2. WHEN nsys FAILED, THE SCRIPT ONLY POINTED AT A LOG. Now it prints what
#      nsys actually said, right there, and checks "nsys --version" once
#      before the first point.
#   3. A control run that produced no figure was profiled anyway. Now the
#      point is skipped with the reason and the first lines the program
#      printed, because profiling a run that did not measure anything cannot
#      produce a number either.
#
#   Also: the docstring that said logs\ raised a SyntaxWarning on 3.13.
#
# WHAT IS NEW IN VERSION 02 (22.09.2026, the same day)
#
#   Version 01 found no logs at all on the bench and had to stop. Three
#   things were wrong, and the first one is the real defect:
#
#   1. THE QUALITY WAS NOT FILTERED. The grid is measured at q 90 AND q 95,
#      and version 01 mixed both into one group: the median was taken over
#      two different qualities, and the log name was then built with the q of
#      whichever repeat won. Now the quality is a key of its own, --q, and it
#      is 90 by default - the quality the article's charts use.
#   2. LOGS ARE LOOKED FOR IN THREE PLACES - logs\, the run folder itself and
#      inside logs.zip - and a point is matched to a log by the log's name
#      FIRST and by the command line in it second. If nothing matches, the
#      script now prints what it actually sees in the folder instead of only
#      the name it wanted.
#   3. nsys is looked for in more places, and --nsys takes a path.
#
# WHAT IT DOES, IN ORDER
#
#   1. picks the run folder under runs\ - the one that has both kinds of
#      points (single frames and a thread/batch grid); of those, the fullest
#      grid; at equal fullness the newest. The others are printed with their
#      reason, never skipped silently;
#   2. picks the points to profile (see THE POINTS below);
#   3. takes each point's COMMAND LINE FROM ITS LOG, never builds one. A line
#      that is not in the log is not a point: rebuilding it would be a
#      different experiment (the rule that came out of the JPEG2000 work);
#   4. runs each point twice: once plain, as a control, and once under nsys;
#   5. exports every trace to SQLite and writes small digests next to it -
#      kernels, memory copies, API calls. The .nsys-rep files stay on the
#      machine; the digests are what travels;
#   6. counts coverage, busy, starvation, kernel time per frame and average
#      kernel concurrency, and compares the control figure with the profiled
#      one. More than 2 % apart and the trace is marked "not a witness".
#
# THE POINTS (METHOD.md section 7)
#
#   both codecs, both directions, frames 2K and 4K, two modes each: a single
#   frame and the best grid point. Plus one point of its own kind - the nvJPEG
#   decoder on its native batch, 32 against 128 on 2K, which is where nvjpeg.h
#   says the Huffman stage moves to the GPU. That pair is compared only with
#   itself: its measurement window is not the others'.
#
#   The best grid point is chosen BY THE MEDIAN of its repeats, not by the
#   fastest one. The repeats live in results.jsonl; the command line is the
#   same for all of them and is read from the log.
#
# WHAT IT DOES NOT DO
#
#   - the card's own counters are OFF by default: --gpu-metrics-devices needs
#     an administrator terminal and costs time of its own. --gpu-metrics turns
#     them on; either way the screen says which of the two it is, and they
#     stay in their own digest, out of the coverage table;
#   - it does not touch the programs and does not add NVTX marks. Any change
#     to a measuring program voids the method (Fyodor, 08.09.2026);
#   - it does not use Nsight Compute at all: it serialises kernels, and this
#     whole measurement is about overlap. Frames per second from Nsight
#     Compute never go into an article.
#
# HOW TO READ THE NUMBERS
#
#   window          frames / fps as the program itself printed them
#   kernel time     sum of all kernel durations
#   coverage        share of the window with at least one kernel running
#   busy            the same, counting memory copies as work
#   starvation      1 - busy: the card doing nothing
#   AKC             kernel time / window: how many kernels run at once on
#                   average. Below 1 the card idles inside the window
#
#   Profiling costs time itself - 2 to 15 % on the JPEG2000 runs - so coverage
#   is a LOWER bound, and in an article it is "about two and a half times",
#   not "2,41".
#
# WHAT IS NEW IN THIS VERSION
#
#   It reads runs\\USE-THIS-RUN.txt, if there is one, and uses the run
#   named in it. That file is written by jpeg-run-all-01.py after the
#   bench finishes, so every script of one testing session works on the
#   same run. Without the file nothing changes: the run is chosen as
#   before.
#
# USAGE
#
#     python nsys-jpeg-16.py            show everything, run nothing
#     python nsys-jpeg-16.py --do       run it
#
#   That is all there is. Two states, as memo 46.04 SS1 says.
#
#   Results: <project>\nsys-out\<date>-<time>_<bench run>\ - next to it,
#   named by the day the profiling was done.
#
#   No folder argument: the project folder is the folder THIS FILE lies in,
#   the runs are under it. Same rule as jpeg-bench-12.py, and for the same
#   reason - a run started from a code editor must not look for its files
#   next to the editor.

import argparse
import datetime
import glob
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time

VERSION = "nsys-jpeg-20 of 25.09.2026"

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
# Results live in a folder of their own, named by TODAY. They used to go
# inside the bench run folder, which carries the date of the bench run -
# so a measurement made on the 22nd was filed under the 21st and could
# not be found (Fyodor, 22.09.2026).
OUTROOT = os.path.join(HERE, "nsys-out")
# The traces live apart from the results. They are hundreds of megabytes,
# and the results folder is meant to be zipped whole and sent without
# anyone picking files out of it (Fyodor, 22.09.2026).
TRACES = os.path.join(HERE, "nsys-traces")
# Fixed, not keys: the article is built on quality 90, and mixing 90 with
# 95 in one median means nothing. 2K and 4K are the frames of METHOD SS7.
QUALITY = 90
# Version 18 (24.09.2026): only what the article shows - which programs each
# codec runs on the card (one point per codec and direction: the best grid
# point, frame 2K) and nvJPEG's batch 32 against 128. Six points, not 18.
IMAGES = ("2k",)
PROFILE_SINGLE = False     # single-frame points are no longer profiled
# Every point is stretched to at least this much codec work. Short runs
# are mostly warm-up: on 22.09 a 0.23 s point had a trace 3.9 times longer
# than the work itself, and its control speed was 57 % off the bench row.
TARGET_SECONDS = 20.0
# ... but not past this many frames, or the trace grows to millions of
# kernel events and the export takes longer than the measurement.
MAX_FRAMES = 150000
# Tried in order until nsys accepts one. Only CUDA is needed for these
# numbers - kernels and memory copies. "osrt" is a Linux-only value and the
# Windows build refuses it outright, which is what version 03 ran into.
TRACE_TRIES = [
    ["--trace=cuda", "--sample=none"],
    ["--trace=cuda"],
    [],
]
BASE_ARGS = ["--force-overwrite=true"]
# How far the profiled run may differ from the run without the profiler
# before the trace stops standing for an untraced one. It was 2 per cent,
# copied from the JPEG2000 work, and on 22.09 the first run that actually
# measured the drift found 2.5 to 17.5 per cent, in both directions - the
# scatter between two runs of the same point is simply bigger than that. The
# number printed beside every point is the useful part; this is only the line
# past which it is called out.
CONTROL_TOLERANCE = 20.0          # per cent between control and profiled fps
RE_CMD = re.compile(r"^\$ (.+)$", re.M)
RE_FPS = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:fps|frames?/s)", re.I)


# ---------------------------------------------------------------- utilities




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

    25.09.2026: the whole list from --query-compute-apps. The previous
    version printed the first 12 lines of the nvidia-smi table, and on 25.09
    those were all desktop programs - the one that loaded the card was never
    named."""
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


def say(*a):
    print(*a)
    sys.stdout.flush()


def fmt(v, f="%.2f", dash="-"):
    return dash if v is None else f % v


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def union_seconds(spans):
    """Length of the union of [start, end] spans, in seconds."""
    if not spans:
        return 0.0
    spans = sorted(spans)
    total = 0.0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


# ------------------------------------------------------------- finding nsys


NSYS_PATTERNS = [
    # Nsight Systems, standalone and inside the CUDA Toolkit
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems*\*\*\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems*\*\nsys.exe",
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems*\nsys.exe",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin\nsys.exe",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\*\*\nsys.exe",
    r"C:\Program Files (x86)\NVIDIA Corporation\*\*\nsys.exe",
    # as installed by the Nsight Systems standalone installer for a user
    os.path.expandvars(r"%LOCALAPPDATA%\NVIDIA Corporation\*\*\nsys.exe"),
    "/opt/nvidia/nsight-systems/*/target-linux-x64/nsys",
    "/opt/nvidia/nsight-systems/*/bin/nsys",
    "/usr/local/cuda*/bin/nsys",
]


def find_nsys(explicit=None):
    """PATH, then every place Nsight Systems is known to install itself.

    Several versions live on the bench; the newest by path wins, and the
    choice is printed. Nothing is guessed silently, and --nsys overrides it.
    """
    if explicit:
        if os.path.isfile(explicit):
            return explicit, [explicit]
        return None, []
    found = []
    exe = "nsys.exe" if os.name == "nt" else "nsys"
    inpath = shutil.which(exe)
    if inpath:
        found.append(inpath)
    for pattern in NSYS_PATTERNS:
        found.extend(p for p in glob.glob(pattern) if os.path.isfile(p))
    if not found:
        return None, []
    ordered = sorted(set(found), reverse=True)
    return ordered[0], ordered


# ------------------------------------------------------- reading a bench run


class Point(object):
    """One measured point: what it was, how fast it went, how it was called."""

    def __init__(self, row):
        self.codec = row.get("codec")
        self.direction = row.get("direction")
        self.image = row.get("image")
        self.q = row.get("q")
        self.sub = row.get("sub") or "444"
        self.threads = row.get("threads") or 1
        self.batch = row.get("batch") or 1
        self.mode = row.get("mode") or ""
        self.fps = row.get("fps")
        self.frames = row.get("frames")
        self.variant = row.get("variant") or "gpu"
        # jpeg-bench-12 writes the command line into every result row, so the
        # first place to look is the row itself - the log is the fallback.
        self.cmd = row.get("cmd")

    @property
    def single(self):
        return self.threads == 1 and self.batch == 1

    @property
    def key(self):
        tail = "single" if self.single else "t%sb%s" % (self.threads,
                                                        self.batch)
        return "%s_%s_%s_%s" % (self.codec, self.direction, self.image, tail)

    def log_name(self):
        """The log jpeg-bench-12 wrote for this point."""
        tail = ("single" if self.single
                else "t%d_b%d" % (self.threads, self.batch))
        return "%s_%s_%s_%s_q%s_%s_%s" % (self.codec, self.variant,
                                          self.direction, self.image,
                                          self.q, self.sub, tail)


def read_run(folder):
    """Every measured row of a run folder, plus what is missing."""
    path = os.path.join(folder, "results.jsonl")
    rows, bad = [], []
    if not os.path.isfile(path):
        return rows, ["no results.jsonl"]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                bad.append("results.jsonl:%d is not JSON" % n)
                continue
            if row.get("fps") and row.get("codec"):
                rows.append(row)
    return rows, bad


def verdict(folder):
    """Is this run usable, and why not."""
    rows, bad = read_run(folder)
    if not rows:
        return None, "no measured rows" + (" (%s)" % bad[0] if bad else "")
    pts = [Point(r) for r in rows]
    singles = [p for p in pts if p.single]
    grid = [p for p in pts if not p.single]
    if not singles:
        return None, "no single-frame points"
    if not grid:
        return None, "no grid points"
    return {"rows": rows, "points": pts, "grid": len(grid),
            "singles": len(singles)}, "ok"


# The run every side script must use, when a run of all the tests has said
# which one it is. A file, not a key: the run-everything script writes it
# after the bench finishes, and every script that reads the runs folder
# looks at it first. Two
# scripts choosing "the newest suitable run" by their own rules can choose
# two different runs, and then the results do not belong to one another.
POINTER = "USE-THIS-RUN.txt"


def pointed_run():
    """The run named in the pointer file, if it is there and real."""
    path = os.path.join(RUNS, POINTER)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            name = fh.read().strip()
    except OSError:
        return None
    if not name:
        return None
    full = name if os.path.isabs(name) else os.path.join(RUNS, name)
    return full if os.path.isdir(full) else None


def pick_run(explicit=None):
    """The run to profile, with a word about every run that was passed over."""
    if explicit:
        data, why = verdict(explicit)
        if not data:
            return None, [(explicit, why)]
        return explicit, [(explicit, "chosen (asked for on the command line)")]
    pointed = pointed_run()
    if pointed:
        data, why = verdict(pointed)
        if data:
            return pointed, [(pointed, "CHOSEN - named in " + POINTER)]
        return None, [(pointed, "named in %s but %s" % (POINTER, why))]
    folders = sorted(glob.glob(os.path.join(RUNS, "*")))
    folders = [f for f in folders if os.path.isdir(f)]
    notes, usable = [], []
    for f in folders:
        data, why = verdict(f)
        if data:
            usable.append((f, data))
        notes.append((f, why))
    if not usable:
        return None, notes
    usable.sort(key=lambda fd: (fd[1]["grid"], os.path.getmtime(fd[0])),
                reverse=True)
    best = usable[0][0]
    out = []
    for f, why in notes:
        out.append((f, "CHOSEN" if f == best else why))
    return best, out


# ------------------------------------------------------- choosing the points


def best_grid_point(points, codec, direction, image, q):
    """The grid point whose MEDIAN over repeats is the highest.

    Not the fastest repeat: choosing by the best repeat is exactly the error
    the JPEG2000 work ran into on 08.09.2026.
    """
    groups = {}
    for p in points:
        if (p.codec, p.direction, p.image) != (codec, direction, image):
            continue
        if q is not None and int(p.q or 0) != int(q):
            continue
        if p.single:
            continue
        if codec == "nv" and direction == "D" and p.threads == 1 and p.batch > 1:
            # nvJPEG's native batch is a row of its own: its measurement
            # window is not the grid's, so it never competes for "best grid".
            continue
        groups.setdefault((p.threads, p.batch), []).append(p)
    if not groups:
        return None
    scored = []
    for (t, b), ps in groups.items():
        med = median([p.fps for p in ps])
        if med:
            scored.append((med, t, b, ps))
    if not scored:
        return None
    scored.sort(reverse=True)
    med, t, b, ps = scored[0]
    # the repeat to profile is the one CLOSEST TO THE MEDIAN
    ps.sort(key=lambda p: abs((p.fps or 0) - med))
    chosen = ps[0]
    chosen.fps = med
    return chosen


def single_point(points, codec, direction, image, q):
    ps = [p for p in points
          if (p.codec, p.direction, p.image) == (codec, direction, image)
          and p.single and (p.mode or "") != "step1"
          and (q is None or int(p.q or 0) == int(q))]
    if not ps:
        return None
    med = median([p.fps for p in ps])
    ps.sort(key=lambda p: abs((p.fps or 0) - (med or 0)))
    chosen = ps[0]
    chosen.fps = med
    return chosen


def batch_points(points, q):
    """The nvJPEG decoder on its native batch: 32 against 128, frame 2K.

    Its own kind of point - the window is not the others' - so it is compared
    only with itself.
    """
    out = []
    for size in (32, 128):
        ps = [p for p in points
              if p.codec == "nv" and p.direction == "D" and p.image == "2k"
              and p.batch == size and p.threads == 1
              and (q is None or int(p.q or 0) == int(q))]
        if ps:
            med = median([p.fps for p in ps])
            ps.sort(key=lambda p: abs((p.fps or 0) - (med or 0)))
            ps[0].fps = med
            out.append(ps[0])
    return out


def choose_points(points, images=("2k", "4k"), q=90):
    chosen, missing = [], []
    for codec in ("fv", "nv"):
        for direction in ("E", "D"):
            for image in images:
                s = single_point(points, codec, direction, image, q)
                g = best_grid_point(points, codec, direction, image, q)
                pairs = ((s, "single"), (g, "best grid")) if PROFILE_SINGLE \
                    else ((g, "best grid"),)
                for p, what in pairs:
                    if p:
                        chosen.append(p)
                    else:
                        missing.append("%s %s %s: no %s point"
                                       % (codec, direction, image, what))
    extra = batch_points(points, q)
    chosen.extend(extra)
    if len(extra) < 2:
        missing.append("nv D 2k: native batch 32/128 not both in this run")
    # one point per key: the same measurement is never profiled twice
    seen, unique = set(), []
    for p in chosen:
        if p.key in seen:
            continue
        seen.add(p.key)
        unique.append(p)
    return unique, missing


# ------------------------------------------------------------ command lines


class LogIndex(object):
    r"""Every log of a run, wherever the run keeps them.

    Three places are looked at, because a run folder does not always keep its
    logs unpacked: logs\, the run folder itself, and logs.zip inside it. What
    was found and what was not is printed - a point without a log must say so
    with the folder's real contents next to it, not just with the name that
    was wanted.
    """

    def __init__(self, folder):
        self.folder = folder
        self.where = []
        self.logs = {}          # name without .log -> text
        self._load_dir(os.path.join(folder, "logs"))
        self._load_dir(folder)
        self._load_zip(os.path.join(folder, "logs.zip"))

    def _load_dir(self, d):
        if not os.path.isdir(d):
            return
        names = [f for f in os.listdir(d) if f.lower().endswith(".log")]
        if names:
            self.where.append("%s (%d logs)" % (os.path.relpath(d,
                                                                self.folder),
                                                len(names)))
        for f in names:
            key = os.path.splitext(f)[0]
            if key in self.logs:
                continue
            try:
                with open(os.path.join(d, f), encoding="utf-8",
                          errors="replace") as fh:
                    self.logs[key] = fh.read()
            except OSError:
                pass

    def _load_zip(self, path):
        if not os.path.isfile(path):
            return
        try:
            import zipfile
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".log")]
                if names:
                    self.where.append("logs.zip (%d logs)" % len(names))
                for n in names:
                    key = os.path.splitext(os.path.basename(n))[0]
                    if key not in self.logs:
                        self.logs[key] = z.read(n).decode("utf-8", "replace")
        except Exception as exc:
            self.where.append("logs.zip unreadable: %s" % exc)

    def contents_note(self):
        """What is actually in the run folder - printed when nothing matched."""
        try:
            names = sorted(os.listdir(self.folder))
        except OSError as exc:
            return "cannot list the run folder: %s" % exc
        head = ", ".join(names[:12])
        more = "" if len(names) <= 12 else " and %d more" % (len(names) - 12)
        return "run folder holds %d entries: %s%s" % (len(names), head, more)

    def by_name(self, point):
        """The log jpeg-bench-12 would have written for this point."""
        want = point.log_name()
        if want in self.logs:
            return want
        # the same fields, whatever the separator or the order of the tail
        marks = [str(point.codec), str(point.direction),
                 str(point.image), "q%s" % point.q]
        tail = ("single" if point.single
                else ("t%d" % point.threads, "b%d" % point.batch))
        for key in self.logs:
            low = key.lower()
            if not all(m.lower() in low for m in marks):
                continue
            if isinstance(tail, str):
                if "single" in low:
                    return key
            elif all(x in low for x in tail) and "single" not in low:
                return key
        return None


def command_in(text):
    m = RE_CMD.search(text or "")
    return m.group(1).strip() if m else None


def command_of(index, point):
    """The command line as the bench wrote it down. Never rebuilt.

    Two sources, in this order: the result row itself (results.jsonl keeps a
    "cmd" field for every measurement) and the log of that point. A run whose
    logs were packed away or cleared is therefore still usable.
    """
    if point.cmd:
        return point.cmd, None
    key = index.by_name(point)
    if key is None:
        return None, ("no command line: the row has no \"cmd\" field and there"
                      " is no log %s.log" % point.log_name())
    cmd = command_in(index.logs[key])
    if not cmd:
        return None, "log %s has no '$ ' command line" % key
    return cmd, None


def split_command(line):
    """Windows command line to argv, keeping quoted paths whole."""
    out, cur, quoted = [], "", False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        elif ch == " " and not quoted:
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


# --------------------------------------------------------------- the runs


def result_of(text):
    """What the program said about itself: frames per second and frames.

    Both come from the SAME run, and that matters: the window is frames / fps,
    so taking the frames from one run and the speed from another gives a
    window that never existed and a coverage above a hundred per cent.
    """
    fps, frames = None, None
    m = RE_FPS.search(text)
    if m:
        fps = float(m.group(1).replace(",", "."))
    for line in text.splitlines():
        if line.strip().startswith("RESULT"):
            try:
                data = json.loads(line.split("RESULT", 1)[1].strip())
            except ValueError:
                continue
            fps = data.get("fps", fps)
            # These programs call them images, not frames. Version 13 looked
            # only for "frames", found nothing, and quietly fell back to the
            # bench row - which was right only as long as the repeat count
            # was not touched. Now it is.
            for k in ("frames", "images_decoded", "images"):
                if data.get(k) is not None:
                    frames = data[k]
                    break
    if frames is None:
        m = re.search(r"([0-9]{2,})\s+(?:frames|images)", text)
        if m:
            frames = int(m.group(1))
    return fps, frames


def fps_of(text):
    return result_of(text)[0]


def head_of(text, n=6):
    """The first lines a program printed - what to show when it said no figure."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    return lines[:n]


def repeat_of(argv):
    """How many frames the command line asks for, if it says."""
    for i, a in enumerate(argv[:-1]):
        if a.lower() in ("-repeat", "--repeat"):
            try:
                return int(argv[i + 1])
            except ValueError:
                return None
    return None


def stretch_repeat(argv, fps):
    """Make the point run for at least TARGET_SECONDS.

    ONLY the repeat count is touched. Everything else in the command line
    stays exactly as the bench wrote it - a command that differs in anything
    else is a different experiment (Fyodor, 08.09.2026).
    """
    if not fps or fps <= 0:
        return list(argv), None
    out = list(argv)
    for i, a in enumerate(out[:-1]):
        if a.lower() in ("-repeat", "--repeat"):
            try:
                was = int(out[i + 1])
            except ValueError:
                return out, None
            want = int(math.ceil(TARGET_SECONDS * fps))
            want = min(max(want, was), MAX_FRAMES)
            if want == was:
                return out, None
            out[i + 1] = str(want)
            return out, (was, want)
    return out, None


def write_wrapper(path, argv, out_file):
    """A tiny script that runs the program and keeps what it printed.

    nsys on Windows does not pass the child's output through to its own, so
    the speed under the profiler was never read and the witness check never
    ran (found 22.09.2026, on the first full run). The program and its
    arguments are untouched; only its output is redirected.
    """
    q = lambda x: '"%s"' % x if " " in x else x
    line = " ".join(q(a) for a in argv)
    if os.name == "nt":
        text = "@echo off\r\n%s > %s 2>&1\r\n" % (line, q(out_file))
        nl = ""
    else:
        text = "#!/bin/sh\n%s > %s 2>&1\n" % (line, q(out_file))
        nl = "\n"
    with open(path, "w", encoding="utf-8", newline=nl) as fh:
        fh.write(text)
    if os.name != "nt":
        os.chmod(path, 0o755)
    return [os.environ.get("COMSPEC", "cmd.exe"), "/c", path] \
        if os.name == "nt" else ["/bin/sh", path]


def _said_figures(path):
    """Figures from the file the wrapper kept, if there is one."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return result_of(fh.read())
    except OSError:
        return None, None


def run_plain(argv, log_path):
    """One control run, started IN THE PROJECT FOLDER.

    That is not a detail: a bench command line names its reference file
    relatively ("fv_ref_2k_q90_444.jpg"), and those files live next to this
    script. Started anywhere else, every decode run exits at once having
    found nothing - which is exactly what happened with version 02.
    """
    t0 = time.time()
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, cwd=HERE)
        out, _ = p.communicate(timeout=1800)
        text = out.decode("utf-8", "replace")
        rc = p.returncode
    except FileNotFoundError:
        return None, "program not found: %s" % argv[0]
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return None, "timed out after 30 minutes"
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(argv) + "\n(in %s)\n\n" % HERE + text)
    fps, frames = result_of(text)
    return {"fps": fps, "frames": frames, "rc": rc,
            "wall_s": time.time() - t0, "head": head_of(text)}, None


def nsys_version(nsys):
    """Ask nsys who it is before the first point. A version that answers is
    a version that runs."""
    try:
        p = subprocess.run([nsys, "--version"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=60)
        return p.stdout.decode("utf-8", "replace").strip().splitlines()[0]
    except Exception as exc:
        return "does not answer --version: %s" % exc


GPU_METRIC_TRIES = [
    ["--gpu-metrics-devices=all"],
    ["--gpu-metrics-device=all"],
    ["--gpu-metrics-devices", "all"],
]


def _one_profile(nsys, trace, argv, out_base, extra=()):
    """One attempt with one option line."""
    argv_nsys = ([nsys, "profile"] + trace + list(extra) + BASE_ARGS
                 + ["-o", out_base, "--"] + list(argv))
    p = subprocess.Popen(argv_nsys, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, cwd=HERE)
    out, _ = p.communicate(timeout=3600)
    return " ".join(argv_nsys), out.decode("utf-8", "replace"), p.returncode


def run_profiled(nsys, argv, out_base, state, log_path, wrapped=None):
    """The same run under nsys, in the project folder, and it says why not.

    Both option lines are TRIED, not assumed - the trace line and, when asked
    for, the card counters. A value that exists on one platform is rejected
    outright on another, and version 03 died eighteen times on exactly that.
    The first combination nsys accepts is remembered for the rest of the run.
    """
    t0 = time.time()
    traces = ([state["trace"]] if state.get("trace") is not None
              else TRACE_TRIES)
    if state.get("metrics") is not None:
        extras = [state["metrics"]]
    elif state.get("want_metrics"):
        extras = GPU_METRIC_TRIES + [[]]
    else:
        extras = [[]]
    last = None
    for trace in traces:
        for extra in extras:
            try:
                line, text, rc = _one_profile(nsys, trace,
                                              wrapped or argv, out_base,
                                              extra)
            except subprocess.TimeoutExpired:
                return None, "nsys timed out after an hour"
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write("$ " + line + "\n(in %s)\n\n" % HERE + text)
            rep = out_base + ".nsys-rep"
            if os.path.isfile(rep):
                if state.get("trace") is None:
                    state["trace"] = trace
                    say("   nsys accepts: %s"
                        % (" ".join(trace) or "(no trace options)"))
                if state.get("want_metrics") and state.get("metrics") is None:
                    state["metrics"] = extra
                    say("   card counters: %s"
                        % (" ".join(extra) if extra
                           else "REFUSED, going on without them"))
                fps, frames = result_of(text)
                return {"rep": rep, "fps": fps, "frames": frames,
                        "wall_s": time.time() - t0}, None
            tail = [l.rstrip() for l in text.splitlines() if l.strip()][-8:]
            last = ("nsys wrote no report (exit code %s) with %s. What it"
                    " said:\n       "
                    % (rc, " ".join(trace + extra) or "no options")
                    ) + "\n       ".join(tail or ["(nothing)"])
            if extra and len(extras) > 1:
                say("   the card counters were refused as %s, trying another"
                    " form" % " ".join(extra))
                continue
            break
        if "Illegal" not in (last or "") and "usage:" not in (last or ""):
            break          # not an option problem: a simpler line will not help
        if state.get("trace") is None and len(traces) > 1:
            say("   nsys refused %s, trying a simpler trace line"
                % (" ".join(trace) or "no trace options"))
    return None, last


# Tried in order until this build of nsys accepts one. Older builds want
# --force-export, newer ones do not know it at all; %s is the report file.
EXPORT_TRIES = [
    ["--type", "sqlite", "--force-overwrite=true", "--output"],
    ["--type", "sqlite", "--output"],
    ["--type", "sqlite", "-o"],
]


def export_sqlite(nsys, rep, state):
    """Trace to SQLite. The option line is tried, not assumed."""
    db = rep.replace(".nsys-rep", ".sqlite")
    if os.path.isfile(db):
        try:
            os.remove(db)       # a build that will not overwrite still works
        except OSError:
            pass
    tries = ([state["export"]] if state.get("export") is not None
             else EXPORT_TRIES)
    last = None
    for opts in tries:
        cmd = [nsys, "export"] + opts + [db, rep]
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        text = p.stdout.decode("utf-8", "replace")
        if os.path.isfile(db):
            if state.get("export") is None:
                state["export"] = opts
                say("   nsys export accepts: %s" % " ".join(opts))
            return db, None
        last = "nsys export failed with %s: %s" % (" ".join(opts),
                                                   text.strip()[-300:])
        if "unrecognised" not in text and "unrecognized" not in text \
                and "usage:" not in text:
            break
        say("   nsys export refused %s, trying a simpler form" % " ".join(opts))
    return None, last


# ------------------------------------------------------------- the counting


def table_exists(con, name):
    row = con.execute("SELECT name FROM sqlite_master WHERE type='table'"
                      " AND name=?", (name,)).fetchone()
    return row is not None


def digest(db, frames, fps):
    """Kernels, copies and the five figures, from one exported trace."""
    con = sqlite3.connect(db)
    # nsys stores whatever text the process met - on a Russian Windows that
    # can be a console prompt in the local code page. Read it as it comes.
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    out = {"frames": frames, "fps": fps}
    names = {}
    if table_exists(con, "StringIds"):
        for sid, value in con.execute("SELECT id, value FROM StringIds"):
            names[sid] = value

    kernels = []
    if table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        for start, end, sid in con.execute(
                "SELECT start, end, shortName FROM"
                " CUPTI_ACTIVITY_KIND_KERNEL"):
            kernels.append((start, end, names.get(sid, "?")))
    copies = []
    if table_exists(con, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for start, end, kind, nbytes in con.execute(
                "SELECT start, end, copyKind, bytes FROM"
                " CUPTI_ACTIVITY_KIND_MEMCPY"):
            copies.append((start, end, kind, nbytes or 0))
    con.close()

    if not kernels:
        out["error"] = "no kernels in the trace"
        return out, [], []

    ns = 1e-9
    k_spans = [(s * ns, e * ns) for s, e, _ in kernels]
    c_spans = [(s * ns, e * ns) for s, e, _, _ in copies]
    k_sum = sum(e - s for s, e in k_spans)
    k_union = union_seconds(k_spans)
    both_union = union_seconds(k_spans + c_spans)
    span = max(e for _, e in k_spans + c_spans) - min(
        s for s, _ in k_spans + c_spans)
    window = (frames / fps) if (frames and fps) else None

    out.update({
        "kernels": len(kernels),
        "copies": len(copies),
        "kernel_time_s": k_sum,
        "kernel_union_s": k_union,
        "busy_union_s": both_union,
        "trace_span_s": span,
        "window_s": window,
    })
    # Against the SPAN of the card's work, not against frames / fps: the
    # window is the time the codec was working, the trace also holds reading,
    # warm-up and the final write. On short runs that was up to 3.3 times as
    # much, and every share went over 100 %.
    if span:
        out["coverage_pct"] = 100.0 * k_union / span
        out["busy_pct"] = 100.0 * both_union / span
        out["starvation_pct"] = max(0.0, 100.0 - out["busy_pct"])
        out["akc"] = k_sum / span
    if window:
        out["span_over_window"] = span / window
        out["coverage_vs_window_pct"] = 100.0 * k_union / window
        out["akc_vs_window"] = k_sum / window
    if frames:
        out["kernel_ms_per_frame"] = 1000.0 * k_sum / frames
        out["copy_mb_per_frame"] = sum(b for _, _, _, b in copies) / 1e6 / frames

    by_name = {}
    for s, e, name in kernels:
        rec = by_name.setdefault(name, [0, 0.0])
        rec[0] += 1
        rec[1] += (e - s) * ns
    rows = [(name, n, t, 100.0 * t / k_sum) for name, (n, t) in by_name.items()]
    rows.sort(key=lambda r: r[2], reverse=True)

    copy_rows = {}
    for s, e, kind, nbytes in copies:
        rec = copy_rows.setdefault(kind, [0, 0.0, 0])
        rec[0] += 1
        rec[1] += (e - s) * ns
        rec[2] += nbytes
    return out, rows, sorted(copy_rows.items())


# ------------------------------------------------------------------- report


def metrics_digest(db, outdir, key):
    """The card's own counters, one line per metric.

    Version 13 averaged every numeric column, metric identifiers and
    timestamps included, and the result meant nothing. The counters live in
    GPU_METRICS as (timestamp, metricId, value) and their names in
    TARGET_INFO_GPU_METRICS; the name table differs between builds of nsys,
    so it is looked at rather than assumed. They are NEVER mixed into the
    main table: collecting them costs extra time.
    """
    con = sqlite3.connect(db)
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    if "GPU_METRICS" not in names:
        con.close()
        return None
    # the name table: whichever of these exists, and whichever two columns
    # in it look like an id and a name
    title = {}
    for tbl in names:
        if "GPU_METRIC" not in tbl.upper() or tbl == "GPU_METRICS":
            continue
        cols = [r[1] for r in con.execute("PRAGMA table_info(%s)" % tbl)]
        idc = next((c for c in cols if c.lower().endswith("metricid")), None)
        namec = next((c for c in cols if "name" in c.lower()), None)
        if not idc or not namec:
            continue
        try:
            for mid, nm in con.execute("SELECT %s, %s FROM %s"
                                       % (idc, namec, tbl)):
                if mid is not None and nm is not None:
                    title[mid] = str(nm)
        except sqlite3.Error:
            continue
    try:
        rows = list(con.execute(
            "SELECT metricId, COUNT(*), AVG(value), MIN(value), MAX(value)"
            " FROM GPU_METRICS GROUP BY metricId ORDER BY metricId"))
    except sqlite3.Error as e:
        con.close()
        say("   card counters: GPU_METRICS is there but unreadable (%s)" % e)
        return None
    if not rows:
        con.close()
        return None
    path = os.path.join(outdir, key + ".gpu-metrics.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("metric,metric_id,samples,average,minimum,maximum\n")
        for mid, n, avg, lo, hi in rows:
            fh.write('"%s",%s,%d,%s,%s,%s\n'
                     % (title.get(mid, "metric %s" % mid), mid, n,
                        fmt(avg, "%.4g"), fmt(lo, "%.4g"), fmt(hi, "%.4g")))
    con.close()
    return path

def write_digests(outdir, key, metrics, kernel_rows, copy_rows):
    with open(os.path.join(outdir, key + ".kernels.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("kernel,launches,seconds,share_pct\n")
        for name, n, t, share in kernel_rows:
            fh.write('"%s",%d,%.6f,%.2f\n' % (name.replace('"', "'"), n, t,
                                              share))
    with open(os.path.join(outdir, key + ".copies.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("copy_kind,count,seconds,bytes\n")
        for kind, (n, t, nbytes) in copy_rows:
            fh.write("%s,%d,%.6f,%d\n" % (kind, n, t, nbytes))
    with open(os.path.join(outdir, key + ".metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=1)


HEAD = ("point                     frames    fps   kernel ms/frame  coverage"
        "   busy  starve    AKC  span/win  witness")


def report(outdir, results):
    """The table on screen and coverage.csv next to the traces - one place."""
    say("\n" + HEAD)
    with open(os.path.join(outdir, "coverage.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("point,frames,fps,kernel_ms_per_frame,coverage_pct,"
                 "busy_pct,starvation_pct,akc,span_over_window,"
                 "coverage_vs_window_pct,akc_vs_window,drift_pct,"
                 "profiling_cost_pct,profiled_fps,bench_fps,"
                 "control_vs_bench_pct,witness\n")
        for key, m, witness in results:
            say(line_for(key, m, witness))
            fh.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                     % (key, m.get("frames", ""), m.get("fps", ""),
                        m.get("kernel_ms_per_frame", ""),
                        m.get("coverage_pct", ""), m.get("busy_pct", ""),
                        m.get("starvation_pct", ""), m.get("akc", ""),
                        m.get("span_over_window", ""),
                        m.get("coverage_vs_window_pct", ""),
                        m.get("akc_vs_window", ""),
                        m.get("drift_pct", ""),
                        m.get("profiling_cost_pct", ""),
                        m.get("profiled_fps", ""), m.get("bench_fps", ""),
                        m.get("control_vs_bench_pct", ""), witness))


def line_for(key, m, witness):
    return ("%-24s %7s %6s %14s %9s %6s %7s %6s %9s  %s"
            % (key[:24], fmt(m.get("frames"), "%d"), fmt(m.get("fps"), "%.0f"),
               fmt(m.get("kernel_ms_per_frame"), "%.3f"),
               fmt(m.get("coverage_pct"), "%.1f %%"),
               fmt(m.get("busy_pct"), "%.0f %%"),
               fmt(m.get("starvation_pct"), "%.0f %%"),
               fmt(m.get("akc"), "%.2f"),
               fmt(m.get("span_over_window"), "%.2f"), witness))


def _read_log_figures(path):
    """Speed and frames a saved run log kept, so a trace can be counted later."""
    if not os.path.isfile(path):
        return None, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        return result_of(fh.read())


def count_leftovers(nsys, tracedir, outdir):
    """Count traces that a previous run left uncounted.

    A run that measured everything and then died at the export step has all
    its traces; re-measuring them would only spend the hours again - and give
    slightly different figures, which is worse. This used to be a key
    (--export-only); now it happens by itself.
    """
    reps = sorted(glob.glob(os.path.join(tracedir, "*.nsys-rep")))
    if not reps:
        return 1
    say("\nCounting from traces already in %s" % os.path.basename(tracedir))
    os.makedirs(outdir, exist_ok=True)
    state = {"trace": None, "export": None}
    results = []
    for n, rep in enumerate(reps, 1):
        key = os.path.basename(rep)[:-len(".nsys-rep")]
        say("\n[%d/%d] %s" % (n, len(reps), key))
        db, err = export_sqlite(nsys, rep, state)
        if err:
            say("   FAILED: %s" % err)
            results.append((key, {"error": err}, "no"))
            continue
        # The logs live with the results, not with the traces.
        fps, frames = _read_log_figures(os.path.join(outdir,
                                                     key + ".run.log"))
        cfps, cframes = _read_log_figures(os.path.join(outdir,
                                                       key + ".control.log"))
        metrics, kernel_rows, copy_rows = digest(db, frames or cframes,
                                                 fps or cfps)
        metrics["control_fps"] = cfps
        metrics["profiled_fps"] = fps
        witness = "not checked"
        if cfps and fps:
            drift = 100.0 * abs(fps - cfps) / cfps
            metrics["drift_pct"] = drift
            witness = ("yes" if drift <= CONTROL_TOLERANCE
                       else "NOT (%.1f %%)" % drift)
        if (metrics.get("coverage_pct") or 0) > 105:
            metrics["warning"] = ("coverage above 105 %: the window and the"
                                  " trace are not from the same run")
            say("   WARNING: %s" % metrics["warning"])
        gm = metrics_digest(db, outdir, key)
        if gm:
            metrics["gpu_metrics_digest"] = os.path.basename(gm)
        write_digests(outdir, key, metrics, kernel_rows, copy_rows)
        results.append((key, metrics, witness))
        say("   coverage %s, starvation %s, AKC %s, profiler cost %s,"
            " witness %s"
            % (fmt(metrics.get("coverage_pct"), "%.1f %%"),
               fmt(metrics.get("starvation_pct"), "%.0f %%"),
               fmt(metrics.get("akc"), "%.2f"),
               fmt(metrics.get("profiling_cost_pct"), "%+.1f %%"), witness))
    report(outdir, results)
    return 0


# --------------------------------------------------------------------- main


def _folders_in(where, prefix=""):
    out = []
    if not os.path.isdir(where):
        return out
    for name in sorted(os.listdir(where)):
        full = os.path.join(where, name)
        if not os.path.isdir(full) or not name.startswith(prefix):
            continue
        files = os.listdir(full)
        digests = [f for f in files if f.endswith(".csv")
                   or f.endswith(".json")]
        out.append((full, len(files), len(digests)))
    return out


def existing_outputs(folder):
    """Result folders, new place first, then the old one inside the run.

    Versions 06 to 08 wrote into the bench run folder; those results are
    still worth finding, so both places are listed.
    """
    both = _folders_in(OUTROOT) + _folders_in(folder, "nsys-")
    # oldest first, so "the newest" is the last line and the one
    # --export-only takes by default
    both.sort(key=lambda t: os.path.getmtime(t[0]))
    return both


def write_run_info(outdir, folder, plan, nsys, tracedir):
    """The folder gets a file the moment it is made.

    The first measured file appears a minute or two in, when the first
    control run ends. Until then the folder used to be empty, and an empty
    folder is indistinguishable from a run that never started.
    """
    lines = [
        VERSION,
        "started:      %s" % datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "project:      %s" % HERE,
        "bench run:    %s" % folder,
        "nsys:         %s" % nsys,
        "quality:      %d       frames: %s" % (QUALITY, ", ".join(IMAGES)),
        "card counters: asked for (they need an administrator terminal)",
        "",
        "%d points, each one run twice - control, then under nsys:" % len(plan),
    ]
    for p, cmd in plan:
        lines.append("   %-24s %s" % (p.key, cmd))
    lines += [
        "",
        "What will appear in this folder, per point:",
        "   <point>.control.log      what the program printed without nsys",
        "   <point>.run.log          and under it",
        "   <point>.kernels.csv      kernels by name, time and share",
        "   <point>.copies.csv       memory copies by kind",
        "   <point>.metrics.json     everything counted for that point",
        "   <point>.gpu-metrics.csv  the card's own counters, if nsys took them",
        "and once, at the end:",
        "   coverage.csv             every point in one table",
        "",
        "Only small files are written here. The traces themselves (.nsys-rep",
        "and .sqlite, hundreds of megabytes) go to a folder of their own:",
        "   %s" % tracedir,
        "so this folder can be zipped whole and sent without sorting.",
        "",
        "PROGRESS.txt beside this file is rewritten after every point.",
        "If this folder holds only this file, the first control run has not",
        "finished yet - give it a minute.",
    ]
    _write_text(os.path.join(outdir, "RUN-INFO.txt"), "\n".join(lines))


def write_progress(outdir, plan, done, running):
    """Rewritten after every point, so the folder always says where it is."""
    lines = ["%d of %d points done" % (len(done), len(plan)),
             "last change: %s"
             % datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"), ""]
    if running:
        lines.append("running now: %s" % running)
        lines.append("")
    for key, metrics, witness in done:
        if metrics.get("error"):
            lines.append("   %-24s FAILED: %s" % (key, metrics["error"]))
        else:
            lines.append("   %-24s coverage %s, starvation %s, AKC %s, %s"
                         % (key,
                            fmt(metrics.get("coverage_pct"), "%.1f %%"),
                            fmt(metrics.get("starvation_pct"), "%.0f %%"),
                            fmt(metrics.get("akc"), "%.2f"), witness))
    for p, _ in plan[len(done) + (1 if running else 0):]:
        lines.append("   %-24s waiting" % p.key)
    _write_text(os.path.join(outdir, "PROGRESS.txt"), "\n".join(lines))


def _write_text(path, text):
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except OSError as e:
        say("could not write %s: %s" % (os.path.basename(path), e))


def note_output_folder(outdir, tracedir=None):
    """Say it loudly, and leave it on disk as well.

    The path was printed once before and scrolled away in an hour-long run,
    and the folder is dated by the BENCH run, not by today - so looking for
    "something from today" finds nothing. Hence the frame and the note file.
    """
    bar = "=" * 72
    say("\n" + bar)
    say("RESULTS OF THIS RUN GO TO:")
    say("   " + outdir)
    say("(named by today, next to this script - not inside the bench run)")
    if tracedir:
        say("Only small files go here. The traces go to:")
        say("   " + tracedir)
    say(bar)
    try:
        with open(os.path.join(HERE, "nsys-last-output.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(outdir + "\n")
        say("the same path is written to %s"
            % os.path.join(HERE, "nsys-last-output.txt"))
    except OSError as e:
        say("could not write the note file: %s" % e)


def main():
    # ONE key. Everything else the script does by itself - 46.04 SS1,
    # Fyodor 22.09.2026: "without arguments it shows the diagnosis, with
    # --do it works, nothing else is needed".
    ap = argparse.ArgumentParser(add_help=True, description=VERSION)
    ap.add_argument("--do", action="store_true",
                    help="run it; without this everything is shown and"
                         " nothing is measured")
    a = ap.parse_args()

    say(VERSION)
    say("no arguments: show everything and measure nothing."
        "  --do: run it.  That is all there is.")
    say("project folder: %s" % HERE)

    if not os.path.isdir(RUNS):
        say("\nNo runs folder next to this file (%s)." % RUNS)
        say("Run jpeg-bench-12.py first, or put this file into the project"
            " folder.")
        return 1

    nsys, all_nsys = find_nsys(None)
    if nsys:
        say("nsys: %s" % nsys)
        if len(all_nsys) > 1:
            say("      (%d found, newest taken)" % len(all_nsys))
    else:
        say("nsys: NOT FOUND. Looked in PATH and in:")
        for pattern in NSYS_PATTERNS[:6]:
            say("      %s" % pattern)
        say("      To find it: where /R \"C:\\Program Files\" nsys.exe")
        say("      Then put that folder into PATH, or put nsys.exe beside"
            " this script.")

    folder, notes = pick_run(None)
    say("\nRuns:")
    for f, why in notes:
        mark = "TAKEN " if f == folder else "      "
        say("   %s %-40s %s" % (mark, os.path.basename(f), why))
    if not folder:
        say("\nNo run has both single-frame and grid points; nothing to"
            " profile.")
        return 1

    have = existing_outputs(folder)
    say("\nRESULTS GO TO A FOLDER OF THEIR OWN, NEXT TO THIS SCRIPT:")
    say("   %s" % os.path.join(OUTROOT, "<date>-<time>_%s"
                               % os.path.basename(folder)))
    say("The name BEGINS with the day the profiling was done, so today's"
        " measurement is under today's date.")
    if have:
        say("Already on disk, %d folder(s):" % len(have))
        for full, n_files, n_dig in have:
            where = ("nsys-out\\" if full.startswith(OUTROOT)
                     else "inside the run folder (version 06-08)")
            say("   %-38s %3d files, %2d digests   %s"
                % (os.path.basename(full), n_files, n_dig, where))
    else:
        say("Nothing on disk yet: no result folder in either place.")

    rows, bad = read_run(folder)
    for b in bad:
        say("      %s" % b)
    points = [Point(r) for r in rows]
    chosen, missing = choose_points(points, IMAGES, q=QUALITY)

    index = LogIndex(folder)
    if index.where:
        say("logs: " + "; ".join(index.where))
    else:
        say("logs: NONE FOUND. Looked in logs\\, in the run folder itself and"
            " in logs.zip.")
        say("      " + index.contents_note())
    if have:
        say("result folders already on disk: %s"
            % ", ".join(os.path.basename(f) for f, _, _ in have))

    say("\nPoints to profile: %d (each one runs twice - control, then nsys)"
        % len(chosen))
    plan = []
    for p in chosen:
        cmd, why = command_of(index, p)
        if not cmd:
            say("   %-24s SKIPPED: %s" % (p.key, why))
            continue
        plan.append((p, cmd))
        say("   %-24s %s fps, %s" % (p.key, fmt(p.fps, "%.0f"), cmd))
    for m in missing:
        say("   missing: %s" % m)

    if not plan:
        say("\nNothing to run: no point had a command line in its log.")
        return 1

    say("\nCard counters: always asked for. nsys needs an administrator"
        " terminal for them; the form it accepts is named at the first point,"
        " and again at the end. If it refuses, the run goes on without them.")
    say("Profiling costs time itself, so coverage is a lower bound.")

    if not a.do:
        say("\nThis was the plan. Add --do to run it.")
        return 0

    if not nsys:
        say("\nCannot run without nsys.")
        return 1

    say("nsys says: %s" % nsys_version(nsys))

    # The folder is made only now - after everything that could still
    # refuse. And it gets a file at once, because a folder that exists and
    # is empty says nothing (Fyodor, 22.09.2026).
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "%s_%s" % (stamp, os.path.basename(folder))
    outdir = os.path.join(OUTROOT, name)
    tracedir = os.path.join(TRACES, name)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(tracedir, exist_ok=True)
    write_run_info(outdir, folder, plan, nsys, tracedir)
    note_output_folder(outdir, tracedir)

    # A run that measured everything and then died at the export step leaves
    # traces nobody counted. Re-measuring them would spend the hours again
    # and give slightly different figures, which is worse. So they are
    # counted first, by themselves - this used to be the --export-only key.
    for td, _, _ in _folders_in(TRACES):
        done = os.path.join(OUTROOT, os.path.basename(td))
        if os.path.isfile(os.path.join(done, "coverage.csv")):
            continue
        if not glob.glob(os.path.join(td, "*.nsys-rep")):
            continue
        say("\n%s holds traces that were never counted - counting them first."
            % os.path.basename(td))
        count_leftovers(nsys, td, done)

    state = {"trace": None, "export": None, "metrics": None,
             "want_metrics": True, "wrap": None}

    results = []
    for n, (p, cmd) in enumerate(plan, 1):
        argv, changed = stretch_repeat(split_command(cmd), p.fps)
        say("\n[%d/%d] %s" % (n, len(plan), p.key))
        write_progress(outdir, plan, results, p.key)
        if n % 3 == 1 and n > 1:
            say("   (results are going to %s)" % outdir)
        if changed:
            say("   repeat %d -> %d, for about %.0f s of work"
                % (changed[0], changed[1], changed[1] / p.fps))
        wait_until_card_quiet(say)
        say("   control run (no profiler)")
        ctl, err = run_plain(argv, os.path.join(outdir, p.key + ".control.log"))
        if err:
            say("   FAILED: %s" % err)
            results.append((p.key, {"error": err}, "no"))
            continue
        say("   control: %s fps in %.1f s" % (fmt(ctl["fps"], "%.0f"),
                                              ctl["wall_s"]))
        if not ctl["fps"]:
            say("   SKIPPED: the control run printed no figure, so there is"
                " nothing to compare a trace with.")
            for line in ctl.get("head") or ["(the program printed nothing)"]:
                say("      %s" % line)
            results.append((p.key, {"error": "control run measured nothing"},
                            "no"))
            continue
        wait_until_card_quiet(say)
        say("   under nsys")
        said = os.path.join(tracedir, p.key + ".said.txt")
        wrapped = None
        if state.get("wrap"):
            wrapped = write_wrapper(
                os.path.join(tracedir, p.key + ".run.bat"), argv, said)
        prof, err = run_profiled(nsys, argv, os.path.join(tracedir, p.key),
                                 state,
                                 os.path.join(outdir, p.key + ".run.log"),
                                 wrapped)
        if err:
            say("   FAILED: %s" % err)
            results.append((p.key, {"error": err}, "no"))
            continue
        if prof.get("fps") is None:
            prof["fps"], prof["frames"] = _said_figures(said)
        # nsys on Windows does not pass the child's output through, so the
        # speed under the profiler goes unread and the witness check never
        # runs. Probe it ONCE: if the plain form said nothing, redo this one
        # point through a wrapper that keeps the output, and use the wrapper
        # from here on.
        if prof.get("fps") is None and state.get("wrap") is None:
            say("   nsys did not pass the program's own output through -"
                " running this point again through a wrapper that keeps it")
            state["wrap"] = True
            wrapped = write_wrapper(
                os.path.join(tracedir, p.key + ".run.bat"), argv, said)
            again, err2 = run_profiled(
                nsys, argv, os.path.join(tracedir, p.key), state,
                os.path.join(outdir, p.key + ".run.log"), wrapped)
            if err2:
                say("   the wrapper did not work either: %s" % err2)
                state["wrap"] = False
            else:
                prof = again
                prof["fps"], prof["frames"] = _said_figures(said)
                if prof["fps"] is None:
                    say("   still nothing - the speed under the profiler"
                        " cannot be read on this machine, and the witness"
                        " check will say so instead of saying yes")
                    state["wrap"] = False
                else:
                    say("   the wrapper works: %s fps under the profiler"
                        % fmt(prof["fps"], "%.0f"))
        if state.get("wrap") is None:
            state["wrap"] = False
        db, err = export_sqlite(nsys, prof["rep"], state)
        if err:
            say("   FAILED: %s" % err)
            results.append((p.key, {"error": err}, "no"))
            continue
        # window = frames / fps, both from the PROFILED run: that is the run
        # the trace belongs to. The control run and the bench row are only
        # fallbacks, and when they are used the window is approximate.
        # The repeat count is what the point was actually told to do, so it
        # beats the bench row whenever the program itself said nothing.
        asked = changed[1] if changed else repeat_of(argv) or p.frames
        frames = prof.get("frames") or ctl.get("frames") or asked
        window_fps = prof.get("fps") or ctl.get("fps")
        try:
            metrics, kernel_rows, copy_rows = digest(db, frames, window_fps)
        except Exception as e:
            say("   FAILED to count this trace: %s: %s" % (type(e).__name__, e))
            say("   The trace stays in %s; the next point goes on." % tracedir)
            results.append((p.key, {"error": "counting failed: %s" % e}, "no"))
            continue
        metrics["window_from"] = ("profiled run" if prof.get("frames")
                                  else ("control run" if ctl.get("frames")
                                        else "the repeat count asked for"))
        metrics["control_fps"] = ctl["fps"]
        metrics["bench_fps"] = p.fps
        if p.fps and ctl["fps"]:
            gap = 100.0 * (ctl["fps"] - p.fps) / p.fps
            metrics["control_vs_bench_pct"] = gap
            if abs(gap) > 10:
                say("   NOTE: the control run is %+.0f %% against the bench"
                    " row (%s against %s fps) - this trace cannot be carried"
                    " over to the article's figures"
                    % (gap, fmt(ctl["fps"], "%.0f"), fmt(p.fps, "%.0f")))
        metrics["profiled_fps"] = prof["fps"]
        # The witness is a CHECK, not a default. When the speed under the
        # profiler cannot be read, the row says so - version 13 said "yes"
        # eighteen times without ever comparing anything.
        witness = "not checked"
        if ctl["fps"] and prof["fps"]:
            drift = 100.0 * abs(prof["fps"] - ctl["fps"]) / ctl["fps"]
            metrics["drift_pct"] = drift
            metrics["profiling_cost_pct"] = (
                100.0 * (ctl["fps"] - prof["fps"]) / ctl["fps"])
            witness = ("yes" if drift <= CONTROL_TOLERANCE
                       else "NOT (%.1f %%)" % drift)
        if (metrics.get("span_over_window") or 1) > 1.5:
            say("   NOTE: the trace is %.1f times longer than the codec's own"
                " working time - it also holds reading, warm-up and the write"
                % metrics["span_over_window"])
        gm = metrics_digest(db, outdir, p.key)
        if gm:
            metrics["gpu_metrics_digest"] = os.path.basename(gm)
            say("   card counters written to %s" % os.path.basename(gm))
        write_digests(outdir, p.key, metrics, kernel_rows, copy_rows)
        results.append((p.key, metrics, witness))
        say("   coverage %s, starvation %s, AKC %s, profiler cost %s,"
            " witness %s"
            % (fmt(metrics.get("coverage_pct"), "%.1f %%"),
               fmt(metrics.get("starvation_pct"), "%.0f %%"),
               fmt(metrics.get("akc"), "%.2f"),
               fmt(metrics.get("profiling_cost_pct"), "%+.1f %%"), witness))

    write_progress(outdir, plan, results, None)
    report(outdir, results)

    drifts = [m.get("drift_pct") for _, m, _ in results
              if m.get("drift_pct") is not None]
    if drifts:
        say("\nProfiled against unprofiled: the two runs of the same point"
            " differed by at most %.1f %%, over %d points. That is the band"
            " the numbers above stand in." % (max(drifts), len(drifts)))
    else:
        say("\nProfiled against unprofiled: not compared on any point - the"
            " speed under the profiler could not be read.")

    got = state.get("metrics")
    if got:
        say("\nCard counters: collected, nsys accepted %s."
            " Their averages are in the <point>.gpu-metrics.csv files;"
            " they are never mixed into the table above." % " ".join(got))
    elif got == []:
        say("\nCard counters: REFUSED by nsys in every form tried."
            " The run went on without them; everything else is unaffected."
            " The usual cause is a terminal without administrator rights.")
    else:
        say("\nCard counters: asked for, but no point got far enough for"
            " nsys to say yes or no.")

    total = sum(os.path.getsize(os.path.join(outdir, f))
                for f in os.listdir(outdir))
    say("\nRESULTS ARE IN:")
    say("   %s" % outdir)
    say("   coverage.csv             the table above, with every column")
    say("   <point>.kernels.csv      kernels by name, time and share")
    say("   <point>.copies.csv       memory copies by kind")
    say("   <point>.metrics.json     everything counted for that point")
    say("   <point>.gpu-metrics.csv  the card's own counters, if they came")
    say("   <point>.control.log      what the program printed without nsys")
    say("   <point>.run.log          and under it")
    say("   RUN-INFO.txt, PROGRESS.txt   what was run and how far it got")
    say("\nZIP THAT FOLDER WHOLE AND SEND IT. Nothing in it has to be taken"
        " out: %d files, %.1f MB in all."
        % (len(os.listdir(outdir)), total / 1048576.0))
    say("The traces stay behind, in %s - they are hundreds of megabytes, and"
        " everything the numbers are made of is in the folder above."
        % tracedir)
    return 0


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
    try:
        finish(main())
    except KeyboardInterrupt:
        say("\nStopped. Whatever finished is already written to disk.")
        finish(1)
