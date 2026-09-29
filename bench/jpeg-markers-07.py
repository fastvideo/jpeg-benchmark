#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jpeg-markers-07.py
# version 2026-09-25.1 of 25.09.2026, cancels version 06
#
# NEW IN VERSION 07 (25.09.2026): frame 8K. The experiment runs on every
# frame the bench run holds - 2K, 4K and, when jpeg-bench-13 measured it,
# 8K. "Who is on the card" lists every compute process (version 06 cut the
# list at 12 lines and never reached the one that loaded the card).
#
# NEW: before every measured repeat the card is checked for someone else's load, and the repeat waits while it is busy.
#
# The direct marker experiment: ONE image, three streams, one decoder.
#
# Everything in the article so far compares two pairs of codec-plus-decoder.
# This measures the thing itself - what restart markers are worth - by giving
# our own decoder the same picture three times over:
#
#   as-is        what our encoder wrote (a marker every 10 MCU rows)
#   no markers   jpegtran -restart 0
#   every MCU    jpegtran -restart 1B - the most the format allows
#
# All three hold THE SAME DCT COEFFICIENTS. jpegtran only re-writes the
# entropy layer, so the decoded picture is identical to the last bit and the
# only difference is how many independent pieces the stream falls into. That
# is what makes this a clean experiment and the section 4 comparison an
# unclean one.
#
# What to expect: "no markers" is the floor - nothing to divide, so the card
# has nothing to fill itself with. "Every MCU" is the ceiling - the most
# parallelism the format can give, paid for in file size. The stream our
# encoder writes sits between them, and the point of the measurement is to
# show HOW MUCH closer to the ceiling it is, and what it costs in bytes.
#
# WHAT IS NEW IN VERSION 05 (24.09.2026)
#
#   1. nvJPEG DECODES THE SAME THREE FILES. Our encoder's file, the file
#      without markers and the file with a marker after every MCU, all made
#      from OUR encoder's output, go to both decoders. For ours it shows what
#      the markers give; for nvJPEG it shows whether its decoder makes any
#      use of them. Points for nvJPEG: one frame at a time, and its native
#      batch of 128 - the mode where it runs Huffman on the card.
#   2. Each variant gets its own image count, so one repeat lasts about
#      TARGET_SECONDS (version 04); TARGET_SECONDS is 8 s, not 20.
#   3. The one-thread point is taken from rows whose path is "single" only.
#      nvJPEG's native batch of 1 also has one thread and one frame, and
#      mixing the two would have given a median of two different things.
#
# WHAT IS NEW IN VERSION 02 (23.09.2026)
#
#   1. JPEGTRAN IS LOOKED FOR WHERE THE MEASURED PROGRAM LIVES. Fyodor keeps
#      it next to JpegSample.exe, which is the obvious place, and version 01
#      looked everywhere but there. The folder is not guessed: it is taken
#      from the command line the bench recorded (46.04 SS10).
#   2. EVERY VARIANT IS MEASURED SEVERAL TIMES AND THE SCATTER IS SHOWN.
#      Version 01 ran each one ONCE. On 23.09 the repeats of a single point
#      of this same program came out 2817, 3015 and 3674 frames per second -
#      a scatter of 30 per cent - so one run is not a measurement at all, and
#      a comparison of three single runs is worthless. Rounds are added while
#      the scatter is above SPREAD_LIMIT; if it stays there, no comparison is
#      printed for that point.
#   3. THE THREE VARIANTS ALTERNATE, one repeat each per round, so anything
#      that drifts through a point is shared between them instead of landing
#      on whichever went last.
#   4. The number of INDEPENDENT PIECES is printed for every variant - MCUs
#      in the frame divided by the interval - because that is the quantity
#      the speed is supposed to follow.
#
# WHAT IT DOES NOT DO
#
#   - it does not re-encode anything. jpegtran is lossless here, and the
#     script checks that: the decoded output of all three variants must match
#     byte for byte, and if it does not, the run stops;
#   - it does not invent the decoder's command line. That is read from the
#     bench's own results.jsonl, exactly as nsys-jpeg-14.py does, and only
#     the input file and the repeat count are substituted (46.04 SS10);
#   - it does not touch either codec.
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
#     python jpeg-markers-03.py            show everything, measure nothing
#     python jpeg-markers-03.py --do       run it
#
#   Two states, as memo 46.04 SS1 says. Results go to
#   <project>\markers-out\<date>-<time>\ - small files only, zip it whole.

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

VERSION = "jpeg-markers-07 of 25.09.2026"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUTROOT = os.path.join(HERE, "markers-out")

# At least this much decoder work per point, same rule as the profiler.
TARGET_SECONDS = 8.0
MAX_FRAMES = 150000
# Each variant gets its own image count, so that one repeat of it lasts about
# TARGET_SECONDS. Version 03 gave all three the count worked out for the
# file as our encoder wrote it; the file without markers decodes 4 to 55
# times slower, so its repeats ran for up to ten minutes each and the test
# took five hours on 23.09.2026 instead of half an hour. The count is found
# by one short trial run of this many images, which is not recorded.
CALIB_FRAMES = 300
REPEATS = 3                 # rounds always done
MAX_ROUNDS = 7              # and at most this many when the scatter is wide
# Between the fastest and the slowest repeat of one variant. Taken from
# bench-07.py, where the rule has been in force since 31.08.2026.
SPREAD_LIMIT = 7.0

# Tried in order. jpegtran comes with libjpeg-turbo; on Windows it is usually
# under Program Files, and it may also simply be on PATH.
JPEGTRAN_PATTERNS = [
    r"C:\libjpeg-turbo64\bin\jpegtran.exe",
    r"C:\libjpeg-turbo\bin\jpegtran.exe",
    r"C:\Program Files\libjpeg-turbo64\bin\jpegtran.exe",
    r"C:\Program Files\libjpeg-turbo\bin\jpegtran.exe",
    r"C:\Program Files (x86)\libjpeg-turbo\bin\jpegtran.exe",
]

# name -> the jpegtran options that make it. None means "leave the file alone".
VARIANTS = [
    ("as-is", None,
     "what our encoder wrote"),
    ("no-markers", ["-restart", "0"],
     "every marker removed - nothing to divide"),
    ("every-mcu", ["-restart", "1B"],
     "a marker after every MCU - the most the format allows"),
]

RE_FPS = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*FPS", re.I)




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
    """Every process that computes on the card, with its memory."""
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


# ------------------------------------------------------------ finding things


def spread_of(values):
    """How far the repeats of one variant are from each other, per cent."""
    if not values:
        return 0.0
    med = sorted(values)[len(values) // 2]
    return 100.0 * (max(values) - min(values)) / med if med else 0.0


def find_tool(name, patterns, beside=()):
    """Beside the measured program first, then PATH, then the usual places.

    "Beside" is not a guess: the folder comes from the command line the bench
    recorded. Fyodor keeps jpegtran next to JpegSample.exe, and version 01
    looked everywhere except there.
    """
    found = []
    for d in beside:
        p = os.path.join(d, name)
        for cand in (p, p + ".exe"):
            if os.path.isfile(cand):
                found.append(cand)
    p = os.path.join(HERE, name)
    for cand in (p, p + ".exe"):
        if os.path.isfile(cand):
            found.append(cand)
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d.strip('"'), name)
        for cand in (p, p + ".exe"):
            if os.path.isfile(cand):
                found.append(cand)
    for pattern in patterns:
        found += [p for p in glob.glob(pattern) if os.path.isfile(p)]
    seen, out = set(), []
    for p in found:
        k = os.path.normcase(os.path.abspath(p))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


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


def pick_run():
    """The newest bench run that holds a Fastvideo decode command line."""
    if not os.path.isdir(RUNS):
        return None, []
    pointed = pointed_run()
    if pointed:
        return pointed, [(pointed, "named in " + POINTER)]
    notes = []
    best = None
    for folder in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if not os.path.isdir(folder):
            continue
        rows = read_rows(folder)
        dec = [r for r in rows
               if r.get("codec") == "fv" and r.get("direction") == "D"
               and r.get("cmd")]
        if not dec:
            notes.append((folder, "no Fastvideo decode rows"))
            continue
        notes.append((folder, "%d decode rows" % len(dec)))
        best = folder
    return best, notes


def read_rows(folder):
    path = os.path.join(folder, "results.jsonl")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


# Frames, in order. 8K is new on 25.09 and is there only when the bench run
# measured it.
IMAGES = ("2k", "4k", "8k")
OPTIONAL_IMAGES = {"8k"}

# nvJPEG's second point: its native batch of this size (Huffman on the card)
NV_BATCH = 128


def decode_commands(folder, image, codec="fv"):
    """Two command lines for this frame: one thread, and the best of the grid.

    Read from the bench's own records, never rebuilt. The best grid point is
    taken by the MEDIAN of its repeats, not by the fastest one. For nvJPEG
    the second point is its native batch of NV_BATCH instead.
    """
    rows = [r for r in read_rows(folder)
            if r.get("codec") == codec and r.get("direction") == "D"
            and r.get("image") == image and r.get("q") == 90 and r.get("cmd")
            and (r.get("sub") or "444") == "444"]
    if not rows:
        return {}
    out = {}
    single = [r for r in rows if r.get("threads") == 1 and r.get("batch") == 1
              and (r.get("path") or "single") == "single"]
    if single:
        out["single"] = (median_row(single), 1)
    if codec == "nv":
        nb = [r for r in rows if r.get("path") == "native_batch"
              and r.get("batch") == NV_BATCH
              and (r.get("variant") or "gpu") == "gpu"]
        if nb:
            out["batch%d" % NV_BATCH] = (median_row(nb), 1)
        return out
    grid = {}
    for r in rows:
        if r.get("threads") == 1 and r.get("batch") == 1:
            continue
        grid.setdefault((r.get("threads"), r.get("batch")), []).append(r)
    if grid:
        best = max(grid.values(), key=lambda g: median_fps(g))
        t = best[0].get("threads")
        out["threads%s" % t] = (median_row(best), t)
    return out


def median_fps(rows):
    v = sorted(r.get("fps") or 0 for r in rows)
    return v[len(v) // 2]


def median_row(rows):
    v = sorted(rows, key=lambda r: r.get("fps") or 0)
    return v[len(v) // 2]


# ------------------------------------------------------------- the variants


def split_command(cmd):
    """Windows-style splitting: quotes hold spaces together."""
    out, cur, quoted = [], "", False
    for ch in cmd:
        if ch == '"':
            quoted = not quoted
        elif ch.isspace() and not quoted:
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def substitute(argv, image_path, frames):
    """Same command line, other input file and other repeat count."""
    out = list(argv)
    for i, a in enumerate(out[:-1]):
        if a.lower() in ("-i", "--input"):
            out[i + 1] = image_path
        elif a.lower() in ("-repeat", "--repeat"):
            out[i + 1] = str(frames)
    return out


def input_of(argv):
    for i, a in enumerate(argv[:-1]):
        if a.lower() in ("-i", "--input"):
            return argv[i + 1]
    return None


def make_variant(jpegtran, src, dst, options):
    """One jpegtran pass. Lossless: only the entropy layer is rewritten."""
    if options is None:
        with open(src, "rb") as a, open(dst, "wb") as b:
            b.write(a.read())
        return None
    argv = [jpegtran, "-copy", "none"] + options + ["-outfile", dst, src]
    p = subprocess.Popen(argv, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    out, _ = p.communicate(timeout=300)
    if p.returncode != 0 or not os.path.isfile(dst):
        return ("jpegtran exit code %s on %s. What it said:\n       %s"
                % (p.returncode, " ".join(argv),
                   out.decode("utf-8", "replace").strip() or "(nothing)"))
    return None


def mcus_of(path):
    """How many MCUs the frame holds, by its own header."""
    w, h, f = frame_of(path)
    if not w or not h or not f:
        return 0
    hmax = max(x[0] for x in f) or 1
    vmax = max(x[1] for x in f) or 1
    return (-(-w // (8 * hmax))) * (-(-h // (8 * vmax)))


def pieces_of(path):
    """Independent pieces: MCUs divided by the interval; 1 without markers."""
    n = mcus_of(path)
    if not n:
        return 0
    ri = restart_interval(path)
    return -(-n // ri) if ri else 1


def frame_of(path):
    """Width, height and sampling factors, out of the file's own SOF marker."""
    with open(path, "rb") as fh:
        data = fh.read()
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:
            break
        length = (data[i + 2] << 8) | data[i + 3]
        if marker in (0xC0, 0xC1, 0xC2):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            f = []
            for c in range(data[i + 9]):
                b = data[i + 10 + c * 3 + 1]
                f.append((b >> 4, b & 15))
            return w, h, f
        i += 2 + length
    return 0, 0, []


def restart_interval(path):
    """The restart interval written in the file, from the DRI marker."""
    with open(path, "rb") as fh:
        data = fh.read()
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:          # start of scan - the headers are over
            break
        length = (data[i + 2] << 8) | data[i + 3]
        if marker == 0xDD and length >= 4:
            return (data[i + 4] << 8) | data[i + 5]
        i += 2 + length
    return 0


# --------------------------------------------------------------- the running


def run(argv, log_path, cwd):
    t0 = time.time()
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, cwd=cwd)
        out, _ = p.communicate(timeout=3600)
    except FileNotFoundError:
        return None, "program not found: %s" % argv[0]
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return None, "timed out after an hour"
    text = out.decode("utf-8", "replace")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(argv) + "\n(in %s)\n\n" % cwd + text)
    fps = None
    m = RE_FPS.search(text)
    if m:
        fps = float(m.group(1).replace(",", "."))
    for line in text.splitlines():
        if line.strip().startswith("RESULT"):
            try:
                fps = json.loads(
                    line.split("RESULT", 1)[1].strip()).get("fps", fps)
            except ValueError:
                pass
    head = [l.rstrip() for l in text.splitlines() if l.strip()][:6]
    return {"fps": fps, "wall_s": time.time() - t0, "head": head}, None


def decode_to_file(argv, out_path, log_path):
    """One decode, one frame, written out - for the byte-for-byte check."""
    argv = list(argv)
    for i, a in enumerate(argv[:-1]):
        if a.lower() in ("-repeat", "--repeat"):
            argv[i + 1] = "1"
        elif a.lower() in ("-o", "--output"):
            argv[i + 1] = out_path
    argv = [a for a in argv if a != "-discard"]
    return run(argv, log_path, HERE)


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(add_help=True, description=VERSION)
    ap.add_argument("--do", action="store_true",
                    help="run it; without this everything is shown and"
                         " nothing is measured")
    a = ap.parse_args()

    say(VERSION)
    say("no arguments: show everything and measure nothing."
        "  --do: run it.  That is all there is.")
    say("project folder: %s" % HERE)

    folder, notes = pick_run()
    say("\nRuns:")
    for f, why in notes:
        say("   %s %-40s %s" % ("TAKEN " if f == folder else "      ",
                                os.path.basename(f), why))
    if not folder:
        say("\nNo bench run holds a Fastvideo decode command line."
            " Run jpeg-bench-13.py first.")
        return 1

    say("\nRESULTS GO TO A FOLDER OF THEIR OWN, NEXT TO THIS SCRIPT:")
    say("   %s" % os.path.join(OUTROOT, "<date>-<time>"))
    say("Small files only - zip that folder whole and send it.")

    plan = []
    for image, codec in [(im, c) for im in IMAGES for c in ("fv", "nv")]:
        cmds = decode_commands(folder, image, codec)
        if not cmds and image in OPTIONAL_IMAGES:
            if codec == "fv":
                say("\n%s: not in this bench run - skipped (it is measured"
                    " only when jpeg-bench-13 had the frame)" % image.upper())
            continue
        if not cmds:
            say("\n%s: no %s decode rows in this run"
                % (image.upper(), "Fastvideo" if codec == "fv" else "nvJPEG"))
            continue
        for how, (row, threads) in sorted(cmds.items()):
            argv = split_command(row["cmd"])
            src = input_of(argv)
            if not src:
                say("   %s %s: the command line has no -i" % (image, how))
                continue
            full = src if os.path.isabs(src) else os.path.join(HERE, src)
            if not os.path.isfile(full):
                say("   %s %s: no such file: %s" % (image, how, full))
                continue
            plan.append({"image": image, "codec": codec, "how": how,
                         "threads": threads, "batch": row.get("batch") or 1,
                         "argv": argv, "src": full, "fps": row.get("fps")})

    if not plan:
        say("\nNothing to measure.")
        return 1

    say("\nPoints: %d (frame, decoder and way to run) x %d variants, %d"
        " rounds and up to %d when the repeats scatter more than %.0f %%"
        % (len(plan), len(VARIANTS), REPEATS, MAX_ROUNDS, SPREAD_LIMIT))
    lo = len(plan) * len(VARIANTS) * REPEATS
    hi = len(plan) * len(VARIANTS) * MAX_ROUNDS
    say("That is %d to %d measurements of about %.0f s each - from %.0f to"
        " %.0f minutes." % (lo, hi, TARGET_SECONDS,
                            lo * (TARGET_SECONDS + 2) / 60.0,
                            hi * (TARGET_SECONDS + 2) / 60.0))
    for p in plan:
        say("   %-3s %-2s %-10s %s" % (p["image"], p["codec"], p["how"],
                                     " ".join(p["argv"])))
    # The folder of the measured program, taken from the command line the
    # bench recorded. The first element may be an interpreter with no path,
    # so the first element that names an existing folder wins.
    beside = set()
    for q in plan:
        for a_ in q["argv"]:
            d = os.path.dirname(a_)
            if d and os.path.isdir(d):
                beside.add(d)
                break
    beside = sorted(beside)
    tran = find_tool("jpegtran", JPEGTRAN_PATTERNS, beside)
    if tran:
        say("\njpegtran: %s" % tran[0])
    else:
        say("\njpegtran: NOT FOUND. Looked next to the measured program:")
        for d in beside:
            say("      %s" % d)
        say("   next to this script, on PATH, and in:")
        for q in JPEGTRAN_PATTERNS:
            say("      %s" % q)
        say("   It comes with libjpeg-turbo:"
            " https://libjpeg-turbo.org/Documentation/OfficialBinaries")

    say("\nVariants of every frame, all three holding the same coefficients:")
    for name, opts, why in VARIANTS:
        say("   %-12s %-22s %s"
            % (name, " ".join(opts) if opts else "(the file itself)", why))

    if not a.do:
        say("\nThis was the plan. Add --do to run it.")
        return 0
    if not tran:
        say("\nCannot run without jpegtran.")
        return 1

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(OUTROOT, stamp)
    imgdir = os.path.join(HERE, "markers-img", stamp)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(imgdir, exist_ok=True)
    say("\n" + "=" * 72)
    say("RESULTS GO TO:")
    say("   " + outdir)
    say("(the variants themselves go to %s - they are big)" % imgdir)
    say("=" * 72)

    # ---- make the variants, and check they really are the same picture
    made = {}
    sizes = []
    for image in sorted({p["image"] for p in plan}):
        fv = [p for p in plan if p["image"] == image and p["codec"] == "fv"]
        if not fv:
            say("\n%s: no Fastvideo point, so no file of our encoder to make"
                " the variants from - this frame is skipped" % image.upper())
            plan = [p for p in plan if p["image"] != image]
            continue
        src = fv[0]["src"]
        base = os.path.splitext(os.path.basename(src))[0]
        for name, opts, _ in VARIANTS:
            dst = os.path.join(imgdir, "%s_%s.jpg" % (base, name))
            err = make_variant(tran[0], src, dst, opts)
            if err:
                say("\nFAILED making %s of %s: %s" % (name, image, err))
                return 1
            made[(image, name)] = dst
            sizes.append((image, name, os.path.getsize(dst),
                          restart_interval(dst), pieces_of(dst)))
            say("   %-3s %-12s %10d bytes, interval %d, %d independent pieces"
                % (image, name, os.path.getsize(dst), restart_interval(dst),
                   pieces_of(dst)))

    with open(os.path.join(outdir, "sizes.csv"), "w", encoding="utf-8") as fh:
        fh.write("image,variant,bytes,restart_interval,independent_pieces,"
                 "vs_as_is_pct\n")
        for image, name, n, ri, pieces in sizes:
            ref = next(x[2] for x in sizes
                       if x[0] == image and x[1] == "as-is")
            fh.write("%s,%s,%d,%d,%d,%.2f\n"
                     % (image, name, n, ri, pieces, 100.0 * (n - ref) / ref))

    # ---- the same picture? decode one frame of each and compare bytes
    say("\nChecking that all three variants decode to the same picture.")
    for image in sorted({p["image"] for p in plan}):
        p0 = next(p for p in plan if p["image"] == image
                  and p["codec"] == "fv")
        ref_bytes = None
        for name, _, _ in VARIANTS:
            out_ppm = os.path.join(imgdir, "check_%s_%s.ppm" % (image, name))
            argv = substitute(p0["argv"], made[(image, name)], 1)
            res, err = decode_to_file(
                argv, out_ppm,
                os.path.join(outdir, "check_%s_%s.log" % (image, name)))
            if err or not os.path.isfile(out_ppm):
                say("   %s %s: could not decode for the check (%s)"
                    % (image, name, err or "no output file"))
                say("   Stopping: without this check the measurement proves"
                    " nothing.")
                return 1
            with open(out_ppm, "rb") as fh:
                data = fh.read()
            if ref_bytes is None:
                ref_bytes = data
                say("   %s as-is: %d bytes decoded" % (image, len(data)))
            elif data != ref_bytes:
                say("   %s %s: THE PICTURE IS NOT THE SAME. jpegtran was"
                    " supposed to be lossless here. Stopping." % (image, name))
                return 1
            else:
                say("   %s %-12s same picture, byte for byte" % (image, name))
            try:
                os.remove(out_ppm)
            except OSError:
                pass

    # ---- measure: the three variants ALTERNATE, one repeat each per round,
    # and rounds are added while the repeats of a variant scatter too widely
    rows = []
    for n, p in enumerate(plan, 1):
        say("\n[%d/%d] %s %s %s" % (n, len(plan), p["image"], p["codec"],
                                   p["how"]))
        low = max(100, int(p.get("batch") or 1))
        base_frames = min(max(int(TARGET_SECONDS * (p["fps"] or 500)), low),
                          MAX_FRAMES)
        frames = {}
        stopped = False
        for name, _, _ in VARIANTS:
            if name == "as-is" and p["fps"] and p["codec"] == "fv":
                frames[name] = base_frames
                continue
            argv = substitute(p["argv"], made[(p["image"], name)],
                              max(CALIB_FRAMES, low))
            res, err = run(argv, os.path.join(
                outdir, "%s_%s_%s_%s_trial.log"
                % (p["image"], p["codec"], p["how"], name)), HERE)
            if err or not res or not res["fps"]:
                say("   %s: the short trial gave no figure (%s) - the count"
                    " for the file as written is used" % (name, err or "no fps"))
                frames[name] = base_frames
                continue
            frames[name] = min(max(int(TARGET_SECONDS * res["fps"]), low),
                               MAX_FRAMES)
        say("   images per repeat: %s  (about %.0f s each)"
            % (", ".join("%s %d" % (nm, frames[nm]) for nm, _, _ in VARIANTS),
               TARGET_SECONDS))
        got = {name: [] for name, _, _ in VARIANTS}
        for rnd in range(MAX_ROUNDS):
            for name, _, _ in VARIANTS:
                argv = substitute(p["argv"], made[(p["image"], name)],
                                  frames[name])
                wait_until_card_quiet(say)
                res, err = run(argv, os.path.join(
                    outdir, "%s_%s_%s_%s_%d.log"
                    % (p["image"], p["codec"], p["how"], name, rnd)), HERE)
                if err:
                    say("   %s FAILED: %s" % (name, err))
                    stopped = True
                    break
                if not res["fps"]:
                    say("   %s: the program printed no figure. What it said:"
                        % name)
                    for line in res["head"] or ["(nothing)"]:
                        say("      %s" % line)
                    stopped = True
                    break
                got[name].append(res["fps"])
            if stopped:
                break
            if rnd + 1 < REPEATS:
                continue
            wide = [nm for nm in got if spread_of(got[nm]) > SPREAD_LIMIT]
            if not wide:
                break
            if rnd + 1 < MAX_ROUNDS:
                say("   scatter %s - one more round (%d of %d)"
                    % (", ".join("%s %.0f %%" % (nm, spread_of(got[nm]))
                                 for nm in wide), rnd + 2, MAX_ROUNDS))
        here = {}
        for name, _, _ in VARIANTS:
            v = got[name]
            if not v:
                continue
            here[name] = (sorted(v)[len(v) // 2], spread_of(v), len(v))
            say("   %-12s %s fps, scatter %.1f %% over %d repeats"
                % (name, fmt(here[name][0], "%.0f"), here[name][1],
                   here[name][2]))
        rows.append((p, here))

    # ---- the table
    say("\n%-3s %-2s %-10s %-12s %10s %8s %10s %11s"
        % ("", "", "how", "variant", "fps", "scatter", "vs as-is", "pieces"))
    with open(os.path.join(outdir, "speeds.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("point,image,decoder,how,threads,variant,fps,spread_pct,repeats,"
                 "vs_as_is_pct,trustworthy,independent_pieces\n")
        for p, here in rows:
            base = here.get("as-is")
            for name, _, _ in VARIANTS:
                cur = here.get(name)
                if not cur:
                    continue
                ok = (base and cur[1] <= SPREAD_LIMIT
                      and base[1] <= SPREAD_LIMIT)
                rel = (100.0 * (cur[0] - base[0]) / base[0]) if ok else None
                pieces = pieces_of(made[(p["image"], name)])
                key = "%s_%s_%s_%s" % (p["image"], p["codec"], p["how"], name)
                say("%-3s %-2s %-10s %-12s %10s %8s %10s %11d"
                    % (p["image"], p["codec"], p["how"], name,
                       fmt(cur[0], "%.0f"),
                       fmt(cur[1], "%.0f %%"),
                       (fmt(rel, "%+.0f %%") if name != "as-is" else "")
                       if ok else "scatter", pieces))
                fh.write("%s,%s,%s,%s,%s,%s,%s,%s,%d,%s,%s,%d\n"
                         % (key, p["image"], p["codec"], p["how"],
                            p["threads"], name,
                            fmt(cur[0], "%.2f", ""), fmt(cur[1], "%.2f", ""),
                            cur[2], fmt(rel, "%.2f", ""),
                            "yes" if ok else "no", pieces))

    bad = ["%s %s %s" % (p["image"], p["codec"], p["how"]) for p, here in rows
           if any(v[1] > SPREAD_LIMIT for v in here.values())]
    if bad:
        say("\n%d point(s) of %d still scattered more than %.0f %% after %d"
            " rounds: %s." % (len(bad), len(rows), SPREAD_LIMIT, MAX_ROUNDS,
                              ", ".join(bad)))
        say("For those no comparison is printed, and none must be worked out"
            " from the figures: they are medians of numbers that do not agree"
            " with each other.")

    say("\n\"pieces\" is how many independent parts the stream falls into -"
        " MCUs in the frame divided by the restart interval. That is the"
        " quantity the speed is supposed to follow, and putting the two"
        " side by side is the whole point of this measurement.")

    say("\nRESULTS ARE IN:")
    say("   %s" % outdir)
    say("   sizes.csv     what each variant costs in bytes")
    say("   speeds.csv    the table above")
    say("   <point>.log   what the program printed, for every measurement")
    say("\nZIP THAT FOLDER WHOLE AND SEND IT: %d files, %.1f MB in all."
        % (len(os.listdir(outdir)),
           sum(os.path.getsize(os.path.join(outdir, f))
               for f in os.listdir(outdir)) / 1048576.0))
    say("The variant images stay behind, in %s." % imgdir)
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
