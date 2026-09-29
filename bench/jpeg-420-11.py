#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jpeg-420-11.py
# version 2026-09-25.1 of 25.09.2026, cancels version 10
#
# NEW IN THIS VERSION (25.09.2026): "who is on the card" lists every compute
# process from nvidia-smi --query-compute-apps; nothing else is changed.
#
# NEW: before every measured repeat the card is checked for someone else's load, and the repeat waits while it is busy.
#
# The 4:2:0 side measurement. Not the whole grid - a short list of points that
# answer one question and stop (Fyodor, 22.09.2026: "a test for 420 can be
# done, but not for every case - for some").
#
# WHY 4:2:0 IS NOT JUST "THE SAME, ONLY SMALLER"
#
# CORRECTION TO VERSION 01 (Fyodor, 22.09.2026): "the restart intervals are
# different for different -s values, that is fixed, nothing in our codec can
# be changed there."
#
# Version 01 said the encoder keeps one setting and 4:2:0 therefore leaves
# four times fewer pieces. That was wrong, and it was written down correctly
# in memo 41.22 SS4 the day before: OUR ENCODER HAS NO INTERVAL SETTING AT
# ALL. It once did: the interval used to be one of the encoder's parameters,
# nobody needed it, the number was made fixed and the encoder got a little
# faster for it (Fyodor, 22.09.2026). The fixed value is its own for each
# subsampling, and it also depends on whether the image is colour or
# monochrome. So the encoder has already chosen a different interval for
# 4:2:0, and what that choice comes to is a thing to MEASURE, not to derive.
#
# What actually changes with 4:2:0, and why the answer cannot be guessed:
#
#   an MCU is 16x16 pixels instead of 8x8, so one frame holds a quarter as
#   many MCUs to begin with;
#   the encoder puts markers at its own fixed interval for this subsampling;
#   there are half as many coefficients to code.
#
# The script therefore reports, for every file it makes, THE NUMBER OF
# INDEPENDENT PIECES THE STREAM FALLS INTO - MCUs in the frame divided by the
# interval the encoder chose. That number, not the interval, is what the
# decoder can spread across the card, and it is the one to put beside the
# speed.
#
# THE SHORT LIST
#
#   quality 90 only            the quality the article is built on
#   frames 2K and 4K           the frames the article is built on
#   both codecs, both ways
#   encoding: one thread       comparable with the table in section 3
#   decoding: one thread and the best point of the grid
#
#   Twelve points, EACH MEASURED AT BOTH 4:4:4 AND 4:2:0 in the same run -
#   twenty four measurements, three repeats each, about twenty seconds of
#   work per repeat. Everything else - 4:2:2, quality 95, the thread sweep,
#   the batch sweep, energy - stays as it is.
#
#   Version 04 measured only 4:2:0 and put the bench row of 21.09 beside it
#   as the 4:4:4 figure. That is two different runs on two different days,
#   and on single-frame points those disagree by tens of per cent - the
#   profiler run of 22.09 measured exactly that. A comparison of two numbers
#   from two runs is not a comparison. Both are measured here now.
#
# WHAT VERSION 05 GOT WRONG, AND WHY IT MATTERS MORE THAN THE RESULT
#
#   Its run of 23.09 showed that on some points the Fastvideo sample's speed
#   SCATTERS between repeats by 30 to 45 per cent. On 2K decoding with 32
#   threads the three repeats came out 2817, 3015 and 3674 frames per second.
#   The median of three numbers that far apart means nothing, and the change
#   computed from two such medians means less than nothing. The nvJPEG points
#   held to two per cent over the same run.
#
#   This rule already existed in our own bench (bench-07.py, RESPREAD_LIMIT):
#   when the repeats of a point scatter more than seven per cent, more
#   repeats are added. I did not carry it over here. Now it is here:
#
#   1. SPREAD IS MEASURED AND SHOWN for every point and every subsampling,
#      and repeats are added while it stays above SPREAD_LIMIT, up to
#      MAX_ROUNDS.
#   2. WHEN THE SPREAD STAYS HIGH, NO CHANGE IS PRINTED for that point - the
#      table says "scatter too wide" instead of a per cent figure that would
#      be read as a result.
#   3. THE TWO SUBSAMPLINGS ALTERNATE, one repeat each, round by round.
#      Version 05 ran all the 4:4:4 repeats first and all the 4:2:0 repeats
#      after, so anything that drifts through a point - the card warming up,
#      another load on the machine - was charged entirely to 4:2:0.
#
# WHERE THE OPTION COMES FROM
#
#   Not from me. Every command line the bench recorded already carries
#   "-s 444" - both programs take the subsampling that way - so the script
#   reads the key out of the data and changes its value, exactly as 46.04 SS10
#   says. Nothing is rebuilt and nothing is invented.
#
# WHAT IT CHECKS RATHER THAN ASSUMES
#
#   - that the files it made really ARE 4:2:0: the sampling factors are read
#     out of the SOF marker of the file itself, not believed because the
#     option was accepted. A program that quietly ignores an unknown value
#     would otherwise hand us 4:4:4 files under a 4:2:0 name, and the
#     measurement would be wrong and look right;
#   - that both codecs got the same subsampling, or it stops;
#   - that the reference file was actually written before anything is
#     measured against it.
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
#     python jpeg-420-07.py            show everything, measure nothing
#     python jpeg-420-07.py --do       run it
#
#   Results go to <project>\sub420-out\<date>-<time>\ - small files only.

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

VERSION = "jpeg-420-11 of 25.09.2026"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUTROOT = os.path.join(HERE, "sub420-out")

QUALITY = 90
IMAGES = ("2k", "4k")
REPEATS = 3                 # rounds always done
MAX_ROUNDS = 7              # and at most this many when the scatter is wide
# Between the fastest and the slowest repeat of one point. Above this the
# point is measured again; if it stays above it after MAX_ROUNDS, no change
# is printed for that point at all. The number is taken from bench-07.py,
# where the same rule has been in force since 31.08.2026.
SPREAD_LIMIT = 7.0
# 8 s since version 09: the article's own runs are 4 s a repeat, and three
# to seven repeats of 8 s answer the question in a quarter of an hour.
TARGET_SECONDS = 8.0
# Version 09: one point per codec and direction, one thread. The best grid
# point of the decoder is no longer measured here: 4:2:0 changes the stream,
# and one thread shows it as well as the grid does.
DECODE_GRID = False
MAX_FRAMES = 150000

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


# ------------------------------------------------------- the bench's records


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
    best, notes = None, []
    if not os.path.isdir(RUNS):
        return None, notes
    pointed = pointed_run()
    if pointed:
        return pointed, [(pointed, "named in " + POINTER)]
    for folder in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if not os.path.isdir(folder):
            continue
        rows = [r for r in read_rows(folder) if r.get("cmd")]
        kinds = {(r.get("codec"), r.get("direction")) for r in rows}
        need = {("fv", "E"), ("fv", "D"), ("nv", "E"), ("nv", "D")}
        if need <= kinds:
            notes.append((folder, "all four codec-direction pairs"))
            best = folder
        else:
            notes.append((folder, "missing %s"
                          % ", ".join("%s %s" % k for k in sorted(need - kinds))
                          or "no command lines"))
    return best, notes


def median_row(rows):
    return sorted(rows, key=lambda r: r.get("fps") or 0)[len(rows) // 2]


def pick_points(folder):
    """One encode point and two decode points per codec and frame."""
    rows = [r for r in read_rows(folder)
            if r.get("q") == QUALITY and r.get("sub") == "444" and r.get("cmd")]
    out = []
    for image in IMAGES:
        for codec in ("fv", "nv"):
            here = [r for r in rows if r.get("image") == image
                    and r.get("codec") == codec]
            # path "single" only: nvJPEG's native batch of 1 also has one
            # thread and one frame, but it is another way of running
            single = [r for r in here if r.get("threads") == 1
                      and r.get("batch") == 1
                      and (r.get("path") or "single") == "single"]
            for direction in ("E", "D"):
                one = [r for r in single if r.get("direction") == direction]
                if one:
                    out.append({"image": image, "codec": codec,
                                "direction": direction, "how": "single",
                                "row": median_row(one)})
            if not DECODE_GRID:
                continue
            grid = {}
            for r in here:
                if r.get("direction") != "D":
                    continue
                if r.get("threads") == 1 and r.get("batch") == 1:
                    continue
                if codec == "nv" and r.get("threads") == 1:
                    continue        # the native batch has its own frame of
                grid.setdefault((r.get("threads"), r.get("batch")),
                                []).append(r)     # reference; not compared here
            if grid:
                best = max(grid.values(),
                           key=lambda g: median_row(g).get("fps") or 0)
                out.append({"image": image, "codec": codec, "direction": "D",
                            "how": "grid", "row": median_row(best)})
    return out


# ------------------------------------------------------- reading a JPEG file


def frame_of(path):
    """Width, height and sampling factors, out of the file's own SOF marker.

    Read from the file rather than believed from the option: an encoder that
    silently ignores an unknown key would otherwise give us 4:4:4 files under
    a 4:2:0 name, and the whole measurement would be wrong and look right.
    """
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
            n = data[i + 9]
            f = []
            for c in range(n):
                b = data[i + 10 + c * 3 + 1]
                f.append((b >> 4, b & 15))
            return w, h, f
        i += 2 + length
    return 0, 0, []


def sampling_of(path):
    return frame_of(path)[2]


def mcus_of(path):
    """How many MCUs the frame holds, by its own header."""
    w, h, f = frame_of(path)
    if not w or not h or not f:
        return 0
    hmax = max(x[0] for x in f) or 1
    vmax = max(x[1] for x in f) or 1
    across = -(-w // (8 * hmax))
    down = -(-h // (8 * vmax))
    return across * down


def pieces_of(path):
    """Independent pieces: how many parts the decoder can work on at once.

    MCUs in the frame divided by the interval the encoder chose. This - not
    the interval - is what a thousand cores can be spread over.
    """
    n = mcus_of(path)
    ri = restart_interval(path)
    if not n:
        return 0
    return -(-n // ri) if ri else 1


def sub_name(factors):
    if not factors:
        return "unknown"
    if len(factors) == 1:
        return "grey"
    y, c1 = factors[0], factors[1]
    if y == (1, 1) and c1 == (1, 1):
        return "444"
    if y == (2, 1) and c1 == (1, 1):
        return "422"
    if y == (2, 2) and c1 == (1, 1):
        return "420"
    return "%dx%d" % y


def restart_interval(path):
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
        if marker == 0xDD and length >= 4:
            return (data[i + 4] << 8) | data[i + 5]
        i += 2 + length
    return 0


# ------------------------------------------------------------- the programs


def split_command(cmd):
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


def run(argv, log_path, cwd=None):
    t0 = time.time()
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, cwd=cwd or HERE)
        out, _ = p.communicate(timeout=3600)
    except FileNotFoundError:
        return None, "program not found: %s" % argv[0]
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return None, "timed out after an hour"
    text = out.decode("utf-8", "replace")
    if log_path:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(argv) + "\n(in %s)\n\n" % (cwd or HERE)
                     + text)
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
    return {"fps": fps, "wall_s": time.time() - t0, "head": head,
            "text": text}, None


def spread_of(values):
    """How far the repeats of one point are from each other, per cent.

    A median only means something when the numbers it sits between agree.
    On 23.09 three repeats of one point came out 2817, 3015 and 3674 - the
    median of that is not a measurement.
    """
    if not values:
        return 0.0
    med = sorted(values)[len(values) // 2]
    return 100.0 * (max(values) - min(values)) / med if med else 0.0


def value_of(argv, flag):
    for i, a in enumerate(argv[:-1]):
        if a.lower() == flag:
            return argv[i + 1]
    return None


def swap(argv, **kw):
    """Same command line with some of its values replaced."""
    out = list(argv)
    for key, value in kw.items():
        flag = "-" + key
        for i, a in enumerate(out[:-1]):
            if a.lower() == flag:
                out[i + 1] = str(value)
                break
        else:
            out += [flag, str(value)]
    return out


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
        say("\nNo bench run holds all four codec-direction pairs.")
        return 1

    points = pick_points(folder)
    if not points:
        say("\nNo points at quality %d, 4:4:4." % QUALITY)
        return 1

    say("\nRESULTS GO TO A FOLDER OF THEIR OWN, NEXT TO THIS SCRIPT:")
    say("   %s" % os.path.join(OUTROOT, "<date>-<time>"))
    say("Small files only - zip that folder whole and send it.")

    say("\nPoints: %d, each measured at 4:4:4 AND at 4:2:0, alternating,"
        " %d rounds and up to %d when the repeats scatter more than %.0f %%"
        % (len(points), REPEATS, MAX_ROUNDS, SPREAD_LIMIT))
    lo = len(points) * 2 * REPEATS
    hi = len(points) * 2 * MAX_ROUNDS
    say("That is %d to %d measurements of about %.0f s each - from %.0f to"
        " %.0f minutes." % (lo, hi, TARGET_SECONDS,
                            lo * (TARGET_SECONDS + 2) / 60.0,
                            hi * (TARGET_SECONDS + 2) / 60.0))
    for p in points:
        say("   %-3s %-3s %-2s %-7s %s"
            % (p["image"], p["codec"], p["direction"], p["how"],
               " ".join(split_command(p["row"]["cmd"]))[:110]))

    # The subsampling key is already in every command line the bench wrote.
    # Only the ENCODE lines need it. A decoder takes the subsampling from
    # the file it is given, not from a key.
    bad = [p for p in points if p["direction"] == "E"
           and not value_of(split_command(p["row"]["cmd"]), "-s")]
    if bad:
        say("\nThese command lines have no -s key, so I do not know how to"
            " ask them for 4:2:0:")
        for p in bad:
            say("   %s %s %s" % (p["image"], p["codec"], p["direction"]))
        say("I will not invent an option for someone else's program"
            " (46.04 SS10).")
        return 1
    say("\nSubsampling is asked for with -s, the key the bench itself uses:"
        " -s 444 becomes -s 420. Nothing else in the command lines changes.")
    say("Every file made this way is checked by reading the sampling factors"
        " out of its own SOF marker before anything is measured against it.")

    if not a.do:
        say("\nThis was the plan. Add --do to run it.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(OUTROOT, stamp)
    imgdir = os.path.join(HERE, "sub420-img", stamp)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(imgdir, exist_ok=True)
    say("\n" + "=" * 72)
    say("RESULTS GO TO:")
    say("   " + outdir)
    say("(the 4:2:0 reference files go to %s)" % imgdir)
    say("=" * 72)

    # ---- make one 4:2:0 reference per codec and frame, and check it
    refs = {}
    sizes = []
    for p in points:
        if p["direction"] != "E" or p["how"] != "single":
            continue
        key = (p["codec"], p["image"])
        dst = os.path.join(imgdir, "%s_ref_%s_q%d_420.jpg"
                           % (p["codec"], p["image"], QUALITY))
        argv = split_command(p["row"]["cmd"])
        argv = swap(argv, s="420", o=dst, repeat="1")
        argv = [x for x in argv if x != "-discard"]
        res, err = run(argv, os.path.join(outdir, "make_%s_%s.log"
                                          % (p["codec"], p["image"])))
        if err or not os.path.isfile(dst):
            say("\nFAILED making the 4:2:0 file for %s %s: %s"
                % (p["codec"], p["image"], err or "no output file"))
            for line in (res or {}).get("head") or ["(nothing printed)"]:
                say("      %s" % line)
            return 1
        got = sub_name(sampling_of(dst))
        ri = restart_interval(dst)
        say("   %s %s -> %s, %d bytes, %d MCUs, interval %d, %d independent"
            " pieces" % (p["codec"], p["image"], got, os.path.getsize(dst),
                         mcus_of(dst), ri, pieces_of(dst)))
        if got != "420":
            say("   THE FILE IS NOT 4:2:0 - it is %s. The option was accepted"
                " and ignored. Stopping." % got)
            return 1
        refs[key] = dst
        sizes.append((p["image"], p["codec"], "420", os.path.getsize(dst),
                      mcus_of(dst), ri, pieces_of(dst)))
        # the 4:4:4 file of the same codec and frame, for comparison - the
        # bench already made it, and the interval it got is the thing being
        # compared against
        old = value_of(split_command(
            next((q["row"]["cmd"] for q in points
                  if q["codec"] == p["codec"] and q["image"] == p["image"]
                  and q["direction"] == "D"), "")), "-i")
        if old:
            full = old if os.path.isabs(old) else os.path.join(HERE, old)
            if os.path.isfile(full):
                say("   %s %s    4:4:4 for comparison: %d bytes, %d MCUs,"
                    " interval %d, %d pieces"
                    % (p["codec"], p["image"], os.path.getsize(full),
                       mcus_of(full), restart_interval(full),
                       pieces_of(full)))
                sizes.append((p["image"], p["codec"], "444",
                              os.path.getsize(full), mcus_of(full),
                              restart_interval(full), pieces_of(full)))

    with open(os.path.join(outdir, "sizes.csv"), "w", encoding="utf-8") as fh:
        fh.write("image,codec,subsampling,bytes,mcus,restart_interval,"
                 "independent_pieces\n")
        for image, codec, sub, n, mcus, ri, pieces in sizes:
            fh.write("%s,%s,%s,%d,%d,%d,%d\n"
                     % (image, codec, sub, n, mcus, ri, pieces))

    # ---- measure: the two subsamplings ALTERNATE, round by round, and
    # rounds are added while the repeats of a point scatter too widely
    rows = []
    for n, p in enumerate(points, 1):
        key = "%s_%s_%s_%s" % (p["image"], p["codec"], p["direction"],
                               p["how"])
        say("\n[%d/%d] %s" % (n, len(points), key))
        base = split_command(p["row"]["cmd"])
        was = p["row"].get("fps")
        frames = min(max(int(TARGET_SECONDS * (was or 500)), 100), MAX_FRAMES)

        def line_for_sub(sub):
            if p["direction"] == "E":
                return swap(base, s=sub, repeat=frames)
            if sub == "420":
                ref = refs.get((p["codec"], p["image"]))
                return swap(base, i=ref, repeat=frames) if ref else None
            return swap(base, repeat=frames)      # its own file, as it is

        got = {"444": [], "420": []}
        stopped = False
        for rnd in range(MAX_ROUNDS):
            for sub in ("444", "420"):
                argv = line_for_sub(sub)
                if argv is None:
                    say("   %s: no file for this codec and frame" % sub)
                    stopped = True
                    break
                wait_until_card_quiet(say)
                res, err = run(argv, os.path.join(
                    outdir, "%s_%s_%d.log" % (key, sub, rnd)))
                if err:
                    say("   %s FAILED: %s" % (sub, err))
                    stopped = True
                    break
                if not res["fps"]:
                    say("   %s: the program printed no figure. What it said:"
                        % sub)
                    for l in res["head"] or ["(nothing)"]:
                        say("      %s" % l)
                    stopped = True
                    break
                got[sub].append(res["fps"])
            if stopped:
                break
            if rnd + 1 < REPEATS:
                continue
            wide = [sub for sub in ("444", "420")
                    if spread_of(got[sub]) > SPREAD_LIMIT]
            if not wide:
                break
            if rnd + 1 < MAX_ROUNDS:
                say("   scatter %s at %s - one more round (%d of %d)"
                    % (", ".join("%.0f %%" % spread_of(got[s_]) for s_ in wide),
                       " and ".join(wide), rnd + 2, MAX_ROUNDS))

        out = {}
        for sub in ("444", "420"):
            v = got[sub]
            if not v:
                continue
            out[sub] = (sorted(v)[len(v) // 2], spread_of(v), len(v))
            say("   %s: %s fps, scatter %.1f %% over %d repeats"
                % (sub, fmt(out[sub][0], "%.0f"), out[sub][1], out[sub][2]))
        rows.append((key, p, out.get("444"), out.get("420"), was))

    # ---- the table
    say("\n%-3s %-3s %-2s %-7s %9s %7s %9s %7s %11s"
        % ("", "", "", "how", "4:4:4", "spread", "4:2:0", "spread", "change"))
    with open(os.path.join(outdir, "speeds.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("point,image,codec,direction,how,fps_444,spread_444_pct,"
                 "repeats_444,fps_420,spread_420_pct,repeats_420,change_pct,"
                 "trustworthy,bench_fps_444,this_vs_bench_pct\n")
        for key, p, a444, a420, was in rows:
            f444 = a444[0] if a444 else None
            f420 = a420[0] if a420 else None
            s444 = a444[1] if a444 else None
            s420 = a420[1] if a420 else None
            ok = (a444 and a420 and s444 <= SPREAD_LIMIT
                  and s420 <= SPREAD_LIMIT)
            rel = (100.0 * (f420 - f444) / f444) if (ok and f444) else None
            chk = (100.0 * (f444 - was) / was) if (f444 and was) else None
            say("%-3s %-3s %-2s %-7s %9s %7s %9s %7s %11s"
                % (p["image"], p["codec"], p["direction"], p["how"],
                   fmt(f444, "%.0f"), fmt(s444, "%.0f %%"),
                   fmt(f420, "%.0f"), fmt(s420, "%.0f %%"),
                   fmt(rel, "%+.0f %%") if ok else "scatter too wide"))
            fh.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                     % (key, p["image"], p["codec"], p["direction"], p["how"],
                        fmt(f444, "%.2f", ""), fmt(s444, "%.2f", ""),
                        a444[2] if a444 else "",
                        fmt(f420, "%.2f", ""), fmt(s420, "%.2f", ""),
                        a420[2] if a420 else "",
                        fmt(rel, "%.2f", ""), "yes" if ok else "no",
                        fmt(was, "%.2f", ""), fmt(chk, "%.2f", "")))

    bad = [k for k, _, a4, a2, _ in rows
           if not (a4 and a2 and a4[1] <= SPREAD_LIMIT
                   and a2[1] <= SPREAD_LIMIT)]
    if bad:
        say("\n%d point(s) of %d scattered more than %.0f %% even after %d"
            " rounds: %s." % (len(bad), len(rows), SPREAD_LIMIT, MAX_ROUNDS,
                              ", ".join(bad)))
        say("For those the change is not printed and must not be worked out"
            " from the two figures: they are medians of numbers that do not"
            " agree with each other.")

    # ---- what the article cares about: the gap between the codecs
    say("\nThe gap between the codecs, same subsampling, same run:")
    for image in sorted({p["image"] for _, p, _, _, _ in rows}):
        for how in ("single", "grid"):
            side = {}
            for key, p, a4, a2, was in rows:
                if (p["image"] == image and p["direction"] == "D"
                        and p["how"] == how):
                    side[p["codec"]] = (a4, a2)
            if "fv" not in side or "nv" not in side:
                continue
            out = []
            for i, sub in enumerate(("4:4:4", "4:2:0")):
                f = side["fv"][i]
                v = side["nv"][i]
                if not f or not v:
                    out.append("%s -" % sub)
                elif max(f[1], v[1]) > SPREAD_LIMIT:
                    out.append("%s (scatter too wide)" % sub)
                else:
                    out.append("%s %.1f x" % (sub, f[0] / v[0]))
            say("   decoding %-3s %-7s %s" % (image, how, ",  ".join(out)))

    say("\nBoth columns are measured in THIS run, alternating one repeat"
        " each, so nothing that drifts through a point lands on one of them"
        " alone.")
    say("The last column is only a sanity check: how far this run's 4:4:4"
        " figure sits from the bench row of 21.09. A few per cent is normal;"
        " tens of per cent means the two runs are not comparable and the"
        " bench row for that point wants looking at.")

    say("\nRESULTS ARE IN:")
    say("   %s" % outdir)
    say("   sizes.csv     bytes, MCUs, interval and independent pieces,"
        " 4:2:0 beside 4:4:4")
    say("   speeds.csv    the table above")
    say("   <point>_N.log what the program printed, every repeat")
    say("\nZIP THAT FOLDER WHOLE AND SEND IT: %d files, %.1f MB in all."
        % (len(os.listdir(outdir)),
           sum(os.path.getsize(os.path.join(outdir, f))
               for f in os.listdir(outdir)) / 1048576.0))
    say("The 4:2:0 reference files stay behind, in %s." % imgdir)
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
