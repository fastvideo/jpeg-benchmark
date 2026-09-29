#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jpeg-big-06.py
# version 2026-09-28.2 of 28.09.2026, cancels versions 01 to 05
#
# NEW IN 06 (Fyodor, 28.09.2026, after the run of 12:13):
#
#   1. Text typed or pasted into the console while the script runs is thrown
#      away at the end. In a cmd window a right click pastes the clipboard;
#      the pasted lines waited in the console's input and, when the script
#      ended, cmd ran them one by one as commands ("... is not recognized as
#      an internal or external command"). Selection with the mouse is also
#      switched off for the run - a selection freezes the program's output -
#      and put back at the end, as jpeg-run-all-04 has done since 24.09.
#   2. Nothing but text goes into the results folder (46.04 par. 13). Version
#      05 left the three JPEG files of step 2 there, 1.5-2 MB each; they now
#      go to a temporary folder. At the end the script lists anything in its
#      folder that is not .txt, .csv, .log or .json, and prints the size.
#   3. The card is watched, not only asked before a repeat. On 28.09 another
#      program (anaconda, "biohub") loaded the card to 98 % and 430 W in
#      bursts, also in the middle of repeats, and half of the repeats came
#      out about twice slower. Now:
#        - before step 3 the card must stay free for 30 s in a row; if it does
#          not, the script names the programs on the card and stops (without
#          --do it only says so);
#        - during step 3 the card is sampled once a second (utilisation,
#          power, SM clock) into card.csv, and every repeat goes into
#          repeats.csv with the power it ran at;
#        - a repeat whose mean power is more than 20 % above the quietest
#          repeat of the same point is marked "disturbed", is left out of the
#          median, and more rounds are run (up to 10) until three clean
#          repeats agree within 7 %.
#   4. Everything printed also goes to console.txt in the results folder, so
#      the waits for the card and who was on it are in the folder you send.
#
# ---- version 05 --------------------------------------------------------
#
# NEW IN 05 (Fyodor, 28.09.2026: "three steps are not needed, do it in one"):
# one command does all three things the article waits for, and the only key
# is --do (46.04 par. 1 and 12).
#
#   python jpeg-big-06.py        steps 1 and 2 in full, step 3 as a plan
#   python jpeg-big-06.py --do   steps 1, 2 and 3, results into big-out\
#
#   1. THE 8K FRAME. The 8K frame the article was measured on (its path is
#      taken from the 8K line of the bench run) is compared, pixels only,
#      with 2 x 2 copies of the 4K frame made by make-big-frames-02.py. The
#      README says the 8K frame is made by that script; this checks it.
#      Nothing is written.
#   2. MCU AND RESTART INTERVAL OF THE FASTVIDEO ENCODER. The encoder of the
#      bench run compresses the 4K frame at 4:4:4, 4:2:2 and 4:2:0, q = 90;
#      the headers of the files are read (SOF and DRI, T.81 A.2.3): sampling
#      factors, MCU size in pixels, data units per MCU, restart interval -
#      and compared with the table of the article (8x8, 3, 10 / 16x8, 4, 8 /
#      16x16, 6, 5). Without --do the files go to a temporary folder that is
#      removed at once; with --do they and the report stay in the results.
#   3. 12K AND 16K, as in version 04: q = 90, 4:4:4 and 4:2:0 alternating,
#      one thread, both codecs, encoding and decoding. Only with --do.
#
# The keys --frames, --q and --subs of version 04 are gone: this script
# serves the benchmark article, where the values are fixed (12K and 16K,
# q = 90, 4:4:4 and 4:2:0). Frames of real aerial cameras at q = 95 are for
# the separate aerial article and will get their own version then.
#
# Everything must lie in one folder: this script, make-big-frames-02.py and
# the runs\ folder of the bench (runs\USE-THIS-RUN.txt names the run).
#
# ---- what version 04 said, still true for step 3 ------------------------
#
# NEW IN THIS VERSION: words only. What the markers split the stream into is
# called by its name in ITU-T T.81 (3.1.51): entropy-coded segments. Version 03
# said "independent pieces"; the file sizes.csv now has the column "segments".
#
# NEW IN THIS VERSION: frames of any size, quality and subsamplings are chosen.
#   --frames 12k,16k or WIDTHxHEIGHT - for aerial cameras, for example
#       11608x8708 (Phase One iXM-RS100F, 100 MPix), 14204x10652 (iXM-RS150F,
#       150 MPix), 19580x12600 (Canon LI8020SA sensor, 250 MPix);
#   --q 95        quality (default 90, the quality of the article);
#   --subs 444    subsamplings (default 444,420).
# Aerial imaging after RAW to RGB is usually JPEG at quality 95, 4:4:4
# (Fyodor, 26.09.2026), hence:
#   (version 04: python jpeg-big-04.py --do --frames 11608x8708,14204x10652,19580x12600 --q 95 --subs 444)
# The frames are made by make-big-frames-02.py (must lie next to this script).
#
# Version 02: #
# NEW IN THIS VERSION: the big frames are made by make-big-frames-01.py, which
# must lie next to this script - the same script that goes to the repository
# instead of the frames, so the frames measured here and the frames anyone
# makes from the repository are byte for byte the same. Nothing else changed.
#
# Frames larger than 8K: 12K (11520 x 6480) and 16K (15360 x 8640), made the
# same way as the 8K frame - copies of the 4K frame laid side by side, 3 x 3
# and 4 x 4. Fyodor, 26.09.2026: "maybe make 12K and 16K pictures for the
# tests, the same way as 8K"; and, on threads: "for big resolutions the
# multithreaded variant makes no sense".
#
# WHAT IS MEASURED
#
#   quality 90, both codecs, encoding and decoding, ONE THREAD, without copies
#   to the card and back - the same points as sections 3, 4 and 4.1 of the
#   article, on the two new frames. Every point at 4:4:4 AND at 4:2:0, the two
#   alternating repeat by repeat, 3 rounds and up to 7 while the repeats
#   scatter more than 7 per cent - exactly the rules of jpeg-420-12.py.
#   Each decoder decodes the file of its own encoder.
#
# THE QUESTION IT ANSWERS
#
#   From 2K to 4K the Fastvideo decoder's frame time grew by 14 per cent only,
#   from 4K to 8K by 3.2 times: on 8K the entropy-coded segments seem to be enough
#   to fill the card. If that is so, from 8K on the time per frame should grow
#   in step with the number of pixels. The script prints milliseconds per frame
#   and per megapixel for 12K and 16K next to the 8K point of the bench run.
#
# WHERE THE COMMAND LINES COME FROM
#
#   From the bench run named in runs\USE-THIS-RUN.txt, as in jpeg-420-12.py:
#   the one-thread 8K lines of each codec and direction. Only -i (the input),
#   -s (subsampling, encoding only) and -repeat change. The 4K frame is found
#   through the 4K encoding line of the same run.
#
# WHAT IT CHECKS FIRST
#
#   Before anything is measured, each codec encodes each big frame once, at
#   4:4:4 and at 4:2:0, and the file is read back: width and height from its
#   own SOF marker must be the big frame's, the sampling must be the one asked
#   for. If a program refuses a frame of this size, the script says so and
#   stops - nothing half measured.
#
# USAGE
#
#     python jpeg-big-06.py            steps 1 and 2, then the plan of step 3
#     python jpeg-big-06.py --do       all three steps
#
#   The big frames are written once to <project>\big-img\ (224 MB and
#   398 MB) and reused on the next run; they can be deleted afterwards.
#   Results go to <project>\big-out\<date>-<time>\ - small files only.

import argparse
import ctypes
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

VERSION = "jpeg-big-06 of 28.09.2026"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUTROOT = os.path.join(HERE, "big-out")
IMGDIR = os.path.join(HERE, "big-img")

QUALITY = 90                # the bench lines are taken at this quality
Q_RUN = 90                  # the quality measured here; --q changes it
SUBS = ("444", "420")       # --subs changes it
DIMS = {}                   # frame name -> (width, height), filled in main
IMAGES = ("12k", "16k")     # default; --frames changes it
ALL_IMAGES = ("12k", "16k", "WIDTHxHEIGHT")
# how many 4K frames across and down make the big frame (8K is 2 x 2)
TILES = {"12k": 3, "16k": 4}
PIXELS = {"8k": 7680 * 4320, "12k": 11520 * 6480, "16k": 15360 * 8640}
MIN_FRAMES = 20
REPEATS = 3                 # rounds always done
MAX_ROUNDS = 10             # and at most this many when the scatter is wide
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


_CONSOLE = {"fh": None}


def say(*a):
    print(*a)
    sys.stdout.flush()
    fh = _CONSOLE["fh"]
    if fh:
        try:
            fh.write(" ".join(str(x) for x in a) + "\n")
            fh.flush()
        except Exception:
            pass


# ------------------------------------------------ the console window
#
# 28.09.2026: text pasted into the cmd window during the run (a right click
# pastes the clipboard there) waited in the console's input and was run by
# cmd as commands after the script ended. So: selection with the mouse off
# for the run (a selection also freezes the output), and at the end the
# console's input is emptied before the mode is put back.

ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080
_CONSOLE_MODE = {"saved": None}


def _console_in():
    k = ctypes.windll.kernel32
    k.GetStdHandle.restype = ctypes.c_void_p
    return k, ctypes.c_void_p(k.GetStdHandle(-10))      # STD_INPUT_HANDLE


def console_start():
    if os.name != "nt":
        return
    try:
        k, h = _console_in()
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return
        new = (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE
        if new != mode.value and k.SetConsoleMode(h, ctypes.c_uint32(new)):
            _CONSOLE_MODE["saved"] = (h, mode.value)
    except Exception:
        pass


def console_end():
    """Throw away whatever was typed or pasted during the run, then put the
    console's mode back. Returns True if something was thrown away."""
    if os.name != "nt":
        return False
    dropped = False
    try:
        k, h = _console_in()
        n = ctypes.c_uint32()
        if k.GetNumberOfConsoleInputEvents(h, ctypes.byref(n)) and n.value:
            dropped = True
        k.FlushConsoleInputBuffer(h)
        if _CONSOLE_MODE["saved"]:
            k.SetConsoleMode(_CONSOLE_MODE["saved"][0],
                             ctypes.c_uint32(_CONSOLE_MODE["saved"][1]))
            _CONSOLE_MODE["saved"] = None
    except Exception:
        pass
    return dropped


# ------------------------------------------------ watching the card

CLEAN_POWER = 1.20      # a repeat above 1.2 x the quietest one is disturbed
CLEAN_NEED = 3          # clean repeats wanted per subsampling
FREE_BEFORE_S = 30      # the card must be free this long before step 3
SAMPLE_S = 1.0


class CardWatch(object):
    """Samples the card once a second in the background: utilisation, power,
    SM clock. NVML is used when nvidia-ml-py is installed (no new process per
    sample); otherwise nvidia-smi."""

    def __init__(self):
        self.samples = []            # (t, label, util, power_w, sm_mhz)
        self.label = "start"
        self._stop = threading.Event()
        self._thread = None
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:
            self._nvml = None

    def how(self):
        return "NVML" if self._nvml else "nvidia-smi"

    def read(self):
        if self._nvml:
            try:
                nv, h = self._nvml
                u = nv.nvmlDeviceGetUtilizationRates(h).gpu
                p = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                c = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
                return float(u), float(p), float(c)
            except Exception:
                pass
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,power.draw,clocks.sm",
                 "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL,
                timeout=10).decode("ascii", "replace")
            f = [x.strip() for x in out.strip().splitlines()[0].split(",")]
            return float(f[0]), float(f[1]), float(f[2])
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            r = self.read()
            if r:
                self.samples.append((time.time(), self.label) + r)
            self._stop.wait(SAMPLE_S)

    def start(self):
        if self.read() is None:
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(5)

    def stats(self, t0, t1):
        """(samples, max util, max power, mean power) between t0 and t1."""
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return 0, None, None, None
        return (len(s), max(x[2] for x in s), max(x[3] for x in s),
                sum(x[3] for x in s) / len(s))

    def write(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("time,label,util_pct,power_w,sm_mhz\n")
            for t, lab, u, p, c in self.samples:
                fh.write("%s,%s,%.0f,%.1f,%.0f\n" % (
                    datetime.datetime.fromtimestamp(t).strftime(
                        "%H:%M:%S.%f")[:-3], lab, u, p, c))


def card_free_for(seconds, tell):
    """Is the card free (utilisation <= QUIET_UTIL) at every check for this
    many seconds in a row? (True/False/None, what was seen). None: cannot
    tell (no nvidia-smi)."""
    seen = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        now = _card_now()
        if now is None:
            return None, seen
        seen.append(now)
        if now[0] > QUIET_UTIL:
            return False, seen
        time.sleep(1.0)
    return True, seen


def clean_flags(powers):
    """Which repeats are clean: mean power not above CLEAN_POWER x the
    quietest. A repeat with no power figure counts as clean."""
    known = [p for p in powers if p is not None]
    if not known:
        return [True] * len(powers)
    base = min(known)
    return [p is None or p <= base * CLEAN_POWER for p in powers]


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
    """Entropy-coded segments (T.81 3.1.51): what the decoder can work on at once.

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


def _load_maker():
    """make-big-frames-02.py from the folder of this script."""
    import importlib.util
    path = os.path.join(HERE, "make-big-frames-02.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("make_big_frames", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------ step 1: the 8K frame


def check_8k(maker, src4k, path8k):
    """Pixels of the 8K frame on disk against 2 x 2 copies of the 4K frame.
    Returns (verdict line, details) - the header of the file is not compared,
    another program may write it differently."""
    if not path8k:
        return "NOT CHECKED - the run names no 8K frame", []
    if not os.path.isfile(path8k):
        return "NOT CHECKED - the 8K frame is not on disk: %s" % path8k, []
    w, h, pix = maker.read_ppm(src4k)
    chunks = maker.big_bytes(w, h, pix, 2)
    next(chunks)                                    # the header
    made, _ = maker.md5_of_chunks(chunks)
    cw, ch, cpix = maker.read_ppm(path8k)
    have = hashlib.md5(cpix).hexdigest()
    lines = ["file on disk: %s" % path8k,
             "   made from the 4K frame: %d x %d, pixel MD5 %s" % (w * 2, h * 2, made),
             "   the file on disk:       %d x %d, pixel MD5 %s" % (cw, ch, have)]
    if made == have:
        return "THE SAME - the 8K frame is 2 x 2 copies of the 4K frame", lines
    return ("DIFFERENT - the 8K frame on disk was made some other way", lines)


# ------------------------------------------------ step 2: MCU of the encoder

# What the article says (table "restart intervals of the Fastvideo encoder"):
# MCU in pixels, data units per MCU, restart interval in MCUs.
ARTICLE_MCU = {"444": ((8, 8), 3, 10), "422": ((16, 8), 4, 8),
               "420": ((16, 16), 6, 5)}


def mcu_facts(path):
    """(sampling name, MCU (w, h), data units per MCU, restart interval)."""
    w, h, f = frame_of(path)
    if not f:
        return None
    hmax = max(x[0] for x in f)
    vmax = max(x[1] for x in f)
    units = sum(x[0] * x[1] for x in f)
    return sub_name(f), (8 * hmax, 8 * vmax), units, restart_interval(path), f


def check_mcu(tpl, src4k, folder, keep_logs):
    """Encode the 4K frame with the Fastvideo encoder of the run at 4:4:4,
    4:2:2 and 4:2:0 and read the headers. Returns (rows, all_as_article)."""
    rows, same = [], True
    tmpdir = tempfile.mkdtemp(prefix="jpeg-big-mcu-")
    for sub in ("444", "422", "420"):
        dst = os.path.join(tmpdir, "fv_mcu_4k_q%d_%s.jpg" % (QUALITY, sub))
        argv = swap(split_command(tpl[("fv", "E")]["cmd"]), i=src4k, s=sub,
                    q=QUALITY, o=dst, repeat="1")
        argv = [x for x in argv if x != "-discard"]
        if os.path.isfile(dst):
            os.remove(dst)
        res, err = run(argv, os.path.join(folder, "mcu_%s.log" % sub)
                       if keep_logs else None)
        if err or not os.path.isfile(dst):
            rows.append((sub, None, "FAILED: %s" % (err or "no output file"),
                         (res or {}).get("head") or []))
            same = False
            continue
        facts = mcu_facts(dst)
        if not facts:
            rows.append((sub, None, "FAILED: no SOF marker in the file", []))
            same = False
            continue
        name, mcu, units, ri, f = facts
        want = ARTICLE_MCU[sub]
        ok = name == sub and (mcu, units, ri) == want
        same = same and ok
        size = os.path.getsize(dst)
        try:
            os.remove(dst)
        except OSError:
            pass
        rows.append((sub, (name, mcu, units, ri, f, size),
                     "as in the article" if ok else
                     "NOT AS IN THE ARTICLE (article: MCU %dx%d, %d units,"
                     " interval %d)" % (want[0][0], want[0][1], want[1],
                                        want[2]), []))
    try:
        for n in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, n))
        os.rmdir(tmpdir)
    except OSError:
        pass
    return rows, same


def say_mcu(rows):
    for sub, facts, verdict, head in rows:
        if facts is None:
            say("   %s: %s" % (sub, verdict))
            for line in head:
                say("      %s" % line)
            continue
        name, mcu, units, ri, f, size = facts
        say("   asked %s -> file %s, factors %s, MCU %d x %d px, %d data"
            " units, restart interval %d MCU  [%s]"
            % (sub, name, " ".join("%dx%d" % x for x in f), mcu[0], mcu[1],
               units, ri, verdict))


def templates(folder):
    """The one-thread 8K rows of the run, and the 4K frame's file."""
    rows = [r for r in read_rows(folder)
            if r.get("q") == QUALITY and r.get("sub") == "444" and r.get("cmd")
            and r.get("threads") == 1 and r.get("batch") == 1
            and (r.get("path") or "single") == "single"]
    out = {}
    for codec in ("fv", "nv"):
        for direction in ("E", "D"):
            one = [r for r in rows if r.get("image") == "8k"
                   and r.get("codec") == codec
                   and r.get("direction") == direction]
            if one:
                out[(codec, direction)] = median_row(one)
    src4k = None
    for r in rows:
        if r.get("image") == "4k" and r.get("direction") == "E":
            src4k = value_of(split_command(r["cmd"]), "-i")
            if src4k:
                break
    return out, src4k


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(add_help=True, description=VERSION)
    ap.add_argument("--do", action="store_true",
                    help="all three steps; without it steps 1 and 2 are done"
                         " and step 3 is only shown")
    a = ap.parse_args()
    console_start()

    say(VERSION)
    say("no arguments: steps 1 and 2, the plan of step 3.  --do: all three.")
    say("project folder: %s" % HERE)
    say("frames: %s; quality %d; subsampling %s"
        % (", ".join(IMAGES), Q_RUN, ", ".join(SUBS)))

    folder, notes = pick_run()
    say("\nRuns:")
    for f, why in notes:
        say("   %s %-40s %s" % ("TAKEN " if f == folder else "      ",
                                os.path.basename(f), why))
    if not folder:
        say("\nNo bench run holds all four codec-direction pairs.")
        return 1
    maker = _load_maker()
    if maker is None:
        say("\nmake-big-frames-02.py is not next to this script (%s)."
            " It makes the big frames; put it here and run again."
            % HERE)
        return 1
    read_ppm, make_frame = maker.read_ppm, maker.make_frame
    tpl, src4k = templates(folder)
    need = [(c, d) for c in ("fv", "nv") for d in ("E", "D")]
    miss = [k for k in need if k not in tpl]
    if miss:
        say("\nThe run has no one-thread 8K line for: %s. Stopping."
            % ", ".join("%s %s" % k for k in miss))
        return 1
    if not src4k or not os.path.isfile(src4k):
        say("\nThe 4K frame named by the run is not on disk: %s" % src4k)
        return 1
    say("\n4K frame, the tile: %s" % src4k)
    for k in need:
        say("   8K %s %s, %s fps in the run: %s"
            % (k[0], k[1], fmt(tpl[k].get("fps"), "%.1f"),
               " ".join(split_command(tpl[k]["cmd"]))[:110]))

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(OUTROOT, stamp)
    if a.do:
        os.makedirs(outdir, exist_ok=True)
        _CONSOLE["fh"] = open(os.path.join(outdir, "console.txt"), "w",
                              encoding="utf-8")
        say(VERSION)
        say("results folder: %s" % outdir)

    # ---- step 1
    say("\n" + "=" * 72)
    say("STEP 1. Is the 8K frame of the article 2 x 2 copies of the 4K frame?")
    path8k = value_of(split_command(tpl[("fv", "E")]["cmd"]), "-i")
    verdict8k, lines8k = check_8k(maker, src4k, path8k)
    for line in lines8k:
        say("   " + line)
    say("   " + verdict8k)

    # ---- step 2
    say("\n" + "=" * 72)
    say("STEP 2. MCU and restart interval of the Fastvideo encoder, 4K frame,"
        " q = %d" % QUALITY)
    mcu_rows, mcu_same = check_mcu(tpl, src4k, outdir, a.do)
    say_mcu(mcu_rows)
    say("   %s" % ("all three as in the article" if mcu_same else
                   "SOMETHING DIFFERS FROM THE ARTICLE - see above"))
    if a.do:
        with open(os.path.join(outdir, "checks.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("%s\n\nStep 1. 8K frame\n" % VERSION)
            for line in lines8k:
                fh.write("   %s\n" % line)
            fh.write("   %s\n\nStep 2. MCU of the Fastvideo encoder,"
                     " 4K frame, q = %d\n" % (verdict8k, QUALITY))
            for sub, facts, verdict, head in mcu_rows:
                if facts is None:
                    fh.write("   %s: %s\n" % (sub, verdict))
                    continue
                name, mcu, units, ri, f, size = facts
                fh.write("   asked %s -> file %s, factors %s, MCU %dx%d,"
                         " units %d, interval %d, %d bytes: %s\n"
                         % (sub, name, " ".join("%dx%d" % x for x in f),
                            mcu[0], mcu[1], units, ri, size, verdict))

    say("\n" + "=" * 72)
    say("STEP 3. 12K and 16K, one thread, q = %d, %s"
        % (Q_RUN, " and ".join(SUBS)))
    w4, h4, _ = read_ppm(src4k)
    bigs = {}
    for image in IMAGES:
        DIMS[image] = maker.size_of(image, w4, h4)
        bigs[image] = os.path.join(IMGDIR, "%s_wild.ppm" % image)
        say("\n%s: %d x %d, %.1f MPix, copies of the 4K frame -> %s"
            % (image, DIMS[image][0], DIMS[image][1],
               DIMS[image][0] * DIMS[image][1] / 1e6, bigs[image]))
    PIXELS.update({k: v[0] * v[1] for k, v in DIMS.items()})

    points = [(image, codec, direction) for image in IMAGES
              for codec in ("fv", "nv") for direction in ("E", "D")]
    lo, hi = len(points) * 2 * REPEATS, len(points) * 2 * MAX_ROUNDS
    say("\nPoints: %d (one thread, no copies), each at %s, alternating;"
        " %d to %d measurements of about %.0f s - from %.0f to"
        " %.0f minutes, plus making the files."
        % (len(points), " and ".join(SUBS), lo * len(SUBS) // 2,
           hi * len(SUBS) // 2, TARGET_SECONDS,
           lo * len(SUBS) // 2 * (TARGET_SECONDS + 3) / 60.0,
           hi * len(SUBS) // 2 * (TARGET_SECONDS + 3) / 60.0))
    say("Results go to %s" % os.path.join(OUTROOT, "<date>-<time>"))

    say("\nIs the card free? Watching it for %d s (nothing of ours running)..."
        % FREE_BEFORE_S)
    free, seen = card_free_for(FREE_BEFORE_S, say)
    if free is None:
        say("   nvidia-smi does not answer - cannot tell whether the card is"
            " free")
    elif free:
        say("   free: %d checks, at most %.0f %% and %.0f W"
            % (len(seen), max(x[0] for x in seen), max(x[1] for x in seen)))
    else:
        say("   NOT FREE: %.0f %% load, %.0f W. On the card now:" % seen[-1])
        for l in _who_is_on_card():
            say("      " + l)
        say("   Another program is computing on the card. Its bursts made half"
            " of the repeats of 28.09 twice slower. Stop it, then run this"
            " again.")
        if a.do:
            return 1

    if not a.do:
        say("\nSteps 1 and 2 are done above; step 3 was only the plan."
            " Add --do to run all three.")
        return 0

    os.makedirs(IMGDIR, exist_ok=True)
    say("\n" + "=" * 72)
    say("RESULTS GO TO:\n   " + outdir)
    say("=" * 72)

    # ---- the big frames: made once, reused while they are the right size
    for image in IMAGES:
        (W, H), dst = DIMS[image], bigs[image]
        size = len(b"P6\n%d %d\n255\n" % (W, H)) + W * H * 3
        if os.path.isfile(dst) and os.path.getsize(dst) == size:
            say("   %s: already there, %d bytes" % (image, size))
            continue
        say("   %s: writing %.0f MB ..." % (image, size / 1e6))
        w, h = make_frame(src4k, W, H, dst)
        say("   %s: %d x %d, %d bytes" % (image, w, h, os.path.getsize(dst)))

    # ---- one file per codec, frame and subsampling; checked before use
    refs, sizes = {}, []
    for image in IMAGES:
        for codec in ("fv", "nv"):
            for sub in SUBS:
                dst = os.path.join(IMGDIR, "%s_ref_%s_q%d_%s.jpg"
                                   % (codec, image, Q_RUN, sub))
                argv = split_command(tpl[(codec, "E")]["cmd"])
                argv = swap(argv, i=bigs[image], s=sub, q=Q_RUN, o=dst,
                            repeat="1")
                argv = [x for x in argv if x != "-discard"]
                if os.path.isfile(dst):
                    os.remove(dst)
                res, err = run(argv, os.path.join(
                    outdir, "make_%s_%s_%s.log" % (codec, image, sub)))
                if err or not os.path.isfile(dst):
                    say("\nFAILED: %s could not encode the %s frame at %s: %s"
                        % (codec, image, sub, err or "no output file"))
                    for line in (res or {}).get("head") or ["(nothing)"]:
                        say("      %s" % line)
                    if err and err.startswith("program not found"):
                        say("The program itself is not there. Stopping.")
                    else:
                        say("The program may not take a frame of this size."
                            " Stopping; the log is in %s" % outdir)
                    return 1
                w, h, f = frame_of(dst)
                got = sub_name(f)
                say("   %s %s %s -> %d x %d, %s, %d bytes, interval %d, %d"
                    " entropy-coded segments" % (codec, image, sub, w, h, got,
                                             os.path.getsize(dst),
                                             restart_interval(dst),
                                             pieces_of(dst)))
                if (w, h) != DIMS[image] or got != sub:
                    say("   THE FILE IS NOT WHAT WAS ASKED FOR. Stopping.")
                    return 1
                refs[(codec, image, sub)] = dst
                sizes.append((image, codec, sub, os.path.getsize(dst),
                              mcus_of(dst), restart_interval(dst),
                              pieces_of(dst)))
    with open(os.path.join(outdir, "sizes.csv"), "w", encoding="utf-8") as fh:
        fh.write("image,codec,subsampling,bytes,mcus,restart_interval,"
                 "segments\n")
        for row in sizes:
            fh.write("%s,%s,%s,%d,%d,%d,%d\n" % row)

    # ---- measure: 4:4:4 and 4:2:0 alternate; the card is watched all along
    watch = CardWatch()
    watching = watch.start()
    say("\nThe card is sampled once a second (%s)%s."
        % (watch.how(), "" if watching else " - NOT AVAILABLE, repeats"
           " cannot be checked for other programs"))
    rep_path = os.path.join(outdir, "repeats.csv")
    rep_rows = []                # every repeat; "clean" is filled per point

    def write_repeats():
        with open(rep_path, "w", encoding="utf-8") as fh:
            fh.write("image,codec,direction,subsampling,round,fps,seconds,"
                     "waited_s,samples,util_max,power_max_w,power_mean_w,"
                     "clean\n")
            for r in rep_rows:
                fh.write(",".join(r) + "\n")
    rows = []
    for k, (image, codec, direction) in enumerate(points, 1):
        key = "%s_%s_%s" % (image, codec, direction)
        say("\n[%d/%d] %s" % (k, len(points), key))
        base = split_command(tpl[(codec, direction)]["cmd"])
        was8 = tpl[(codec, direction)].get("fps") or 50.0
        guess = was8 * PIXELS["8k"] / PIXELS[image]
        frames = min(max(int(TARGET_SECONDS * guess), MIN_FRAMES), MAX_FRAMES)

        def line_for_sub(sub):
            if direction == "E":
                return swap(base, i=bigs[image], s=sub, q=Q_RUN,
                            repeat=frames)
            return swap(base, i=refs[(codec, image, sub)], repeat=frames)

        got = {s: [] for s in SUBS}          # (fps, mean power)
        stopped = False
        for rnd in range(MAX_ROUNDS):
            for sub in SUBS:
                watch.label = "wait"
                waited = wait_until_card_quiet(say)
                watch.label = "%s_%s_%d" % (key, sub, rnd)
                t0 = time.time()
                res, err = run(line_for_sub(sub), os.path.join(
                    outdir, "%s_%s_%d.log" % (key, sub, rnd)))
                t1 = time.time()
                watch.label = "gap"
                if err or not (res or {}).get("fps"):
                    say("   %s FAILED: %s" % (sub, err or "no figure printed"))
                    for l in (res or {}).get("head") or []:
                        say("      %s" % l)
                    stopped = True
                    break
                n, umax, pmax, pmean = watch.stats(t0, t1)
                got[sub].append((res["fps"], pmean))
                rep_rows.append([
                    image, codec, direction, sub, "%d" % rnd,
                    "%.3f" % res["fps"], "%.1f" % (t1 - t0),
                    "%.0f" % (waited or 0), "%d" % n,
                    "" if umax is None else "%.0f" % umax,
                    "" if pmax is None else "%.1f" % pmax,
                    "" if pmean is None else "%.1f" % pmean, ""])
                write_repeats()
            if stopped:
                break
            if rnd + 1 < REPEATS:
                continue
            short = []
            for s in SUBS:
                flags = clean_flags([p for _f, p in got[s]])
                clean = [f for (f, _p), c in zip(got[s], flags) if c]
                if len(clean) < CLEAN_NEED or spread_of(clean) > SPREAD_LIMIT:
                    short.append("%s: %d clean of %d, scatter %.0f %%"
                                 % (s, len(clean), len(got[s]),
                                    spread_of(clean)))
            if not short:
                break
            if rnd + 1 < MAX_ROUNDS:
                say("   %s - one more round (%d of %d)"
                    % ("; ".join(short), rnd + 2, MAX_ROUNDS))
        out = {}
        for sub in SUBS:
            v = got[sub]
            if not v:
                continue
            flags = clean_flags([p for _f, p in v])
            mine = [r for r in rep_rows if r[:4] == [image, codec, direction,
                                                      sub]]
            for r, c in zip(mine, flags):
                r[12] = "yes" if c else "no"
            write_repeats()
            clean = [f for (f, _p), c in zip(v, flags) if c]
            use = clean if clean else [f for f, _p in v]
            out[sub] = (sorted(use)[len(use) // 2], spread_of(use), len(use),
                        len(v))
            say("   %s: %.1f fps, scatter %.1f %% over %d clean repeats of %d"
                % (sub, out[sub][0], out[sub][1], out[sub][2], out[sub][3]))
        rows.append((image, codec, direction, out, was8))
    watch.stop()
    watch.write(os.path.join(outdir, "card.csv"))

    # ---- the table
    say("\nQuality %d." % Q_RUN)
    say("%-11s %-3s %-2s %-4s %9s %8s %8s %9s %7s"
        % ("", "", "", "sub", "fps", "ms", "ms/MPix", "GB/s", "spread"))
    with open(os.path.join(outdir, "speeds.csv"), "w", encoding="utf-8") as fh:
        fh.write("image,width,height,quality,codec,direction,subsampling,fps,ms_per_frame,"
                 "ms_per_mpix,gb_per_s_uncompressed,spread_pct,repeats,"
                 "trustworthy,repeats_all\n")
        for image, codec, direction, out, was8 in rows:
            for sub in SUBS:
                if sub not in out:
                    continue
                fps, spread, nrep, nall = out[sub]
                ms = 1000.0 / fps
                mpix = PIXELS[image] / 1e6
                gbs = fps * PIXELS[image] * 3 / 1e9
                ok = spread <= SPREAD_LIMIT
                say("%-11s %-3s %-2s %-4s %9.1f %8.3f %8.4f %9.1f %6.1f%%%s"
                    % (image, codec, direction, sub, fps, ms, ms / mpix, gbs,
                       spread, "" if ok else "  scatter too wide"))
                fh.write("%s,%d,%d,%d,%s,%s,%s,%.3f,%.4f,%.5f,%.2f,%.2f,%d,%s,%d\n"
                         % (image, DIMS[image][0], DIMS[image][1], Q_RUN,
                            codec, direction, sub, fps, ms, ms / mpix,
                            gbs, spread, nrep, "yes" if ok else "no", nall))
    say("\nFor comparison, 8K 4:4:4, quality %d, one thread in the bench run"
        " (ms per megapixel)%s:" % (QUALITY, "" if Q_RUN == QUALITY else
                                    " - another quality than measured here"))
    for (codec, direction), r in sorted(tpl.items()):
        f = r.get("fps")
        if f:
            say("   8k   %-3s %-2s 444 %9.1f fps %8.4f ms/MPix"
                % (codec, direction, f, 1000.0 / f / (PIXELS["8k"] / 1e6)))
    say("\nIf the time per megapixel stays about the same from 8K to the"
        " bigger frames, the card is already full on 8K and the time grows with the"
        " number of pixels.")
    bad, total = [], 0
    for dp, _dn, fn in os.walk(outdir):
        for f in fn:
            total += os.path.getsize(os.path.join(dp, f))
            if not f.lower().endswith((".txt", ".csv", ".log", ".json")):
                bad.append(f)
    say("\nRESULTS ARE IN:\n   %s" % outdir)
    say("   %d files, %.1f MB, text only%s" % (
        sum(len(fn) for _dp, _dn, fn in os.walk(outdir)), total / 1e6,
        "" if not bad else " - NOT SO: " + ", ".join(bad)))
    say("   checks.txt  steps 1 and 2: the 8K frame, MCU of the encoder")
    say("   sizes.csv   bytes, MCUs, restart interval, entropy-coded segments")
    say("   repeats.csv every repeat: speed, card power, clean or disturbed")
    say("   card.csv    the card once a second during step 3")
    say("   console.txt everything printed here")
    say("   speeds.csv  the table above")
    say("   *.log       what the programs printed, every repeat")
    say("ZIP THAT FOLDER WHOLE AND SEND IT. The big frames stay in %s."
        % IMGDIR)
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
    if console_end():
        print("(text typed or pasted into this window during the run was"
              " thrown away, so that cmd does not run it)")
    if _CONSOLE["fh"]:
        try:
            _CONSOLE["fh"].close()
        except Exception:
            pass
        _CONSOLE["fh"] = None
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
