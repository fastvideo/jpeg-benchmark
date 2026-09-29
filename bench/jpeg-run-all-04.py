#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jpeg-run-all-04.py
# version 2026-09-24.2 of 24.09.2026, cancels version 03
#
# One run for the whole article. Starts every measuring script in turn, on
# ONE bench run, and leaves one folder to zip and send.
#
# WHY IT EXISTS
#
# There are five scripts now, three of them feed off the bench's run folder,
# and each of them picks that folder by its own rules. Started by hand on
# different days they can pick different runs, and then the numbers do not
# belong to one another. This starts them in the right order, on the same
# run, and gathers the results.
#
# THE ORDER, AND WHY IT IS THAT ORDER
#
#   1. jpeg-bench --probe         every program once, so that a mistake in a
#                                 path costs a minute and not four hours
#   2. jpeg-bench --final         the measurement the article stands on
#   3. runs\USE-THIS-RUN.txt      written here: the run everything else uses
#   4. jpeg-spread-check --do     did the repeats of that run agree?
#   5. nsys-jpeg --do             what the card actually does
#   6. jpeg-420 --do              4:2:0 beside 4:4:4
#   7. jpeg-markers --do          the restart markers themselves
#   8. all-out\<date>\            every small result folder, copied together
#
# A CLICK IN THE WINDOW DOES NOT STOP IT
#
# On Windows a console with "Quick Edit" on freezes the program writing to
# it while any text in it is selected - one stray click and the run stands
# still until someone presses a key, and the pause lands in a measurement.
# The run switches Quick Edit off for its own window at the start and puts
# it back at the end. Nothing else in the console's settings is touched.
#
# WHICH FILE OF EACH
#
# Every script is looked for by its name without the number, and the one with
# the highest number next to this file is taken: jpeg-bench-10.py over
# jpeg-bench-09.py. A new version of any of them therefore needs no change
# here. Which file was taken is printed before anything starts.
#
# Step 1 is the whole point of having a plan stage: it is the only cheap way
# to find out that a program moved or a reference frame is missing.
#
# WHAT IT DOES NOT DO
#
#   - it does not change any script's own behaviour and passes no options of
#     its own invention: every step is the command those scripts document;
#   - it does not stop the rest when one step fails. A profiler that died
#     must not cost us the markers experiment; what failed is said at the end;
#   - it does not delete anything, ever.
#
# USAGE
#
#     python jpeg-run-all-01.py            show the plan and check readiness
#     python jpeg-run-all-01.py --do       run it
#
#   Without arguments it checks that every script and every program is where
#   it should be, and measures nothing. That check takes a minute or two
#   (step 1 runs each program once) and is worth doing before a long night.

import argparse
import ctypes
import datetime
import glob
import os
import shutil
import subprocess
import sys
import time

VERSION = "jpeg-run-all-04 of 24.09.2026"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
POINTER = os.path.join(RUNS, "USE-THIS-RUN.txt")
OUTROOT = os.path.join(HERE, "all-out")

# name, script, arguments, what it is, roughly how long
STEPS = [
    ("probe", "jpeg-bench", ["--probe"],
     "every program once, to catch a wrong path early", "a minute or two"),
    ("bench", "jpeg-bench", ["--final"],
     "the measurement the article stands on: q 90, one thread and three"
     " plateau points, nvJPEG batch 1/32/64/128", "half an hour to an hour"),
    ("spread", "jpeg-spread-check", ["--do"],
     "did the repeats of that run agree with each other", "seconds"),
    ("profiler", "nsys-jpeg", ["--do"],
     "what the card does: 4 points of frame 2K plus nvJPEG batch 32 and"
     " 128", "ten to twenty minutes"),
    ("sub420", "jpeg-420", ["--do"],
     "4:2:0 beside 4:4:4, one thread, both codecs and directions",
     "five to fifteen minutes"),
    ("markers", "jpeg-markers", ["--do"],
     "our file with markers, without them and with one per MCU, through"
     " both decoders", "ten to twenty-five minutes"),
]

# folders whose results are gathered at the end; each one's newest subfolder
GATHER = [
    ("nsys-out", "the profiler"),
    ("sub420-out", "4:2:0"),
    ("markers-out", "the markers"),
]


ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080


def quick_edit_off():
    """Switch off selection with the mouse for this console; return how to
    put it back, or None where there is nothing to do (not Windows, no
    console, output redirected)."""
    if os.name != "nt":
        return None
    try:
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)                 # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return None
        new = (mode.value | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE
        if new != mode.value and k.SetConsoleMode(h, new):
            return (h, mode.value)
    except Exception:
        pass
    return None


def quick_edit_back(saved):
    if not saved:
        return
    try:
        ctypes.windll.kernel32.SetConsoleMode(saved[0], saved[1])
    except Exception:
        pass


def say(*a):
    print(*a)
    sys.stdout.flush()


def newest(pattern):
    items = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not items:
        return None
    return max(items, key=os.path.getmtime)


def latest(prefix):
    """The highest-numbered <prefix>-NN.py next to this file, or None."""
    best, best_n = None, -1
    for p in glob.glob(os.path.join(HERE, prefix + "-*.py")):
        tail = os.path.basename(p)[len(prefix) + 1:-3]
        if tail.isdigit() and int(tail) > best_n:
            best, best_n = os.path.basename(p), int(tail)
    return best


def script_path(name):
    return os.path.join(HERE, name)


def run_step(name, script, args, log_dir):
    """One script, its output on the screen and in a file at the same time."""
    path = script_path(script)
    argv = [sys.executable, path] + list(args)
    log = os.path.join(log_dir, "%s.log" % name)
    say("\n" + "=" * 72)
    say("%s   %s" % (name.upper(), " ".join([script] + list(args))))
    say("started %s" % datetime.datetime.now().strftime("%H:%M:%S"))
    say("=" * 72)
    t0 = time.time()
    try:
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(argv) + "\n(in %s)\n\n" % HERE)
            fh.flush()
            # The step writes into a pipe, not a console, and on Windows it
            # would then use the local code page; the log is read as UTF-8.
            env = dict(os.environ, PYTHONIOENCODING="utf-8",
                       PYTHONUNBUFFERED="1")
            p = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, cwd=HERE, env=env)
            for raw in iter(p.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                print(line)
                sys.stdout.flush()
                fh.write(line + "\n")
            p.stdout.close()
            rc = p.wait()
    except FileNotFoundError:
        return False, 0.0, "script not found: %s" % path
    except KeyboardInterrupt:
        raise
    except OSError as e:
        return False, time.time() - t0, str(e)
    took = time.time() - t0
    say("\n%s finished in %s, exit code %d"
        % (name, human(took), rc))
    return rc == 0, took, ("exit code %d" % rc) if rc else ""


def human(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return "%d h %02d min" % (h, m)
    if m:
        return "%d min %02d s" % (m, s)
    return "%d s" % s


def main():
    ap = argparse.ArgumentParser(add_help=True, description=VERSION)
    ap.add_argument("--do", action="store_true",
                    help="run it; without this the plan is shown, readiness"
                         " is checked and nothing is measured")
    a = ap.parse_args()

    global STEPS
    resolved = []
    for name, prefix, args, what, howlong in STEPS:
        resolved.append((name, latest(prefix) or (prefix + "-NN.py"), args,
                         what, howlong))
    STEPS = resolved

    say(VERSION)
    say("no arguments: show the plan and check readiness."
        "  --do: run everything.  That is all there is.")
    say("project folder: %s" % HERE)

    # ---- is everything here?
    say("\nScripts:")
    missing = []
    for name, script, args, what, howlong in STEPS:
        there = os.path.isfile(script_path(script))
        say("   %-9s %-26s %s" % (name, script,
                                  "found" if there else "NOT FOUND"))
        if not there:
            missing.append(script)
    if missing:
        say("\nMissing: %s" % ", ".join(sorted(set(missing))))
        say("They all live in _scripts\\bench\\ of the site folder and are"
            " copied here beside this file.")

    say("\nOrder of the run:")
    for i, (name, script, args, what, howlong) in enumerate(STEPS, 1):
        say("   %d. %-9s %-28s %s" % (i, name,
                                      " ".join([script] + list(args)), what))
        say("      %s" % howlong)
    say("   7. all-out\\<date>\\   every small result folder, copied together")
    say("\nAltogether: one to two hours, depending on how many repeats the"
        " points need before they agree.")

    # ---- the two things the run of 24.09 12:52 was missing
    say("\nThe card, right now:")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader"], stderr=subprocess.DEVNULL,
            timeout=15).decode("ascii", "replace").strip()
        say("   load, power, memory in use: %s" % out)
        full = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL,
                                       timeout=15).decode("utf-8", "replace")
        tail = full.split("Processes", 1)[1].splitlines()[1:] \
            if "Processes" in full else []
        tail = [l for l in tail if l.strip().strip("|=+-")]
        if tail:
            say("   on the card:")
            for l in tail[:15]:
                say("      " + l)
        say("   Nothing but the desktop should be there. Every measuring"
            " script now waits before a repeat while the card is busy with"
            " someone else's work, and says so.")
    except Exception:
        say("   nvidia-smi does not answer - the card cannot be checked.")
    try:
        import pynvml  # noqa: F401
        say("\nEnergy counter (nvidia-ml-py): present.")
    except Exception:
        say("\nEnergy counter (nvidia-ml-py): NOT installed for this Python"
            " (%s)." % sys.executable)
        say("   Without it the energy per frame is not measured. To install:")
        say('   "%s" -m pip install nvidia-ml-py' % sys.executable)

    if os.path.isdir(RUNS):
        old = newest(os.path.join(RUNS, "*"))
        say("\nNewest run now: %s"
            % (os.path.basename(old) if old else "none"))
    else:
        say("\nNo runs folder yet - the bench makes it.")
    if os.path.isfile(POINTER):
        with open(POINTER, encoding="utf-8") as fh:
            say("USE-THIS-RUN.txt says: %s" % fh.read().strip())
        say("It will be rewritten by this run.")

    if not a.do:
        if missing:
            say("\nNOT READY: the scripts above are missing.")
            return 1
        say("\nThis was the plan. Run the first step by hand to check the"
            " programs and paths:")
        say("   python %s --probe" % STEPS[0][1])
        say("and when it looks right, start everything:")
        say("   python jpeg-run-all-01.py --do")
        return 0
    if missing:
        say("\nCannot run: the scripts above are missing.")
        return 1

    saved = quick_edit_off()
    if saved:
        say("\nSelection with the mouse is off in this window while the run"
            " goes: a click here can no longer pause it.")
    try:
        return run_all(saved)
    finally:
        quick_edit_back(saved)


def run_all(saved):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(OUTROOT, stamp)
    os.makedirs(outdir, exist_ok=True)
    logs = os.path.join(outdir, "logs")
    os.makedirs(logs, exist_ok=True)
    say("\n" + "=" * 72)
    say("EVERYTHING FROM THIS RUN GOES TO:")
    say("   " + outdir)
    say("The screen output of every step is written there as it goes, so if"
        " the machine is left alone the log is still complete.")
    say("=" * 72)

    done = []
    started = time.time()
    for name, script, args, what, howlong in STEPS:
        ok, took, why = run_step(name, script, args, logs)
        done.append((name, ok, took, why))
        if name == "probe" and not ok:
            say("\nThe probe did not go through. Nothing else is started:"
                " every step below depends on those programs.")
            break
        if name == "bench":
            run = newest(os.path.join(RUNS, "*"))
            if run and not os.path.basename(run).startswith("USE-THIS"):
                try:
                    with open(POINTER, "w", encoding="utf-8") as fh:
                        fh.write(os.path.basename(run) + "\n")
                    say("\nUSE-THIS-RUN.txt -> %s" % os.path.basename(run))
                    say("Every script below will work on that run, and not on"
                        " whichever one it would have chosen by itself.")
                except OSError as e:
                    say("\nCould not write %s: %s" % (POINTER, e))
                    say("The scripts below will each choose a run by"
                        " themselves - check in their output that they all"
                        " chose the same one.")
            else:
                say("\nNo run folder appeared - the scripts below will choose"
                    " by themselves.")

    # ---- gather
    say("\n" + "=" * 72)
    say("GATHERING THE RESULTS")
    say("=" * 72)
    run = newest(os.path.join(RUNS, "*"))
    if run:
        dst = os.path.join(outdir, "bench-" + os.path.basename(run))
        try:
            os.makedirs(dst, exist_ok=True)
            for f in sorted(os.listdir(run)):
                src = os.path.join(run, f)
                if os.path.isfile(src) and not f.endswith(".zip"):
                    shutil.copy2(src, os.path.join(dst, f))
            say("   bench run: %d files from %s"
                % (len(os.listdir(dst)), os.path.basename(run)))
        except OSError as e:
            say("   bench run NOT copied: %s" % e)
    for folder, what in GATHER:
        src = newest(os.path.join(HERE, folder, "*"))
        if not src:
            say("   %-12s nothing to copy (%s did not finish?)"
                % (folder, what))
            continue
        dst = os.path.join(outdir, folder + "-" + os.path.basename(src))
        try:
            shutil.copytree(src, dst)
            say("   %-12s %d files" % (folder, len(os.listdir(dst))))
        except OSError as e:
            say("   %-12s NOT copied: %s" % (folder, e))

    # The spread check writes its csv beside this script, not into a folder of
    # its own, so it has to be picked up by name. Only the ones this run made.
    took_spread = 0
    for src in sorted(glob.glob(os.path.join(HERE, "spread-check-*.csv"))):
        try:
            if os.path.getmtime(src) < started - 5:
                continue
            shutil.copy2(src, os.path.join(outdir, os.path.basename(src)))
            took_spread += 1
        except OSError as e:
            say("   spread csv NOT copied: %s" % e)
    say("   %-12s %s" % ("spread", "%d csv" % took_spread if took_spread
                         else "nothing to copy (the check did not finish?)"))

    # ---- the verdict
    say("\n" + "=" * 72)
    say("WHAT HAPPENED, %s in all" % human(time.time() - started))
    say("=" * 72)
    for name, ok, took, why in done:
        say("   %-9s %-10s %s" % (name, human(took),
                                  "done" if ok else ("FAILED - " + why)))
    failed = [n for n, ok, _, _ in done if not ok]
    if failed:
        say("\nFailed: %s. Their logs are in %s." % (", ".join(failed), logs))
        say("Everything that did finish is still in the folder and still"
            " usable.")

    total = 0
    for root, _, files in os.walk(outdir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    say("\nZIP THIS FOLDER WHOLE AND SEND IT:")
    say("   %s" % outdir)
    say("   %.1f MB, nothing in it needs taking out."
        % (total / 1048576.0))
    say("The traces and the variant images stay behind on the machine - they"
        " are large, and everything the numbers are made of is in the folder"
        " above.")
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
