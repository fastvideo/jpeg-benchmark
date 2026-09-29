#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# jpeg-spread-check-02.py
# version 2026-09-24.1 of 24.09.2026, cancels version 01
#
# Does the run the article stands on hold together? Answered from the files
# that are already on disk - nothing is measured and nothing is run.
#
# WHY
#
# On 23.09.2026 a side script found that the repeats of one Fastvideo point
# came out 2817, 3015 and 3674 frames per second. The median of numbers that
# far apart is not a measurement. The bench writes EVERY repeat into
# results.jsonl as its own row, so the same question can be asked of every
# run already made, without measuring anything again:
#
#   for each point, how far apart are its repeats, and which points never
#   settled?
#
# WHAT A POINT IS
#
# Rows that agree on codec, variant, direction, image, quality, subsampling,
# threads, batch and path are repeats of one point. They differ only in the
# speed they came out at - and in how far apart those speeds are.
#
# WHAT IT DOES NOT DO
#
#   - it does not run anything and does not touch the run folders;
#   - it does not decide which points may go into an article. It says how far
#     apart the repeats are; what to do with a point whose repeats disagree
#     is a decision, not a calculation.
#
# USAGE
#
#     python jpeg-spread-check-01.py            show the answer
#     python jpeg-spread-check-01.py --do       and write it to a csv file
#
#   No folder argument: the runs are looked for under runs\ next to this
#   file, and every one of them is checked.

import argparse
import datetime
import glob
import json
import os
import sys

VERSION = "jpeg-spread-check-02 of 24.09.2026"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

# The same limit the bench uses, so that the two agree on what "wide" means.
SPREAD_LIMIT = 7.0

# What makes one point one point.
KEYS = ("codec", "variant", "direction", "image", "q", "sub",
        "threads", "batch", "path")


def say(*a):
    print(*a)
    sys.stdout.flush()


def median(values):
    v = sorted(values)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def spread_pct(values):
    if len(values) < 2:
        return 0.0
    m = median(values)
    return (100.0 * (max(values) - min(values)) / m) if m else 0.0


def read_rows(path):
    rows, bad = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    return rows, bad


def group(rows):
    """Repeats of one point, gathered."""
    out = {}
    for r in rows:
        if not r.get("fps"):
            continue
        key = tuple(r.get(k) for k in KEYS)
        out.setdefault(key, []).append(float(r["fps"]))
    return out


def name_of(key):
    codec, variant, direction, image, q, sub, threads, batch, path = key
    return ("%-3s %-8s %s %-3s q%-3s %-4s %sx%s%s"
            % (codec or "?", variant or "", direction or "?", image or "?",
               q if q is not None else "?", sub or "",
               threads if threads is not None else "?",
               batch if batch is not None else "?",
               (" " + path) if path else ""))


def main():
    ap = argparse.ArgumentParser(add_help=True, description=VERSION)
    ap.add_argument("--do", action="store_true",
                    help="also write the answer to a csv file next to this"
                         " script; without this nothing is written")
    a = ap.parse_args()

    say(VERSION)
    say("no arguments: show the answer.  --do: and write it to a csv."
        "  Nothing is measured either way.")
    say("project folder: %s" % HERE)

    if not os.path.isdir(RUNS):
        say("\nNo runs folder next to this file (%s)." % RUNS)
        return 1

    folders = [f for f in sorted(glob.glob(os.path.join(RUNS, "*")))
               if os.path.isfile(os.path.join(f, "results.jsonl"))]
    if not folders:
        say("\nNo run under %s holds a results.jsonl." % RUNS)
        return 1

    say("\nRuns with results.jsonl: %d" % len(folders))
    all_rows = []
    for folder in folders:
        rows, bad = read_rows(os.path.join(folder, "results.jsonl"))
        points = group(rows)
        wide = {k: v for k, v in points.items()
                if len(v) > 1 and spread_pct(v) > SPREAD_LIMIT}
        once = [k for k, v in points.items() if len(v) == 1]
        say("\n%s" % os.path.basename(folder))
        say("   %d rows%s, %d points, %d measured once only"
            % (len(rows), (", %d unreadable" % bad) if bad else "",
               len(points), len(once)))
        if not points:
            continue
        say("   repeats disagree by more than %.0f %%: %d point(s) of %d"
            % (SPREAD_LIMIT, len(wide), len(points)))
        if wide:
            say("")
            say("   %-46s %6s %5s  %s"
                % ("point", "spread", "reps", "values"))
            for k in sorted(wide, key=lambda k: -spread_pct(points[k])):
                v = points[k]
                say("   %-46s %5.1f %% %5d  %s"
                    % (name_of(k), spread_pct(v), len(v),
                       ", ".join("%.0f" % x for x in v)))
        for k, v in points.items():
            all_rows.append((os.path.basename(folder), k, v))

    total = len(all_rows)
    wide_all = [x for x in all_rows
                if len(x[2]) > 1 and spread_pct(x[2]) > SPREAD_LIMIT]
    once_all = [x for x in all_rows if len(x[2]) == 1]
    say("\n" + "=" * 72)
    say("Points in all: %d. Never settled to within %.0f %%: %d."
        " Measured once only, so not checkable: %d."
        % (total, SPREAD_LIMIT, len(wide_all), len(once_all)))
    if wide_all:
        say("A median of repeats that disagree by tens of per cent is not a"
            " measurement. Every point above wants either more repeats or an"
            " explanation before its number is used anywhere.")
    say("=" * 72)

    if not a.do:
        say("\nNothing written. Add --do to put this into a csv file.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(HERE, "spread-check-%s.csv" % stamp)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("run," + ",".join(KEYS)
                 + ",repeats,median_fps,spread_pct,verdict,values\n")
        for run, key, v in all_rows:
            sp = spread_pct(v)
            fh.write("%s,%s,%d,%.2f,%.2f,%s,%s\n"
                     % (run,
                        ",".join("" if x is None else str(x) for x in key),
                        len(v), median(v), sp,
                        "once" if len(v) == 1
                        else ("wide" if sp > SPREAD_LIMIT else "ok"),
                        " ".join("%.2f" % x for x in v)))
    say("\nwritten: %s" % path)
    say("One line per point, every run. Send it over and we will go through"
        " the wide ones together.")
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
        say("\nStopped.")
        finish(1)
