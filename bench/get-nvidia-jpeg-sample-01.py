#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# get-nvidia-jpeg-sample-01.py
# version 01 of 20.09.2026
#
# Fetches NVIDIA's own nvJPEG sample programs, prints the size and md5 of
# every file, and builds them. Changes nothing in NVIDIA's files - that is the
# whole point. The JPEG counterpart of get-nvidia-sample-02.py (JPEG2000).
#
# WHY THE FILES ARE DOWNLOADED AND NOT SHIPPED
#
# Step 1 of the benchmark answers "what does NVIDIA's product give as it is".
# The answer holds only while the program is theirs, word for word. A copy
# that passed through our hands does not carry that weight: any reader may
# ask whether we edited it. So the files come from NVIDIA's repositories and
# the size and md5 of each is printed - it is visible what exactly was taken.
#
# WHICH PROGRAMS, AND WHY THESE (checked on 20.09.2026)
#
#   nvJPEG          decoder  NVIDIA/cuda-samples, tag v13.2
#   nvJPEG_encoder  encoder  NVIDIA/cuda-samples, tag v13.2
#   nvjpegDecoder   decoder  NVIDIA/CUDALibrarySamples, master - optional
#
#   - cuda-samples is pinned to the tag v13.2 on purpose: both nvJPEG samples
#     exist in the tags v12.4 ... v13.2 and are GONE from v13.3 and from the
#     master branch. A tag does not move; a branch does.
#   - nvJPEG_encoder is the only stand-alone encoder NVIDIA publishes. It
#     reads JPEG files, decodes each outside its timer and times the encode
#     plus the copy of the stream to host memory (two calls of
#     nvjpegEncodeRetrieveBitstream). It has no -repeat: one file, one
#     measurement, and the first call's start-up lands in the average - so the
#     benchmark runs it on N and on 2N copies and takes the difference.
#   - nvjpegDecoder (CUDALibrarySamples) is the only NVIDIA program that tries
#     the HARDWARE JPEG decoder by itself: it creates the hardware backend and,
#     if the card has no engine, prints "Hardware Decoder not supported.
#     Falling back to default backend". On the RTX 4090 its words are a fourth,
#     independent witness of whether the engine is there. It is optional: it
#     is not written for Windows (see below).
#
# HOW THEY ARE BUILT, AND WHY NOT WITH THEIR CMAKE
#
# With one compiler command each, from their unmodified files - the same way
# jpeg-bench-01.py builds our harness. Their CMake files cannot be used as
# they are on Windows: the CUDALibrarySamples one hard-codes the library path
# "CUDA/v10.1/lib/x64", and the cuda-samples ones expect the whole repository
# tree around them. The commands below were verified on 20.09.2026 against
# the real nvjpeg.h 12.4 (Linux, g++): all three compile and link.
#
#   - cuda-samples programs need five headers of its Common folder; they are
#     downloaded next to the samples.
#   - nvjpegDecoder is written for nvcc, which quietly includes cuda_runtime.h
#     (its templated cudaMalloc accepts any pointer). A plain C++ compiler is
#     given the same include by a flag: /FIcuda_runtime.h (cl) or
#     -include cuda_runtime.h (g++). NVIDIA's code is not touched.
#   - nvjpegDecoder includes <dirent.h> unconditionally; Visual C++ has no
#     such header. On Windows the single-file MIT implementation by Toni Ronkko
#     (github.com/tronkko/dirent) is put into a separate folder "shim" and put
#     on the include path. Again, not a change to NVIDIA's code; if the build
#     still fails, this program is simply left out - it is a witness, not a
#     measurement the article depends on.
#
#     python get-nvidia-jpeg-sample-01.py                 download only
#     python get-nvidia-jpeg-sample-01.py --build         download and build
#     python get-nvidia-jpeg-sample-01.py --dir D:\nvjpeg-sample --build
#
# Running it again is safe and gives the same result: the files are simply
# overwritten with the same bytes. Without network access the script prints
# the addresses; the files can then be taken with a browser into the same
# folders.

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request

VERSION = "get-nvidia-jpeg-sample-01 of 20.09.2026"

RAW = "https://raw.githubusercontent.com/NVIDIA/%s/%s/%s"
CS, CS_REF = "cuda-samples", "v13.2"
CLS, CLS_REF = "CUDALibrarySamples", "master"
DIRENT = "https://raw.githubusercontent.com/tronkko/dirent/master/%s"

EXE = ".exe" if os.name == "nt" else ""

# The program names are those of add_executable in their CMakeLists.txt, not
# of a README: for JPEG2000 the two disagreed at NVIDIA, so the rule is to
# take the name from what actually builds.
SAMPLES = [
    {"key": "decoder", "repo": CS, "ref": CS_REF,
     "path": "Samples/4_CUDA_Libraries/nvJPEG",
     "files": ["nvJPEG.cpp", "CMakeLists.txt"], "optional": ["README.md"],
     "exe": "nvJPEG", "source": "nvJPEG.cpp", "common": True,
     "what": "decoding: single, -batched, -pipelined; CUDA path only",
     "required": True},
    {"key": "encoder", "repo": CS, "ref": CS_REF,
     "path": "Samples/4_CUDA_Libraries/nvJPEG_encoder",
     "files": ["nvJPEG_encoder.cpp", "CMakeLists.txt"],
     "optional": ["README.md"],
     "exe": "nvJPEG_encoder", "source": "nvJPEG_encoder.cpp", "common": True,
     "what": "encoding, one image per file", "required": True},
    {"key": "decoder-hw", "repo": CLS, "ref": CLS_REF,
     "path": "nvJPEG/nvJPEG-Decoder",
     "files": ["nvjpegDecoder.cpp", "nvjpegDecoder.h", "CMakeLists.txt"],
     "optional": ["README.md"],
     "exe": "nvjpegDecoder", "source": "nvjpegDecoder.cpp", "common": False,
     "what": "decoding; tries the hardware JPEG engine by itself",
     "required": False},
]
COMMON = ["helper_nvJPEG.hxx", "helper_cuda.h", "helper_timer.h",
          "helper_string.h", "exception.h"]
LICENSES = [(CS, CS_REF, ["LICENSE", "LICENSE.md", "LICENSE.txt"]),
            (CLS, CLS_REF, ["LICENSE.TXT", "LICENSE", "LICENSE.md"])]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fastvideo-bench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def save(folder, name, data):
    with open(os.path.join(folder, name), "wb") as fh:
        fh.write(data)
    print("  %-26s %8d bytes   md5 %s" % (name, len(data),
                                          hashlib.md5(data).hexdigest()))


def fetch(out):
    """Every file, with its md5. Returns (all required present, failed urls)."""
    ok, failed = True, []
    for s in SAMPLES:
        folder = os.path.join(out, s["key"])
        os.makedirs(folder, exist_ok=True)
        print("\n%s - %s\n  from %s @ %s, %s" % (s["key"], s["what"], s["repo"],
                                                s["ref"], s["path"]))
        for name in s["files"] + s["optional"]:
            url = RAW % (s["repo"], s["ref"], s["path"] + "/" + name)
            try:
                save(folder, name, get(url))
            except Exception as exc:
                if name in s["optional"]:
                    print("  %-26s not in the repository; not needed to build"
                          % name)
                    continue
                print("  %-26s NOT DOWNLOADED: %s" % (name, exc))
                failed.append(url)
                if s["required"]:
                    ok = False
    folder = os.path.join(out, "Common")
    os.makedirs(folder, exist_ok=True)
    print("\nCommon - the five helper headers the cuda-samples programs use\n"
          "  from %s @ %s, Common" % (CS, CS_REF))
    for name in COMMON:
        url = RAW % (CS, CS_REF, "Common/" + name)
        try:
            save(folder, name, get(url))
        except Exception as exc:
            print("  %-26s NOT DOWNLOADED: %s" % (name, exc))
            failed.append(url)
            ok = False
    print("\nLicences")
    for repo, ref, names in LICENSES:
        for name in names:
            try:
                data = get(RAW % (repo, ref, name))
            except Exception:
                continue
            save(out, "%s-%s" % (repo, name), data)
            break
        else:
            print("  %s: licence file not found - see the repository" % repo)
    if os.name == "nt":
        folder = os.path.join(out, "shim")
        os.makedirs(folder, exist_ok=True)
        print("\nshim - dirent.h for Visual C++ (MIT, Toni Ronkko), only for "
              "nvjpegDecoder")
        for name, target in (("include/dirent.h", "dirent.h"),
                             ("LICENSE", "dirent-LICENSE")):
            try:
                save(folder, target, get(DIRENT % name))
            except Exception as exc:
                print("  %-26s NOT DOWNLOADED: %s (nvjpegDecoder will not "
                      "build; it is optional)" % (target, exc))
    return ok, failed


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

def _vcvars():
    pf = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = os.path.join(pf, "Microsoft Visual Studio", "Installer",
                           "vswhere.exe")
    if not os.path.exists(vswhere):
        return None
    try:
        out = subprocess.run([vswhere, "-latest", "-products", "*", "-requires",
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


def command_for(s, out, cuda):
    """The one compiler command for one program: (argv or string, shell)."""
    folder = os.path.join(out, s["key"])
    src = os.path.join(folder, s["source"])
    target = os.path.join(out, s["exe"] + EXE)
    if os.name == "nt":
        inc = os.path.join(cuda, "include")
        lib = os.path.join(cuda, "lib", "x64")
        cl = ["cl", "/nologo", "/EHsc", "/O2", "/MD", "/std:c++17",
              "/D_CRT_SECURE_NO_WARNINGS", "/I" + inc, "/I" + folder]
        if s["common"]:
            cl += ["/I" + os.path.join(out, "Common")]
        else:
            cl += ["/FIcuda_runtime.h", "/I" + os.path.join(out, "shim")]
        cl += [src, "/Fo" + os.path.join(folder, ""), "/Fe:" + target,
               "/link", "/LIBPATH:" + lib, "cudart.lib", "nvjpeg.lib"]
        return cl, target
    inc = os.path.join(cuda, "include")
    libs = [d for d in (os.path.join(cuda, "lib64"), os.path.join(cuda, "lib"),
                        os.path.join(cuda, "targets", "aarch64-linux", "lib"))
            if os.path.isdir(d)]
    cmd = [os.environ.get("CXX", "g++"), "-O2", "-std=c++17", "-I" + inc,
           "-I" + folder]
    if s["common"]:
        cmd += ["-I" + os.path.join(out, "Common")]
    else:
        cmd += ["-include", "cuda_runtime.h"]
    cmd += [src, "-o", target]
    for d in libs:
        cmd += ["-L" + d, "-Wl,-rpath," + d]
    cmd += ["-lnvjpeg", "-lcudart"]
    return cmd, target


def build(out):
    if os.name == "nt":
        cuda = os.environ.get("CUDA_PATH", "")
    else:
        cuda = next((r for r in (os.environ.get("CUDA_HOME", ""),
                                 os.environ.get("CUDA_PATH", ""),
                                 "/usr/local/cuda")
                     if r and os.path.isdir(os.path.join(r, "include"))), "")
    if not cuda or not os.path.exists(os.path.join(cuda, "include",
                                                   "nvjpeg.h")):
        print("\nnvjpeg.h not found under %s. nvJPEG comes with the CUDA "
              "Toolkit; set %s to it." % (cuda or "(nothing)",
                                          "CUDA_PATH" if os.name == "nt"
                                          else "CUDA_HOME"))
        return 1
    vcvars = None
    if os.name == "nt" and not shutil.which("cl"):
        vcvars = _vcvars()
        if not vcvars:
            print("\nThe Microsoft compiler was not found. Run this from an "
                  "'x64 Native Tools Command Prompt for VS'.")
            return 1
    logdir = os.path.join(out, "build-logs")
    os.makedirs(logdir, exist_ok=True)
    print("\nBUILDING (CUDA at %s)" % cuda)
    rc = 0
    for s in SAMPLES:
        cmd, target = command_for(s, out, cuda)
        if not os.path.exists(os.path.join(out, s["key"], s["source"])):
            print("  %-16s source missing, skipped" % s["exe"])
            rc = rc or (1 if s["required"] else 0)
            continue
        if os.path.exists(target):
            os.remove(target)          # a failed build must not leave an old one
        if vcvars:
            run, shell = 'call "%s" >nul && %s' % (
                vcvars, subprocess.list2cmdline(cmd)), True
        else:
            run, shell = cmd, False
        p = subprocess.run(run, shell=shell, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=900)
        with open(os.path.join(logdir, "build_%s.log" % s["exe"]), "w",
                  encoding="utf-8") as fh:
            fh.write(subprocess.list2cmdline(cmd) + "\n\n"
                     + p.stdout.decode("utf-8", "replace"))
        if os.path.exists(target):
            print("  %-16s built: %s" % (s["exe"], target))
        else:
            print("  %-16s NOT BUILT - see %s%s"
                  % (s["exe"], os.path.join(logdir, "build_%s.log" % s["exe"]),
                     "" if s["required"] else " (optional, the run goes on "
                     "without it)"))
            if s["required"]:
                rc = 1
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="nvidia-jpeg-sample",
                    help="where to put the sources and the built programs; "
                         "jpeg-bench-01.py looks in ./nvidia-jpeg-sample")
    ap.add_argument("--build", action="store_true",
                    help="build the programs after downloading")
    ap.add_argument("--no-download", action="store_true",
                    help="build what is already in --dir")
    args = ap.parse_args()

    print(VERSION)
    out = os.path.abspath(args.dir)
    os.makedirs(out, exist_ok=True)
    print("Folder: %s" % out)

    if not args.no_download:
        ok, failed = fetch(out)
        if not ok:
            print("\nSome files did not download. Addresses, for a browser:")
            for u in failed:
                print("  %s" % u)
            print("Put them into the same folders and run again with "
                  "--no-download --build.")
            return 1
    if args.build:
        rc = build(out)
        if rc:
            return rc
    else:
        print("\nDownloaded. To build: the same command with --build.")
        return 0

    print("""
CHECK THAT THEY ARE THERE (no card needed for -h)

    %(d)s%(s)snvJPEG%(e)s -h
    %(d)s%(s)snvJPEG_encoder%(e)s -h

WHAT THE BENCHMARK DOES WITH THEM (step 1)

    jpeg-bench-01.py finds them in ./nvidia-jpeg-sample and runs, next to
    our harness, in the same boundary (the pixels stay on the card, the
    compressed stream is on the host side):
      nvJPEG          single, -batched -b B          "Avg images per sec"
      nvJPEG_encoder  N and 2N copies of one frame   "Total time spent on
                                                      encoding"
      nvjpegDecoder   whether it says "Hardware Decoder not supported"

    Their keys, from their own usage lines:
      nvJPEG          -i dir [-b batch] [-t total] [-w warmup] [-o dir]
                      [-pipelined] [-batched] [-fmt format]
      nvJPEG_encoder  -i dir [-o dir] [-q quality] [-s 420/444]
                      [-fmt format] [-huf 0]
                      defaults: -q 70, -s 420, -fmt yuv - all three are set
                      explicitly by the benchmark, or it compares something
                      else
""" % {"d": out, "s": os.sep, "e": EXE})
    return 0


if __name__ == "__main__":
    sys.exit(main())
