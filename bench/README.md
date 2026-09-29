# bench — how to repeat the measurements

Everything here ran on Windows 11 with an RTX 4090 on 25.09.2026; `jpeg-big-06.py` on
29.09.2026. The scripts also
contain a Linux path (g++ build of `nvjpeg_bench`), which was not used for these results.

## What to install

| What | Why | Where from |
|---|---|---|
| Python 3 (the run used 3.13) | every script | python.org |
| `pip install nvidia-ml-py` | GPU energy counter; without it energy is not measured | PyPI |
| CUDA Toolkit 13.x | nvJPEG comes with it (`nvjpeg.h`, `nvjpeg.lib`) | NVIDIA |
| Visual Studio C++ compiler (`cl`) | builds `nvjpeg_bench`; run from an "x64 Native Tools Command Prompt" | Microsoft |
| Fastvideo SDK with `JpegSample` and the frames `2k_wild.ppm`, `4k_wild.ppm` | the Fastvideo side | on request, fastcompression.com |
| `jpegtran` from libjpeg-turbo | markers test: the same image without markers and with a marker after every MCU | libjpeg-turbo.org |
| Nsight Systems | profiler run | NVIDIA |

## One project folder

```
D:\_Test\jpeg\                  the scripts; every run writes its own folder here
  nvjpeg_bench\nvjpeg_bench-01.cpp   built into nvjpegEncoderSample / nvjpegDecoderSample
  nvidia-jpeg-sample\           NVIDIA's programs, fetched by get-nvidia-jpeg-sample-01.py
```

The Fastvideo SDK stays where it is installed; `jpeg-bench-13.py --fv <folder>` names it.

## Order

1. `python make-big-frames-02.py --src <SDK folder>\4k_wild.ppm --do` — makes
   `8k_wild.ppm`, `12k_wild.ppm` and `16k_wild.ppm` next to the 4K frame: 2×2, 3×3 and 4×4
   copies of it. The frames are 100, 224 and 398 MB, so they are not in the repository.
2. `python jpeg-bench-13.py --build` — builds `nvjpeg_bench`.
3. `python jpeg-bench-13.py --probe` — runs every program once and shows what was read
   from its output.
4. `python jpeg-run-all-04.py` — shows the plan and checks readiness; with `--do` runs
   bench, spread check, profiler, 4:2:0 and markers in order and collects the small result
   folders into `all-out\<date>\`.
5. `python jpeg-big-06.py --do` — 12K and 16K, one thread, both codecs, both ways, 4:4:4
   and 4:2:0. Before that it checks that the 8K frame of the run is 2×2 copies of the 4K
   frame and reads the MCU and restart interval of the Fastvideo encoder at 4:4:4, 4:2:2
   and 4:2:0. Needs `make-big-frames-02.py` in the same folder and the run of step 4
   under `runs\` (`runs\USE-THIS-RUN.txt` names it); the command lines are taken from it.
   The card must be free of other programs: the script waits for 30 s of an idle card
   before measuring, samples the card once a second during the run (`card.csv`), and
   leaves out of the median any repeat run at more than 20 % above the quietest power
   (`repeats.csv`, column `clean`). Its results folder holds text files only.

Every script started without arguments only shows what it would do and measures nothing.

## Two things to know

- **`jpeg-run-all-04.py` takes the script with the highest number** of each kind. With
  both `jpeg-420-11.py` and `jpeg-420-12.py` in the folder it takes 12, which measures
  8K only by default. On 25.09 the 2K and 4K points were measured by version 11 inside
  the full run, and 8K separately by `jpeg-420-12.py --do`. To get all three frames in one
  go: `python jpeg-420-12.py --do --frames 2k,4k,8k`.
- **Before every measured repeat the card is checked for someone else's load** through
  `nvidia-smi`, and the repeat waits while it is busy. Close other GPU programs first.
