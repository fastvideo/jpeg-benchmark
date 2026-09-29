# JPEG on the GPU: Fastvideo JPEG vs NVIDIA nvJPEG on RTX 4090

A reproducible benchmark of two baseline JPEG codecs on one NVIDIA GeForce RTX 4090:
the Fastvideo JPEG codec from the Fastvideo SDK and the nvJPEG library from the CUDA
Toolkit. Same frames, same quantization tables, same card, same day. The test scripts,
our nvJPEG measurement program, every summary and every log of 25 and 29 September 2026
are here.

**Article (Russian):** [Бенчмарки кодеков Fastvideo JPEG и NVIDIA nvJPEG на RTX 4090](https://www.fastvideo.ru/blog/fastvideo-jpeg-vs-nvjpeg.htm?utm_source=github&utm_medium=readme&utm_campaign=jpeg-benchmark)
**Article (English):** coming soon on fastcompression.com.

## Key results

RTX 4090, Fastvideo SDK 0.23.1.0, nvJPEG 13.2.1.68, quality 90, 8 bit, three channels,
4:4:4 unless stated otherwise. Frames per second, one thread, no transfers between host
memory and the card.

| | Fastvideo JPEG | nvJPEG | Difference |
|---|---:|---:|---|
| Encoding 2K (1920×1080) | 5595 | 4313 | Fastvideo +30 % |
| Encoding 4K (3840×2160) | 2805 | 2014 | Fastvideo +39 % |
| Encoding 8K (7680×4320) | 715 | 541 | Fastvideo +32 % |
| Decoding 2K | 1862 | 191 | Fastvideo ×9.7 |
| Decoding 4K | 1640 | 59.1 | Fastvideo ×28 |
| Decoding 8K | 515 | 14.9 | Fastvideo ×35 |
| Encoding 12K (11520×6480) | 335 | 248 | Fastvideo +35 % |
| Encoding 16K (15360×8640) | 191 | 137 | Fastvideo +39 % |
| Decoding 12K | 221 | 6.05 | Fastvideo ×37 |
| Decoding 16K | 118 | 3.43 | Fastvideo ×34 |

12K and 16K were measured in a separate run on 29 September 2026 on the same machine
(`results/2026-09-29/`); all other rows on 25 September 2026.


- **Encoding 8K with 4:2:0 subsampling:** Fastvideo JPEG 963 fps, nvJPEG 749 fps.
- **Big frames, 12K and 16K:** in GB/s of uncompressed data the Fastvideo codec runs about as
  fast as on 8K (encoding 75 and 76 GB/s against 71; decoding 49 and 47 GB/s against 51):
  its performance saturates already on the 8K frame, so processing several big frames at
  once gives nothing. With 4:2:0 every codec gets faster on 12K and 16K, the Fastvideo
  decoder included (+33 and +40 %).
- **The Fastvideo encoder itself is much faster than PCIe:** 963 fps on 8K, q = 90, 4:2:0 is
  96 GB/s of uncompressed data, 3.8 times the PCIe bandwidth of this machine (25.5 GB/s).
  When the source image is already in GPU memory, as in a camera processing pipeline,
  encoding is not limited by the bus: only the compressed image, several times smaller,
  goes over PCIe.
- **Why the Fastvideo decoder is so fast: restart markers.** The Fastvideo encoder always
  writes restart markers, with a fixed restart interval for each subsampling: 10 MCUs at
  4:4:4, 8 at 4:2:2, 5 at 4:2:0 and 32 for greyscale (30–32 data units, 8×8 blocks, per
  segment); nvJPEG never writes them. They split the scan into restart intervals, and the
  compressed data of each interval - an entropy-coded segment (ITU-T T.81, 3.1.51) - can be
  decoded on its own: DC coding starts from zero after every marker. The GPU decodes the
  segments at once. For 2.1–2.9 % of file size the Fastvideo decoder runs 19–75 times
  faster than on the same image without markers.
- **Where this decoding speed applies:** only on images with restart markers — encoded by
  the Fastvideo encoder, or prepared beforehand by other software such as `jpegtran`. On an
  image without markers (or with a restart interval above 255) the Fastvideo decoder runs
  entropy decoding on the CPU, as its manual says, and has no advantage. The fastest variant, a marker
  after every MCU, is not made by the Fastvideo encoder (it puts one every 10 MCUs); it is
  prepared offline and costs 21–29 % of file size.
- **Where nvJPEG spends its decoding time:** 91–93 % of the frame goes to stream parsing
  and entropy decoding on the CPU; the GPU gets 0.7–1.9 %.
- **Maximum performance in multithreaded mode, with transfers to the card and back**
  (optimal parameters for each codec): Fastvideo decodes 2K at 3632 fps, 4K at 895 fps, 8K at 193 fps; nvJPEG decodes 2K at 1161 fps,
  4K at 268 fps, 8K at 65 fps. Fastvideo is 3.0–3.3 times faster.
- **Energy consumed by the GPU, encoding:** to encode one frame, the GPU uses 1.1–1.3 times
  less energy with Fastvideo than with nvJPEG. Decoding energy is not compared: most of the nvJPEG decoder's work runs on the CPU,
  which the card's counter does not see.

## Test setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, driver 610.88, no hardware JPEG engine |
| CPU / OS | 32 logical cores, Windows 11 |
| Fastvideo JPEG | Fastvideo SDK 0.23.1.0, measured with `JpegSample` from the SDK |
| nvJPEG | CUDA Toolkit, library 13.2.1.68, measured with our `nvjpeg_bench` (source here) |
| Frames | 2K, 4K and 8K; 12K and 16K in a separate run. 8K, 12K and 16K are copies of the 4K frame, 2×2, 3×3 and 4×4, made by `bench/make-big-frames-02.py` |
| Quality | 90, same quantization tables for both codecs |
| Repeats | 3 per point, median; up to 7 when repeats scatter more than 7 %. 12K and 16K: at least 3 clean repeats, up to 10, the card sampled once a second and repeats with outside load left out (none occurred); the widest scatter is 7.6 % (nvJPEG decoding 16K, 10 repeats) |

The method is described on the page
[Методика тестирования кодеков JPEG и JPEG2000 на CPU и GPU](https://www.fastvideo.ru/benchmarks/codec-benchmark-methodology.htm?utm_source=github&utm_medium=readme&utm_campaign=jpeg-benchmark).

## What is in this repository

```
bench/                            the test scripts, as they ran on 25.09.2026 (jpeg-big-06.py on 29.09)
  README.md                       what to install and in which order to run
  jpeg-run-all-04.py              runs the steps below in order
  jpeg-tidy-02.py                 clears the project folder before a full run (moves, never deletes)
  jpeg-bench-13.py                speed, threads and batches, energy, stages; builds nvjpeg_bench
  jpeg-spread-check-02.py         how far the repeats of every point scatter
  nsys-jpeg-20.py                 Nsight Systems profiler run
  jpeg-420-11.py                  4:2:0 against 4:4:4 on 2K and 4K
  jpeg-420-12.py                  the same on 8K (--frames chooses the frames)
  jpeg-markers-07.py              the same image with and without restart markers
  make-big-frames-02.py           makes big frames (8K, 12K, 16K or any size) out of the 4K frame
  jpeg-big-06.py                  big frames, one thread: 12K and 16K; first checks the 8K frame
                                  and the MCU and restart interval of the Fastvideo encoder
  jpeg-sof-01.py                  reads a JPEG header: sampling factors, MCU, restart interval, segments
  get-nvidia-jpeg-sample-01.py    fetches NVIDIA's own JPEG sample programs
  nvjpeg_bench/nvjpeg_bench-01.cpp  our nvJPEG measurement program
results/2026-09-25/
  bench/                          summary.txt, results.csv / .json / .jsonl, spread.csv,
                                  logs.zip - every launch of the main run, 356 logs
  markers/                        speeds and file sizes with and without markers
  sub420/                         4:2:0 against 4:4:4, 2K and 4K
  sub420-8k/                      4:2:0 against 4:4:4, 8K
  profiler/                       kernels, copies and GPU metrics per point
  spread-check.csv
  logs.zip                        markers, 4:2:0 and profiler runs, every repeat,
                                  and the six general logs of the whole run
results/2026-09-29/
  big/                            12K and 16K, jpeg-big-06.py: speeds.csv (the table),
                                  repeats.csv (every repeat with card power), sizes.csv
                                  (bytes, MCUs, restart interval, segments), checks.txt
                                  (8K frame and encoder MCU checks), card.csv (the card once
                                  a second), console.txt, logs.zip - every launch, 73 logs
```

`JpegSample` is part of the Fastvideo SDK and is not included. To run the Fastvideo side,
[request the SDK](https://www.fastcompression.com/products/sdk.htm?utm_source=github&utm_medium=readme&utm_campaign=jpeg-benchmark).

## How to reproduce

See `bench/README.md`. In short: put `bench/` into one project folder, install what it
lists, run every script without arguments first — it prints what it found and what it
would do, and measures nothing — then start the real run by name (`--final` or `--do`).

## Licenses

Code: `LICENSE` (MIT). Texts, tables and logs: `CONTENT-LICENSE.md` (CC BY 4.0).
Third-party software: `THIRD-PARTY.md`.

## How to cite

See `CITATION.cff`. A DOI on Zenodo will be added with the first release.
