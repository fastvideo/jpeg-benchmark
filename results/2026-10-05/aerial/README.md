# Aerial frames 65–151 MPix, RTX 4090 and CPU, 5 October 2026

Article (Russian): [Декодирование больших JPEG на видеокарте: аэрофотоснимки 50–150 Мпикс](https://www.fastvideo.ru/blog/aerial-jpeg-gpu-decoding.htm?utm_source=github&utm_medium=readme&utm_campaign=jpeg-benchmark)

Machine: NVIDIA GeForce RTX 4090 (24 GB), AMD Ryzen 9 7950X (16 cores, 32 threads), Windows 11.
Fastvideo SDK 0.23.1.0 (`JpegSample`), nvJPEG 13.2.1.68 (our `nvjpeg_bench`), libjpeg-turbo 3.2.0
(`tjbench`, output BGR).

## Two runs, one data set

| Folder | Script | What it measured |
|---|---|---|
| `run-20261005-111244/` | `bench/jpeg-aerial-07.py --do` | one thread without transfers (step 3) for frames 02, 06, 03; stopped by hand in step 4 — see below |
| `run-20261005-135612/` | `bench/jpeg-aerial-08.py --do` | step 3 for frame 04; with transfers in one thread (step 4a) and in several threads (step 4); CPU in 1, 8 and 16 processes (step 5), all four frames |

Each run folder holds the files exactly as the script wrote them, with `logs/` zipped into `logs.zip`.

Version 07 stopped in step 4 because the encoded variants of frame 04 left in the work folder by an
earlier, interrupted run were 4096-byte files with a header and no data; version 07 trusted the header and
fed them to the decoders. Version 08 checks the whole file (end-of-image marker and at least 0.05 byte per
pixel), sets a timeout on every launch and checks that the card's memory is free before it starts. The
measuring code of step 3 is the same in both versions, so the step-3 numbers of the two runs are merged.

The files in this folder are that merge, the numbers of the article:

- `frames.csv` — the four source files: size, subsampling, restart interval, segments;
- `variants.csv` — the source file and the three files of the Fastvideo encoder (q = 95, 4:4:4 / 4:2:2 /
  4:2:0) for every frame: bytes, restart interval, segments;
- `speeds.csv` — step 3, one thread without transfers, rows of both runs (02, 06, 03 from the first,
  04 from the second);
- `copies1.csv` — with transfers to the card and back, one thread;
- `copies.csv` — with transfers, several threads: 8, or fewer when 8 threads of the Fastvideo decoder
  would ask for more GPU memory than is free (6, 5 and 4 threads for frames 06, 03 and 04);
- `cpu.csv`, `cpu-version.txt` — libjpeg-turbo, 1, 8 and 16 processes;
- `logs.zip` — the step-3 logs of both runs (178 files); the Fastvideo decoder's speed is taken from the
  time these logs print, not from the rounded FPS.

`python bench/jpeg-aerial-charts-02.py results/2026-10-05/aerial <folder>` draws the four pictures of
the article from this folder; they come out byte for byte the same as on the page.

## Frames

Frame numbers are the names of the source files: 02 (65 MPix, 9344×7000), 06 (104 MPix, 11272×9200),
03 (129 MPix, 13468×9564), 04 (151 MPix, 14192×10640). All are baseline, 8 bit, 4:4:4. Only 02 has restart
markers (every 10 MCUs). The frames themselves are not published.

## Main numbers

Milliseconds per frame, the file with restart markers (q = 95, 4:4:4), the same file on the card and on the CPU.

| Frame, MPix | 65 | 104 | 129 | 151 |
|---|---:|---:|---:|---:|
| Fastvideo decoder, file and picture in GPU memory | 8.3 | 10 | 14 | 12 |
| nvJPEG, file and picture in GPU memory | 300 | 420 | 530 | 470 |
| Fastvideo decoder, with transfers from host memory and back, one thread | 16 | 22 | 29 | 29 |
| libjpeg-turbo, 1 process | 330 | 430 | 560 | 540 |
| libjpeg-turbo, 8 processes (total time / frames) | 43 | 56 | 73 | 71 |
| libjpeg-turbo, 16 processes (total time / frames) | 23 | 30 | 39 | 38 |

On a file without restart markers the Fastvideo decoder does entropy decoding on the CPU in one thread and
has no advantage over the CPU; the same 151 MPix picture decodes 69 times faster from the file with markers.
