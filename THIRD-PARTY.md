# Third-party software

This repository contains no third-party binaries. Everything listed below is
obtained by the user directly from its vendor and is governed by that vendor's
own licence terms.

| Component                   | Where it comes from                       | Terms             |
|-----------------------------|-------------------------------------------|-------------------|
| Fastvideo SDK (`JpegSample`) | fastcompression.com                       | Fastvideo licence |
| nvJPEG                      | NVIDIA, part of the CUDA Toolkit          | NVIDIA licence    |
| CUDA Toolkit, driver        | NVIDIA                                    | NVIDIA licence    |
| Nsight Systems              | NVIDIA                                    | NVIDIA licence    |
| NVIDIA JPEG samples         | NVIDIA `cuda-samples`, `CUDALibrarySamples` | BSD 3-Clause     |
| `jpegtran`                  | libjpeg-turbo                             | IJG / BSD-style   |
| `nvidia-ml-py` (pynvml)     | PyPI                                      | BSD               |

**Fastvideo SDK.** `JpegSample` is part of the SDK and is not redistributed
here. The speed figures can be reproduced with the freely downloadable demo
build; the SDK itself is provided on request.

**nvJPEG** ships with the CUDA Toolkit: `nvjpeg.h` and `nvjpeg.lib` come with
it, nothing has to be installed separately. The measurement program
`nvjpeg_bench` in this repository is our own source code and is covered by
`LICENSE`.

**NVIDIA's own JPEG samples** are fetched from NVIDIA's repositories by
`bench/get-nvidia-jpeg-sample-01.py`; nothing of theirs is copied here.

**jpegtran** from libjpeg-turbo is used by `bench/jpeg-markers-07.py` to write
the same image without restart markers and with a marker after every MCU. It
re-writes the file without re-encoding.

**Test images** (`2k_wild.ppm`, `4k_wild.ppm`, `8k_wild.ppm`) ship with the
Fastvideo SDK and are not redistributed here.
