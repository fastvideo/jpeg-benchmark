// nvjpeg_bench version 01, 20 September 2026 - first version
// ---------------------------------------------------------------------------
// nvJPEG (baseline 8-bit JPEG) benchmark harness
//
// THE VERSION IS IN THREE PLACES AT ONCE, ON PURPOSE:
//   - in the file name          nvjpeg_bench-01.cpp, visible in the folder
//   - on the first line above    visible the moment the file is opened
//   - in the built program       nvjpegDecoderSample -version prints it and
//                                stops, measuring nothing
// An older executable does not know -version and prints its usage instead,
// which is answer enough. jpeg-bench-01.py asks both executables for their
// version before the first measurement and stops the run if it does not match
// this file. That is what makes "which build produced these numbers" a
// question with an answer rather than a guess. This mirrors nvj2k_bench.
//
// Build two executables from this one source:
//   nvjpegEncoderSample.exe   (default)
//   nvjpegDecoderSample.exe   (compile with /DBUILD_DECODER, or -DBUILD_DECODER)
//
// It mirrors the command line of the Fastvideo JPEG sample (JpegSample), so
// that one and the same run script and parser can drive both codecs:
//   JpegSample -i in.ppm -o out.jpg -q 90 -s 444 [-info] [-repeat N]        (encode)
//   JpegSample -i in.ppm -o out.jpg -async -thread T -threadR R -threadW W -b B  (encode, MT)
//   JpegSample -i in.jpg -o out.ppm            [-info] [-repeat N]          (decode)
// Fastvideo's JPEG sample has fewer keys than the JPEG2000 one for the codec
// itself (quality -q, chroma subsampling -s) but the same threading machinery
// (-async -thread -threadR -threadW -b). This harness accepts those and adds
// the keys that only make sense on the NVIDIA side (backend, native batch,
// our multi-stream scheme, the measurement window). -threadR / -threadW are
// reader / writer thread counts on the Fastvideo side; here they are accepted
// and ignored, because this harness has no separate reader / writer threads.
//
// HOW nvJPEG DIFFERS FROM nvJPEG2000, AND WHY THIS HARNESS IS SHAPED AS IT IS
//   - Decoding has a NATIVE batch (nvjpegDecodeBatched) and, on Ampere / Ada /
//     Hopper / Blackwell, a dedicated HARDWARE decoder (NVJPEG_BACKEND_HARDWARE).
//     So on the decode side the "several codec states per thread" trick that
//     the JPEG2000 harness needed is not needed here: the library already keeps
//     the card busy. This harness can measure the native batch directly
//     (-batched) and can select the backend (-backend hardware|gpuhybrid|
//     default).
//   - Encoding has NO batch API: nvjpegEncodeImage takes one image per call.
//     Here the multi-stream scheme still applies, and it is switched on exactly
//     as in the JPEG2000 harness: -async -thread T -b B keeps T threads with B
//     encoder states and CUDA streams each in flight. The scheme is ours; the
//     library has no such call.
//
// MEASUREMENT BOUNDARY, named explicitly (a lesson from the JPEG2000 harness,
// where "including all transfers" once lived four months apart from the code
// that made it true):
//   -window host2host   pixels in host memory  <-> compressed stream in host
//                       memory. Both directions include the pixel-side copy.
//                       This is the throughput / energy window.
//   -window codec       the pixels stay on the card: the encoder starts with
//                       them already there, the decoder stops with them still
//                       there. The COMPRESSED stream is on the host side in
//                       both windows - the encoder copies it back inside the
//                       timer. This is exactly the boundary of NVIDIA's own
//                       samples (nvJPEG_encoder, nvjpegDecoder), so it is the
//                       window of the step-1 cross-check against them.
// The printed line always states the window in words ("including all
// transfers" / "excluding the host-to-device transfer" / "excluding the
// device-to-host transfer") AND the harness prints one machine-readable RESULT
// line whose "window" field is set from the very same flag. -info changes only
// what is printed, never what is measured.
// ---------------------------------------------------------------------------

#include <nvjpeg.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------

static void die(const char* what, int code, const char* file, int line) {
    std::fprintf(stderr, "ERROR: %s failed with code %d at %s:%d\n",
                 what, code, file, line);
    std::exit(1);
}

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t e_ = (call);                                               \
        if (e_ != cudaSuccess) die(#call, (int)e_, __FILE__, __LINE__);        \
    } while (0)

#define CHECK_NVJPEG(call)                                                     \
    do {                                                                       \
        nvjpegStatus_t s_ = (call);                                            \
        if (s_ != NVJPEG_STATUS_SUCCESS)                                       \
            die(#call, (int)s_, __FILE__, __LINE__);                           \
    } while (0)

// Same as CHECK_NVJPEG but returns the status instead of dying: used where a
// failure is a fact to report (a backend that this card does not have), not a
// crash.
static nvjpegStatus_t tryNvjpeg(nvjpegStatus_t s) { return s; }

static std::atomic<size_t> g_device_bytes(0);

static void* devAlloc(size_t bytes) {
    void* p = nullptr;
    CHECK_CUDA(cudaMalloc(&p, bytes));
    g_device_bytes += bytes;
    return p;
}

// ---------------------------------------------------------------------------
// clock
// ---------------------------------------------------------------------------

typedef std::chrono::high_resolution_clock Clock;

static double msSince(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// PPM / PGM reader, planar output (identical to the JPEG2000 harness, so that
// both codecs are fed byte for byte the same pixels)
// ---------------------------------------------------------------------------

struct Image {
    int width = 0;
    int height = 0;
    int comps = 0;
    int precision = 8;
    int bytesPerSample = 1;
    std::vector<std::vector<unsigned char> > plane;   // one per component
};

static void skipWhitespaceAndComments(FILE* f) {
    int c;
    for (;;) {
        c = std::fgetc(f);
        if (c == '#') {
            while (c != '\n' && c != EOF) c = std::fgetc(f);
        } else if (c == EOF) {
            return;
        } else if (!std::isspace(c)) {
            std::ungetc(c, f);
            return;
        }
    }
}

static int readInt(FILE* f) {
    skipWhitespaceAndComments(f);
    int v = 0;
    if (std::fscanf(f, "%d", &v) != 1) {
        std::fprintf(stderr, "ERROR: malformed PPM/PGM header\n");
        std::exit(1);
    }
    return v;
}

static Image readImage(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    char magic[3] = {0, 0, 0};
    if (std::fread(magic, 1, 2, f) != 2 || magic[0] != 'P' ||
        (magic[1] != '5' && magic[1] != '6')) {
        std::fprintf(stderr, "ERROR: %s is not a binary PGM (P5) or PPM (P6)\n",
                     path.c_str());
        std::exit(1);
    }
    Image im;
    im.comps = (magic[1] == '6') ? 3 : 1;
    im.width = readInt(f);
    im.height = readInt(f);
    int maxval = readInt(f);
    std::fgetc(f);                     // single whitespace after maxval

    im.bytesPerSample = (maxval > 255) ? 2 : 1;
    im.precision = 8;
    // Baseline JPEG is 8-bit. A 16-bit PPM is refused rather than silently
    // truncated: nvjpeg's 8-bit path would encode the high byte only, and the
    // comparison would stop being like for like.
    if (im.bytesPerSample != 1) {
        std::fprintf(stderr,
                     "ERROR: %s is a 16-bit image; this harness measures "
                     "baseline 8-bit JPEG only.\n", path.c_str());
        std::exit(1);
    }

    const size_t pixels = (size_t)im.width * im.height;
    const size_t interleavedBytes = pixels * im.comps;
    std::vector<unsigned char> raw(interleavedBytes);
    if (std::fread(&raw[0], 1, interleavedBytes, f) != interleavedBytes) {
        std::fprintf(stderr, "ERROR: %s is shorter than its header claims\n",
                     path.c_str());
        std::exit(1);
    }
    std::fclose(f);

    im.plane.resize(im.comps);
    for (int c = 0; c < im.comps; ++c) im.plane[c].resize(pixels);
    for (size_t i = 0; i < pixels; ++i)
        for (int c = 0; c < im.comps; ++c)
            im.plane[c][i] = raw[i * im.comps + c];
    return im;
}

static std::vector<unsigned char> readFile(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> buf((size_t)(n > 0 ? n : 0));
    if (n > 0 && std::fread(&buf[0], 1, (size_t)n, f) != (size_t)n) {
        std::fprintf(stderr, "ERROR: short read on %s\n", path.c_str());
        std::exit(1);
    }
    std::fclose(f);
    return buf;
}

// ---------------------------------------------------------------------------
// command line
// ---------------------------------------------------------------------------

enum Window { WIN_HOST2HOST, WIN_CODEC };

struct Args {
    std::string input;
    std::string output;
    int quality = 90;               // JPEG quality 0..100
    std::string subsampling = "444"; // 444 | 422 | 420 | gray
    int repeat = 1;
    int batch = 1;
    int threads = 1;
    bool async = false;
    bool batched = false;           // decode: use the native nvjpegDecodeBatched
    bool discard = false;
    bool info = false;
    bool showFrames = false;
    bool optHuffman = false;        // encode: optimized Huffman tables
    int device = 0;
    std::string backend = "default"; // decode backend: default|gpuhybrid|hardware|hybrid
    std::string encBackend = "default"; // encode backend: default|gpu|hardware
    bool hwinfo = false;            // report the hardware engines and stop
    long targetSize = 0;            // bytes; non-zero switches on calibration
    double tol = 0.01;              // calibration: allowed miss, share of target
    int qlo = 1;                    // calibration: lower end of the search
    int qhi = 100;                  // calibration: upper end of the search
    // The window is set from -window, or from the -noupload / -nodownload
    // diagnostics, which are kept as synonyms so the run script can share code
    // with the JPEG2000 harness.
    Window window = WIN_HOST2HOST;
    bool windowSet = false;
    bool decode = false;
};

static bool eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

static Args parseArgs(int argc, char** argv) {
    Args a;
#ifdef BUILD_DECODER
    a.decode = true;
#endif
    for (int i = 1; i < argc; ++i) {
        const char* k = argv[i];
        const char* v = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (eq(k, "-i") && v) { a.input = v; ++i; }
        else if (eq(k, "-o") && v) { a.output = v; ++i; }
        else if (eq(k, "-q") && v) { a.quality = std::atoi(v); ++i; }
        else if (eq(k, "-s") && v) { a.subsampling = v; ++i; }
        else if (eq(k, "-repeat") && v) { a.repeat = std::atoi(v); ++i; }
        else if (eq(k, "-b") && v) { a.batch = std::atoi(v); ++i; }
        else if (eq(k, "-thread") && v) { a.threads = std::atoi(v); ++i; }
        else if (eq(k, "-d") && v) { a.device = std::atoi(v); ++i; }
        else if (eq(k, "-backend") && v) { a.backend = v; ++i; }
        else if (eq(k, "-encbackend") && v) { a.encBackend = v; ++i; }
        else if (eq(k, "-hwinfo")) a.hwinfo = true;
        else if (eq(k, "-targetsize") && v) { a.targetSize = std::atol(v); ++i; }
        else if (eq(k, "-tol") && v) { a.tol = std::atof(v); ++i; }
        else if (eq(k, "-qlo") && v) { a.qlo = std::atoi(v); ++i; }
        else if (eq(k, "-qhi") && v) { a.qhi = std::atoi(v); ++i; }
        else if (eq(k, "-window") && v) {
            a.window = eq(v, "codec") ? WIN_CODEC : WIN_HOST2HOST;
            a.windowSet = true; ++i;
        }
        else if (eq(k, "-async")) a.async = true;
        else if (eq(k, "-batched")) a.batched = true;
        else if (eq(k, "-discard")) a.discard = true;
        else if (eq(k, "-info")) a.info = true;
        else if (eq(k, "-showFrames")) a.showFrames = true;
        else if (eq(k, "-optHuffman") || eq(k, "-opthuff")) a.optHuffman = true;
        else if (eq(k, "-decode")) a.decode = true;
        // Diagnostics kept as synonyms of -window, so the JPEG2000 run script
        // works unchanged: -noupload/-nodownload mean the codec-only window.
        else if (eq(k, "-noupload") || eq(k, "-nodownload")) {
            a.window = WIN_CODEC; a.windowSet = true;
        }
        else if (eq(k, "-download") || eq(k, "-upload")) {
            a.window = WIN_HOST2HOST; a.windowSet = true;
        }
        else if (eq(k, "-version")) {
            std::printf("nvjpeg_bench version 01, 20 September 2026\n");
            std::exit(0);
        }
        // Host-side plumbing keys of the Fastvideo sample: accepted and ignored
        // so the same command line drives both codecs.
        else if (eq(k, "-threadR") || eq(k, "-threadW") ||
                 eq(k, "-log") || eq(k, "-repeatTime")) {
            ++i;
        }
    }
    if (a.threads < 1) a.threads = 1;
    if (a.batch < 1) a.batch = 1;
    if (a.repeat < 1) a.repeat = 1;
    if (a.quality < 1) a.quality = 1;
    if (a.quality > 100) a.quality = 100;
    // Default window: single frame (no -async) is the codec-only latency
    // window unless the caller said otherwise; the multi-stream / batched path
    // is the host-to-host throughput window.
    if (!a.windowSet) a.window = (a.async || a.batched) ? WIN_HOST2HOST
                                                        : WIN_CODEC;
    return a;
}

static nvjpegChromaSubsampling_t cssOf(const std::string& s) {
    if (s == "444") return NVJPEG_CSS_444;
    if (s == "422") return NVJPEG_CSS_422;
    if (s == "420") return NVJPEG_CSS_420;
    if (s == "440") return NVJPEG_CSS_440;
    if (s == "411") return NVJPEG_CSS_411;
    if (s == "410") return NVJPEG_CSS_410;
    if (s == "gray" || s == "400") return NVJPEG_CSS_GRAY;
    return NVJPEG_CSS_444;
}

static nvjpegBackend_t backendOf(const std::string& s) {
    if (s == "hardware" || s == "hw") return NVJPEG_BACKEND_HARDWARE;
    if (s == "gpuhybrid" || s == "gpu") return NVJPEG_BACKEND_GPU_HYBRID;
    if (s == "hybrid" || s == "cpu") return NVJPEG_BACKEND_HYBRID;
    return NVJPEG_BACKEND_DEFAULT;
}

static nvjpegEncBackend_t encBackendOf(const std::string& s) {
    if (s == "hardware" || s == "hw") return NVJPEG_ENC_BACKEND_HARDWARE;
    if (s == "gpu") return NVJPEG_ENC_BACKEND_GPU;
    return NVJPEG_ENC_BACKEND_DEFAULT;
}

// Where the Huffman stage of a decode runs. nvjpeg.h (12.4) says of
// NVJPEG_BACKEND_GPU_HYBRID: "nvjpegDecodeBatched will use GPU decoding for
// baseline JPEG bitstreams with interleaved scan when batch size is bigger
// than 50". Below that the Huffman stage runs on the CPU. A grid that stops
// at batch 32 would therefore measure the CPU Huffman path without saying so;
// every RESULT line carries this field so that cannot happen silently.
static const char* huffmanWhere(const Args& a, bool batchedPath) {
    nvjpegBackend_t b = backendOf(a.backend);
    if (b == NVJPEG_BACKEND_HARDWARE) return "hardware";
    if (b == NVJPEG_BACKEND_HYBRID) return "cpu";
    // The header states the threshold for the batched call only. For the
    // single-image call it says nothing, and a label must not claim more than
    // its source does: the -info stage breakdown shows where the time goes.
    if (b == NVJPEG_BACKEND_GPU_HYBRID)
        return batchedPath ? (a.batch > 50 ? "gpu" : "cpu") : "not_stated";
    return "not_stated";   // the library decides
}

// ---------------------------------------------------------------------------
// header, printed in the same shape as the Fastvideo / JPEG2000 samples, so
// the run script's parser reads one set of lines for both codecs
// ---------------------------------------------------------------------------

static void printHeader(const Args& a) {
    std::printf("SDK version: nvJPEG-%d.%d.%d.%d\n",
                NVJPEG_VER_MAJOR, NVJPEG_VER_MINOR,
                NVJPEG_VER_PATCH, NVJPEG_VER_BUILD);

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, a.device));
    std::printf("Processing unit: %s (device id = %d)\n", prop.name, a.device);

    size_t freeB = 0, totalB = 0;
    CHECK_CUDA(cudaMemGetInfo(&freeB, &totalB));
    std::printf("Available GPU memory size: %.2f GB\n",
                (double)freeB / (1024.0 * 1024.0 * 1024.0));

    const size_t testBytes = 64u * 1024u * 1024u;
    void* hostPinned = nullptr;
    void* devBuf = nullptr;
    CHECK_CUDA(cudaHostAlloc(&hostPinned, testBytes, cudaHostAllocDefault));
    CHECK_CUDA(cudaMalloc(&devBuf, testBytes));
    double best = 0.0;
    for (int i = 0; i < 5; ++i) {
        Clock::time_point t0 = Clock::now();
        CHECK_CUDA(cudaMemcpy(devBuf, hostPinned, testBytes,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaDeviceSynchronize());
        double ms = msSince(t0);
        double mbs = (double)testBytes / (1024.0 * 1024.0) / (ms / 1000.0);
        if (mbs > best) best = mbs;
    }
    CHECK_CUDA(cudaFree(devBuf));
    CHECK_CUDA(cudaFreeHost(hostPinned));
    std::printf("PCI-Express bandwidth test (host to device): %.0f MByte/s\n\n",
                best);
}

static void printMemory() {
    std::printf("Requested GPU memory size: %.2f GB\n",
                (double)g_device_bytes.load() / (1024.0 * 1024.0 * 1024.0));
}

// One machine-readable line per run: the run script prefers it over parsing
// prose, so a change of wording or a typo cannot move a number (a lesson
// written down after the JPEG2000 runs). Printed in addition to the human line.
static void printResult(const char* dir, const Args& a, int images,
                        double ms, double mbs, const char* windowWord,
                        const char* extra) {
    double fps = ms > 0 ? images * 1000.0 / ms : 0.0;
    std::printf("RESULT {\"dir\": \"%s\", \"images\": %d, \"threads\": %d, "
                "\"batch\": %d, \"async\": %s, \"batched\": %s, "
                "\"backend\": \"%s\", \"quality\": %d, \"subsampling\": \"%s\", "
                "\"ms\": %.3f, \"ms_per_frame\": %.4f, \"mb_s\": %.1f, "
                "\"fps\": %.2f, \"window\": \"%s\"%s}\n",
                dir, images, a.threads, a.batch,
                a.async ? "true" : "false", a.batched ? "true" : "false",
                a.decode ? a.backend.c_str() : a.encBackend.c_str(), a.quality,
                a.subsampling.c_str(), ms, images ? ms / images : 0.0,
                mbs, fps, windowWord, extra ? extra : "");
}

static const char* windowWordEncode(Window w) {
    // The encoder's copy is host-to-device; "codec" excludes it.
    return (w == WIN_CODEC) ? "excluding the host-to-device transfer"
                            : "including all transfers";
}
static const char* windowWordDecode(Window w) {
    // The decoder's copy is device-to-host; "codec" excludes it.
    return (w == WIN_CODEC) ? "excluding the device-to-host transfer"
                            : "including all transfers";
}

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------

struct EncSlot {
    cudaStream_t stream = nullptr;
    nvjpegEncoderState_t state = nullptr;
    nvjpegImage_t img;              // device planes
    std::vector<unsigned char*> dplane;
    // Pinned host buffer the compressed stream is copied into, INSIDE the
    // measured region. Grown if a frame ever needs more (outside any rule of
    // thumb about compression ratios).
    unsigned char* hbits = nullptr;
    size_t hcap = 0;
    size_t bitstreamLen = 0;
};

// Pinned copy of the source planes, so uploading from host does not serialise
// the threads through the driver's staging buffer (that would look like "the
// encoder does not scale" - an artefact of the harness, not the library).
struct PinnedSource {
    std::vector<void*> plane;
    size_t planeBytes = 0;
    int comps = 0;
    int width = 0;
    int height = 0;
};

static PinnedSource pinSource(const Image& im) {
    PinnedSource p;
    p.comps = im.comps;
    p.width = im.width;
    p.height = im.height;
    p.planeBytes = (size_t)im.width * im.height;
    p.plane.resize((size_t)im.comps);
    for (int c = 0; c < im.comps; ++c) {
        CHECK_CUDA(cudaHostAlloc(&p.plane[c], p.planeBytes,
                                 cudaHostAllocDefault));
        std::memcpy(p.plane[c], &im.plane[c][0], p.planeBytes);
    }
    return p;
}

static void encoderSlotInit(EncSlot& s, nvjpegHandle_t handle,
                            const PinnedSource& src,
                            nvjpegEncBackend_t backend = NVJPEG_ENC_BACKEND_DEFAULT) {
    CHECK_CUDA(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
    if (backend == NVJPEG_ENC_BACKEND_DEFAULT) {
        CHECK_NVJPEG(nvjpegEncoderStateCreate(handle, &s.state, s.stream));
    } else {
        nvjpegStatus_t st = nvjpegEncoderStateCreateWithBackend(
            handle, &s.state, backend, s.stream);
        if (st != NVJPEG_STATUS_SUCCESS) {
            std::fprintf(stderr,
                         "ERROR: encoder backend %d is not available on this "
                         "GPU (nvjpegEncoderStateCreateWithBackend returned "
                         "%d). Nothing was measured.\n", (int)backend, (int)st);
            std::exit(3);
        }
    }
    std::memset(&s.img, 0, sizeof(s.img));
    s.dplane.resize((size_t)src.comps);
    for (int c = 0; c < src.comps; ++c) {
        s.dplane[c] = (unsigned char*)devAlloc(src.planeBytes);
        s.img.channel[c] = s.dplane[c];
        s.img.pitch[c] = (size_t)src.width;
    }
    // Half the raw frame plus a megabyte covers every quality this comparison
    // uses; encoderRetrieve grows it if a frame ever needs more.
    s.hcap = src.planeBytes * src.comps / 2 + (1u << 20);
    CHECK_CUDA(cudaHostAlloc((void**)&s.hbits, s.hcap, cudaHostAllocDefault));
}

static void encoderUpload(EncSlot& s, const PinnedSource& src) {
    for (int c = 0; c < src.comps; ++c)
        CHECK_CUDA(cudaMemcpyAsync(s.dplane[c], src.plane[c], src.planeBytes,
                                   cudaMemcpyHostToDevice, s.stream));
}

static void encoderOne(nvjpegHandle_t handle, nvjpegEncoderParams_t params,
                       EncSlot& s, const PinnedSource& src,
                       nvjpegInputFormat_t fmt) {
    CHECK_NVJPEG(nvjpegEncodeImage(handle, s.state, params, &s.img, fmt,
                                   src.width, src.height, s.stream));
}

// The compressed stream is brought into host memory here, and this function
// is called INSIDE the measured region - so an encode ends with the stream in
// host memory, as the window says. Found on 20.09.2026 while reading NVIDIA's
// own nvJPEG_encoder sample: it calls nvjpegEncodeRetrieveBitstream twice
// inside its timer (length, then the copy). The first draft of this harness
// asked only for the length inside the timer and never copied the stream -
// one transfer fewer than the label "including all transfers" claimed.
static size_t encoderRetrieve(nvjpegHandle_t handle, EncSlot& s,
                              std::vector<unsigned char>* out) {
    size_t len = 0;
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, s.state, nullptr, &len,
                                               s.stream));
    if (len > s.hcap) {                    // rare: grow, then copy
        CHECK_CUDA(cudaStreamSynchronize(s.stream));
        CHECK_CUDA(cudaFreeHost(s.hbits));
        s.hcap = len + len / 4;
        CHECK_CUDA(cudaHostAlloc((void**)&s.hbits, s.hcap,
                                 cudaHostAllocDefault));
    }
    CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle, s.state, s.hbits, &len,
                                               s.stream));
    CHECK_CUDA(cudaStreamSynchronize(s.stream));
    s.bitstreamLen = len;
    if (out) out->assign(s.hbits, s.hbits + len);   // outside any timer use
    return len;
}

static nvjpegInputFormat_t inputFormatOf(int comps) {
    // Planar RGB: our reader emits planes, so this needs no repacking.
    // Colour frames only - requireColour() refuses anything else before this.
    (void)comps;
    return NVJPEG_INPUT_RGB;
}

// The comparison is made on colour frames. nvjpegEncodeImage takes RGB; a
// one-channel frame would need nvjpegEncodeYUV with NVJPEG_CSS_GRAY, and
// feeding it to the RGB path would not fail - it would encode the wrong
// thing. So a gray frame is refused, loudly.
static void requireColour(const Image& im, const Args& a) {
    if (im.comps != 3) {
        std::fprintf(stderr, "ERROR: %s has %d channel(s); this harness "
                     "encodes colour (P6) frames only. Nothing was "
                     "measured.\n", a.input.c_str(), im.comps);
        std::exit(2);
    }
}

static void makeEncoderParams(nvjpegHandle_t handle,
                              nvjpegEncoderParams_t* params, const Args& a,
                              cudaStream_t stream) {
    CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle, params, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(*params, a.quality, stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(
        *params, cssOf(a.subsampling), stream));
    CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(
        *params, a.optHuffman ? 1 : 0, stream));
}

// ---------------------------------------------------------------------------
// calibration: find the JPEG quality that hits a target compressed size
// (JPEG quality is an integer 1..100, so the search is over integers)
// ---------------------------------------------------------------------------

static void runCalibration(const Args& a, const Image& im) {
    PinnedSource src = pinSource(im);
    nvjpegHandle_t handle = nullptr;
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));

    int lo = a.qlo, hi = a.qhi, q = a.quality;
    int bestQ = a.quality;
    size_t bestSize = 0;
    double bestRel = 1e9;

    for (int iter = 0; iter < 20 && lo <= hi; ++iter) {
        q = (lo + hi) / 2;
        Args tmp = a;
        tmp.quality = q;
        nvjpegEncoderParams_t params = nullptr;
        cudaStream_t probeStream = nullptr;
        CHECK_CUDA(cudaStreamCreateWithFlags(&probeStream,
                                             cudaStreamNonBlocking));
        makeEncoderParams(handle, &params, tmp, probeStream);
        EncSlot s;
        encoderSlotInit(s, handle, src, encBackendOf(a.encBackend));
        encoderUpload(s, src);
        CHECK_CUDA(cudaStreamSynchronize(s.stream));
        encoderOne(handle, params, s, src, inputFormatOf(src.comps));
        std::vector<unsigned char> bits;
        size_t sz = encoderRetrieve(handle, s, &bits);

        double rel = (double)((long)sz - a.targetSize) / (double)a.targetSize;
        double mag = rel < 0 ? -rel : rel;
        if (mag < bestRel) { bestRel = mag; bestQ = q; bestSize = sz; }

        // Larger quality -> larger file. Walk towards the target.
        if ((long)sz > a.targetSize) hi = q - 1; else lo = q + 1;

        for (int c = 0; c < (int)s.dplane.size(); ++c) cudaFree(s.dplane[c]);
        cudaFreeHost(s.hbits);
        cudaStreamDestroy(s.stream);
        nvjpegEncoderStateDestroy(s.state);
        nvjpegEncoderParamsDestroy(params);
        cudaStreamDestroy(probeStream);

        if (mag < a.tol) break;
    }

    const double uncompressed = (double)im.width * im.height * im.comps;
    std::printf("(excluded) 8) Buffer write disabled; size = %d KB (%.1f:1)\n",
                (int)(bestSize / 1024),
                bestSize ? uncompressed / (double)bestSize : 0.0);
    std::printf("Calibration: q = %.4f; size = %d bytes; target = %ld bytes; "
                "miss = %.3f %%; ratio = %.2f:1; search = [%d, %d]; "
                "tol = %.3f %%\n",
                (double)bestQ, (int)bestSize, a.targetSize, 100.0 * bestRel,
                bestSize ? uncompressed / (double)bestSize : 0.0,
                a.qlo, a.qhi, 100.0 * a.tol);
    for (int c = 0; c < src.comps; ++c) cudaFreeHost(src.plane[c]);
    nvjpegDestroy(handle);
}

// ---------------------------------------------------------------------------
// encoder benchmark
// ---------------------------------------------------------------------------

static void benchEncode(const Args& a, const Image& im) {
    const double uncompressedMB =
        (double)im.width * im.height * im.comps / (1024.0 * 1024.0);

    std::printf("Input image: %s (%dx%d pixels; %dx8-bit channel(s)) - %.1f MB\n",
                a.input.c_str(), im.width, im.height, im.comps, uncompressedMB);

    PinnedSource src = pinSource(im);
    nvjpegHandle_t handle = nullptr;
    CHECK_NVJPEG(nvjpegCreateSimple(&handle));
    const nvjpegInputFormat_t fmt = inputFormatOf(src.comps);

    if (!a.async) {
        // ------------------------- synchronous (latency) -------------------
        nvjpegEncoderParams_t params = nullptr;
        cudaStream_t s0 = nullptr;
        CHECK_CUDA(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
        makeEncoderParams(handle, &params, a, s0);
        EncSlot s;
        encoderSlotInit(s, handle, src, encBackendOf(a.encBackend));
        printMemory();

        // warm-up
        encoderUpload(s, src);
        CHECK_CUDA(cudaStreamSynchronize(s.stream));
        std::vector<unsigned char> bits;
        encoderOne(handle, params, s, src, fmt);
        encoderRetrieve(handle, s, &bits);

        double codecMs = 0.0, totalMs = 0.0;
        for (int i = 0; i < a.repeat; ++i) {
            Clock::time_point t0 = Clock::now();
            if (a.window == WIN_HOST2HOST) {
                encoderUpload(s, src);
                CHECK_CUDA(cudaStreamSynchronize(s.stream));
            }
            Clock::time_point t1 = Clock::now();
            encoderOne(handle, params, s, src, fmt);
            encoderRetrieve(handle, s, nullptr);
            double frame = msSince(t1);
            codecMs += frame;
            totalMs += msSince(t0);
            if (a.showFrames) std::printf("  %6.2f ms Total time\n", frame);
        }
        // The measured region is codecMs (codec only) or totalMs (host2host),
        // chosen by the window - not by -info. -info only adds the extra line.
        double measured = (a.window == WIN_HOST2HOST) ? totalMs : codecMs;

        const double ratio = uncompressedMB * 1024.0 * 1024.0 /
                             (double)(s.bitstreamLen ? s.bitstreamLen : 1);
        std::printf("(excluded) 8) Buffer write disabled; size = %d KB (%.1f:1)\n",
                    (int)(s.bitstreamLen / 1024), ratio);

        const double mbs = uncompressedMB * a.repeat / (measured / 1000.0);
        std::printf("Total encode time %s for %d images = %.1f ms; "
                    "%.0f MB/s; %.1f FPS;\n",
                    windowWordEncode(a.window), a.repeat, measured, mbs,
                    a.repeat * 1000.0 / measured);
        printResult("E", a, a.repeat, measured, mbs,
                    a.window == WIN_HOST2HOST ? "host2host" : "codec", nullptr);
        if (a.info) {
            std::printf("Total encode time including all transfers for %d "
                        "images = %.1f ms; %.0f MB/s; %.1f FPS;\n",
                        a.repeat, totalMs,
                        uncompressedMB * a.repeat / (totalMs / 1000.0),
                        a.repeat * 1000.0 / totalMs);
            std::printf("  %6.2f ms 1) encode (nvJPEG gives no sub-stage "
                        "timing)\n", codecMs / a.repeat);
        }
        if (!a.discard && !a.output.empty()) {
            std::vector<unsigned char> outbits;
            encoderOne(handle, params, s, src, fmt);
            encoderRetrieve(handle, s, &outbits);
            FILE* f = std::fopen(a.output.c_str(), "wb");
            if (f) { std::fwrite(&outbits[0], 1, outbits.size(), f);
                     std::fclose(f); }
        }
        return;
    }

    // ------------------------- asynchronous (our multi-stream scheme) ------
    const int T = a.threads;
    const int B = a.batch;
    std::vector<nvjpegEncoderParams_t> params((size_t)T, nullptr);
    std::vector<std::vector<EncSlot> > slots((size_t)T);
    for (int t = 0; t < T; ++t) {
        cudaStream_t tmp = nullptr;
        CHECK_CUDA(cudaStreamCreateWithFlags(&tmp, cudaStreamNonBlocking));
        makeEncoderParams(handle, &params[t], a, tmp);
        cudaStreamDestroy(tmp);
        slots[t].resize((size_t)B);
        for (int b = 0; b < B; ++b)
            encoderSlotInit(slots[t][b], handle, src,
                            encBackendOf(a.encBackend));
    }
    printMemory();

    if (a.window == WIN_CODEC)
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < B; ++b) {
                encoderUpload(slots[t][b], src);
                CHECK_CUDA(cudaStreamSynchronize(slots[t][b].stream));
            }

    // warm-up on the first slot of every thread
    for (int t = 0; t < T; ++t) {
        encoderUpload(slots[t][0], src);
        CHECK_CUDA(cudaStreamSynchronize(slots[t][0].stream));
        encoderOne(handle, params[t], slots[t][0], src, fmt);
        encoderRetrieve(handle, slots[t][0], nullptr);
    }

    std::atomic<int> next(0);
    const int total = a.repeat;
    Clock::time_point t0 = Clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < T; ++t) {
        workers.push_back(std::thread([&, t]() {
            for (;;) {
                int start = next.fetch_add(B);
                if (start >= total) return;
                int n = total - start; if (n > B) n = B;
                if (a.window == WIN_HOST2HOST)
                    for (int b = 0; b < n; ++b) encoderUpload(slots[t][b], src);
                for (int b = 0; b < n; ++b)
                    encoderOne(handle, params[t], slots[t][b], src, fmt);
                for (int b = 0; b < n; ++b)
                    encoderRetrieve(handle, slots[t][b], nullptr);
            }
        }));
    }
    for (size_t i = 0; i < workers.size(); ++i) workers[i].join();
    double wall = msSince(t0);

    const double mbs = uncompressedMB * total / (wall / 1000.0);
    std::printf("Total JPEG Encode time:\n");
    std::printf("- GPU pipeline %s for %d images per %d thread%s = %.1f ms; "
                "%.0f MB/s; %.1f FPS;\n",
                windowWordEncode(a.window), total, T, (T == 1 ? "" : "s"),
                wall, mbs, total * 1000.0 / wall);
    printResult("E", a, total, wall, mbs,
                a.window == WIN_HOST2HOST ? "host2host" : "codec", nullptr);
}

// ---------------------------------------------------------------------------
// decoder
// ---------------------------------------------------------------------------

struct DecSlot {
    cudaStream_t stream = nullptr;
    nvjpegJpegState_t state = nullptr;
    nvjpegImage_t img;
    std::vector<unsigned char*> dplane;
    std::vector<void*> hplane;      // pinned host planes for the copy-back
    size_t planeBytes = 0;
    int comps = 0;
};

static void decoderSlotInit(DecSlot& s, nvjpegHandle_t handle,
                            int width, int height, int comps) {
    CHECK_CUDA(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
    CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &s.state));
    std::memset(&s.img, 0, sizeof(s.img));
    s.comps = comps;
    s.planeBytes = (size_t)width * height;
    s.dplane.resize((size_t)comps);
    s.hplane.resize((size_t)comps);
    for (int c = 0; c < comps; ++c) {
        s.dplane[c] = (unsigned char*)devAlloc(s.planeBytes);
        s.img.channel[c] = s.dplane[c];
        s.img.pitch[c] = (size_t)width;
        CHECK_CUDA(cudaHostAlloc(&s.hplane[c], s.planeBytes,
                                 cudaHostAllocDefault));
    }
}

static void decoderDownload(DecSlot& s) {
    for (int c = 0; c < s.comps; ++c)
        CHECK_CUDA(cudaMemcpyAsync(s.hplane[c], s.dplane[c], s.planeBytes,
                                   cudaMemcpyDeviceToHost, s.stream));
}

static const char* subsamplingName(nvjpegChromaSubsampling_t css) {
    switch (css) {
        case NVJPEG_CSS_444: return "444";
        case NVJPEG_CSS_422: return "422";
        case NVJPEG_CSS_420: return "420";
        case NVJPEG_CSS_440: return "440";
        case NVJPEG_CSS_411: return "411";
        case NVJPEG_CSS_410: return "410";
        case NVJPEG_CSS_GRAY: return "gray";
        default: return "unknown";
    }
}

// Stage breakdown for one frame, using the decoupled decode API: the three
// phases (host parse + Huffman, host-to-device transfer, device inverse
// transform) are the JPEG analogue of the JPEG2000 stage breakdown. Printed
// only under -info, and it does not touch the throughput numbers.
static void decodeStages(nvjpegHandle_t handle, nvjpegBackend_t backend,
                         const std::vector<unsigned char>& bits,
                         int width, int height, int comps) {
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegJpegState_t dstate = nullptr;
    if (tryNvjpeg(nvjpegDecoderCreate(handle, backend, &decoder))
            != NVJPEG_STATUS_SUCCESS)
        return;
    CHECK_NVJPEG(nvjpegDecoderStateCreate(handle, decoder, &dstate));
    nvjpegBufferPinned_t pinned = nullptr;
    nvjpegBufferDevice_t devbuf = nullptr;
    CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle, nullptr, &pinned));
    CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle, nullptr, &devbuf));
    CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(dstate, pinned));
    CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(dstate, devbuf));
    nvjpegJpegStream_t jstream = nullptr;
    CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &jstream));
    nvjpegDecodeParams_t dparams = nullptr;
    CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle, &dparams));
    CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(dparams, NVJPEG_OUTPUT_RGB));

    DecSlot s;
    decoderSlotInit(s, handle, width, height, comps);

    double host = 0, xfer = 0, dev = 0;
    const int runs = 20;
    for (int i = 0; i < runs; ++i) {
        CHECK_NVJPEG(nvjpegJpegStreamParse(handle, &bits[0], bits.size(), 0, 0,
                                           jstream));
        Clock::time_point t0 = Clock::now();
        CHECK_NVJPEG(nvjpegDecodeJpegHost(handle, decoder, dstate, dparams,
                                          jstream));
        double th = msSince(t0);
        Clock::time_point t1 = Clock::now();
        CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle, decoder, dstate,
                                                      jstream, s.stream));
        CHECK_CUDA(cudaStreamSynchronize(s.stream));
        double tx = msSince(t1);
        Clock::time_point t2 = Clock::now();
        CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle, decoder, dstate, &s.img,
                                            s.stream));
        CHECK_CUDA(cudaStreamSynchronize(s.stream));
        double td = msSince(t2);
        if (i > 0) { host += th; xfer += tx; dev += td; }  // drop warm-up
    }
    host /= (runs - 1); xfer /= (runs - 1); dev /= (runs - 1);
    std::printf("  %6.3f ms 1) host parse and Huffman\n", host);
    std::printf("  %6.3f ms 2) host-to-device transfer\n", xfer);
    std::printf("  %6.3f ms 3) device inverse transform\n", dev);

    for (int c = 0; c < s.comps; ++c) { cudaFree(s.dplane[c]);
                                        cudaFreeHost(s.hplane[c]); }
    cudaStreamDestroy(s.stream);
    nvjpegDecodeParamsDestroy(dparams);
    nvjpegJpegStreamDestroy(jstream);
    nvjpegBufferPinnedDestroy(pinned);
    nvjpegBufferDeviceDestroy(devbuf);
    nvjpegJpegStateDestroy(dstate);
    nvjpegDecoderDestroy(decoder);
}

// Throughput in MB/s is always megabytes of UNCOMPRESSED pixels per second,
// in both directions, exactly as the encoder and the JPEG2000 harness count
// it. Megapixels per second are derived by the run script from fps and the
// frame geometry, so there is one definition of each unit, not two.
static double uncompressedMiB(int width, int height, int comps) {
    return (double)width * height * comps / (1024.0 * 1024.0);
}

static std::string decodeExtra(const char* path, const Args& a,
                               bool batchedPath, int images) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  ", \"path\": \"%s\", \"huffman\": \"%s\", "
                  "\"images_decoded\": %d",
                  path, huffmanWhere(a, batchedPath), images);
    return std::string(buf);
}

static void benchDecode(const Args& a) {
    std::vector<unsigned char> bitstream = readFile(a.input);

    // Create the handle with the requested backend. A backend this card does
    // not have is reported plainly and stops the run; it is never silently
    // replaced, because "which backend produced these numbers" must have an
    // answer.
    nvjpegBackend_t backend = backendOf(a.backend);
    nvjpegHandle_t handle = nullptr;
    nvjpegStatus_t cs = tryNvjpeg(
        nvjpegCreateEx(backend, nullptr, nullptr, 0, &handle));
    if (cs != NVJPEG_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "ERROR: backend '%s' is not available on this GPU "
                     "(nvjpegCreateEx returned %d%s). Nothing was measured.\n",
                     a.backend.c_str(), (int)cs,
                     cs == NVJPEG_STATUS_ARCH_MISMATCH ? " = ARCH_MISMATCH" : "");
        std::exit(3);
    }
    if (backend == NVJPEG_BACKEND_HARDWARE) {
        unsigned int engines = 0, cores = 0;
        nvjpegStatus_t hs = tryNvjpeg(
            nvjpegGetHardwareDecoderInfo(handle, &engines, &cores));
        std::printf("Hardware decoder: engines = %u; cores per engine = %u "
                    "(status %d)\n", engines, cores, (int)hs);
        if (hs != NVJPEG_STATUS_SUCCESS || engines == 0) {
            std::fprintf(stderr, "ERROR: the hardware backend was created but "
                         "reports no engines. Nothing was measured.\n");
            std::exit(3);
        }
    }

    int comps = 0;
    nvjpegChromaSubsampling_t css;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};
    CHECK_NVJPEG(nvjpegGetImageInfo(handle, &bitstream[0], bitstream.size(),
                                    &comps, &css, widths, heights));
    const int width = widths[0];
    const int height = heights[0];
    const int outComps = (comps == 1) ? 1 : 3;
    const double frameMiB = uncompressedMiB(width, height, outComps);

    std::printf("Input image : %s (%dx%d pixels; %d channel(s); %s)\n",
                a.input.c_str(), width, height, comps, subsamplingName(css));

    if (a.info) {
        std::printf("Stage breakdown (decoupled API, backend %s):\n",
                    a.backend.c_str());
        decodeStages(handle, backend, bitstream, width, height, outComps);
    }

    // The hardware engine is driven through the batched call: that is how
    // NVIDIA's own A100 sample uses it, and it keeps one code path for
    // everything the engine does. A single-frame run on the hardware backend
    // is therefore a batch of one; with -async it is a batch of -b.
    const bool hardware = (backend == NVJPEG_BACKEND_HARDWARE);
    const bool useBatched = a.batched || hardware;

    // -------------------- native batched decode ---------------------------
    if (useBatched) {
        const int B = a.batched ? a.batch : (a.async ? a.batch : 1);
        Args eff = a;
        eff.batch = B;
        nvjpegJpegState_t state = nullptr;
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &state));
        CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle, state, B, 1,
                                                   NVJPEG_OUTPUT_RGB));
        if (hardware) {
            // is_supported == 0 means SUPPORTED (see nvjpeg.h). A stream the
            // engine refuses is reported, not measured on some other path.
            nvjpegJpegStream_t js = nullptr;
            CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &js));
            CHECK_NVJPEG(nvjpegJpegStreamParse(handle, &bitstream[0],
                                               bitstream.size(), 0, 0, js));
            int isSupported = -1;
            CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle, js,
                                                      &isSupported));
            nvjpegJpegStreamDestroy(js);
            std::printf("Hardware decode supported for this stream: %s "
                        "(nvjpegDecodeBatchedSupported = %d)\n",
                        isSupported == 0 ? "yes" : "NO", isSupported);
            if (isSupported != 0) {
                std::fprintf(stderr, "ERROR: the hardware decoder does not "
                             "support this stream. Nothing was measured.\n");
                std::exit(3);
            }
        }
        std::vector<DecSlot> slots((size_t)B);
        std::vector<const unsigned char*> ptrs((size_t)B, &bitstream[0]);
        std::vector<size_t> lens((size_t)B, bitstream.size());
        std::vector<nvjpegImage_t> dests((size_t)B);
        cudaStream_t stream = nullptr;
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (int b = 0; b < B; ++b) {
            decoderSlotInit(slots[b], handle, width, height, outComps);
            dests[b] = slots[b].img;
        }
        printMemory();
        if (!hardware && backend == NVJPEG_BACKEND_GPU_HYBRID && B <= 50)
            std::printf("Note: GPU hybrid backend with batch %d <= 50 - "
                        "nvjpeg.h says the Huffman stage then runs on the "
                        "CPU\n", B);

        // warm-up
        CHECK_NVJPEG(nvjpegDecodeBatched(handle, state, &ptrs[0], &lens[0],
                                         &dests[0], stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // The batch always decodes exactly B frames, so the count is the
        // number of calls times B, not the number that was asked for.
        const int calls = (a.repeat + B - 1) / B;
        const int decoded = calls * B;
        Clock::time_point t0 = Clock::now();
        for (int k = 0; k < calls; ++k) {
            CHECK_NVJPEG(nvjpegDecodeBatched(handle, state, &ptrs[0], &lens[0],
                                             &dests[0], stream));
            if (a.window == WIN_HOST2HOST)
                for (int b = 0; b < B; ++b) {
                    for (int c = 0; c < slots[b].comps; ++c)
                        CHECK_CUDA(cudaMemcpyAsync(slots[b].hplane[c],
                                                   slots[b].dplane[c],
                                                   slots[b].planeBytes,
                                                   cudaMemcpyDeviceToHost,
                                                   stream));
                }
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
        double wall = msSince(t0);
        const double mbs = frameMiB * decoded / (wall / 1000.0);
        std::printf("Total JPEG Decode time (native batch%s):\n",
                    hardware ? ", hardware engine" : "");
        std::printf("- GPU pipeline %s for %d images per %d batch = %.1f ms; "
                    "%.0f MB/s; %.1f FPS;\n",
                    windowWordDecode(a.window), decoded, B, wall, mbs,
                    decoded * 1000.0 / wall);
        std::string ex = decodeExtra("native_batch", eff, true, decoded);
        printResult("D", eff, decoded, wall, mbs,
                    a.window == WIN_HOST2HOST ? "host2host" : "codec",
                    ex.c_str());
        return;
    }

    if (!a.async) {
        // -------------------- synchronous single frame (latency) ----------
        DecSlot s;
        decoderSlotInit(s, handle, width, height, outComps);
        printMemory();
        CHECK_NVJPEG(nvjpegDecode(handle, s.state, &bitstream[0],
                                  bitstream.size(), NVJPEG_OUTPUT_RGB, &s.img,
                                  s.stream));
        CHECK_CUDA(cudaStreamSynchronize(s.stream));   // warm-up

        double ms = 0.0;
        for (int i = 0; i < a.repeat; ++i) {
            Clock::time_point t1 = Clock::now();
            CHECK_NVJPEG(nvjpegDecode(handle, s.state, &bitstream[0],
                                      bitstream.size(), NVJPEG_OUTPUT_RGB,
                                      &s.img, s.stream));
            if (a.window == WIN_HOST2HOST) decoderDownload(s);
            CHECK_CUDA(cudaStreamSynchronize(s.stream));
            double frame = msSince(t1);
            ms += frame;
            if (a.showFrames) std::printf("  %6.2f ms Total time\n", frame);
        }
        const double mbs = frameMiB * a.repeat / (ms / 1000.0);
        std::printf("Total decode time %s for %d images = %.1f ms; "
                    "%.0f MB/s; %.1f FPS;\n",
                    windowWordDecode(a.window), a.repeat, ms, mbs,
                    a.repeat * 1000.0 / ms);
        std::string ex = decodeExtra("single", a, false, a.repeat);
        printResult("D", a, a.repeat, ms, mbs,
                    a.window == WIN_HOST2HOST ? "host2host" : "codec",
                    ex.c_str());

        if (!a.discard && !a.output.empty()) {
            std::vector<std::vector<unsigned char> > host(outComps);
            for (int c = 0; c < outComps; ++c) {
                host[c].resize(s.planeBytes);
                CHECK_CUDA(cudaMemcpy(&host[c][0], s.dplane[c], s.planeBytes,
                                      cudaMemcpyDeviceToHost));
            }
            FILE* f = std::fopen(a.output.c_str(), "wb");
            if (f) {
                std::fprintf(f, "P%d\n%d %d\n255\n", outComps == 1 ? 5 : 6,
                             width, height);
                std::vector<unsigned char> row((size_t)width * outComps);
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x)
                        for (int c = 0; c < outComps; ++c)
                            row[(size_t)x * outComps + c] =
                                host[c][(size_t)y * width + x];
                    std::fwrite(&row[0], 1, row.size(), f);
                }
                std::fclose(f);
            }
        }
        return;
    }

    // -------------------- asynchronous, our multi-stream scheme -----------
    // Kept for parity with the JPEG2000 harness and to answer "would the trick
    // help here too". On decode the native batch is the product's own way; this
    // path is several single decodes overlapped across threads and states.
    const int T = a.threads;
    const int B = a.batch;
    std::vector<std::vector<DecSlot> > slots((size_t)T);
    for (int t = 0; t < T; ++t) {
        slots[t].resize((size_t)B);
        for (int b = 0; b < B; ++b)
            decoderSlotInit(slots[t][b], handle, width, height, outComps);
    }
    printMemory();
    for (int t = 0; t < T; ++t) {
        CHECK_NVJPEG(nvjpegDecode(handle, slots[t][0].state, &bitstream[0],
                                  bitstream.size(), NVJPEG_OUTPUT_RGB,
                                  &slots[t][0].img, slots[t][0].stream));
        CHECK_CUDA(cudaStreamSynchronize(slots[t][0].stream));
    }
    std::atomic<int> next(0);
    const int total = a.repeat;
    Clock::time_point t0 = Clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < T; ++t) {
        workers.push_back(std::thread([&, t]() {
            for (;;) {
                int start = next.fetch_add(B);
                if (start >= total) return;
                int n = total - start; if (n > B) n = B;
                for (int b = 0; b < n; ++b)
                    CHECK_NVJPEG(nvjpegDecode(handle, slots[t][b].state,
                                              &bitstream[0], bitstream.size(),
                                              NVJPEG_OUTPUT_RGB,
                                              &slots[t][b].img,
                                              slots[t][b].stream));
                if (a.window == WIN_HOST2HOST)
                    for (int b = 0; b < n; ++b) decoderDownload(slots[t][b]);
                for (int b = 0; b < n; ++b)
                    CHECK_CUDA(cudaStreamSynchronize(slots[t][b].stream));
            }
        }));
    }
    for (size_t i = 0; i < workers.size(); ++i) workers[i].join();
    double wall = msSince(t0);
    const double mbs = frameMiB * total / (wall / 1000.0);
    std::printf("Total JPEG Decode time:\n");
    std::printf("- GPU pipeline %s for %d images per %d thread%s = %.1f ms; "
                "%.0f MB/s; %.1f FPS;\n",
                windowWordDecode(a.window), total, T, (T == 1 ? "" : "s"),
                wall, mbs, total * 1000.0 / wall);
    std::string ex = decodeExtra("streams", a, false, total);
    printResult("D", a, total, wall, mbs,
                a.window == WIN_HOST2HOST ? "host2host" : "codec",
                ex.c_str());
}

// ---------------------------------------------------------------------------
// -hwinfo: does this card really have the hardware JPEG engines, and does the
// hardware decoder produce the same picture as the CUDA decoder?
//
// Measures nothing. Three independent answers, because a return code alone
// cannot tell a working engine from a silent fallback:
//   1. the library: nvjpegCreateEx(HARDWARE) and the engine count from
//      nvjpegGetHardwareDecoderInfo / nvjpegGetHardwareEncoderInfo;
//   2. the stream: nvjpegDecodeBatchedSupported for the given file;
//   3. the picture: the frame decoded by the engine against the frame decoded
//      by the CUDA (GPU hybrid) path - identical, or how far apart.
// The fourth answer - the engine's own utilisation counter read through NVML
// while it works - is taken by the run script, from outside this program.
// Exit code: 0 hardware decoder usable, 4 not available, 5 available but the
// picture differs beyond rounding.
// ---------------------------------------------------------------------------

static int decodeToHost(nvjpegHandle_t handle, bool batched,
                        const std::vector<unsigned char>& bits,
                        int width, int height, int comps,
                        std::vector<std::vector<unsigned char> >* out) {
    DecSlot s;
    decoderSlotInit(s, handle, width, height, comps);
    if (batched) {
        nvjpegJpegState_t st = nullptr;
        CHECK_NVJPEG(nvjpegJpegStateCreate(handle, &st));
        CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle, st, 1, 1,
                                                   NVJPEG_OUTPUT_RGB));
        const unsigned char* p = &bits[0];
        size_t len = bits.size();
        nvjpegStatus_t r = nvjpegDecodeBatched(handle, st, &p, &len, &s.img,
                                               s.stream);
        if (r != NVJPEG_STATUS_SUCCESS) return (int)r;
    } else {
        nvjpegStatus_t r = nvjpegDecode(handle, s.state, &bits[0], bits.size(),
                                        NVJPEG_OUTPUT_RGB, &s.img, s.stream);
        if (r != NVJPEG_STATUS_SUCCESS) return (int)r;
    }
    CHECK_CUDA(cudaStreamSynchronize(s.stream));
    out->assign((size_t)comps, std::vector<unsigned char>(s.planeBytes));
    for (int c = 0; c < comps; ++c)
        CHECK_CUDA(cudaMemcpy(&(*out)[c][0], s.dplane[c], s.planeBytes,
                              cudaMemcpyDeviceToHost));
    return 0;
}

static int hwInfo(const Args& a) {
    int rc = 0;

    // --- the decoder engine ---
    nvjpegHandle_t hw = nullptr;
    nvjpegStatus_t cs = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr,
                                       nullptr, 0, &hw);
    unsigned int engines = 0, cores = 0;
    if (cs == NVJPEG_STATUS_SUCCESS)
        nvjpegGetHardwareDecoderInfo(hw, &engines, &cores);
    std::printf("Hardware decoder: %s; nvjpegCreateEx(HARDWARE) = %d%s; "
                "engines = %u; cores per engine = %u\n",
                (cs == NVJPEG_STATUS_SUCCESS && engines > 0) ? "AVAILABLE"
                                                             : "NOT available",
                (int)cs,
                cs == NVJPEG_STATUS_ARCH_MISMATCH ? " (ARCH_MISMATCH)" : "",
                engines, cores);
    if (cs != NVJPEG_STATUS_SUCCESS || engines == 0) rc = 4;

    // --- the encoder engine ---
    nvjpegHandle_t def = nullptr;
    CHECK_NVJPEG(nvjpegCreateSimple(&def));
    unsigned int encEngines = 0;
    nvjpegStatus_t es = nvjpegGetHardwareEncoderInfo(def, &encEngines);
    cudaStream_t st0 = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&st0, cudaStreamNonBlocking));
    nvjpegEncoderState_t probe = nullptr;
    nvjpegStatus_t ec = nvjpegEncoderStateCreateWithBackend(
        def, &probe, NVJPEG_ENC_BACKEND_HARDWARE, st0);
    std::printf("Hardware encoder: %s; nvjpegGetHardwareEncoderInfo = %d, "
                "engines = %u; encoder state with HARDWARE backend = %d\n",
                (ec == NVJPEG_STATUS_SUCCESS && encEngines > 0)
                    ? "AVAILABLE" : "NOT available",
                (int)es, encEngines, (int)ec);
    if (ec == NVJPEG_STATUS_SUCCESS) nvjpegEncoderStateDestroy(probe);

    // --- the stream and the picture, if a .jpg was given ---
    if (!a.input.empty() && rc == 0) {
        std::vector<unsigned char> bits = readFile(a.input);
        int comps = 0;
        nvjpegChromaSubsampling_t css;
        int w[NVJPEG_MAX_COMPONENT] = {0}, h[NVJPEG_MAX_COMPONENT] = {0};
        CHECK_NVJPEG(nvjpegGetImageInfo(hw, &bits[0], bits.size(), &comps,
                                        &css, w, h));
        const int outComps = (comps == 1) ? 1 : 3;

        nvjpegJpegStream_t js = nullptr;
        CHECK_NVJPEG(nvjpegJpegStreamCreate(hw, &js));
        CHECK_NVJPEG(nvjpegJpegStreamParse(hw, &bits[0], bits.size(), 0, 0,
                                           js));
        int isSupported = -1;
        CHECK_NVJPEG(nvjpegDecodeBatchedSupported(hw, js, &isSupported));
        nvjpegJpegStreamDestroy(js);
        std::printf("Stream %s (%dx%d, %s): hardware decode %s "
                    "(nvjpegDecodeBatchedSupported = %d, 0 means yes)\n",
                    a.input.c_str(), w[0], h[0], subsamplingName(css),
                    isSupported == 0 ? "SUPPORTED" : "NOT supported",
                    isSupported);

        if (isSupported == 0) {
            nvjpegHandle_t gpu = nullptr;
            CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, nullptr,
                                        nullptr, 0, &gpu));
            std::vector<std::vector<unsigned char> > ph, pg;
            int r1 = decodeToHost(hw, true, bits, w[0], h[0], outComps, &ph);
            int r2 = decodeToHost(gpu, false, bits, w[0], h[0], outComps, &pg);
            if (r1 || r2) {
                std::printf("Picture check: decode failed (hardware %d, "
                            "GPU %d)\n", r1, r2);
                rc = 5;
            } else {
                long diff = 0; int maxd = 0; double se = 0.0;
                size_t n = 0;
                for (int c = 0; c < outComps; ++c)
                    for (size_t i = 0; i < ph[c].size(); ++i) {
                        int d = (int)ph[c][i] - (int)pg[c][i];
                        if (d) ++diff;
                        if (d < 0) d = -d;
                        if (d > maxd) maxd = d;
                        se += (double)d * d; ++n;
                    }
                double mse = n ? se / n : 0.0;
                if (diff == 0)
                    std::printf("Picture check: hardware and GPU decode are "
                                "IDENTICAL, byte for byte\n");
                else
                    std::printf("Picture check: %ld of %zu samples differ, "
                                "max difference %d, PSNR %.2f dB\n",
                                diff, n, maxd,
                                10.0 * std::log10(255.0 * 255.0 / mse));
                // Two IDCT implementations may round differently by one
                // level; more than that is a different picture.
                if (maxd > 2) rc = 5;
            }
            nvjpegDestroy(gpu);
        }
    }
    std::printf("hwinfo exit code: %d (0 usable, 4 not available, "
                "5 picture differs)\n", rc);
    return rc;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args a = parseArgs(argc, argv);
    if (a.hwinfo) {
        // Works from either executable; the .jpg after -i is optional.
        CHECK_CUDA(cudaSetDevice(a.device));
        printHeader(a);
        return hwInfo(a);
    }
    if (a.input.empty()) {
        std::printf("nvjpeg_bench version 01, 20 September 2026\n");
        std::printf("usage: %s -i input -o output [-q 90] [-s 444|422|420] "
                    "[-repeat N] [-async -thread T -b B] [-batched -b B] "
                    "[-backend default|gpuhybrid|hardware] "
                    "[-encbackend default|gpu|hardware] [-optHuffman] "
                    "[-window host2host|codec] [-discard] [-info] "
                    "[-showFrames] [-targetsize BYTES] [-tol SHARE] "
                    "[-qlo Q] [-qhi Q] [-hwinfo] [-version]\n", argv[0]);
        std::printf("  encoder: .ppm -> .jpg ; decoder: .jpg -> .ppm "
                    "(build with -DBUILD_DECODER)\n");
        std::printf("  -window codec       codec only, no copy of the pixels "
                    "across the bus\n");
        std::printf("  -window host2host   host memory to host memory "
                    "(throughput and energy)\n");
        std::printf("  -batched            decode with the native "
                    "nvjpegDecodeBatched\n");
        std::printf("  -backend hardware   the dedicated JPEG decoder engine; "
                    "always driven through\n"
                    "                      nvjpegDecodeBatched (a single frame "
                    "is a batch of one)\n");
        std::printf("  -hwinfo [-i x.jpg]  are the hardware engines there, and "
                    "does the hardware\n"
                    "                      decoder give the same picture as "
                    "the GPU one? measures nothing\n");
        std::printf("  -version            print the version and stop\n");
        return 1;
    }
    CHECK_CUDA(cudaSetDevice(a.device));
    printHeader(a);

    if (a.decode) {
        benchDecode(a);
    } else {
        Image im = readImage(a.input);
        requireColour(im, a);
        if (a.targetSize > 0) runCalibration(a, im);
        else benchEncode(a, im);
    }
    return 0;
}
