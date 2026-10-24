#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "coreml_backend.h"
#include "cpu_conv.h"
#include "cpu_fast.h"
#include "image.h"
#include "mps_backend.h"
#include "timer.h"
#include "weights.h"

static const std::string MODEL_DIR = "models/";
static const int SIZES[4][2] = {{256, 256}, {512, 512}, {1024, 1024}, {1080, 1920}};

static std::string model_path(int h, int w, int bits) {
    return MODEL_DIR + "sr_" + std::to_string(h) + "x" + std::to_string(w) + "_fp" + std::to_string(bits) + ".mlpackage";
}

static double maxDiff(const Tensor& a, const Tensor& b) {
    double m = 0;
    for (size_t i = 0; i < a.data.size(); i++) {
        double d = std::fabs(a.data[i] - b.data[i]);
        if (d > m) m = d;
    }
    return m;
}

static double psnr(const Tensor &a, const Tensor &b)
{
    double mse = 0;
    for (size_t i = 0; i < a.data.size(); i++) {
        double d = a.data[i] - b.data[i];
        mse += d * d;
    }
    mse /= a.data.size();
    if (mse == 0) return 999; // identical
    return 10 * std::log10(1.0 / mse);
}

static Tensor randInput(int h, int w) {
    Tensor t(1, h, w);
    srand(1);
    for (float& v : t.data) v = (float)rand() / (float)RAND_MAX;
    return t;
}

static void check(const char* name, const Tensor& out, const Tensor& ref) {
    if (out.data.size() != ref.data.size()) {
        printf("%-34s WRONG SHAPE\n", name);
        return;
    }
    printf("%-34s max abs err %.2e   psnr %6.1f dB\n", name, maxDiff(out, ref), psnr(out, ref));
}

static int cmd_verify(const Weights& net) {
    Tensor in, ref;
    if (!read_tensor(MODEL_DIR + "ref_in.bin", in) || !read_tensor(MODEL_DIR + "ref_out.bin", ref)) {
        printf("missing reference tensors, run train/export.py\n");
        return 1;
    }
    printf("comparing against PyTorch output, input %dx%d\n\n", in.h, in.w);

    check("cpu_naive", run_naive(net, in), ref);
    check("cpu_accelerate", run_accel(net, in), ref);

    MPSNet* mps = mps_create(net);
    check("mpsgraph_gpu", mps_run(mps, in), ref);
    mps_destroy(mps);

    for (int bits : {32, 16}) {
        std::string compiled = coreml_compile(model_path(in.h, in.w, bits), nullptr);
        if (compiled.empty()) continue;
        for (int units = 0; units < 4; units++) {
            CoreMLModel* m = coreml_load(compiled, units, nullptr);
            if (!m) continue;
            Tensor out;
            std::string name = "coreml_fp" + std::to_string(bits) + "_" + units_name(units);
            if (coreml_run(m, in, out)) check(name.c_str(), out, ref);
            coreml_destroy(m);
        }
    }

    // no PyTorch reference at the bigger sizes so compare to the Accelerate output instead
    for (int i = 1; i < 4; i++) {
        int h = SIZES[i][0], w = SIZES[i][1];
        Tensor big = randInput(h, w);
        Tensor expect = run_accel(net, big);
        printf("\ncomparing against cpu_accelerate, input %dx%d\n\n", w, h);

        mps = mps_create(net);
        check("mpsgraph_gpu", mps_run(mps, big), expect);
        mps_destroy(mps);

        for (int bits : {32, 16}) {
            std::string compiled = coreml_compile(model_path(h, w, bits), nullptr);
            if (compiled.empty()) continue;
            for (int units = 0; units < 4; units++) {
                CoreMLModel* m = coreml_load(compiled, units, nullptr);
                if (!m) continue;
                Tensor out;
                std::string name = "coreml_fp" + std::to_string(bits) + "_" + units_name(units);
                if (coreml_run(m, big, out)) check(name.c_str(), out, expect);
                coreml_destroy(m);
            }
        }
    }
    return 0;
}

static void report(FILE* csv, const std::string& name, int h, int w, double load_ms, const Stats& s) {
    printf("  %-34s load %8.1f ms  first %9.1f ms  median %9.2f ms  p95 %9.2f ms  %7.1f fps\n",
           name.c_str(), load_ms, s.first, s.median, s.p95, 1000.0 / s.median);
    fprintf(csv, "%s,%d,%d,%.2f,%.3f,%.3f,%.3f,%.2f\n", name.c_str(), h, w, load_ms, s.first, s.median, s.p95, 1000.0 / s.median);
    fflush(csv);
}

static int cmd_bench(const Weights& net, int runs) {
    FILE* csv = fopen("results/results.csv", "w");
    if (!csv) {
        printf("could not open results/results.csv\n");
        return 1;
    }
    fprintf(csv, "backend,h,w,load_ms,first_ms,median_ms,p95_ms,fps\n");

    for (int i = 0; i < 4; i++) {
        int h = SIZES[i][0], w = SIZES[i][1];
        Tensor in = randInput(h,w);
        printf("\ninput %dx%d -> output %dx%d\n", w, h, w * 2, h * 2);

        Stats s = time_it([&] { run_naive(net, in); }, 0, 3);
        report(csv, "cpu_naive", h, w, 0, s);

        Stats s = time_it([&] { run_accel(net, in); }, 2, runs);
        report(csv, "cpu_accelerate", h, w, 0, s);

        // new graph for each size so first run includes the graph compile
        // TODO fp16 version of the mps graph?
        double t0 = now_ms();
        MPSNet* mps = mps_create(net);
        double mps_load = now_ms() - t0;
        s = time_it([&] { mps_run(mps, in); }, 3, runs);
        report(csv, "mpsgraph_gpu", h, w, mps_load, s);
        mps_destroy(mps);

        for (int bits : {32, 16}) {
            double compile_ms = 0;
            std::string compiled = coreml_compile(model_path(h, w, bits), &compile_ms);
            if (compiled.empty()) continue;
            printf("  (coreml fp%d compile %.1f ms)\n", bits, compile_ms);
            for (int units = 0; units < 4; units++) {
                double load_ms = 0;
                CoreMLModel* m = coreml_load(compiled, units, &load_ms);
                if (!m) continue;
                Tensor out;
                s = time_it([&] { coreml_run(m, in, out); }, 3, runs);
                report(csv, "coreml_fp" + std::to_string(bits) + "_" + units_name(units), h, w, load_ms, s);
                coreml_destroy(m);
            }
        }
    }
    fclose(csv);
    printf("\nwrote results/results.csv\n");
    return 0;
}

static int cmd_plan() {
    for (int i = 0; i < 4; i++) {
        int h = SIZES[i][0], w = SIZES[i][1];
        for (int bits : {32, 16}) {
            std::string compiled = coreml_compile(model_path(h, w, bits), nullptr);
            if (compiled.empty()) continue;
            printf("\n%dx%d fp%d\n", w, h, bits);
            for (int units = 0; units < 4; units++) {
                auto ops = coreml_plan(compiled, units);
                std::map<std::string, int> counts;
                std::string detail;
                for (auto& op : ops) {
                    counts[op.second]++;
                    detail += op.first + ":" + op.second + " ";
                }
                printf("  %-20s", units_name(units));
                for (auto& c : counts) printf(" %s x%d", c.first.c_str(), c.second);
                printf("   [ %s]\n", detail.c_str());
            }
        }
    }
    return 0;
}

static int run_image(const Weights& net, const std::string& in_path, const std::string& out_path, const std::string& backend) {
    Tensor rgb;
    if (!load_image(in_path, rgb)) {
        printf("could not read %s\n", in_path.c_str());
        return 1;
    }
    Tensor y, cb, cr;
    rgb_to_ycbcr(rgb, y, cb, cr);

    Tensor y2;
    double t0 = now_ms();
    if (backend == "naive") {
        y2 = run_naive(net, y);
    } else if (backend == "accel") {
        y2 = run_accel(net, y);
    } else if (backend == "bilinear") {
        y2 = bilinear_up(y, net.scale);
    } else {
        MPSNet* mps = mps_create(net);
        y2 = mps_run(mps, y);
        mps_destroy(mps);
    }
    double ms = now_ms() - t0;

    Tensor out = ycbcr_to_rgb(y2, bilinear_up(cb, net.scale), bilinear_up(cr, net.scale));
    if (!save_image(out_path, out)) {
        printf("could not write %s\n", out_path.c_str());
        return 1;
    }
    printf("%dx%d -> %dx%d with %s in %.1f ms\n", rgb.w, rgb.h, out.w, out.h, backend.c_str(), ms);
    return 0;
}

static void usage() {
    printf("usage:\n");
    printf("  upscale-bench verify\n");
    printf("  upscale-bench bench [--runs N]\n");
    printf("  upscale-bench plan\n");
    printf("  upscale-bench run in.png out.png [mps|accel|naive|bilinear]\n");
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string cmd = argv[1];

    Weights net;
    if (!load_weights(MODEL_DIR + "weights.bin", net)) {
        printf("could not load models/weights.bin, run train/export.py first\n");
        return 1;
    }

    if (cmd == "verify") return cmd_verify(net);
    if (cmd == "plan") return cmd_plan();
    if (cmd == "bench") {
        int runs = 30;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "--runs") && i + 1 < argc) runs = atoi(argv[++i]);
        }
        return cmd_bench(net, runs);
    }
    if (cmd == "run" && argc >= 4) {
        return run_image(net, argv[2], argv[3], argc >= 5 ? argv[4] : "mps");
    }
    usage();
    return 1;
}
