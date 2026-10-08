#pragma once
#include <algorithm>
#include <chrono>
#include <vector>

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Stats {
    double first = 0, median = 0, p95 = 0, min = 0;
};

template <typename F>
Stats time_it(F fn, int warmup, int runs) {
    Stats s;
    double t0 = now_ms();
    fn();
    s.first = now_ms() - t0;
    for (int i = 0; i < warmup; i++) fn();

    std::vector<double> t;
    for (int i = 0; i < runs; i++) {
        t0 = now_ms();
        fn();
        t.push_back(now_ms() - t0);
    }
    std::sort(t.begin(), t.end());
    s.min = t[0];
    s.median = t[t.size()/2];
    s.p95 = t[std::min(t.size() - 1, (size_t)(t.size() * 0.95))];
    return s;
}
