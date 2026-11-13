# upscale-bench

2× image super-resolution CNN trained in PyTorch, run from C++ on Apple silicon with plain loops, Accelerate, MPSGraph (GPU) and Core ML (CPU / GPU / Neural Engine), then verified and benchmarked.

The model is a 3-layer ESPCN-style network (21,284 parameters) that works on the luma channel. Full analysis is in [REPORT.md](REPORT.md).

## Results — Apple M4, 1080p → 4K

| Backend | median ms | vs naive |
|---|---|---|
| C++ naive loops | 36,200 | 1× |
| C++ im2col + Accelerate | 1,097 | 33× |
| MPSGraph GPU | 38.2 | 948× |
| Core ML FP32 GPU | 35.4 | 1,023× |
| Core ML FP16 Neural Engine | 14.1 | **2,567×** |

## Key findings

- **The Neural Engine needs FP16.** An FP32 model silently falls back to the CPU even when you ask for cpuAndNeuralEngine, there is no warning. I only caught this by checking with MLComputePlan.
- **FP16 quality loss is negligible.** - Max error on the ANE is ~1.4e-3, well under one 8-bit pixel step (3.9e-3). You wouldn't see the difference.
- **First-run latency matters.** The ANE model takes ~118 ms to load and the first inference is 2× slower than steady state. MPSGraph is even worse: first run is 6× slower because the graph compiles for the input shape on the fly.
- **MPSGraph has a gotcha with command queues.**- Using runWithFeeds: in a loop caused latency to climb from 55 ms to over 1 second after ~10 frames. Switching to runWithMTLCommandQueue: with one reused queue fixed it.

## Run

```
make
./upscale-bench verify
./upscale-bench plan
./upscale-bench bench
./upscale-bench run results/sample_lr.png out.png
```

Trained models are in `models/`. To retrain, extract [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz) into `data/`, install `train/requirements.txt` and run `train.py` then `export.py` from `train/`.
