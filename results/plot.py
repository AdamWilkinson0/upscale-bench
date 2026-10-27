import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/results.csv")))
sizes = ["256x256", "512x512", "1024x1024", "1920x1080"]
backends = []
for r in rows:
    if r["backend"] not in backends:
        backends.append(r["backend"])

plt.figure(figsize=(9, 5))
colors = plt.get_cmap("tab20").colors
for i, b in enumerate(backends):
    xs, ys = [], []
    for r in rows:
        if r["backend"] == b:
            xs.append(f'{r["w"]}x{r["h"]}')
            ys.append(float(r["median_ms"]))
    style = "--" if "fp32" in b else "-"
    plt.plot(xs, ys, style, marker="o", label=b, color=colors[i])
plt.yscale("log")
plt.ylabel("median latency (ms, log scale)")
plt.xlabel("input size")
plt.title("2x super-resolution latency by backend (M4)")
plt.legend(fontsize=7)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/latency.png", dpi=150)

hd = [r for r in rows if r["w"] == "1920" and r["backend"] != "cpu_naive"]
plt.figure(figsize=(9, 5))
names = [r["backend"] for r in hd]
plt.barh(names, [float(r["first_ms"]) for r in hd], label="first run")
plt.barh(names, [float(r["median_ms"]) for r in hd], label="steady state (median)")
plt.xscale("log")
plt.xlabel("ms (log scale)")
plt.title("First run vs steady state at 1920x1080")
plt.legend()
plt.tight_layout()
plt.savefig("results/first_run.png", dpi=150)
