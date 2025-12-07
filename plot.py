from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)

T_SEQ = 4871.66

times_mpi = {
    2: 3139.61,
    4: 2039.91,
    8: 1579.15,
    16: 1371.66,
    32: 1935.12,
}

times_island = {
    2: 2142.26,
    4: 1007.49,
    8: 492.01,
    16: 443.44,
    32: 256.71,
}

p_all = [1, 2, 4, 8, 16, 32]
pos_all = list(range(len(p_all)))

speedup_mpi = [T_SEQ / times_mpi[p] if p in times_mpi else 1.0 for p in p_all]
speedup_island = [T_SEQ / times_island[p] if p in times_island else 1.0 for p in p_all]
ideal_speedup = p_all

plt.figure(figsize=(6, 4))
plt.plot(pos_all, speedup_mpi, marker="o", label="naive MPI")
plt.plot(pos_all, speedup_island, marker="s", label="island MPI (ours)")
plt.plot(pos_all, ideal_speedup, linestyle="--", label="Ideal linear speedup")

plt.xlabel("Number of processes $p$")
plt.ylabel("Speedup $S(p) = T_{seq} / T_p$")
plt.title("Speedup vs Number of Processes")
plt.xticks(pos_all, p_all)
plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "u2319_speedup_SIMD_nodes4.png", dpi=300, bbox_inches="tight")
plt.show()

eff_mpi = [s / p for s, p in zip(speedup_mpi, p_all)]
eff_island = [s / p for s, p in zip(speedup_island, p_all)]

plt.figure(figsize=(6, 4))
plt.plot(pos_all, eff_mpi, marker="o", label="naive MPI")
plt.plot(pos_all, eff_island, marker="s", label="island MPI (ours)")

plt.xlabel("Number of processes $p$")
plt.ylabel("Parallel efficiency $E(p) = S(p) / p$")
plt.title("Parallel Efficiency vs Number of Processes")
plt.xticks(pos_all, p_all)
plt.ylim(0, max(eff_mpi + eff_island) * 1.1)
plt.grid(True, linestyle=":", linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "u2319_efficiency_SIMD_nodes4.png", dpi=300, bbox_inches="tight")
plt.show()

data = {
    "MPI naive":     (1579.15, 18.63),
    "lazy-sync MPI": (690.43,  18.70),
    "island MPI (ours)":    (492.01,  17.06),
}

t_seq_quality = T_SEQ

labels = []
speedups_q = []
errors_q = []

for name, (t_ms, err) in data.items():
    labels.append(name)
    speedups_q.append(t_seq_quality / t_ms)
    errors_q.append(err)

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.scatter(speedups_q, errors_q)

offsets = {
    "MPI naive":     (-45, -10),
    "lazy-sync MPI": (5, -15),
    "island MPI":    (-55, 10),
}

colors = ["tab:blue", "tab:orange", "tab:green"]
ax.scatter(speedups_q, errors_q, c=colors[: len(labels)], s=70, edgecolors="k", zorder=3)

for i, name in enumerate(labels):
    dx, dy = offsets.get(name, (5, 5))
    ax.annotate(
        name,
        (speedups_q[i], errors_q[i]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="left",
        color=colors[i],
        fontsize=9,
    )

ax.set_xlabel("Speedup over sequential (×)")
ax.set_ylabel("Mean tour error (%)")
ax.set_title("Solution Quality vs Speed")

min_s, max_s = min(speedups_q), max(speedups_q)
min_e, max_e = min(errors_q), max(errors_q)

pad_s = 0.25 * (max_s - min_s if max_s > min_s else 1.0)
pad_e = 0.25 * (max_e - min_e if max_e > min_e else 1.0)

ax.set_xlim(min_s - pad_s, max_s + pad_s)
ax.set_ylim(max_e + pad_e, min_e - pad_e)

ax.grid(True, linestyle=":", linewidth=0.8)

plt.tight_layout()
plt.savefig(out_dir / "u2319_quality_vs_speed_SIMD_nodes4_p8.png",
            dpi=300, bbox_inches="tight")
plt.show()

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def plot_perf_analysis_3way():
    labels = ['Sequential\n(P=1)', 'Naive MPI\n(P=4)', 'Island MPI (Ours)\n(P=4)']
    
    instructions = [548, 559, 489] 
    
    l1_miss_rate = [1.83, 1.87, 1.67]

    colors_inst = ['#7f7f7f', '#d62728', '#2ca02c'] 
    colors_cache = ['#7f7f7f', '#d62728', '#2ca02c']

    x = np.arange(len(labels))
    width = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    bars1 = ax1.bar(x, instructions, width, color=colors_inst, alpha=0.85, edgecolor='black')
    
    ax1.set_ylabel('Total Instructions (Billions)', fontsize=13, fontweight='bold')
    ax1.set_title('Algorithmic Efficiency', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax1.set_ylim(450, 600)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    bars2 = ax2.bar(x, l1_miss_rate, width, color=colors_cache, alpha=0.85, edgecolor='black')
    
    ax2.set_ylabel('L1 Data Cache Miss Rate (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Cache Locality', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax2.set_ylim(1.5, 2.0)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Micro-architectural Drivers of Superlinear Speedup (u2319)', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('results/perf_analysis_3way.png', dpi=300, bbox_inches='tight')
if __name__ == "__main__":
    plot_perf_analysis_3way() 
