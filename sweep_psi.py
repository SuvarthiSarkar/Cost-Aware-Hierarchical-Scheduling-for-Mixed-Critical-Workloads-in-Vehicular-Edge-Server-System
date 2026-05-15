#!/usr/bin/env python3
"""Sweep ψ weight combinations and plot results"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 300, 'savefig.bbox': 'tight'
})

from read_config import get_config
_CFG = get_config()

OUT = 'output/graphs'
os.makedirs(OUT, exist_ok=True)

# Parse results
def parse_csv(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 4: continue
            data[p[0]] = {'C_drop': float(p[1]), 'C_dis': float(p[2]), 'C_energy': float(p[3])}
            if len(p) >= 9:
                data[p[0]].update({'H_done': int(p[4]), 'H_drop': int(p[6])})
    return data

ALGOS = ['Proposed', 'NO_DROP', 'AGA', 'LB', 'Uc', 'LOCAL']
LABELS = {'Proposed':'Proposed','NO_DROP':'No-Drop','AGA':'AGA','LB':'LB','Uc':'Uc','LOCAL':'Local'}
COLORS = {'Proposed':'#2196F3','NO_DROP':'#4CAF50','AGA':'#FF9800','LB':'#F44336','Uc':'#9C27B0','LOCAL':'#607D8B'}

# ψ weight combinations to sweep
PSI_COMBOS = [
    # (ψ_drop, ψ_dis, ψ_e, label)
    (0.80, 0.10, 0.10, "Safety-first\n(0.80,0.10,0.10)"),
    (0.70, 0.10, 0.20, "Safety-heavy\n(0.70,0.10,0.20)"),
    (_CFG["PSI_DROP"], _CFG["PSI_DIS"], _CFG["PSI_E"], f"Default\n({_CFG['PSI_DROP']},{_CFG['PSI_DIS']},{_CFG['PSI_E']})"),
    (0.40, 0.20, 0.40, "Balanced\n(0.40,0.20,0.40)"),
    (0.33, 0.33, 0.34, "Equal\n(0.33,0.33,0.34)"),
    (0.20, 0.20, 0.60, "Energy-first\n(0.20,0.20,0.60)"),
    (0.20, 0.40, 0.40, "Distance-heavy\n(0.20,0.40,0.40)"),
    (0.10, 0.10, 0.80, "Energy-only\n(0.10,0.10,0.80)"),
]

scenarios = {
    '80av_10bs': '80 AVs, 10 BSs',
    '80av_35bs': '80 AVs, 35 BSs (Hotspot)',
}

for sc, sc_label in scenarios.items():
    try:
        data = parse_csv(f'output/{sc}_results.csv')
    except:
        print(f"Missing {sc}, skipping"); continue

    # ════════════════════════════════════════════════════════════
    # FIGURE A: C_total across all ψ combos (grouped bar)
    # ════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(PSI_COMBOS))
    width = 0.11
    for i, algo in enumerate(ALGOS):
        if algo not in data: continue
        d = data[algo]
        vals = [wd*d['C_drop'] + wi*d['C_dis'] + we*d['C_energy']
                for wd, wi, we, _ in PSI_COMBOS]
        offset = (i - len(ALGOS)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=LABELS[algo], color=COLORS[algo],
               edgecolor='black', linewidth=0.4)

    ax.set_xlabel('Weight Combination ($\\psi_{drop}$, $\\psi_{dis}$, $\\psi_e$)')
    ax.set_ylabel('$\\hat{C}_{total}$')
    ax.set_title(f'Sensitivity to Weight Selection — {sc_label}')
    ax.set_xticks(x)
    ax.set_xticklabels([l for _,_,_,l in PSI_COMBOS], fontsize=8)
    ax.legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=True)
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(f'{OUT}/fig_psi_sweep_{sc}.png')
    plt.savefig(f'{OUT}/fig_psi_sweep_{sc}.pdf')
    plt.close()
    print(f"Fig A ({sc}): ψ sweep bar chart ✓")

    # ════════════════════════════════════════════════════════════
    # FIGURE B: Winner at each ψ combo
    # ════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 4))
    winner_data = []
    for wd, wi, we, label in PSI_COMBOS:
        best_algo = None; best_cost = 1e9
        for algo in ALGOS:
            if algo not in data: continue
            d = data[algo]
            ct = wd*d['C_drop'] + wi*d['C_dis'] + we*d['C_energy']
            if ct < best_cost: best_cost = ct; best_algo = algo
        winner_data.append((label, best_algo, best_cost))

    x = np.arange(len(winner_data))
    bars = ax.bar(x, [w[2] for w in winner_data], 0.6,
                  color=[COLORS[w[1]] for w in winner_data],
                  edgecolor='black', linewidth=0.5)
    for i, (label, algo, cost) in enumerate(winner_data):
        ax.text(i, cost + 0.005, f'{LABELS[algo]}\n{cost:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Weight Combination')
    ax.set_ylabel('Best $\\hat{C}_{total}$')
    ax.set_title(f'Winning Algorithm per Weight Combo — {sc_label}')
    ax.set_xticks(x)
    ax.set_xticklabels([w[0] for w in winner_data], fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(f'{OUT}/fig_psi_winner_{sc}.png')
    plt.savefig(f'{OUT}/fig_psi_winner_{sc}.pdf')
    plt.close()
    print(f"Fig B ({sc}): Winner per ψ combo ✓")

# ════════════════════════════════════════════════════════════
# FIGURE C: Ternary-style heatmap (ψ_drop vs ψ_e, ψ_dis = 1-both)
# ════════════════════════════════════════════════════════════
sc = '80av_10bs'
if sc in scenarios:
    try:
        data = parse_csv(f'output/{sc}_results.csv')
    except:
        data = None

if data:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Which algorithm wins
    # Panel 2: Proposed's C_total
    resolution = 20
    wd_range = np.linspace(0.05, 0.90, resolution)
    we_range = np.linspace(0.05, 0.90, resolution)

    win_grid = np.full((resolution, resolution), -1.0)
    proposed_grid = np.full((resolution, resolution), np.nan)
    algo_to_id = {a: i for i, a in enumerate(ALGOS)}

    for i, wd in enumerate(wd_range):
        for j, we in enumerate(we_range):
            wi = 1.0 - wd - we
            if wi < 0.01: continue
            best_algo = None; best_cost = 1e9
            for algo in ALGOS:
                if algo not in data: continue
                d = data[algo]
                ct = wd*d['C_drop'] + wi*d['C_dis'] + we*d['C_energy']
                if ct < best_cost: best_cost = ct; best_algo = algo
                if algo == 'Proposed':
                    proposed_grid[j, i] = ct
            win_grid[j, i] = algo_to_id.get(best_algo, -1)

    # Panel 1: Winner map
    ax = axes[0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([COLORS[a] for a in ALGOS])
    im = ax.imshow(win_grid, origin='lower', aspect='auto', cmap=cmap,
                   vmin=0, vmax=len(ALGOS)-1,
                   extent=[wd_range[0], wd_range[-1], we_range[0], we_range[-1]])
    ax.set_xlabel('$\\psi_{drop}$')
    ax.set_ylabel('$\\psi_e$')
    ax.set_title('Winning Algorithm\n($\\psi_{dis}$ = 1 - $\\psi_{drop}$ - $\\psi_e$)')
    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=COLORS[a], label=LABELS[a]) for a in ALGOS]
    ax.legend(handles=patches, loc='upper right', fontsize=7)

    # Panel 2: Proposed C_total heatmap
    ax = axes[1]
    im2 = ax.imshow(proposed_grid, origin='lower', aspect='auto', cmap='RdYlGn_r',
                    extent=[wd_range[0], wd_range[-1], we_range[0], we_range[-1]])
    ax.set_xlabel('$\\psi_{drop}$')
    ax.set_ylabel('$\\psi_e$')
    ax.set_title('Proposed $\\hat{C}_{total}$\n($\\psi_{dis}$ = 1 - $\\psi_{drop}$ - $\\psi_e$)')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    plt.suptitle(f'Weight Sensitivity Analysis — {scenarios[sc]}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_psi_heatmap.png')
    plt.savefig(f'{OUT}/fig_psi_heatmap.pdf')
    plt.close()
    print("Fig C: ψ heatmap ✓")

# ════════════════════════════════════════════════════════════
# FIGURE D: Proposed rank across all ψ combos
# ════════════════════════════════════════════════════════════
sc = '80av_10bs'
if data:
    resolution = 50
    wd_range = np.linspace(0.05, 0.90, resolution)
    we_range = np.linspace(0.05, 0.90, resolution)

    rank_counts = {1: 0, 2: 0, 3: 0}  # rank 1=best, 2=second, 3+=worse
    total_combos = 0
    win_count = {a: 0 for a in ALGOS}

    for wd in wd_range:
        for we in we_range:
            wi = 1.0 - wd - we
            if wi < 0.01: continue
            total_combos += 1
            costs = {}
            for algo in ALGOS:
                if algo not in data: continue
                d = data[algo]
                costs[algo] = wd*d['C_drop'] + wi*d['C_dis'] + we*d['C_energy']
            ranked = sorted(costs.items(), key=lambda x: x[1])
            win_count[ranked[0][0]] += 1
            # Proposed rank
            for rank, (algo, _) in enumerate(ranked, 1):
                if algo == 'Proposed':
                    if rank <= 3:
                        rank_counts[rank] += 1
                    break

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Win count
    ax = axes[0]
    algos_sorted = sorted(win_count.items(), key=lambda x: -x[1])
    names = [LABELS[a] for a, _ in algos_sorted]
    counts = [c for _, c in algos_sorted]
    colors = [COLORS[a] for a, _ in algos_sorted]
    ax.bar(range(len(names)), [c/total_combos*100 for c in counts], 0.6,
           color=colors, edgecolor='black', linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c/total_combos*100 + 1, f'{c/total_combos*100:.1f}%',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'How often each algorithm wins\n({total_combos} ψ combinations)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Proposed rank distribution
    ax = axes[1]
    labels_r = ['1st (Best)', '2nd', '3rd']
    vals_r = [rank_counts.get(i, 0)/total_combos*100 for i in [1, 2, 3]]
    ax.bar(range(3), vals_r, 0.6, color=['gold', 'silver', '#CD7F32'],
           edgecolor='black', linewidth=0.5)
    for i, v in enumerate(vals_r):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Fraction of ψ combinations (%)')
    ax.set_title(f'Proposed Ranking Across {total_combos} Weight Combos')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels_r)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Robustness to Weight Selection — {scenarios[sc]}', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_psi_robustness.png')
    plt.savefig(f'{OUT}/fig_psi_robustness.pdf')
    plt.close()
    print("Fig D: ψ robustness ✓")

print(f"\n=== All ψ-sweep figures saved to {OUT}/ ===")
