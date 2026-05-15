#!/usr/bin/env python3
"""
Generate combined stacked bar chart from simulation results.
Similar style to the reference figure: grouped bars, stacked costs, method names on top.

Reads from output/*_results.csv files.
Usage: python3 make_combined_graph.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess, os
from read_config import get_config

CFG = get_config()
PSI_D = CFG['PSI_DROP']
PSI_I = CFG['PSI_DIS']
PSI_E = CFG['PSI_E']

OUT = 'output/graphs'
os.makedirs(OUT, exist_ok=True)

# ── Method display names ──
METHOD_MAP = {
    'Proposed': 'Proposed',
    'NO_DROP': 'No-Drop',
    'AGA': 'AGA',
    'LB': 'LB',
    'Uc': '$U_c$',
    'LOCAL': 'Local',
}
METHOD_ORDER = ['Proposed', 'NO_DROP', 'AGA', 'LB', 'Uc', 'LOCAL']

# ── Colors and hatches ──
colors = {
    'Drop': '#ff6b6b',
    'Dis': '#caffbf',
    'Energy': '#78d9f6',
}
hatches = {
    'Drop': '',
    'Dis': '..',
    'Energy': '//',
}
edge_colors = {
    'Drop': '#cc0000',
    'Dis': '#007f5f',
    'Energy': '#0066cc',
}

def parse_csv(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 4: continue
            data[p[0]] = {
                'Drop': float(p[1]),
                'Dis': float(p[2]),
                'Energy': float(p[3]),
            }
    return data

def run_cmd(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return r.stdout.strip()

def run_all(datafile, d_hop=10):
    results = {}
    for algo, binary in [('Proposed', 'proposed'), ('NO_DROP', 'no_drop')]:
        out = run_cmd(f'bin/{binary} {datafile} {d_hop} --detailed')
        for line in out.split('\n'):
            p = line.strip().split(',')
            if len(p) >= 4:
                results[p[0]] = {'Drop': float(p[1]), 'Dis': float(p[2]), 'Energy': float(p[3])}
    for algo, binary in [('AGA','aga'),('LB','lb'),('Uc','uc'),('LOCAL','local_only')]:
        out = run_cmd(f'bin/{binary} {datafile} --detailed')
        for line in out.split('\n'):
            p = line.strip().split(',')
            if len(p) >= 4:
                results[p[0]] = {'Drop': float(p[1]), 'Dis': float(p[2]), 'Energy': float(p[3])}
    return results

# ═══════════════════════════════════════════════════
# GENERATE DATA FOR EACH GROUP
# ═══════════════════════════════════════════════════

print("Generating data for combined graph...")

# Group A: Number of AVs (load variation)
group_a = {}
for nav in [50, 80, 100]:
    fname = f'/tmp/cg_{nav}av.txt'
    run_cmd(f'bin/gen_data {nav} 10 10000 {fname}')
    group_a[f'{nav} AVs'] = run_all(fname, 10)
    print(f"  {nav} AVs done")

# Group B: Number of BSs
group_b = {}
for nbs in [10, 25, 35]:
    fname = f'/tmp/cg_{nbs}bs.txt'
    run_cmd(f'bin/gen_data 80 {nbs} 10000 {fname}')
    um = {10: 0.85, 25: 0.90, 35: 0.95}[nbs]
    group_b[f'{nbs} BSs'] = run_all(fname, 10)
    print(f"  {nbs} BSs done")

# Group C: Hard:Soft ratio
group_c = {}
for ratio in [5, 15, 29]:
    fname = f'/tmp/cg_r{ratio}.txt'
    import shutil
    shutil.copy('config.txt', '/tmp/cg_cfg.txt')
    with open('/tmp/cg_cfg.txt', 'r') as f:
        cfg_text = f.read()
    cfg_text = cfg_text.replace(
        f'HARD_SOFT_RATIO     = {CFG["HARD_SOFT_RATIO"]}',
        f'HARD_SOFT_RATIO     = {ratio}')
    # Handle case where format might differ
    import re
    cfg_text = re.sub(r'HARD_SOFT_RATIO\s*=\s*\d+', f'HARD_SOFT_RATIO     = {ratio}', cfg_text)
    with open('/tmp/cg_cfg.txt', 'w') as f:
        f.write(cfg_text)
    run_cmd(f'bin/gen_data {fname} /tmp/cg_cfg.txt')
    group_c[f'{ratio}:1'] = run_all(fname, 10)
    print(f"  Ratio {ratio}:1 done")

# ═══════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════

all_groups = [
    ('(A) Number of AVs ($N$)', group_a),
    ('(B) Number of BSs ($M$)', group_b),
    ('(C) HC:LC ratio', group_c),
]

# Count total scenarios
total_scenarios = sum(len(g) for _, g in all_groups)
methods_present = [m for m in METHOD_ORDER if any(
    m in g[s] for _, g in all_groups for s in g)]
num_methods = len(methods_present)

fig, ax = plt.subplots(figsize=(14, 3.5))

bar_width = 0.13
group_spacing = 0.35
section_spacing = 0.6

# Compute x positions
x_positions = []
x_pos = 0
group_boundaries = []
scenario_labels = []
section_centers = []

for gi, (group_label, group_data) in enumerate(all_groups):
    section_start = x_pos
    for si, scenario_name in enumerate(group_data):
        x_positions.append(x_pos)
        scenario_labels.append(scenario_name)
        x_pos += num_methods * bar_width + group_spacing
    section_end = x_pos - group_spacing
    section_centers.append((section_start + section_end) / 2)
    if gi < len(all_groups) - 1:
        group_boundaries.append(x_pos - group_spacing / 2 + section_spacing / 2)
        x_pos += section_spacing

x_positions = np.array(x_positions)

# Draw bars
legend_added = {'Drop': False, 'Dis': False, 'Energy': False}
scenario_idx = 0

for gi, (group_label, group_data) in enumerate(all_groups):
    for si, scenario_name in enumerate(group_data):
        results = group_data[scenario_name]
        for j, method in enumerate(methods_present):
            if method not in results:
                scenario_idx_local = x_positions[scenario_idx]
                continue
            costs = results[method]
            bottom = 0
            x = x_positions[scenario_idx] + j * bar_width

            for key in ['Drop', 'Dis', 'Energy']:
                val = costs[key]
                label = ''
                if not legend_added[key]:
                    label_map = {
                        'Drop': r'$\hat{C}_{drop}$',
                        'Dis': r'$\hat{C}_{dis}$',
                        'Energy': r'$\hat{C}_{e}$'
                    }
                    label = label_map[key]
                    legend_added[key] = True

                ax.bar(x, val, bar_width, bottom=bottom,
                       color=colors[key], hatch=hatches[key],
                       edgecolor=edge_colors[key], linewidth=0.3,
                       label=label)
                ax.bar(x, val, bar_width, bottom=bottom,
                       color='none', edgecolor='k', linewidth=0.5)
                bottom += val

            # Method name on top
            total = sum(costs.values())
            display_name = METHOD_MAP.get(method, method)
            ax.text(x + bar_width / 2, total + 0.01, display_name,
                    ha='center', va='bottom', rotation=90, fontsize=5.5)

        scenario_idx += 1

# X axis labels
ax.set_xticks(x_positions + (num_methods - 1) * bar_width / 2)
ax.set_xticklabels(scenario_labels, fontsize=8)

# Section dividers
for bx in group_boundaries:
    ax.axvline(x=bx, color='black', linestyle='--', linewidth=1.5)

# Section labels at bottom
for ci, (group_label, _) in enumerate(all_groups):
    fig.text(
        0.15 + ci * 0.28,
        0.01, group_label, ha='center', fontsize=9)

# Y axis
ax.set_ylabel(r'$\hat{C}_{total}$', fontsize=11)
ax.set_ylim(0, ax.get_ylim()[1] * 1.25)

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(
    by_label.values(), by_label.keys(),
    fontsize=9,
    bbox_to_anchor=(0, 1.02, 1, 0.2),
    loc='lower left', mode='expand',
    borderaxespad=0, ncol=3, edgecolor='k')

ax.set_xlim(x_positions[0] - 0.2,
            x_positions[-1] + num_methods * bar_width + 0.2)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f'{OUT}/fig_combined_stacked.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig_combined_stacked.png', bbox_inches='tight', dpi=300)
plt.close()
print(f"\nSaved: {OUT}/fig_combined_stacked.pdf")
print(f"Saved: {OUT}/fig_combined_stacked.png")
