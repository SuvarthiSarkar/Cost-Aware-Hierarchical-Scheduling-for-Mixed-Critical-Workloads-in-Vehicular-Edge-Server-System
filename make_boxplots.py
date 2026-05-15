#!/usr/bin/env python3
"""Box plots: (A) Per-BS average utilization, (B) Per-task distance.
Reads DIST:/UTIL: lines from --stats output of each algorithm.
Usage: python3 make_boxplots.py [datafile]
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess, sys, os

OUT = 'output/graphs'
os.makedirs(OUT, exist_ok=True)

datafile = sys.argv[1] if len(sys.argv) > 1 else 'data/80av_10bs.txt'

METHOD_ORDER = [
    ('Proposed', 'proposed', True),
    ('NO_DROP',  'no_drop',  True),
    ('LOCAL',    'local_only', False),
    ('AGA',      'aga',      False),
    ('LB',       'lb',       False),
    ('Uc',       'uc',       False)
]
LABELS = ['Proposed', 'No-Drop', 'Local', 'AGA', 'LB', '$U_c$']

util_data = {}
dist_data = {}

for name, binary, needs_args in METHOD_ORDER:
    if needs_args:
        cmd = f'bin/{binary} {datafile} 10 --detailed --stats'
    else:
        cmd = f'bin/{binary} {datafile} --detailed --stats'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    for line in r.stdout.strip().split('\n'):
        if line.startswith('DIST:'):
            parts = line.split(':', 2)
            vals = [float(x) for x in parts[2].split(',') if x]
            dist_data[name] = vals
        elif line.startswith('UTIL:'):
            parts = line.split(':', 2)
            vals = [float(x) for x in parts[2].split(',') if x]
            util_data[name] = vals
    print(f"  {name}: {len(dist_data.get(name,[]))} tasks, {len(util_data.get(name,[]))} BSs")

def make_boxplot_bxp(data_dict, order, labels, ylabel, xlabel, filename):
    """Use ax.bxp() with precomputed stats, matching reference style."""
    box_data = []
    for i, (name, _, _) in enumerate(order):
        vals = data_dict.get(name, [])
        if not vals:
            box_data.append({
                'label': labels[i], 'whislo': 0, 'q1': 0,
                'med': 0, 'q3': 0, 'whishi': 0
            })
            continue
        arr = np.array(vals)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        iqr = q3 - q1
        whislo = max(arr.min(), q1 - 1.5 * iqr)
        whishi = min(arr.max(), q3 + 1.5 * iqr)
        box_data.append({
            'label': labels[i],
            'whislo': whislo,
            'q1': q1,
            'med': med,
            'q3': q3,
            'whishi': whishi,
        })

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bp = ax.bxp(box_data, showfliers=False, patch_artist=True)

    colors = ['#78d9f6', '#caffbf', '#ffd6a5', '#ffe0b2',
              '#d1c4e9', '#e0e0e0']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(2.0)
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1.5)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.savefig(f'{OUT}/{filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{OUT}/{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Saved: {OUT}/{filename}.pdf")

# Normalize distances to [0,1] using grid diagonal
GRID = 200
diagonal = GRID * np.sqrt(2)
dist_norm = {}
for name in dist_data:
    dist_norm[name] = [d / diagonal for d in dist_data[name]]

print("\nGenerating box plots...")
make_boxplot_bxp(util_data, METHOD_ORDER, LABELS,
    r'Average utilization per BS', 'Approaches',
    'fig_boxplot_util')

make_boxplot_bxp(dist_norm, METHOD_ORDER, LABELS,
    r'Normalized $\hat{C}_{dis}$ per task', 'Approaches',
    'fig_boxplot_distance')

print("Done.")
