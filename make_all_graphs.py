#!/usr/bin/env python3
"""Generate all paper figures. Reads weights from config.txt."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from read_config import get_config

CFG = get_config()
PSI_D = CFG['PSI_DROP']
PSI_I = CFG['PSI_DIS']
PSI_E = CFG['PSI_E']

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05
})

OUT = 'output/graphs'
os.makedirs(OUT, exist_ok=True)

def parse_csv(filename):
    data = {}
    with open(filename) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 4: continue
            data[p[0]] = {
                'C_drop': float(p[1]), 'C_dis': float(p[2]), 'C_energy': float(p[3]),
                'C_total': PSI_D*float(p[1]) + PSI_I*float(p[2]) + PSI_E*float(p[3])
            }
            if len(p) >= 9:
                data[p[0]].update({
                    'H_done': int(p[4]), 'S_done': int(p[5]),
                    'H_drop': int(p[6]), 'S_drop': int(p[7]), 'AvgUtil': float(p[8])
                })
    return data

ALGO_ORDER = ['Proposed', 'NO_DROP', 'AGA', 'LB', 'Uc', 'LOCAL']
LABELS = {'Proposed':'Proposed','NO_DROP':'No-Drop','AGA':'AGA','LB':'LB','Uc':'Uc','LOCAL':'Local'}

COLORS = {'Proposed':'#2196F3','NO_DROP':'#4CAF50','AGA':'#FF9800','LB':'#F44336','Uc':'#9C27B0','LOCAL':'#607D8B'}
HATCHES = {'Proposed':'','NO_DROP':'//','AGA':'\\','LB':'xx','Uc':'..','LOCAL':'++'}

scenarios = {
    '50av_10bs':  ('50 AVs, 10 BSs', str(CFG['MAX_UTIL'])),
    '80av_10bs':  ('80 AVs, 10 BSs', str(CFG['MAX_UTIL'])),
    '100av_10bs': ('100 AVs, 10 BSs', str(CFG['MAX_UTIL'])),
    '80av_25bs':  ('80 AVs, 25 BSs', '0.90'),
    '80av_35bs':  ('80 AVs, 35 BSs (Hotspot)', '0.95'),
}

all_data = {}
for sc in scenarios:
    try: all_data[sc] = parse_csv(f'output/{sc}_results.csv')
    except: pass

# Fig 1: C_total bar chart
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(scenarios)); width = 0.11
for i, algo in enumerate(ALGO_ORDER):
    vals = [all_data.get(sc, {}).get(algo, {}).get('C_total', 0) for sc in scenarios]
    offset = (i - len(ALGO_ORDER)/2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=LABELS[algo], color=COLORS[algo],
           hatch=HATCHES[algo], edgecolor='black', linewidth=0.5)
ax.set_xlabel('Scenario'); ax.set_ylabel('$\\hat{C}_{total}$')
ax.set_title(f'Weighted Total Cost ($\\psi_{{drop}}$={PSI_D}, $\\psi_{{dis}}$={PSI_I}, $\\psi_e$={PSI_E})')
ax.set_xticks(x)
ax.set_xticklabels([f'{scenarios[s][0]}\n$\\hat{{U}}^{{max}}$={scenarios[s][1]}' for s in scenarios], fontsize=9)
ax.legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.18), frameon=True)
ax.grid(axis='y', alpha=0.3)
plt.savefig(f'{OUT}/fig_ctotal_all.png'); plt.savefig(f'{OUT}/fig_ctotal_all.pdf'); plt.close()
print("Fig 1: C_total ✓")

# Fig 2: Stacked cost 80av/10bs
sc = '80av_10bs'
if sc in all_data:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    algos = [a for a in ALGO_ORDER if a in all_data[sc]]
    labels = [LABELS[a] for a in algos]; x = np.arange(len(algos))
    cd = [all_data[sc][a]['C_drop']*PSI_D for a in algos]
    ci = [all_data[sc][a]['C_dis']*PSI_I for a in algos]
    ce = [all_data[sc][a]['C_energy']*PSI_E for a in algos]
    ax.bar(x, cd, 0.5, label=f'$\\psi_{{drop}}\\cdot\\hat{{C}}_{{drop}}$', color='#EF5350', edgecolor='black', linewidth=0.5)
    ax.bar(x, ci, 0.5, bottom=cd, label=f'$\\psi_{{dis}}\\cdot\\hat{{C}}_{{dis}}$', color='#42A5F5', edgecolor='black', linewidth=0.5)
    ax.bar(x, ce, 0.5, bottom=[a+b for a,b in zip(cd,ci)], label=f'$\\psi_e\\cdot\\hat{{C}}_e$', color='#66BB6A', edgecolor='black', linewidth=0.5)
    for i in range(len(algos)):
        t = cd[i]+ci[i]+ce[i]; ax.text(i, t+0.005, f'{t:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xlabel('Algorithm'); ax.set_ylabel('Weighted Cost')
    ax.set_title(f'Cost Breakdown — {scenarios[sc][0]}'); ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15); ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
    plt.savefig(f'{OUT}/fig_stacked_80av10bs.png'); plt.savefig(f'{OUT}/fig_stacked_80av10bs.pdf'); plt.close()
    print("Fig 2: Stacked (80av/10bs) ✓")

# Fig 3: Hard tasks
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for idx, met, yl, tl in [(0,'H_done','Completed','Hard Task Completion'),(1,'H_drop','Dropped','Hard Task Drops')]:
    ax = axes[idx]; x = np.arange(len(scenarios))
    for i, algo in enumerate(ALGO_ORDER):
        vals = [all_data.get(sc,{}).get(algo,{}).get(met,0) for sc in scenarios]
        offset = (i - len(ALGO_ORDER)/2 + 0.5) * width
        ax.bar(x+offset, vals, width, label=LABELS[algo], color=COLORS[algo], hatch=HATCHES[algo], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Scenario'); ax.set_ylabel(yl); ax.set_title(tl); ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_','/') for s in scenarios], fontsize=8, rotation=15); ax.grid(axis='y', alpha=0.3)
axes[0].legend(ncol=4, loc='upper center', bbox_to_anchor=(1.1, 1.22), frameon=True)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_hard_tasks.png'); plt.savefig(f'{OUT}/fig_hard_tasks.pdf'); plt.close()
print("Fig 3: Hard tasks ✓")

# Fig 4: Three costs 80av/10bs
sc = '80av_10bs'
if sc in all_data:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [('C_drop','$\\hat{C}_{drop}$','Drop Cost'),('C_dis','$\\hat{C}_{dis}$','Distance Cost'),('C_energy','$\\hat{C}_e$','Energy Cost')]
    algos = [a for a in ALGO_ORDER if a in all_data[sc]]; labels = [LABELS[a] for a in algos]; x = np.arange(len(algos))
    for ax,(met,yl,tl) in zip(axes,metrics):
        vals = [all_data[sc][a][met] for a in algos]
        ax.bar(x, vals, 0.6, color=[COLORS[a] for a in algos], hatch=[HATCHES[a] for a in algos], edgecolor='black', linewidth=0.5)
        ax.set_ylabel(yl); ax.set_title(tl); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=9); ax.grid(axis='y', alpha=0.3)
        for i,v in enumerate(vals): ax.text(i, v+0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    plt.suptitle(f'{scenarios[sc][0]}', fontsize=13, y=1.02); plt.tight_layout()
    plt.savefig(f'{OUT}/fig_three_costs_80av10bs.png'); plt.savefig(f'{OUT}/fig_three_costs_80av10bs.pdf'); plt.close()
    print("Fig 4: Three costs ✓")

# Fig 5: Stacked hotspot
sc = '80av_35bs'
if sc in all_data:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    algos = [a for a in ALGO_ORDER if a in all_data[sc]]; labels = [LABELS[a] for a in algos]; x = np.arange(len(algos))
    cd = [all_data[sc][a]['C_drop']*PSI_D for a in algos]
    ci = [all_data[sc][a]['C_dis']*PSI_I for a in algos]
    ce = [all_data[sc][a]['C_energy']*PSI_E for a in algos]
    ax.bar(x, cd, 0.5, label=f'$\\psi_{{drop}}\\cdot\\hat{{C}}_{{drop}}$', color='#EF5350', edgecolor='black', linewidth=0.5)
    ax.bar(x, ci, 0.5, bottom=cd, label=f'$\\psi_{{dis}}\\cdot\\hat{{C}}_{{dis}}$', color='#42A5F5', edgecolor='black', linewidth=0.5)
    ax.bar(x, ce, 0.5, bottom=[a+b for a,b in zip(cd,ci)], label=f'$\\psi_e\\cdot\\hat{{C}}_e$', color='#66BB6A', edgecolor='black', linewidth=0.5)
    for i in range(len(algos)):
        t = cd[i]+ci[i]+ce[i]; ax.text(i, t+0.005, f'{t:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xlabel('Algorithm'); ax.set_ylabel('Weighted Cost'); ax.set_title(f'Cost Breakdown — {scenarios[sc][0]}')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15); ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
    plt.savefig(f'{OUT}/fig_stacked_hotspot.png'); plt.savefig(f'{OUT}/fig_stacked_hotspot.pdf'); plt.close()
    print("Fig 5: Stacked hotspot ✓")

# Fig 6: Scalability AVs
fig, ax = plt.subplots(figsize=(8, 4.5))
av_sc = ['50av_10bs','80av_10bs','100av_10bs']; av_lb = [50,80,100]
for algo in ALGO_ORDER:
    vals = [all_data.get(sc,{}).get(algo,{}).get('C_total',0) for sc in av_sc]
    ax.plot(av_lb, vals, '-', label=LABELS[algo], color=COLORS[algo], linewidth=2, markersize=7, marker='o')
ax.set_xlabel('Number of AVs'); ax.set_ylabel('$\\hat{C}_{total}$')
ax.set_title(f'Scalability: Total Cost vs AVs ({CFG["NUM_BS"]} BSs, $\\hat{{U}}^{{max}}$={CFG["MAX_UTIL"]})')
ax.legend(ncol=4, loc='upper left'); ax.grid(alpha=0.3); ax.set_xticks(av_lb)
plt.savefig(f'{OUT}/fig_scalability_avs.png'); plt.savefig(f'{OUT}/fig_scalability_avs.pdf'); plt.close()
print("Fig 6: Scalability AVs ✓")

# Fig 7: Scalability BSs
fig, ax = plt.subplots(figsize=(8, 4.5))
bs_sc = ['80av_10bs','80av_25bs','80av_35bs']; bs_lb = [10,25,35]
for algo in ALGO_ORDER:
    vals = [all_data.get(sc,{}).get(algo,{}).get('C_total',0) for sc in bs_sc]
    ax.plot(bs_lb, vals, '-', label=LABELS[algo], color=COLORS[algo], linewidth=2, markersize=7, marker='o')
ax.set_xlabel('Number of BSs'); ax.set_ylabel('$\\hat{C}_{total}$'); ax.set_title('Scalability: Total Cost vs BSs (80 AVs)')
ax.legend(ncol=4, loc='upper right'); ax.grid(alpha=0.3); ax.set_xticks(bs_lb)
plt.savefig(f'{OUT}/fig_scalability_bs.png'); plt.savefig(f'{OUT}/fig_scalability_bs.pdf'); plt.close()
print("Fig 7: Scalability BSs ✓")

# Fig 8: Hard drop rate
fig, ax = plt.subplots(figsize=(8, 4.5))
for algo in ALGO_ORDER:
    vals = []
    for sc in scenarios:
        d = all_data.get(sc,{}).get(algo,{})
        ht = d.get('H_done',0)+d.get('H_drop',0)
        vals.append(d.get('H_drop',0)/max(ht,1)*100)
    ax.plot(range(len(scenarios)), vals, '-', label=LABELS[algo], color=COLORS[algo], linewidth=2, markersize=7, marker='o')
ax.set_xlabel('Scenario'); ax.set_ylabel('Hard Drop Rate (%)'); ax.set_title('Hard Task Drop Rate')
ax.set_xticks(range(len(scenarios))); ax.set_xticklabels([s.replace('_','/') for s in scenarios], fontsize=8, rotation=15)
ax.legend(ncol=4, loc='upper left'); ax.grid(alpha=0.3)
plt.savefig(f'{OUT}/fig_hard_drop_rate.png'); plt.savefig(f'{OUT}/fig_hard_drop_rate.pdf'); plt.close()
print("Fig 8: Hard drop rate ✓")

# Fig 9: Tradeoff scatter
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for idx, sc in enumerate(['80av_10bs','80av_35bs']):
    if sc not in all_data: continue
    ax = axes[idx]
    for algo in ALGO_ORDER:
        if algo not in all_data[sc]: continue
        d = all_data[sc][algo]
        ax.scatter(d['C_energy'], d['C_drop'], s=120, color=COLORS[algo], edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(LABELS[algo], (d['C_energy'], d['C_drop']), textcoords='offset points', xytext=(5,5), fontsize=8)
    ax.set_xlabel('$\\hat{C}_e$'); ax.set_ylabel('$\\hat{C}_{drop}$'); ax.set_title(f'{scenarios[sc][0]}'); ax.grid(alpha=0.3)
plt.suptitle('Energy–Drop Tradeoff', fontsize=13, y=1.02); plt.tight_layout()
plt.savefig(f'{OUT}/fig_tradeoff.png'); plt.savefig(f'{OUT}/fig_tradeoff.pdf'); plt.close()
print("Fig 9: Tradeoff ✓")

# Fig 10: Improvement
sc = '80av_10bs'
if sc in all_data and 'Proposed' in all_data[sc]:
    fig, ax = plt.subplots(figsize=(8, 4))
    baselines = [a for a in ALGO_ORDER if a != 'Proposed' and a in all_data[sc]]
    pct = all_data[sc]['Proposed']['C_total']
    imps = [(all_data[sc][b]['C_total']-pct)/all_data[sc][b]['C_total']*100 for b in baselines]
    x = np.arange(len(baselines)); colors_i = ['#4CAF50' if v>0 else '#F44336' for v in imps]
    ax.bar(x, imps, 0.6, color=colors_i, edgecolor='black', linewidth=0.5)
    for i,v in enumerate(imps): ax.text(i, v+0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('Baseline'); ax.set_ylabel('Improvement (%)')
    ax.set_title(f'Proposed Improvement over Baselines — {scenarios[sc][0]}')
    ax.set_xticks(x); ax.set_xticklabels([LABELS[b] for b in baselines], rotation=15)
    ax.axhline(y=0, color='black', linewidth=0.5); ax.grid(axis='y', alpha=0.3)
    plt.savefig(f'{OUT}/fig_improvement.png'); plt.savefig(f'{OUT}/fig_improvement.pdf'); plt.close()
    print("Fig 10: Improvement ✓")

print(f"\nDone. ψ_drop={PSI_D}, ψ_dis={PSI_I}, ψ_e={PSI_E} (from config.txt)")
