#!/usr/bin/env python3
"""
Sweep ONE parameter at a time, run all algorithms, plot results.

Usage:
  python3 sweep_param.py                    # runs ALL sweeps
  python3 sweep_param.py num_av             # sweep only num_av
  python3 sweep_param.py max_util           # sweep only max_util
  python3 sweep_param.py list               # show available sweeps

Edit the SWEEPS dict below to change values.
"""
import subprocess, os, sys, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.bbox': 'tight'
})

OUT = 'output/graphs'
os.makedirs(OUT, exist_ok=True)

# ════════════════════════════════════════════════════════════
# EDIT THESE: parameter sweeps and their values
# ════════════════════════════════════════════════════════════

SWEEPS = {
    # name: (values, xlabel, defaults_override)
    # defaults: num_av=80, num_bs=10, max_util=0.85, d_hop=10, pidle=150, pmax=400,
    #           psi_drop=0.55, psi_dis=0.15, psi_e=0.30

    'num_av': {
        'values': [30, 50, 80, 100, 120],
        'xlabel': 'Number of Autonomous Vehicles',
        'type': 'generator',  # needs data regeneration
    },
    'num_bs': {
        'values': [5, 10, 15, 25, 35, 50],
        'xlabel': 'Number of Base Stations',
        'type': 'generator',
    },
    'umax': {
        'values': [0.2, 0.3, 0.4, 0.5, 0.6, 0.70, 0.80, 0.90, 1.00],
        'xlabel': '$\\hat{U}^{max}$ (BS Capacity Limit)',
        'type': 'power',  # all algorithms via --umax
    },
    'd_hop': {
        'values': [1, 2, 3, 5, 7, 10],
        'xlabel': '$\\hat{D}^{max}$ (Distance Rank Limit)',
        'type': 'proposed_arg',
    },
    'pidle': {
        'values': [15, 50, 80, 120, 150, 200, 300],
        'xlabel': '$P^s$ (Static Power, Watts)',
        'type': 'power',  # all algorithms via --pidle
    },
    'pmax': {
        'values': [200, 300, 400, 500, 700, 1000],
        'xlabel': '$P_{max}$ (Maximum Power, Watts)',
        'type': 'power',
    },
    'psi_drop': {
        'values': [0.10, 0.20, 0.33, 0.40, 0.55, 0.70, 0.80],
        'xlabel': '$\\psi_{drop}$ (Drop Penalty Weight)',
        'type': 'weight',  # post-processing only
    },
    'psi_e': {
        'values': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
        'xlabel': '$\\psi_e$ (Energy Weight)',
        'type': 'weight',
    },
}

# Read defaults from config.txt
from read_config import get_config
_CFG = get_config()
DEF = {
    'num_av': _CFG['NUM_AV'], 'num_bs': _CFG['NUM_BS'], 'sim_ms': _CFG['SIM_MS'],
    'umax': _CFG['MAX_UTIL'], 'd_hop': _CFG['DIS_HOP'],
    'pidle': _CFG['P_IDLE'], 'pmax': _CFG['P_MAX'],
    'psi_drop': _CFG['PSI_DROP'], 'psi_dis': _CFG['PSI_DIS'], 'psi_e': _CFG['PSI_E'],
}

ALGOS = ['Proposed', 'NO_DROP',  'LOCAL', 'AGA', 'LB', 'Uc']
LABELS = {'Proposed':'Proposed','NO_DROP':'No-Drop','LOCAL':'Local','AGA':'AGA','LB':'LB','Uc':'Uc'}
COLORS = {'Proposed':'#2196F3','NO_DROP':'#4CAF50','LOCAL':'#607D8B','AGA':'#FF9800','LB':'#F44336','Uc':'#9C27B0'}
MARKERS = {'Proposed':'o','NO_DROP':'s','LOCAL':'>','AGA':'^','LB':'v','Uc':'D'}

def run_cmd(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return r.stdout.strip()

def parse_line(line):
    p = line.strip().split(',')
    if len(p) < 4: return None
    d = {'name': p[0], 'C_drop': float(p[1]), 'C_dis': float(p[2]), 'C_energy': float(p[3])}
    if len(p) >= 9:
        d.update({'H_done': int(p[4]), 'H_drop': int(p[6]), 'S_drop': int(p[7])})
    return d

def run_all(datafile, d_hop, pidle, pmax, umax):
    """Run all algorithms, return dict of results"""
    results = {}
    power_flags = f'--pidle {pidle} --pmax {pmax} --umax {umax}'

    # Proposed and no_drop take d_hop as positional, power as flags
    for algo, binary in [('Proposed', 'proposed'), ('NO_DROP', 'no_drop')]:
        out = run_cmd(f'bin/{binary} {datafile} {d_hop} --detailed {power_flags}')
        for line in out.split('\n'):
            d = parse_line(line)
            if d: results[d['name']] = d

    # Baselines: power flags only (no d_hop)
    for algo, binary in [('AGA','aga'),('LB','lb'),('Uc','uc'),('LOCAL','local_only')]:
        out = run_cmd(f'bin/{binary} {datafile} --detailed {power_flags}')
        for line in out.split('\n'):
            d = parse_line(line)
            if d: results[d['name']] = d

    return results

def sweep(param_name):
    info = SWEEPS[param_name]
    values = info['values']
    xlabel = info['xlabel']
    ptype = info['type']

    print(f"\n{'='*60}")
    print(f"  Sweeping: {param_name} = {values}")
    print(f"{'='*60}")

    all_results = {}  # {value: {algo: {metrics}}}

    for val in values:
        # Build params from defaults, override the swept one
        p = dict(DEF)
        p[param_name] = val

        # For weight sweeps: adjust other weights to sum to 1
        if ptype == 'weight':
            if param_name == 'psi_drop':
                remaining = 1.0 - val
                p['psi_dis'] = remaining * 0.15 / 0.45  # keep ratio
                p['psi_e'] = remaining - p['psi_dis']
            elif param_name == 'psi_e':
                remaining = 1.0 - val
                p['psi_drop'] = remaining * 0.55 / 0.70
                p['psi_dis'] = remaining - p['psi_drop']

        # Generate data if needed
        if ptype == 'generator':
            datafile = f'/tmp/sweep_{param_name}_{val}.txt'
            run_cmd(f'bin/gen_data {p["num_av"]} {p["num_bs"]} {p["sim_ms"]} {datafile}')
        else:
            datafile = 'data/80av_10bs.txt'

        # Run
        results = run_all(datafile, p['d_hop'], p['pidle'], p['pmax'], p['umax'])

        # Compute C_total with current weights
        for algo in results:
            d = results[algo]
            d['C_total'] = p['psi_drop']*d['C_drop'] + p['psi_dis']*d['C_dis'] + p['psi_e']*d['C_energy']

        all_results[val] = results
        print(f"  {param_name}={val}: ", end='')
        for algo in ALGOS:
            if algo in results:
                print(f"{LABELS[algo]}={results[algo]['C_total']:.3f} ", end='')
        print()

    # ── Plot 1: C_total line chart ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGOS:
        vals_y = [all_results[v].get(algo, {}).get('C_total', np.nan) for v in values]
        ax.plot(values, vals_y, 'o-', label=LABELS[algo], color=COLORS[algo],
                marker=MARKERS[algo], linewidth=2, markersize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$\\hat{C}_{total}$')
    ax.set_title(f'Effect of {param_name} on Total Cost')
    ax.legend(ncol=4, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xticks(values)
    plt.savefig(f'{OUT}/sweep_{param_name}_ctotal.png')
    plt.savefig(f'{OUT}/sweep_{param_name}_ctotal.pdf')
    plt.close()
    print(f"  → sweep_{param_name}_ctotal.png ✓")

    # ── Plot 2: Three costs stacked for Proposed ──
    fig, ax = plt.subplots(figsize=(9, 5))
    p = dict(DEF)
    cdrop = [all_results[v].get('Proposed', {}).get('C_drop', 0) * p['psi_drop'] for v in values]
    cdis = [all_results[v].get('Proposed', {}).get('C_dis', 0) * p['psi_dis'] for v in values]
    ce = [all_results[v].get('Proposed', {}).get('C_energy', 0) * p['psi_e'] for v in values]
    x = np.arange(len(values))
    ax.bar(x, cdrop, 0.5, label='$\\psi_{drop} \\cdot \\hat{C}_{drop}$', color='#EF5350')
    ax.bar(x, cdis, 0.5, bottom=cdrop, label='$\\psi_{dis} \\cdot \\hat{C}_{dis}$', color='#42A5F5')
    ax.bar(x, ce, 0.5, bottom=[a+b for a,b in zip(cdrop,cdis)], label='$\\psi_e \\cdot \\hat{C}_e$', color='#66BB6A')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Weighted Cost')
    ax.set_title(f'Proposed Cost Breakdown vs {param_name}')
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in values])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(f'{OUT}/sweep_{param_name}_breakdown.png')
    plt.savefig(f'{OUT}/sweep_{param_name}_breakdown.pdf')
    plt.close()
    print(f"  → sweep_{param_name}_breakdown.png ✓")

    # ── Plot 3: Hard task drops ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for algo in ALGOS:
        vals_y = [all_results[v].get(algo, {}).get('H_drop', np.nan) for v in values]
        ax.plot(values, vals_y, 'o-', label=LABELS[algo], color=COLORS[algo],
                marker=MARKERS[algo], linewidth=2, markersize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Hard Tasks Dropped')
    ax.set_title(f'Hard Task Drops vs {param_name}')
    ax.legend(ncol=4, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xticks(values)
    plt.savefig(f'{OUT}/sweep_{param_name}_hdrop.png')
    plt.savefig(f'{OUT}/sweep_{param_name}_hdrop.pdf')
    plt.close()
    print(f"  → sweep_{param_name}_hdrop.png ✓")

    # ── Save CSV ──
    with open(f'output/sweep_{param_name}.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow([param_name, 'Algorithm', 'C_drop', 'C_dis', 'C_energy', 'C_total', 'H_done', 'H_drop'])
        for val in values:
            for algo in ALGOS:
                if algo in all_results[val]:
                    d = all_results[val][algo]
                    w.writerow([val, algo, f"{d['C_drop']:.6f}", f"{d['C_dis']:.6f}",
                               f"{d['C_energy']:.6f}", f"{d['C_total']:.6f}",
                               d.get('H_done',''), d.get('H_drop','')])
    print(f"  → output/sweep_{param_name}.csv ✓")

# ── Main ──
if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == 'list':
            print("Available sweeps:")
            for k, v in SWEEPS.items():
                print(f"  {k:12s} → {v['values']}")
            sys.exit(0)
        elif arg in SWEEPS:
            sweep(arg)
        else:
            print(f"Unknown: '{arg}'. Use 'list' to see options.")
            sys.exit(1)
    else:
        for param in SWEEPS:
            sweep(param)

    print(f"\nAll graphs in {OUT}/")
