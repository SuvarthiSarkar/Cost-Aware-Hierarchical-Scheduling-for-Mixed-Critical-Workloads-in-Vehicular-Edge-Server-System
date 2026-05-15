#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
 FFE-VESS: Unified Experiment Runner
 Generates ALL paper figures from scratch.

 Usage:
   python3 run_experiments.py              # everything
   python3 run_experiments.py default      # default-data graphs only
   python3 run_experiments.py sweep        # parameter sweeps only
   python3 run_experiments.py sweep d_hop  # single sweep
   python3 run_experiments.py list         # list available sweeps
═══════════════════════════════════════════════════════════════
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import subprocess, sys, os, csv, time

OUT = 'output/graphs'
os.makedirs(OUT, exist_ok=True)
os.makedirs('output', exist_ok=True)

from read_config import get_config
CFG = get_config()

# ─────────────────────────────────────────────
# Algorithm definitions
# ─────────────────────────────────────────────
ALGOS = ['Proposed', 'NO_DROP', 'AGA', 'LB', 'Uc', 'LOCAL']
BINS  = {'Proposed':'proposed','NO_DROP':'no_drop','AGA':'aga','LB':'lb','Uc':'uc','LOCAL':'local_only'}
LABELS = {'Proposed':'Proposed','NO_DROP':'No-Drop','AGA':'AGA','LB':'LB','Uc':'$U_c$','LOCAL':'Local'}
COLORS = {'Proposed':'#2196F3','NO_DROP':'#4CAF50','AGA':'#FF9800','LB':'#F44336','Uc':'#9C27B0','LOCAL':'#607D8B'}
MARKERS = {'Proposed':'o','NO_DROP':'s','AGA':'^','LB':'v','Uc':'D','LOCAL':'>'}
HAS_DHOP = {'Proposed', 'NO_DROP'}  # algorithms that take d_hop arg

DEFAULT_DATA = 'data/80av_10bs.txt'
DEFAULT_DHOP = CFG.get('DIS_HOP', 10)

# ─────────────────────────────────────────────
# Sweep definitions
# ─────────────────────────────────────────────
SWEEPS = {

    'd_hop': {
        'values': [1, 2, 3, 5, 7, 10, 20, 50, 100, 200],
        'xlabel': '$\\hat{D}^{max}$ (Distance Rank Limit)',
        'type': 'proposed_arg',
    },
    'umax': {
        'values': [0.2, 0.3, 0.4, 0.5, 0.6, 0.70, 0.80, 0.90, 1.00],
        'xlabel': '$\\hat{U}^{max}$ (BS Capacity Limit)',
        'type': 'power',  # all algorithms via --umax
    },
    'delta': {
        'values': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        'xlabel': '$\\delta$ (Drop Penalty Amplifier)',
        'type': 'delta',
    },
    'load': {
        'values': ['light', 'mid_light', 'mid', 'heavy', 'hotspot'],
        'xlabel': 'Load Scenario',
        'type': 'load',
    },
    'pidle': {
        'values': [15, 50, 80, 120, 150, 200, 300],
        'xlabel': '$P^{s}$ (Static Power, Watts)',
        'type': 'power',
    },
    'pmax': {
        'values': [200, 300, 400, 500, 700, 1000],
        'xlabel': '$P_{max}$ (Maximum Power, Watts)',
        'type': 'power',
    },
    'num_av': {
        'values': [30, 50, 80, 100, 120],
        'xlabel': 'Number of Autonomous Vehicles',
        'type': 'generator',
    },
    'num_bs': {
        'values': [5, 10, 15, 25, 35],
        'xlabel': 'Number of Base Stations',
        'type': 'generator',
    },
    'psi_drop': {
        'values': [0.10, 0.20, 0.33, 0.40, 0.55, 0.70, 0.80],
        'xlabel': '$\\psi_{drop}$ (Drop Penalty Weight)',
        'type': 'weight',
    },
    'psi_e': {
        'values': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80],
        'xlabel': '$\\psi_e$ (Energy Weight)',
        'type': 'weight',
    },
}

# Load scenario definitions: {name: (num_av, hotspot_prob, hotspot_cluster, hotspot_radius, label)}
LOAD_SCENARIOS = {
    'light':      (30,  0.50, 0.30, 30, 'Light\n(30 AVs)'),
    'mid_light':  (50,  0.50, 0.30, 30, 'Mid-Light\n(50 AVs)'),
    'mid':        (80,  0.50, 0.30, 30, 'Mid\n(80 AVs)'),
    'heavy':      (120, 0.50, 0.30, 30, 'Heavy\n(120 AVs)'),
    'hotspot':    (80,  0.80, 0.60, 20, 'Hotspot\n(80 AVs, clustered)'),
}

DEF = {
    'num_av': CFG['NUM_AV'], 'num_bs': CFG['NUM_BS'], 'sim_ms': CFG['SIM_MS'],
    'umax': CFG['MAX_UTIL'], 'max_util': CFG['MAX_UTIL'], 'd_hop': CFG['DIS_HOP'],
    'pidle': CFG['P_IDLE'], 'pmax': CFG['P_MAX'],
    'delta': CFG.get('DELTA', 0.0),
    'psi_drop': CFG['PSI_DROP'], 'psi_dis': CFG['PSI_DIS'], 'psi_e': CFG['PSI_E'],
}

# ═════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════
def run_cmd(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()

def parse_csv_line(line):
    """Parse: Name,C_drop,C_dis,C_energy,H_done,S_done,H_drop,S_drop,C_total"""
    p = line.strip().split(',')
    if len(p) < 4: return None
    d = {'name': p[0], 'C_drop': float(p[1]), 'C_dis': float(p[2]), 'C_energy': float(p[3])}
    if len(p) >= 8:
        d.update({'H_done': int(p[4]), 'S_done': int(p[5]),
                  'H_drop': int(p[6]), 'S_drop': int(p[7])})
    if len(p) >= 9:
        d['C_total'] = float(p[8])
    return d

def run_algo(binary, datafile, d_hop=None, extra_flags=''):
    """Run one algorithm binary, return parsed dict or None."""
    cmd = f'bin/{binary} {datafile}'
    if d_hop is not None:
        cmd += f' {d_hop}'
    cmd += f' --detailed {extra_flags}'
    out = run_cmd(cmd)
    for line in out.split('\n'):
        d = parse_csv_line(line)
        if d: return d
    return None

def run_all_algos(datafile, d_hop=DEFAULT_DHOP, extra_flags=''):
    """Run all 6 algorithms, return {algo_name: metrics_dict}."""
    results = {}
    for algo in ALGOS:
        dh = d_hop if algo in HAS_DHOP else None
        d = run_algo(BINS[algo], datafile, dh, extra_flags)
        if d:
            results[algo] = d
    return results

def run_all_with_params(datafile, d_hop, pidle, pmax, umax, delta=0.0, max_util=0.85):
    """Run all algorithms with CLI overrides."""
    flags = f'--pidle {pidle} --pmax {pmax} --umax {umax} --delta {delta} --max-util {max_util}'
    return run_all_algos(datafile, d_hop, flags)

def compute_ctotal(results, psi_drop, psi_dis, psi_e):
    """Recompute C_total with given weights."""
    for algo in results:
        d = results[algo]
        d['C_total'] = psi_drop*d['C_drop'] + psi_dis*d['C_dis'] + psi_e*d['C_energy']

def run_algo_stats(binary, datafile, d_hop=None):
    """Run with --stats and parse DIST:/UTIL: lines."""
    cmd = f'bin/{binary} {datafile}'
    if d_hop is not None:
        cmd += f' {d_hop}'
    cmd += ' --detailed --stats'
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    dist_vals, util_vals = [], []
    for line in r.stdout.strip().split('\n'):
        if line.startswith('DIST:'):
            parts = line.split(':', 2)
            dist_vals = [float(x) for x in parts[2].split(',') if x]
        elif line.startswith('UTIL:'):
            parts = line.split(':', 2)
            util_vals = [float(x) for x in parts[2].split(',') if x]
    return dist_vals, util_vals


# ═════════════════════════════════════════════
#  PART 1: DEFAULT DATA GRAPHS
# ═════════════════════════════════════════════
def generate_default_graphs(datafile=DEFAULT_DATA):
    print(f"\n{'='*60}")
    print(f"  PART 1: Default Data Graphs ({datafile})")
    print(f"{'='*60}\n")

    # ── 1A. Run all algorithms ──
    print("Running all 6 algorithms...")
    results = run_all_algos(datafile)
    for algo in ALGOS:
        if algo in results:
            d = results[algo]
            print(f"  {LABELS[algo]:10s}  C_drop={d['C_drop']:.3f}  C_dis={d['C_dis']:.3f}  "
                  f"C_energy={d['C_energy']:.3f}  C_total={d.get('C_total',0):.3f}")

    # ── 1B. Bar chart: C_total comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ALGOS))
    ctotals = [results.get(a, {}).get('C_total', 0) for a in ALGOS]
    bars = ax.bar(x, ctotals, 0.55, color=[COLORS[a] for a in ALGOS], edgecolor='white', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=11)
    ax.set_ylabel('$\\hat{C}_{total}$', fontsize=12)
    ax.set_title('Total Cost Comparison (Default Scenario)', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, ctotals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_ctotal_comparison.pdf')
    plt.savefig(f'{OUT}/fig_ctotal_comparison.png', dpi=200)
    plt.close()
    print(f"  → fig_ctotal_comparison.pdf ✓")

    # ── 1C. Stacked bar: cost breakdown for all algorithms ──
    fig, ax = plt.subplots(figsize=(10, 5))
    psi_d, psi_dis, psi_e = DEF['psi_drop'], DEF['psi_dis'], DEF['psi_e']
    cd = [results.get(a, {}).get('C_drop', 0) * psi_d for a in ALGOS]
    ci = [results.get(a, {}).get('C_dis', 0) * psi_dis for a in ALGOS]
    ce = [results.get(a, {}).get('C_energy', 0) * psi_e for a in ALGOS]
    ax.bar(x, cd, 0.55, label=f'$\\psi_{{drop}}={psi_d}$ · $\\hat{{C}}_{{drop}}$', color='#EF5350')
    ax.bar(x, ci, 0.55, bottom=cd, label=f'$\\psi_{{dis}}={psi_dis}$ · $\\hat{{C}}_{{dis}}$', color='#42A5F5')
    ax.bar(x, ce, 0.55, bottom=[a+b for a,b in zip(cd,ci)],
           label=f'$\\psi_e={psi_e}$ · $\\hat{{C}}_e$', color='#66BB6A')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=11)
    ax.set_ylabel('Weighted Cost Components', fontsize=12)
    ax.set_title('Cost Breakdown by Algorithm', fontsize=13)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_cost_breakdown.pdf')
    plt.savefig(f'{OUT}/fig_cost_breakdown.png', dpi=200)
    plt.close()
    print(f"  → fig_cost_breakdown.pdf ✓")

    # ── 1D. Hard task completion / drops bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    h_done = [results.get(a, {}).get('H_done', 0) for a in ALGOS]
    h_drop = [results.get(a, {}).get('H_drop', 0) for a in ALGOS]
    w = 0.35
    ax.bar(x - w/2, h_done, w, label='HC Completed', color='#43A047', edgecolor='white')
    ax.bar(x + w/2, h_drop, w, label='HC Dropped',   color='#E53935', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=11)
    ax.set_ylabel('Number of Hard (HC) Tasks', fontsize=12)
    ax.set_title('HC Task Completion vs Drops', fontsize=13)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_hc_tasks.pdf')
    plt.savefig(f'{OUT}/fig_hc_tasks.png', dpi=200)
    plt.close()
    print(f"  → fig_hc_tasks.pdf ✓")

    # ── 1E. Three individual cost metrics comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (metric, ylabel, title) in enumerate([
        ('C_drop', '$\\hat{C}_{drop}$', 'Drop Cost'),
        ('C_dis',  '$\\hat{C}_{dis}$',  'Distance Cost'),
        ('C_energy','$\\hat{C}_e$',      'Energy Cost'),
    ]):
        ax = axes[idx]
        vals = [results.get(a, {}).get(metric, 0) for a in ALGOS]
        bars = ax.bar(np.arange(len(ALGOS)), vals, 0.55,
                      color=[COLORS[a] for a in ALGOS], edgecolor='white')
        ax.set_xticks(np.arange(len(ALGOS)))
        ax.set_xticklabels([LABELS[a] for a in ALGOS], fontsize=9, rotation=20, ha='right')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_three_costs.pdf')
    plt.savefig(f'{OUT}/fig_three_costs.png', dpi=200)
    plt.close()
    print(f"  → fig_three_costs.pdf ✓")

    # ── 1F. Box plots (utilization + distance) ──
    print("\nGenerating box plots...")
    util_data, dist_data = {}, {}
    for algo in ALGOS:
        binary = BINS[algo]
        dh = DEFAULT_DHOP if algo in HAS_DHOP else None
        dvals, uvals = run_algo_stats(binary, datafile, dh)
        if dvals: dist_data[algo] = dvals
        if uvals: util_data[algo] = uvals
        print(f"  {LABELS[algo]:10s}  {len(dvals)} tasks, {len(uvals)} BSs")

    label_list = [LABELS[a] for a in ALGOS]
    color_list = [COLORS[a] for a in ALGOS]

    def make_boxplot(data_dict, filename, ylabel, title):
        fig, ax = plt.subplots(figsize=(10, 5))
        bxp_data = []
        for algo in ALGOS:
            vals = data_dict.get(algo, [])
            if not vals:
                bxp_data.append({'med': 0, 'q1': 0, 'q3': 0, 'whislo': 0, 'whishi': 0})
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            wlo = max(min(vals), q1 - 1.5*iqr)
            whi = min(max(vals), q3 + 1.5*iqr)
            bxp_data.append({'med': med, 'q1': q1, 'q3': q3, 'whislo': wlo, 'whishi': whi})
        bp = ax.bxp(bxp_data, positions=range(len(ALGOS)), widths=0.5,
                     patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], color_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
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
        ax.set_xticks(range(len(ALGOS)))
        ax.set_xticklabels(label_list, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUT}/{filename}.pdf')
        plt.savefig(f'{OUT}/{filename}.png', dpi=200)
        plt.close()
        print(f"  → {filename}.pdf ✓")

    make_boxplot(util_data, 'fig_boxplot_util',
                 'Average BS Utilization', 'BS Utilization Distribution')
    make_boxplot(dist_data, 'fig_boxplot_distance',
                 'Normalized Distance $\\hat{C}_{dis}$', 'Per-Task Distance Distribution')

    # ── 1G. Save default results CSV ──
    with open('output/default_results.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['Algorithm','C_drop','C_dis','C_energy','C_total','H_done','S_done','H_drop','S_drop'])
        for algo in ALGOS:
            if algo in results:
                d = results[algo]
                w.writerow([algo, f"{d['C_drop']:.6f}", f"{d['C_dis']:.6f}",
                           f"{d['C_energy']:.6f}", f"{d.get('C_total',0):.6f}",
                           d.get('H_done',''), d.get('S_done',''),
                           d.get('H_drop',''), d.get('S_drop','')])
    print(f"\n  → output/default_results.csv ✓")
    return results


# ═════════════════════════════════════════════
#  PART 2: PARAMETER SWEEPS
# ═════════════════════════════════════════════
def do_sweep(param_name):
    info = SWEEPS[param_name]
    values = info['values']
    xlabel = info['xlabel']
    ptype = info['type']

    print(f"\n{'─'*55}")
    print(f"  Sweep: {param_name} = {values}")
    print(f"{'─'*55}")

    all_results = {}

    for val in values:
        p = dict(DEF)

        # Weight sweeps: adjust others to sum to 1
        if ptype == 'weight':
            p[param_name] = val
            if param_name == 'psi_drop':
                remaining = 1.0 - val
                p['psi_dis'] = remaining * DEF['psi_dis'] / (DEF['psi_dis'] + DEF['psi_e'])
                p['psi_e'] = remaining - p['psi_dis']
            elif param_name == 'psi_e':
                remaining = 1.0 - val
                p['psi_drop'] = remaining * DEF['psi_drop'] / (DEF['psi_drop'] + DEF['psi_dis'])
                p['psi_dis'] = remaining - p['psi_drop']
        elif ptype != 'load':
            p[param_name] = val

        # ── Generate data ──
        if ptype == 'generator':
            datafile = f'/tmp/sweep_{param_name}_{val}.txt'
            run_cmd(f'bin/gen_data {p["num_av"]} {p["num_bs"]} {p["sim_ms"]} {datafile}')
        elif ptype == 'load':
            sc = LOAD_SCENARIOS[val]
            nav, hp, hc, hr, label = sc
            datafile = f'/tmp/sweep_load_{val}.txt'
            # Create temp config with hotspot params
            import shutil
            tmp_cfg = f'/tmp/sweep_load_{val}_cfg.txt'
            shutil.copy('config.txt', tmp_cfg)
            with open(tmp_cfg, 'a') as f:
                f.write(f'\nHOTSPOT_PROB    = {hp}\n')
                f.write(f'HOTSPOT_CLUSTER = {hc}\n')
                f.write(f'HOTSPOT_RADIUS  = {hr}\n')
            run_cmd(f'bin/gen_data {nav} {p["num_bs"]} {p["sim_ms"]} {datafile} {tmp_cfg}')
        else:
            datafile = DEFAULT_DATA

        # ── Run algorithms ──
        delta_val = p.get('delta', DEF['delta'])
        max_util_val = p.get('max_util', DEF['max_util'])
        results = run_all_with_params(datafile, p['d_hop'], p['pidle'], p['pmax'],
                                       p['umax'], delta_val, max_util_val)
        compute_ctotal(results, p['psi_drop'], p['psi_dis'], p['psi_e'])
        all_results[val] = results

        print(f"  {param_name}={val}: ", end='')
        for algo in ALGOS:
            if algo in results:
                print(f"{LABELS[algo]}={results[algo]['C_total']:.3f} ", end='')
        print()

    # ── Determine x-axis type ──
    is_numeric = all(isinstance(v, (int, float)) for v in values)
    xpos = np.arange(len(values))
    xlabels = [LOAD_SCENARIOS[v][4] if ptype == 'load' else str(v) for v in values]

    # ── Plot 1: C_total ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in ALGOS:
        y = [all_results[v].get(algo, {}).get('C_total', np.nan) for v in values]
        if is_numeric:
            ax.plot(values, y, '-', label=LABELS[algo], color=COLORS[algo],
                    marker=MARKERS[algo], linewidth=2, markersize=7)
        else:
            ax.plot(xpos, y, '-', label=LABELS[algo], color=COLORS[algo],
                    marker=MARKERS[algo], linewidth=2, markersize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('$\\hat{C}_{total}$', fontsize=12)
    ax.set_title(f'Total Cost vs {xlabel}', fontsize=13)
    ax.legend(ncol=3, loc='best')
    ax.grid(alpha=0.3)
    if is_numeric: ax.set_xticks(values)
    else: ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT}/sweep_{param_name}_ctotal.pdf')
    plt.savefig(f'{OUT}/sweep_{param_name}_ctotal.png', dpi=200)
    plt.close()
    print(f"  → sweep_{param_name}_ctotal.pdf ✓")

    # ── Plot 2: cost breakdown (Proposed) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    psi_d, psi_dis, psi_e = DEF['psi_drop'], DEF['psi_dis'], DEF['psi_e']
    cd = [all_results[v].get('Proposed', {}).get('C_drop', 0) * psi_d for v in values]
    ci = [all_results[v].get('Proposed', {}).get('C_dis', 0) * psi_dis for v in values]
    ce = [all_results[v].get('Proposed', {}).get('C_energy', 0) * psi_e for v in values]
    ax.bar(xpos, cd, 0.5, label='$\\psi_{drop} \\cdot \\hat{C}_{drop}$', color='#EF5350')
    ax.bar(xpos, ci, 0.5, bottom=cd, label='$\\psi_{dis} \\cdot \\hat{C}_{dis}$', color='#42A5F5')
    ax.bar(xpos, ce, 0.5, bottom=[a+b for a,b in zip(cd,ci)],
           label='$\\psi_e \\cdot \\hat{C}_e$', color='#66BB6A')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Weighted Cost', fontsize=12)
    ax.set_title(f'Proposed: Cost Breakdown vs {xlabel}', fontsize=13)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/sweep_{param_name}_breakdown.pdf')
    plt.savefig(f'{OUT}/sweep_{param_name}_breakdown.png', dpi=200)
    plt.close()
    print(f"  → sweep_{param_name}_breakdown.pdf ✓")

    # ── Plot 3: HC task drops ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in ALGOS:
        y = [all_results[v].get(algo, {}).get('H_drop', np.nan) for v in values]
        if is_numeric:
            ax.plot(values, y, '-', label=LABELS[algo], color=COLORS[algo],
                    marker=MARKERS[algo], linewidth=2, markersize=7)
        else:
            ax.plot(xpos, y, '-', label=LABELS[algo], color=COLORS[algo],
                    marker=MARKERS[algo], linewidth=2, markersize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('HC Tasks Dropped', fontsize=12)
    ax.set_title(f'HC Task Drops vs {xlabel}', fontsize=13)
    ax.legend(ncol=3, loc='best')
    ax.grid(alpha=0.3)
    if is_numeric: ax.set_xticks(values)
    else: ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT}/sweep_{param_name}_hdrop.pdf')
    plt.savefig(f'{OUT}/sweep_{param_name}_hdrop.png', dpi=200)
    plt.close()
    print(f"  → sweep_{param_name}_hdrop.pdf ✓")

    # ── Plot 4: C_dis ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in ALGOS:
        y = [all_results[v].get(algo, {}).get('C_dis', np.nan) for v in values]
        if is_numeric:
            ax.plot(values, y, '-', label=LABELS[algo], color=COLORS[algo],
                    marker=MARKERS[algo], linewidth=2, markersize=7)
        else:
            ax.plot(xpos, y, '-', label=LABELS[algo], color=COLORS[algo],
                    marker=MARKERS[algo], linewidth=2, markersize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('$\\hat{C}_{dis}$', fontsize=12)
    ax.set_title(f'Distance Cost vs {xlabel}', fontsize=13)
    ax.legend(ncol=3, loc='best')
    ax.grid(alpha=0.3)
    if is_numeric: ax.set_xticks(values)
    else: ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT}/sweep_{param_name}_cdis.pdf')
    plt.savefig(f'{OUT}/sweep_{param_name}_cdis.png', dpi=200)
    plt.close()
    print(f"  → sweep_{param_name}_cdis.pdf ✓")

    # ── Plot 5: C_energy ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in ALGOS:
        y = [all_results[v].get(algo, {}).get('C_energy', np.nan) for v in values]
        if is_numeric:
            ax.plot(values, y, '-', label=LABELS[algo], color=COLORS[algo],
                    marker=MARKERS[algo], linewidth=2, markersize=7)
        else:
            ax.plot(xpos, y, '-', label=LABELS[algo], color=COLORS[algo],
                    marker=MARKERS[algo], linewidth=2, markersize=7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('$\\hat{C}_e$', fontsize=12)
    ax.set_title(f'Energy Cost vs {xlabel}', fontsize=13)
    ax.legend(ncol=3, loc='best')
    ax.grid(alpha=0.3)
    if is_numeric: ax.set_xticks(values)
    else: ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{OUT}/sweep_{param_name}_cenergy.pdf')
    plt.savefig(f'{OUT}/sweep_{param_name}_cenergy.png', dpi=200)
    plt.close()
    print(f"  → sweep_{param_name}_cenergy.pdf ✓")

    # ── Save CSV ──
    with open(f'output/sweep_{param_name}.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow([param_name, 'Algorithm', 'C_drop', 'C_dis', 'C_energy',
                     'C_total', 'H_done', 'H_drop'])
        for val in values:
            for algo in ALGOS:
                if algo in all_results[val]:
                    d = all_results[val][algo]
                    w.writerow([val, algo, f"{d['C_drop']:.6f}", f"{d['C_dis']:.6f}",
                               f"{d['C_energy']:.6f}", f"{d['C_total']:.6f}",
                               d.get('H_done',''), d.get('H_drop','')])
    print(f"  → output/sweep_{param_name}.csv ✓")


def generate_all_sweeps(sweep_list=None):
    print(f"\n{'='*60}")
    print(f"  PART 2: Parameter Sweeps")
    print(f"{'='*60}")

    if sweep_list is None:
        sweep_list = list(SWEEPS.keys())

    for param in sweep_list:
        if param not in SWEEPS:
            print(f"  ⚠ Unknown sweep: '{param}' — skipping")
            continue
        do_sweep(param)


# ═════════════════════════════════════════════
#  PART 3: PSI WEIGHT ANALYSIS
#  (heatmap, winner chart, robustness)
# ═════════════════════════════════════════════
PSI_COMBOS = [
    (0.80, 0.10, 0.10, "Safety-first\n(0.80,0.10,0.10)"),
    (0.70, 0.10, 0.20, "Safety-heavy\n(0.70,0.10,0.20)"),
    (DEF['psi_drop'], DEF['psi_dis'], DEF['psi_e'],
     f"Default\n({DEF['psi_drop']},{DEF['psi_dis']},{DEF['psi_e']})"),
    (0.40, 0.20, 0.40, "Balanced\n(0.40,0.20,0.40)"),
    (0.33, 0.33, 0.34, "Equal\n(0.33,0.33,0.34)"),
    (0.20, 0.20, 0.60, "Energy-first\n(0.20,0.20,0.60)"),
    (0.20, 0.40, 0.40, "Distance-heavy\n(0.20,0.40,0.40)"),
    (0.10, 0.10, 0.80, "Energy-only\n(0.10,0.10,0.80)"),
]

def generate_psi_analysis(datafile=DEFAULT_DATA):
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    print(f"\n{'='*60}")
    print(f"  PART 3: ψ Weight Sensitivity Analysis")
    print(f"{'='*60}\n")

    # Run all algorithms once to get raw costs
    print("Running algorithms for ψ analysis...")
    results = run_all_algos(datafile)
    if not results:
        print("  ⚠ No results — skipping ψ analysis")
        return

    # ── Fig A: C_total across all ψ combos (grouped bar) ──
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(PSI_COMBOS))
    width = 0.11
    for i, algo in enumerate(ALGOS):
        if algo not in results: continue
        d = results[algo]
        vals = [wd*d['C_drop'] + wi*d['C_dis'] + we*d['C_energy']
                for wd, wi, we, _ in PSI_COMBOS]
        offset = (i - len(ALGOS)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=LABELS[algo], color=COLORS[algo],
               edgecolor='black', linewidth=0.4)
    ax.set_xlabel('Weight Combination ($\\psi_{drop}$, $\\psi_{dis}$, $\\psi_e$)')
    ax.set_ylabel('$\\hat{C}_{total}$')
    ax.set_title('Sensitivity to Weight Selection')
    ax.set_xticks(x)
    ax.set_xticklabels([l for _,_,_,l in PSI_COMBOS], fontsize=8)
    ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=True)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_psi_sweep.pdf')
    plt.savefig(f'{OUT}/fig_psi_sweep.png', dpi=200)
    plt.close()
    print(f"  → fig_psi_sweep.pdf ✓  (grouped bar: C_total × 8 weight combos)")

    # ── Fig B: Winner at each ψ combo ──
    fig, ax = plt.subplots(figsize=(12, 4.5))
    winner_data = []
    for wd, wi, we, label in PSI_COMBOS:
        best_algo, best_cost = None, 1e9
        for algo in ALGOS:
            if algo not in results: continue
            d = results[algo]
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
    ax.set_title('Winning Algorithm per Weight Combination')
    ax.set_xticks(x)
    ax.set_xticklabels([w[0] for w in winner_data], fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_psi_winner.pdf')
    plt.savefig(f'{OUT}/fig_psi_winner.png', dpi=200)
    plt.close()
    print(f"  → fig_psi_winner.pdf ✓  (best algorithm per weight combo)")

    # ── Fig C: ψ_drop × ψ_e heatmap ──
    print("  Computing heatmap (ψ_drop × ψ_e grid)...")
    resolution = 40
    wd_range = np.linspace(0.05, 0.90, resolution)
    we_range = np.linspace(0.05, 0.90, resolution)
    algo_to_id = {a: i for i, a in enumerate(ALGOS)}

    win_grid = np.full((resolution, resolution), np.nan)
    proposed_grid = np.full((resolution, resolution), np.nan)

    for i, wd in enumerate(wd_range):
        for j, we in enumerate(we_range):
            wi = 1.0 - wd - we
            if wi < 0.01: continue
            best_algo, best_cost = None, 1e9
            for algo in ALGOS:
                if algo not in results: continue
                d = results[algo]
                ct = wd*d['C_drop'] + wi*d['C_dis'] + we*d['C_energy']
                if ct < best_cost: best_cost = ct; best_algo = algo
                if algo == 'Proposed':
                    proposed_grid[j, i] = ct
            win_grid[j, i] = algo_to_id.get(best_algo, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Winner map
    ax = axes[0]
    cmap = ListedColormap([COLORS[a] for a in ALGOS])
    ax.imshow(win_grid, origin='lower', aspect='auto', cmap=cmap,
              vmin=0, vmax=len(ALGOS)-1,
              extent=[wd_range[0], wd_range[-1], we_range[0], we_range[-1]])
    ax.set_xlabel('$\\psi_{drop}$', fontsize=12)
    ax.set_ylabel('$\\psi_e$', fontsize=12)
    ax.set_title('Winning Algorithm\n($\\psi_{dis}$ = 1 − $\\psi_{drop}$ − $\\psi_e$)')
    patches = [mpatches.Patch(color=COLORS[a], label=LABELS[a]) for a in ALGOS]
    ax.legend(handles=patches, loc='upper right', fontsize=7)

    # Panel 2: Proposed C_total heatmap
    ax = axes[1]
    im2 = ax.imshow(proposed_grid, origin='lower', aspect='auto', cmap='RdYlGn_r',
                    extent=[wd_range[0], wd_range[-1], we_range[0], we_range[-1]])
    ax.set_xlabel('$\\psi_{drop}$', fontsize=12)
    ax.set_ylabel('$\\psi_e$', fontsize=12)
    ax.set_title('Proposed $\\hat{C}_{total}$\n($\\psi_{dis}$ = 1 − $\\psi_{drop}$ − $\\psi_e$)')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    plt.suptitle('Weight Sensitivity Heatmap', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_psi_heatmap.pdf')
    plt.savefig(f'{OUT}/fig_psi_heatmap.png', dpi=200)
    plt.close()
    print(f"  → fig_psi_heatmap.pdf ✓  (2-panel: winner map + Proposed cost)")

    # ── Fig D: Robustness (win rate + ranking) ──
    resolution = 50
    wd_range = np.linspace(0.05, 0.90, resolution)
    we_range = np.linspace(0.05, 0.90, resolution)
    rank_counts = {1: 0, 2: 0, 3: 0}
    total_combos = 0
    win_count = {a: 0 for a in ALGOS}

    for wd in wd_range:
        for we in we_range:
            wi = 1.0 - wd - we
            if wi < 0.01: continue
            total_combos += 1
            costs = {}
            for algo in ALGOS:
                if algo not in results: continue
                d = results[algo]
                costs[algo] = wd*d['C_drop'] + wi*d['C_dis'] + we*d['C_energy']
            ranked = sorted(costs.items(), key=lambda x: x[1])
            win_count[ranked[0][0]] += 1
            for rank, (algo, _) in enumerate(ranked, 1):
                if algo == 'Proposed':
                    if rank <= 3: rank_counts[rank] += 1
                    break

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Win count
    ax = axes[0]
    algos_sorted = sorted(win_count.items(), key=lambda x: -x[1])
    names = [LABELS[a] for a, _ in algos_sorted]
    counts = [c for _, c in algos_sorted]
    colors_sorted = [COLORS[a] for a, _ in algos_sorted]
    ax.bar(range(len(names)), [c/total_combos*100 for c in counts], 0.6,
           color=colors_sorted, edgecolor='black', linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c/total_combos*100 + 1, f'{c/total_combos*100:.1f}%',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_title(f'Algorithm Win Rate\n({total_combos} ψ combinations)')
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
    ax.set_ylabel('Fraction of ψ combinations (%)', fontsize=11)
    ax.set_title(f'Proposed Ranking Distribution\n({total_combos} weight combos)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels_r)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Robustness to Weight Selection', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/fig_psi_robustness.pdf')
    plt.savefig(f'{OUT}/fig_psi_robustness.png', dpi=200)
    plt.close()
    print(f"  → fig_psi_robustness.pdf ✓  (win rate + Proposed ranking)")


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()

    args = sys.argv[1:]

    if not args or args[0] == 'all':
        generate_default_graphs()
        generate_psi_analysis()
        generate_all_sweeps()
    elif args[0] == 'default':
        generate_default_graphs()
    elif args[0] == 'psi':
        generate_psi_analysis()
    elif args[0] == 'sweep':
        if len(args) > 1:
            generate_all_sweeps(args[1:])
        else:
            generate_all_sweeps()
    elif args[0] == 'list':
        print("Available commands:")
        print(f"  python3 {sys.argv[0]}              # run everything")
        print(f"  python3 {sys.argv[0]} default      # default-data graphs + box plots")
        print(f"  python3 {sys.argv[0]} psi          # ψ weight heatmap + robustness")
        print(f"  python3 {sys.argv[0]} sweep        # all parameter sweeps")
        print(f"  python3 {sys.argv[0]} sweep d_hop  # single sweep")
        print(f"\nAvailable sweeps:")
        for k, v in SWEEPS.items():
            print(f"  {k:12s}  {v['xlabel']:40s}  {v['values']}")
        sys.exit(0)
    else:
        # Try as sweep name directly
        if args[0] in SWEEPS:
            generate_all_sweeps(args)
        else:
            print(f"Unknown: '{args[0]}'. Use 'list' for options.")
            sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done! {elapsed:.0f}s elapsed")
    print(f"  All graphs in: {OUT}/")
    print(f"  All CSVs in:   output/")
    print(f"{'='*60}")
