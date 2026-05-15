"""Shared config reader for all Python scripts. Reads config.txt."""
import os

def read_config(path='config.txt'):
    """Read config.txt into a dict. Returns defaults if file missing."""
    cfg = {}
    if not os.path.exists(path):
        return cfg
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if '=' not in line: continue
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            if k and v:
                # Auto-convert to int/float/string
                try: cfg[k] = int(v)
                except ValueError:
                    try: cfg[k] = float(v)
                    except ValueError: cfg[k] = v
    return cfg

def get_config(path='config.txt'):
    """Read config and return with defaults filled in."""
    c = read_config(path)
    D = {
        'NUM_AV': 80, 'NUM_BS': 10, 'SIM_MS': 10000, 'GRID_SIZE': 200,
        'MAX_UTIL': 0.85, 'DIS_HOP': 10, 'BATCH': 15,
        'P_IDLE': 150, 'P_MAX': 400,
        'PSI_DROP': 0.55, 'PSI_DIS': 0.15, 'PSI_E': 0.30,
        'R_HARD': 0.9, 'R_SOFT': 0.1, 'DELTA': 0.9,
        'HARD_SOFT_RATIO': 29, 'HARD_DEADLINE': 150, 'SOFT_DEADLINE': 3000,
        'HOTSPOT_PROB': 0.50, 'HOTSPOT_RADIUS': 30, 'HOTSPOT_CLUSTER': 0.30,
    }
    for k, v in D.items():
        if k not in c: c[k] = v
    return c

if __name__ == '__main__':
    c = get_config()
    for k in sorted(c): print(f"  {k:20s} = {c[k]}")
