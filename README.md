# FFE-VESS: Fast, Fair, and Efficient Scheduling for Vehicular Edge Server Systems

## Build
```bash
make          # builds all binaries into bin/
make clean    # removes bin/
```

## Generate Data
```bash
bin/gen_data <num_av> <num_bs> <sim_duration_ms> <output_file>
# Example:
bin/gen_data 80 10 10000 data/80av_10bs.txt
```

## Run Individual Algorithms
```bash
# Proposed (with Local/Global mode, threshold U_T)
bin/proposed <data_file> [d_hop] [max_util] [--pidle W] [--pmax W] [--detailed] [--stats]
bin/proposed data/80av_10bs.txt 10 0.85 --detailed --stats

# Baselines (no d_hop/max_util params)
bin/md data/80av_10bs.txt --detailed
bin/lb data/80av_10bs.txt --detailed
bin/uc data/80av_10bs.txt --detailed
bin/local_only data/80av_10bs.txt --detailed
bin/nearest_only data/80av_10bs.txt --detailed
bin/aga data/80av_10bs.txt --detailed

# No-drop variant (same params as proposed)
bin/no_drop data/80av_10bs.txt 10 0.85 --detailed
```

## Output Format
CSV: `Name,C_drop,C_dis,C_energy[,H_done,S_done,H_drop,S_drop,AvgUtil]`

## Normalization
- `C_drop`: drops / worst-case (all dropped), weighted by R_h=0.9, R_l=0.1 → [0, 1]
- `C_dis`: total distance / (N × diagonal of grid) → [0, 1]
- `C_energy`: total energy / (M × T × P_max_normalized) → [P_idle/P_max, 1]

## Key Parameters
- `U_crit = (P_idle / 2α)^(1/3)` — energy-optimal BS utilization
- `U_T = U_crit - E[u_task]` — threshold between Local and Global mode
- `MAX_UTIL` (default 0.85), `DIS_HOP` (default 10)
- `P_IDLE` (default 150W), `P_MAX` (default 400W)

## Algorithms
| Binary | Paper Name | Description |
|--------|-----------|-------------|
| proposed | FFE-VESS | Local/Global + packing + MC-Drop |
| md | MD | Min-distance first |
| lb | LB | Load balancing |
| uc | Uc | Urgency-critical |
| no_drop | NO_DROP | Proposed without MC-Drop |
| local_only | LOCAL | Always nearest BS |
| nearest_only | NEAREST | Nearest with deadline check |
| aga | AGA | Adaptive genetic algorithm |

## Run All Experiments
```bash
bash run_all_experiments.sh
python3 plot_all.py
```
