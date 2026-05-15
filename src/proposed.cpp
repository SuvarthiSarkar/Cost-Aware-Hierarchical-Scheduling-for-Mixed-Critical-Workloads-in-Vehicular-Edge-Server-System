// Proposed Approach: Two-tier Local/Global scheduling (HETEROGENEOUS)
// Each BS has its own P_idle, P_max → own U_crit, U_T
// Step 1: Local Mode — nearest BS handles task if U_j(t) < U_T_j
// Step 2: Global Mode — CS packing heuristic (highest-loaded-first) + MC-Drop
// Usage: ./proposed <input> [d_hop] [max_util] [--pidle W] [--pmax W] [--detailed] [--stats]
#include "common.h"

static int local_count = 0, global_count = 0;

// Compute E[u_task] from the task set
double compute_expected_util(const vector<Task> &tasks) {
    double sum = 0; int count = 0;
    for (auto &t : tasks) {
        int slack = t.deadline - t.arrival_time - 1;
        if (slack > 0) {
            sum += (double)t.process_time / (double)slack;
            count++;
        }
    }
    return (count > 0) ? sum / count : 0.1;
}

// ════════════════════════════════════════════════════════
// Packing Heuristic: highest-loaded BS first, distance tie-break
// Uses per-BS U_crit_j for speed assignment
// ════════════════════════════════════════════════════════
int bestmatch(int cur, Task &t, vector<vector<double>> &u) {
    double ut = (double)t.process_time / (double)(t.deadline - cur - 1);
    int st = cur + 1, et = t.deadline, m = (int)u.size() - 1;

    // Pass 1: critical frequency for HC tasks with slack
    // Try each BS at ITS OWN U_crit_j
    if (t.flag == 0) {
        double cl = 0; int pr = -1;
        for (int j = 1; j <= m; j++) {
            // D_max: prefer BSs within d_hop range
            if (!d_hop_check(t.x, t.y, j, DIS_HOP)) continue;
            double uc_j = bs_power[j].u_crit;
            if ((ut - uc_j) > 0.000001) continue;  // task needs more than u_crit on this BS

            int pt = (int)ceil((double)t.process_time / uc_j);
            int ee = st + pt;
            bool ok = true; double ma = 0;
            for (int i = st; i <= ee; i++) {
                if (i >= (int)u[j].size()) { ok = false; break; }
                if ((u[j][i] + uc_j) > bs_cap(j) + 0.000001) { ok = false; break; }
                ma = max(ma, u[j][i]);
            }
            if (ok) {
                if ((cl - ma) <= -0.000001) { pr = j; cl = ma; }
                else if (pr == -1) { pr = j; cl = ma; }
                if (fabs(cl - ma) <= 0.0000001 && pr != -1)
                    if (eu_calc_dis(t.x, t.y, j) < eu_calc_dis(t.x, t.y, pr)) pr = j;
            }
        }
        if (pr != -1) {
            double uc_pr = bs_power[pr].u_crit;
            int pt = (int)ceil((double)t.process_time / uc_pr);
            t.start_time = st; t.end_time = st + pt; t.utilisation = uc_pr;
            return pr;
        }
    }

    // Pass 2: minimum frequency
    double cl = 0; int pr = -1;
    for (int j = 1; j <= m; j++) {
        // D_max: all tasks prefer BSs within d_hop range
        if (!d_hop_check(t.x, t.y, j, DIS_HOP)) continue;
        double val = bs_cap(j) - ut;
        bool ok = true; double ma = 0;
        for (int i = st; i <= et; i++) {
            if (i >= (int)u[j].size()) { ok = false; break; }
            if ((u[j][i] - val) <= -0.000001) ma = max(ma, u[j][i]);
            else { ok = false; break; }
        }
        if (ok) {
            if ((cl - ma) <= -0.000001) { pr = j; cl = ma; }
            else if (pr == -1) { pr = j; cl = ma; }
            if (fabs(cl - ma) <= 0.0000001 && pr != -1)
                if (eu_calc_dis(t.x, t.y, j) < eu_calc_dis(t.x, t.y, pr)) pr = j;
        }
    }
    if (pr != -1) {
        t.start_time = st; t.end_time = et; t.utilisation = ut;
        return pr;
    }
    return -1;
}

// ════════════════════════════════════════════════════════
// Local Mode: try nearest BS if below ITS threshold U_T_j
// ════════════════════════════════════════════════════════
int try_local(int cur, Task &t, vector<vector<double>> &u) {
    int nearest = find_nearest_bs(t.x, t.y);
    double uc_near = bs_power[nearest].u_crit;
    double ut_near = bs_power[nearest].u_thresh;

    // Threshold check: is nearest BS below ITS threshold?
    int st = cur + 1, et = t.deadline;
    double peak = 0;
    for (int i = st; i <= et && i < (int)u[nearest].size(); i++)
        peak = max(peak, u[nearest][i]);

    if (peak >= ut_near) return -1;  // BS is loaded → escalate

    double ut = (double)t.process_time / (double)(t.deadline - cur - 1);

    // HC with slack: try this BS's critical frequency
    if (t.flag == 0 && (ut - uc_near) <= -0.000001) {
        int pt = (int)ceil((double)t.process_time / uc_near);
        int ee = st + pt;
        bool ok = true;
        for (int i = st; i <= ee && ok; i++) {
            if (i >= (int)u[nearest].size()) { ok = false; break; }
            if (u[nearest][i] + uc_near > bs_cap(nearest) + 0.000001) ok = false;
        }
        if (ok) {
            t.start_time = st; t.end_time = ee; t.utilisation = uc_near;
            return nearest;
        }
    }

    // Fallback: minimum frequency
    bool ok = true;
    for (int i = st; i <= et && ok; i++) {
        if (i >= (int)u[nearest].size()) { ok = false; break; }
        if (u[nearest][i] + ut > bs_cap(nearest) + 0.000001) ok = false;
    }
    if (ok) {
        t.start_time = st; t.end_time = et; t.utilisation = ut;
        return nearest;
    }

    return -1;
}

// ════════════════════════════════════════════════════════
// Overall Per-Batch Scheduling
// ════════════════════════════════════════════════════════
void run(vector<Task> &tasks, int m) {
    int t = 0, i = 0;
    vector<Task> in;
    vector<vector<double>> u(m + 1, vector<double>(MX + 5, 0.0));
    vector<vector<Task>> p(m + 1);
    CostResult c(m);
    c.set_task_counts(tasks);

    while (t <= MX) {
        sim_collect(t, i, tasks, in);
        sim_complete(t, m, p, c);

        if (t % BATCH == 0 && !in.empty()) {
            sort(in.begin(), in.end(), comp_priority);
            vector<Task> ns;

            for (auto &x : in) {
                if (x.process_time + t + 1 > x.deadline) {
                    if (x.flag == 0) c.drophard++; else c.dropsoft++;
                    continue;
                }
                x.utilisation = (double)x.process_time / (double)(x.deadline - t - 1);

                // Step 1: Local Mode (per-BS threshold)
                int local_bs = try_local(t, x, u);
                if (local_bs != -1) {
                    if (!d_hop_check(x.x, x.y, local_bs, DIS_HOP)) {
                        // distance fail → escalate to global
                    } else {
                        assign_task(x, local_bs, u, p);
                        c.add_distance(x, local_bs);
                        local_count++;
                        continue;
                    }
                }

                // Step 2: Global Mode — Packing (per-BS U_crit)
                // bestmatch already filters BSs by d_hop for ALL tasks
                int pr = bestmatch(t, x, u);
                if (pr != -1) {
                    assign_task(x, pr, u, p);
                    c.add_distance(x, pr);
                    global_count++;
                } else {
                    // Step 3: MC-Drop (HC only)
                    if (x.flag == 0) {
                        set<int> dr;
                        pr = bestprocdrop(p, x, u, t, dr);
                        if (pr == -1) ns.push_back(x);
                        else { execute_drop(pr, x, t, p, u, dr, c); global_count++; }
                    } else {
                        ns.push_back(x);
                    }
                }
            }
            in = ns;
        }
        t++;
        sim_energy(t, m, u, c);
    }

    for (auto &x : in) {
        if (x.flag == 0) c.drophard++; else c.dropsoft++;
    }

    if (STATS_OUTPUT) {
        int total = local_count + global_count;
        cerr << "Scheduling: Local=" << local_count
             << "  Global=" << global_count
             << "  (Local%=" << fixed << setprecision(1)
             << (total > 0 ? 100.0 * local_count / total : 0) << "%)" << endl;
        cerr << "Per-BS power profiles:" << endl;
        for (int j = 1; j <= m; j++) {
            auto &bp = bs_power[j];
            cerr << "  BS" << j << ": P_idle=" << fixed << setprecision(1) << bp.p_idle
                 << "W P_max=" << bp.p_max << "W alpha=" << bp.alpha
                 << " U_crit=" << setprecision(4) << bp.u_crit
                 << " U_T=" << bp.u_thresh
                 << " U_max=" << bp.u_max << endl;
        }
    }
    print_stats_if_needed("Proposed", c);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./proposed <input> [d_hop] "
             << "[--server-cfg F] [--pidle W] [--pmax W] [--detailed] [--stats]\n";
        return 1;
    }
    check_detailed_flag(argc, argv);
    if (argc >= 3 && string(argv[2]).substr(0,2) != "--")
        DIS_HOP = stoi(argv[2]);

    vector<Task> tasks;
    int m = read_input(argv[1], tasks);  // also calls init_bs_power(m)

    // Compute E[u_task] and set per-BS thresholds
    double eu = compute_expected_util(tasks);
    update_bs_thresholds(eu);

    if (STATS_OUTPUT) {
        cerr << "E[u_task]=" << fixed << setprecision(4) << eu << endl;
    }

    run(tasks, m);
    return 0;
}
