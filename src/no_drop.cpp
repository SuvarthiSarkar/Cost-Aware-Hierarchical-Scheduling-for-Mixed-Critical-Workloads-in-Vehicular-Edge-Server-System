// NO_DROP: Same as Proposed (local/global mode, U_crit packing, U_max, D_max)
// but NO MC-Drop. Tests the contribution of mixed-criticality dropping.
// Usage: ./no_drop <input> [d_hop] [max_util] [--detailed] [--stats]
#include "common.h"

static int local_count = 0, global_count = 0;

double compute_expected_util_nd(const vector<Task> &tasks) {
    double sum = 0; int count = 0;
    for (auto &t : tasks) {
        int slack = t.deadline - t.arrival_time - 1;
        if (slack > 0) { sum += (double)t.process_time / (double)slack; count++; }
    }
    return (count > 0) ? sum / count : 0.1;
}

// Packing: highest-loaded BS, per-BS U_crit
int bestmatch_nd(int cur, Task &t, vector<vector<double>> &u) {
    double ut = (double)t.process_time / (double)(t.deadline - cur - 1);
    int st = cur + 1, et = t.deadline, m = (int)u.size() - 1;

    // Pass 1: critical frequency for HC tasks with slack
    if (t.flag == 0) {
        double cl = 0; int pr = -1;
        for (int j = 1; j <= m; j++) {
            if (!d_hop_check(t.x, t.y, j, DIS_HOP)) continue;
            double uc_j = bs_power[j].u_crit;
            if ((ut - uc_j) > 0.000001) continue;
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
        // D_max: all tasks only consider BSs within d_hop range
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
    if (pr != -1) { t.start_time = st; t.end_time = et; t.utilisation = ut; return pr; }
    return -1;
}

// Local Mode: same as proposed
int try_local_nd(int cur, Task &t, vector<vector<double>> &u) {
    int nearest = find_nearest_bs(t.x, t.y);
    double uc_near = bs_power[nearest].u_crit;
    double ut_near = bs_power[nearest].u_thresh;

    int st = cur + 1, et = t.deadline;
    double peak = 0;
    for (int i = st; i <= et && i < (int)u[nearest].size(); i++)
        peak = max(peak, u[nearest][i]);

    if (peak >= ut_near) return -1;

    double ut = (double)t.process_time / (double)(t.deadline - cur - 1);

    if (t.flag == 0 && (ut - uc_near) <= -0.000001) {
        int pt = (int)ceil((double)t.process_time / uc_near);
        int ee = st + pt;
        bool ok = true;
        for (int i = st; i <= ee && ok; i++) {
            if (i >= (int)u[nearest].size()) { ok = false; break; }
            if (u[nearest][i] + uc_near > bs_cap(nearest) + 0.000001) ok = false;
        }
        if (ok) { t.start_time = st; t.end_time = ee; t.utilisation = uc_near; return nearest; }
    }

    bool ok = true;
    for (int i = st; i <= et && ok; i++) {
        if (i >= (int)u[nearest].size()) { ok = false; break; }
        if (u[nearest][i] + ut > bs_cap(nearest) + 0.000001) ok = false;
    }
    if (ok) { t.start_time = st; t.end_time = et; t.utilisation = ut; return nearest; }
    return -1;
}

void run(vector<Task> &tasks, int m) {
    int t = 0, i = 0; vector<Task> in;
    vector<vector<double>> u(m + 1, vector<double>(MX + 5, 0.0));
    vector<vector<Task>> p(m + 1); CostResult c(m); c.set_task_counts(tasks);

    while (t <= MX) {
        sim_collect(t, i, tasks, in); sim_complete(t, m, p, c);
        if (t % BATCH == 0 && !in.empty()) {
            sort(in.begin(), in.end(), comp_priority); vector<Task> ns;
            for (auto &x : in) {
                if (x.process_time + t + 1 > x.deadline) {
                    if (x.flag == 0) c.drophard++; else c.dropsoft++; continue;
                }
                x.utilisation = (double)x.process_time / (double)(x.deadline - t - 1);

                // Step 1: Local Mode
                int local_bs = try_local_nd(t, x, u);
                if (local_bs != -1) {
                    if (!d_hop_check(x.x, x.y, local_bs, DIS_HOP)) {
                        // distance fail → escalate to global
                    } else {
                        assign_task(x, local_bs, u, p);
                        c.add_distance(x, local_bs);
                        local_count++; continue;
                    }
                }

                // Step 2: Global Mode — Packing (NO MC-Drop)
                int pr = bestmatch_nd(t, x, u);
                if (pr != -1) {
                    assign_task(x, pr, u, p);
                    c.add_distance(x, pr);
                    global_count++;
                } else {
                    // NO DROP: all failed tasks deferred
                    ns.push_back(x);
                }
            }
            in = ns;
        }
        t++; sim_energy(t, m, u, c);
    }
    for (auto &x : in) { if (x.flag == 0) c.drophard++; else c.dropsoft++; }

    if (STATS_OUTPUT) {
        int total = local_count + global_count;
        cerr << "Scheduling: Local=" << local_count
             << "  Global=" << global_count
             << "  (Local%=" << fixed << setprecision(1)
             << (total > 0 ? 100.0 * local_count / total : 0) << "%)" << endl;
    }
    print_stats_if_needed("NO_DROP", c);
}

int main(int argc, char *argv[]) {
    if (argc < 2) { cerr << "Usage: ./no_drop <input> [d_hop] [--server-cfg F] [--detailed] [--stats]\n"; return 1; }
    check_detailed_flag(argc, argv);
    if (argc >= 3 && string(argv[2]).substr(0,2) != "--") DIS_HOP = stoi(argv[2]);

    vector<Task> tasks;
    int m = read_input(argv[1], tasks);  // also calls init_bs_power(m)
    double eu = compute_expected_util_nd(tasks);
    update_bs_thresholds(eu);

    run(tasks, m);
    return 0;
}
