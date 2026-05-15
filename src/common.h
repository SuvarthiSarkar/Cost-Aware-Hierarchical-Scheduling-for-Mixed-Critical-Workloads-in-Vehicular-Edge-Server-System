#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <random>
using namespace std;

// ============================================================
// System model based on real YOLO inference experiments
//
// TIME UNIT: 1 millisecond
//
// HARD TASKS (YOLO object detection):
//   Generation: every 15ms per AV, probability 0.21
//   Deadline: 150ms from generation
//   Compute: ~30-50ms on one GPU slot
//   Each BS GPU: 20 concurrent slots
//
// SOFT TASKS (nav/infotainment):
//   Generation: every 1000-5000ms per AV
//   Deadline: 3000ms from generation
//   Compute: ~100-300ms on CPU
//
// COMMUNICATION (from real 5G/LTE measurements):
//   AV ↔ BS (wireless): RTT 10-15ms Gaussian(12.5, 1.25)
//   BS ↔ BS (wired via CS): RTT 15-18ms Gaussian(16.5, 0.75)
//   Scheduling overhead: ~1-2ms
//
// TIMING BREAKDOWN (hard task, local):
//   AV→BS: ~6ms | sched: ~1ms | compute: ~35ms | BS→AV: ~6ms
//   Total: ~48ms out of 150ms budget
//
// TIMING BREAKDOWN (hard task, forwarded):
//   AV→BS1: ~6ms | sched: ~2ms | BS1→BS2: ~8ms | compute: ~35ms
//   BS2→BS1: ~8ms | BS1→AV: ~6ms
//   Total: ~65ms out of 150ms budget
// ============================================================

// Global simulation parameters (all in milliseconds)
[[maybe_unused]] static int    MX       = 10000;  // 10 seconds sim
[[maybe_unused]] static int    BATCH    = 15;     // match hard task interval
[[maybe_unused]] static double MAX_UTIL = 0.85;    // Ubmax
[[maybe_unused]] static int    DIS_HOP  = 10;     // Dbmax rank
[[maybe_unused]] static int    GRID_SIZE= 200;    // city grid dimension
[[maybe_unused]] static double R_HARD   = 0.9;    // importance weight for HC tasks
[[maybe_unused]] static double R_SOFT   = 0.1;    // importance weight for LC tasks
[[maybe_unused]] static double DELTA    = 0.0;    // drop penalty amplifier: cost = (1+δ) × base_drop
[[maybe_unused]] static double PSI_DROP = 0.55;   // cost weight: task drops
[[maybe_unused]] static double PSI_DIS  = 0.15;   // cost weight: distance
[[maybe_unused]] static double PSI_E    = 0.30;   // cost weight: energy
[[maybe_unused]] static bool DETAILED_OUTPUT = false;
[[maybe_unused]] static bool STATS_OUTPUT = false;
[[maybe_unused]] static string config_path = "config.txt";

// ============================================================
// Power model — based on real edge server measurements
//
// P(u) = P_idle + (P_max - P_idle) * u^3
//
// Real-life BS/Edge server power profiles:
//   NVIDIA Jetson AGX:     P_idle=15W,  P_max=30W   (ratio=0.50)
//   NVIDIA T4 GPU server:  P_idle=70W,  P_max=300W  (ratio=0.23)
//   Intel Xeon edge:       P_idle=150W, P_max=400W  (ratio=0.38)
//   Dell PowerEdge rack:   P_idle=200W, P_max=500W  (ratio=0.40)
//
// Default: P_idle=150W, P_max=400W (typical edge server with GPU)
// The key parameter is r = P_idle/P_max (idle-to-peak ratio)
// Higher r → more static waste → packing matters more
// Lower r  → dynamic dominates → spreading is cheaper
// ============================================================
[[maybe_unused]] static double P_IDLE = 150.0;    // Watts (default, overridden per-BS)
[[maybe_unused]] static double P_MAX  = 400.0;    // Watts (default, overridden per-BS)

// ============================================================
// Read config.txt key=value pairs and set ALL globals
// Priority: hardcoded defaults → config.txt → CLI flags
// ============================================================
inline void load_config(const string &path = "config.txt") {
    ifstream f(path);
    if (!f.is_open()) return;  // use compiled defaults
    string line;
    while (getline(f, line)) {
        auto pos = line.find('#');
        if (pos != string::npos) line = line.substr(0, pos);
        pos = line.find('=');
        if (pos == string::npos) continue;
        string key = line.substr(0, pos);
        string val = line.substr(pos + 1);
        auto trim = [](string &s) {
            while (!s.empty() && isspace(s.front())) s.erase(s.begin());
            while (!s.empty() && isspace(s.back())) s.pop_back();
        };
        trim(key); trim(val);
        if (key.empty() || val.empty()) continue;
        if      (key=="SIM_MS")     MX        = stoi(val);
        else if (key=="BATCH")      BATCH     = stoi(val);
        else if (key=="MAX_UTIL")   MAX_UTIL  = stod(val);
        else if (key=="DIS_HOP")    DIS_HOP   = stoi(val);
        else if (key=="GRID_SIZE")  GRID_SIZE = stoi(val);
        else if (key=="R_HARD")     R_HARD    = stod(val);
        else if (key=="R_SOFT")     R_SOFT    = stod(val);
        else if (key=="P_IDLE")     P_IDLE    = stod(val);
        else if (key=="P_MAX")      P_MAX     = stod(val);
        else if (key=="PSI_DROP")   PSI_DROP  = stod(val);
        else if (key=="PSI_DIS")    PSI_DIS   = stod(val);
        else if (key=="PSI_E")      PSI_E     = stod(val);
        else if (key=="DELTA")      DELTA     = stod(val);
    }
}

// ============================================================
// Per-BS power model — heterogeneous servers
// Each BS can have different P_idle, P_max, alpha, U_crit, U_T, U_max
// ============================================================
struct BSPower {
    double p_idle, p_max, alpha;
    double u_crit;    // (p_idle / 2*alpha)^(1/3)
    double u_thresh;  // u_crit - E[u_task] (set after reading tasks)
    double u_max;     // per-BS maximum utilization

    BSPower() : p_idle(150), p_max(400), alpha(250), u_crit(0.669), u_thresh(0.57), u_max(0.85) {}
    BSPower(double pi, double pm, double al, double um)
        : p_idle(pi), p_max(pm), alpha(al), u_max(um) {
        u_crit = (alpha > 1.0) ? pow(pi / (2.0 * alpha), 1.0/3.0) : u_max;
        u_crit = min(u_crit, u_max);  // clamp: U_crit cannot exceed U_max
        u_thresh = u_crit;  // updated later with E[u_task]
    }
    void set_threshold(double eu) { u_thresh = max(0.0, u_crit - eu); }
};

// Server config ranges for uniform sampling
struct ServerConfig {
    double p_idle_lo=15, p_idle_hi=200;
    double alpha_lo=50,  alpha_hi=350;
    double umax_lo=0.80, umax_hi=0.95;
    int    seed=12345;
};

static ServerConfig srv_cfg;  // global server config

// Read server.cfg key=value file
inline ServerConfig read_server_cfg(const string &path) {
    ServerConfig sc;
    ifstream f(path);
    if (!f.is_open()) return sc;  // use defaults
    string line;
    while (getline(f, line)) {
        auto pos = line.find('#');
        if (pos != string::npos) line = line.substr(0, pos);
        pos = line.find('=');
        if (pos == string::npos) continue;
        string key = line.substr(0, pos);
        string val = line.substr(pos + 1);
        auto trim = [](string &s) {
            while (!s.empty() && isspace(s.front())) s.erase(s.begin());
            while (!s.empty() && isspace(s.back())) s.pop_back();
        };
        trim(key); trim(val);
        if (key.empty() || val.empty()) continue;
        if      (key=="P_IDLE_LO")   sc.p_idle_lo = stod(val);
        else if (key=="P_IDLE_HI")   sc.p_idle_hi = stod(val);
        else if (key=="ALPHA_LO")    sc.alpha_lo  = stod(val);
        else if (key=="ALPHA_HI")    sc.alpha_hi  = stod(val);
        else if (key=="UMAX_LO")     sc.umax_lo   = stod(val);
        else if (key=="UMAX_HI")     sc.umax_hi   = stod(val);
        else if (key=="SERVER_SEED") sc.seed      = stoi(val);
    }
    return sc;
}

static map<int, BSPower> bs_power;  // per-BS power info
static string server_cfg_path = "server.cfg";  // default path

// Effective BS capacity = MAX_UTIL (operating fraction) × u_max (physical capacity)
inline double bs_cap(int j) {
    return MAX_UTIL * bs_power[j].u_max;
}

// Initialize heterogeneous BS power profiles — uniform sampling from ranges
inline void init_bs_power(int m) {
    srv_cfg = read_server_cfg(server_cfg_path);
    mt19937 rng(srv_cfg.seed);
    uniform_real_distribution<double> d_pidle(srv_cfg.p_idle_lo, srv_cfg.p_idle_hi);
    uniform_real_distribution<double> d_alpha(srv_cfg.alpha_lo,  srv_cfg.alpha_hi);
    uniform_real_distribution<double> d_umax(srv_cfg.umax_lo,    srv_cfg.umax_hi);

    for (int j = 1; j <= m; j++) {
        double pi = d_pidle(rng);
        double al = d_alpha(rng);
        double pm = pi + al;  // P_max = P_idle + alpha (always well separated)
        double um = d_umax(rng);
        bs_power[j] = BSPower(pi, pm, al, um);
    }

    if (STATS_OUTPUT) {
        cerr << "  Server heterogeneity (from " << server_cfg_path << "):\n";
        for (int j = 1; j <= m; j++) {
            auto &b = bs_power[j];
            cerr << "    BS" << j << ": P_idle=" << fixed << setprecision(1) << b.p_idle
                 << "W  P_max=" << b.p_max << "W  alpha=" << b.alpha
                 << "  U_crit=" << setprecision(3) << b.u_crit
                 << "  U_max=" << b.u_max << "\n";
        }
    }
}

// Update all thresholds after E[u_task] is known
inline void update_bs_thresholds(double eu) {
    for (auto &[j, bp] : bs_power) bp.set_threshold(eu);
}

// Derived: per-BS power at utilization u
inline double power_at_util_bs(int j, double u) {
    auto &bp = bs_power[j];
    return bp.p_idle + bp.alpha * u * u * u;
}

// Global convenience (uses default P_IDLE/P_MAX for non-per-BS callers)
inline double power_at_util(double u) {
    return P_IDLE + (P_MAX - P_IDLE) * u * u * u;
}

// Critical utilization using global defaults (for baselines that don't use per-BS)
inline double critical_utilization() {
    double alpha = P_MAX - P_IDLE;
    if (alpha < 0.001) return 1.0;
    return pow(P_IDLE / (2.0 * alpha), 1.0/3.0);
}

// Communication delays (ms) — from real experiments
static const double COMM_AV_BS_RTT_MEAN  = 12.5;  // AV↔BS RTT mean
static const double COMM_AV_BS_RTT_STD   = 1.25;  // AV↔BS RTT std
static const double COMM_BS_BS_RTT_MEAN  = 16.5;  // BS↔BS RTT mean
static const double COMM_BS_BS_RTT_STD   = 0.75;  // BS↔BS RTT std
static const double SCHED_LOCAL_MS       = 1.0;   // local scheduling
static const double SCHED_GLOBAL_MS      = 2.0;   // CS scheduling

// GPU capacity per BS
static const int    GPU_SLOTS            = 20;     // concurrent YOLO

// Task compute parameters
static const double YOLO_COMPUTE_MS      = 35.0;  // avg YOLO inference
static const double SOFT_COMPUTE_MS      = 200.0; // avg soft task compute

static const int    HARD_PROC_TIME       = 12;  // avg of 3-20 range     // at u=1.0
static const int    SOFT_PROC_TIME       = 100;    // at u=1.0

static const int    HARD_DEADLINE_BUDGET = 137;
static const int    SOFT_DEADLINE_BUDGET = 2987;

static map<int, pair<int,int>> base_station;

// ============================================================
// Task structure
// ============================================================
struct Task {
    int    arrival_time, deadline, process_time, tid;
    bool   flag;          // 0=high-critical(YOLO), 1=low-critical(soft)
    int    start_time, end_time;
    double utilisation;
    int    x, y;          // AV location at generation time

    Task() : arrival_time(0), deadline(0), process_time(0), tid(0),
             flag(false), start_time(-1), end_time(-1), utilisation(0.0),
             x(0), y(0) {}
    Task(int a, int d, int pr, int id, bool fl, int px, int py)
        : arrival_time(a), deadline(d), process_time(pr), tid(id),
          flag(fl), start_time(-1), end_time(-1), utilisation(0.0),
          x(px), y(py) {}
};

// Sort: high-critical first (YOLO before soft), then EDF
inline bool comp_priority(Task &a, Task &b) {
    if (a.flag == b.flag) return a.deadline < b.deadline;
    return a.flag < b.flag;
}
inline bool comp_util_desc(Task &a, Task &b) {
    return a.utilisation > b.utilisation;
}

// ============================================================
// Distance
// ============================================================
inline int eu_calc_dis(int x, int y, int id) {
    int dx = x - base_station[id].first;
    int dy = y - base_station[id].second;
    return dx * dx + dy * dy;
}
inline int find_nearest_bs(int x, int y) {
    int best = 1, best_d = eu_calc_dis(x, y, 1);
    for (auto &it : base_station) {
        int d = eu_calc_dis(x, y, it.first);
        if (d < best_d) { best_d = d; best = it.first; }
    }
    return best;
}
inline set<int> find_nearest_bs_set(int x, int y) {
    vector<pair<int,int>> ds;
    for (auto &it : base_station) ds.push_back({eu_calc_dis(x, y, it.first), it.first});
    sort(ds.begin(), ds.end());
    set<int> r;
    for (auto &p : ds) { if (p.first == ds[0].first) r.insert(p.second); else break; }
    return r;
}
inline bool d_hop_check(int x, int y, int bs_id, int d) {
    int m = (int)base_station.size();
    d = max(1, min(d, m));  // clamp to [1, num_BSs]
    vector<pair<int,int>> ck;
    for (auto &it : base_station) {
        int dx2 = x - it.second.first, dy2 = y - it.second.second;
        ck.push_back({dx2*dx2 + dy2*dy2, it.first});
    }
    sort(ck.begin(), ck.end());
    if (d - 1 >= (int)ck.size()) return true;
    return eu_calc_dis(x, y, bs_id) <= ck[d - 1].first;
}

// ============================================================
// Cost tracking
// ============================================================
struct CostResult {
    int    drophard, dropsoft, hard_completed, soft_completed;
    double eusqrt_dis_hard, eusqrt_dis_soft, util_cost;
    double total_util_sum; int total_util_samples;
    int    m;
    int    total_hard_input, total_soft_input;  // total tasks in input

    // Per-BS load tracking: sum of utilization samples per BS
    vector<double> bs_util_sum;
    vector<int>    bs_util_cnt;
    vector<int>    bs_task_cnt;  // tasks assigned to each BS

    // Per-task response time tracking (end_time - arrival_time)
    vector<int> hard_response_times;
    vector<int> soft_response_times;

    // Per-task distance tracking (for box plots)
    vector<double> task_distances;

    CostResult(int m_) : drophard(0), dropsoft(0), hard_completed(0),
        soft_completed(0), eusqrt_dis_hard(0), eusqrt_dis_soft(0),
        util_cost(0.0), total_util_sum(0), total_util_samples(0), m(m_),
        total_hard_input(0), total_soft_input(0),
        bs_util_sum(m_+1, 0.0), bs_util_cnt(m_+1, 0), bs_task_cnt(m_+1, 0) {}

    void set_task_counts(int hard, int soft) { total_hard_input = hard; total_soft_input = soft; }
    void set_task_counts(const vector<Task> &tasks) {
        total_hard_input = total_soft_input = 0;
        for (auto &t : tasks) { if (t.flag == 0) total_hard_input++; else total_soft_input++; }
    }

    int completed() const { return hard_completed + soft_completed; }
    int total_tasks() const { return completed() + drophard + dropsoft; }

    void add_distance(Task &x, int bs) {
        double d = sqrt((double)eu_calc_dis(x.x, x.y, bs));
        if (x.flag == 0) eusqrt_dis_hard += d; else eusqrt_dis_soft += d;
        bs_task_cnt[bs]++;
        task_distances.push_back(d);
    }
    void task_completed(Task &x) {
        int rt = x.end_time - x.arrival_time;
        if (x.flag == 0) { hard_completed++; hard_response_times.push_back(rt); }
        else             { soft_completed++;  soft_response_times.push_back(rt); }
    }

    // ============================================================
    // Normalized costs (each in [0, 1])
    // ============================================================

    // C_drop: penalty for dropped tasks
    //   base = (drophard * R_HARD + dropsoft * R_SOFT) / (total * R)
    //   amplified by (1+δ): missing a task costs (1+δ) × its profit
    //   When all dropped → (1+δ), when none → 0
    double cost_drop() const {
        double max_drop = (double)total_hard_input * R_HARD + (double)total_soft_input * R_SOFT;
        if (max_drop < 0.001) return 0;
        double base = ((double)drophard * R_HARD + (double)dropsoft * R_SOFT) / max_drop;
        return (1.0 + DELTA) * base;
    }

    // C_dis: normalized by max distance = all tasks at diagonal
    //   diagonal = GRID_SIZE * sqrt(2)
    //   max_dis = total_tasks * diagonal
    //   When all tasks at maximum distance → 1.0
    double cost_dis() const {
        int n = total_hard_input + total_soft_input; if (n == 0) return 0;
        double diagonal = (double)GRID_SIZE * sqrt(2.0);
        return (eusqrt_dis_hard + eusqrt_dis_soft) / ((double)n * diagonal);
    }

    // C_energy: normalized by max energy = all BSs at P_max for all time
    //   util_cost accumulates (P_idle/P_max + alpha_norm * u^3) per BS per step
    //   max = m * T_total * 1.0 (all BSs at full speed: P_norm = 1.0)
    //   When all BSs at full speed → 1.0
    double cost_energy() const {
        int T = (total_util_samples > 0) ? total_util_samples / m : MX;
        double max_energy = (double)m * (double)T;
        if (max_energy < 0.001) return 0;
        return util_cost / max_energy;
    }

    double avg_util() const {
        if (total_util_samples==0) return 0;
        return total_util_sum/(double)total_util_samples;
    }

    double cost_total() const {
        return PSI_DROP * cost_drop() + PSI_DIS * cost_dis() + PSI_E * cost_energy();
    }

    // Brief output
    void print(const string &name) const {
        if (!DETAILED_OUTPUT)
            cout<<name<<","<<cost_drop()<<","<<cost_dis()<<","<<cost_energy()<<endl;
        else
            cout<<name<<","<<cost_drop()<<","<<cost_dis()<<","<<cost_energy()
                <<","<<hard_completed<<","<<soft_completed
                <<","<<drophard<<","<<dropsoft<<","<<cost_total()<<endl;
    }

    // Print per-BS load distribution
    void print_bs_load(const string &name) const {
        cerr << "\n=== [" << name << "] Per-BS Load Distribution ===\n";
        cerr << "BS_ID  AvgUtil   Tasks   LoadBucket\n";
        // Buckets: [0-0.2) [0.2-0.4) [0.4-0.6) [0.6-0.8) [0.8-1.0]
        int buckets[5] = {0,0,0,0,0};
        for (int j = 1; j <= m; j++) {
            double avg = (bs_util_cnt[j] > 0) ? bs_util_sum[j] / bs_util_cnt[j] : 0.0;
            int b = min(4, (int)(avg * 5));
            buckets[b]++;
            if (m <= 20) // only print individual BS if small count
                cerr << "  " << setw(3) << j << "    "
                     << fixed << setprecision(3) << avg << "    "
                     << setw(5) << bs_task_cnt[j] << "   "
                     << string((int)(avg*40), '#') << "\n";
        }
        cerr << "Load histogram:  ";
        cerr << "[0-0.2):" << buckets[0] << "  [0.2-0.4):" << buckets[1]
             << "  [0.4-0.6):" << buckets[2] << "  [0.6-0.8):" << buckets[3]
             << "  [0.8-1.0]:" << buckets[4] << "\n";

        // Summary stats
        double mn=1e9, mx=0, sum=0; int active=0;
        for (int j=1;j<=m;j++) {
            double avg = (bs_util_cnt[j]>0) ? bs_util_sum[j]/bs_util_cnt[j] : 0.0;
            if (avg > 0.001) { mn=min(mn,avg); mx=max(mx,avg); sum+=avg; active++; }
        }
        if (active > 0)
            cerr << "Active BSs: " << active << "/" << m
                 << "  MinUtil: " << fixed << setprecision(3) << mn
                 << "  MaxUtil: " << mx
                 << "  MeanUtil: " << sum/active << "\n";
    }

    // Print response time distribution
    void print_timing(const string &name) const {
        cerr << "\n=== [" << name << "] Response Time Distribution ===\n";
        auto print_dist = [](const string &label, const vector<int> &v) {
            if (v.empty()) { cerr << label << ": no completed tasks\n"; return; }
            vector<int> s = v;
            sort(s.begin(), s.end());
            int n = s.size();
            double sum = 0; for (int x : s) sum += x;
            cerr << label << " (n=" << n << "):\n"
                 << "  Min: " << s[0] << "ms  "
                 << "  P25: " << s[n/4] << "ms  "
                 << "  P50: " << s[n/2] << "ms  "
                 << "  P75: " << s[3*n/4] << "ms  "
                 << "  P95: " << s[(int)(n*0.95)] << "ms  "
                 << "  Max: " << s[n-1] << "ms  "
                 << "  Mean: " << fixed << setprecision(1) << sum/n << "ms\n";
            // Histogram buckets
            cerr << "  Histogram: ";
            if (label.find("Hard") != string::npos) {
                // Hard: 0-30, 30-60, 60-90, 90-120, 120-150, >150
                int b[6]={0,0,0,0,0,0};
                for (int x:s) { int i=min(5,x/30); b[i]++; }
                cerr << "[0-30):" << b[0] << " [30-60):" << b[1]
                     << " [60-90):" << b[2] << " [90-120):" << b[3]
                     << " [120-150):" << b[4] << " [150+]:" << b[5] << "\n";
            } else {
                // Soft: 0-500, 500-1000, 1000-2000, 2000-3000, >3000
                int b[5]={0,0,0,0,0};
                for (int x:s) {
                    if (x<500) b[0]++;
                    else if (x<1000) b[1]++;
                    else if (x<2000) b[2]++;
                    else if (x<3000) b[3]++;
                    else b[4]++;
                }
                cerr << "[0-500):" << b[0] << " [500-1k):" << b[1]
                     << " [1k-2k):" << b[2] << " [2k-3k):" << b[3]
                     << " [3k+]:" << b[4] << "\n";
            }
        };
        print_dist("  Hard tasks", hard_response_times);
        print_dist("  Soft tasks", soft_response_times);
    }
};

// ============================================================
// CLI override flags (-1 = not set, use server.cfg)
// ============================================================
[[maybe_unused]] static double CLI_PIDLE = -1;
[[maybe_unused]] static double CLI_PMAX  = -1;
[[maybe_unused]] static double CLI_UMAX  = -1;
[[maybe_unused]] static double CLI_DELTA = -1;
[[maybe_unused]] static double CLI_MAXUTIL = -1;

inline void check_detailed_flag(int argc, char *argv[]) {
    for (int i=1; i<argc; i++) {
        if (string(argv[i])=="--detailed") DETAILED_OUTPUT=true;
        if (string(argv[i])=="--stats") STATS_OUTPUT=true;
        if (string(argv[i])=="--config" && i+1<argc) { config_path=string(argv[i+1]); i++; }
        if (string(argv[i])=="--pidle" && i+1<argc) { CLI_PIDLE=stod(argv[i+1]); P_IDLE=CLI_PIDLE; i++; }
        if (string(argv[i])=="--pmax"  && i+1<argc) { CLI_PMAX=stod(argv[i+1]);  P_MAX=CLI_PMAX;  i++; }
        if (string(argv[i])=="--umax"  && i+1<argc) { CLI_UMAX=stod(argv[i+1]);  i++; }
        if (string(argv[i])=="--max-util" && i+1<argc) { CLI_MAXUTIL=stod(argv[i+1]); i++; }
        if (string(argv[i])=="--delta" && i+1<argc) { CLI_DELTA=stod(argv[i+1]); i++; }
        if (string(argv[i])=="--server-cfg" && i+1<argc) { server_cfg_path=string(argv[i+1]); i++; }
    }
}

// Re-apply CLI overrides to globals (after load_config may have overwritten them)
inline void apply_cli_globals() {
    if (CLI_PIDLE >= 0) P_IDLE = CLI_PIDLE;
    if (CLI_PMAX  >= 0) P_MAX  = CLI_PMAX;
    if (CLI_DELTA >= 0) { DELTA = CLI_DELTA; }
    if (CLI_MAXUTIL >= 0) { MAX_UTIL = CLI_MAXUTIL; }
}

// After init_bs_power: apply CLI overrides to all BSs
inline void apply_cli_overrides(int m) {
    for (int j = 1; j <= m; j++) {
        auto &bp = bs_power[j];
        if (CLI_PIDLE >= 0) bp.p_idle = CLI_PIDLE;
        if (CLI_PMAX  >= 0) bp.p_max  = CLI_PMAX;
        if (CLI_PIDLE >= 0 || CLI_PMAX >= 0) {
            bp.alpha = bp.p_max - bp.p_idle;
            bp.u_crit = (bp.alpha > 0.001) ? pow(bp.p_idle / (2.0 * bp.alpha), 1.0/3.0) : 1.0;
        }
        if (CLI_UMAX  >= 0) bp.u_max  = CLI_UMAX;
    }
}

// ============================================================
// File I/O
// ============================================================
inline int read_input(const string &filename, vector<Task> &tasks) {
    // Step 1: Load defaults from config.txt (before anything else)
    load_config(config_path);
    // Step 1b: CLI overrides config.txt for globals
    apply_cli_globals();

    // Step 2: Read data file
    ifstream fin(filename);
    if (!fin.is_open()) { cerr<<"Error: "<<filename<<endl; exit(1); }
    int m; fin>>m;
    for (int ct=1; ct<=m; ct++) { int x,y; fin>>x>>y; base_station[ct]={x,y}; }
    while (fin.good()) {
        int a,pr,d,fl,px,py,tid;
        if (!(fin>>a>>pr>>d>>fl>>px>>py>>tid)) break;
        tasks.emplace_back(a,d,pr,tid,(bool)fl,px,py);
    }
    // Initialize heterogeneous per-BS power profiles for all algorithms
    init_bs_power(m);
    apply_cli_overrides(m);
    return m;
}

// ============================================================
// Scheduling helpers
// ============================================================
inline bool can_fit(vector<vector<double>>&u, int j, double ut, int st, int et, double cap) {
    for (int i=st; i<=et && i<(int)u[j].size(); i++) if (u[j][i]+ut>cap+0.000001) return false;
    return true;
}
inline double peak_util(vector<vector<double>>&u, int j, int st, int et) {
    double mx=0; for (int i=st; i<=et && i<(int)u[j].size(); i++) mx=max(mx,u[j][i]); return mx;
}
inline void assign_task(Task &x, int bs, vector<vector<double>>&u, vector<vector<Task>>&p) {
    p[bs].push_back(x);
    for (int t=x.start_time; t<=x.end_time && t<(int)u[bs].size(); t++) u[bs][t]+=x.utilisation;
}

inline int bestprocdrop(vector<vector<Task>>&proc, Task &t,
    vector<vector<double>>&util, int cur, set<int>&dr) {
    int st=cur+1, et=t.deadline;
    double req=(double)t.process_time/(double)(et-st);
    int best_n=1000000, best_j=-1;
    for (int j=1; j<(int)util.size(); j++) {
        // D_max: only consider BSs within d_hop range
        if (!d_hop_check(t.x, t.y, j, DIS_HOP)) continue;
        double mx_u = bs_cap(j);  // per-BS utilization cap
        set<int> s; bool ok=true;
        for (int i=st; i<=et; i++) {
            if ((req+util[j][i])>=mx_u) {
                vector<Task> c;
                for (auto &x:proc[j]) { if(x.flag==0)continue; if(!(i>=x.start_time&&i<=x.end_time))continue; c.push_back(x); }
                if (c.empty()){ok=false;break;}
                sort(c.begin(),c.end(),comp_util_desc);
                double v=util[j][i]; bool f=false; int n=0;
                for (auto &x:c) { v-=x.utilisation; n++; if((v+req)<=mx_u){f=true;break;} }
                if (f) { for(int q=0;q<n;q++) s.insert(c[q].tid); } else {ok=false;break;}
            }
        }
        if (ok&&(int)s.size()<best_n) { best_n=s.size(); best_j=j; dr=s; }
    }
    return best_j;
}
inline void execute_drop(int pr, Task &x, int cur,
    vector<vector<Task>>&proc, vector<vector<double>>&util, set<int>&dr, CostResult &cost) {
    vector<Task> kept;
    for (auto &t:proc[pr]) {
        if (dr.count(t.tid)) { cost.dropsoft++; for(int tm=cur;tm<=t.end_time;tm++) util[pr][tm]-=t.utilisation; }
        else kept.push_back(t);
    }
    x.utilisation=(double)x.process_time/(double)(x.deadline-cur-1);
    proc[pr]=kept; x.start_time=cur+1; x.end_time=x.deadline;
    for (int tm=x.start_time;tm<=x.end_time;tm++) util[pr][tm]+=x.utilisation;
    proc[pr].push_back(x); cost.add_distance(x,pr);
}

inline void sim_collect(int t, int&i, vector<Task>&tasks, vector<Task>&intask) {
    while (i<(int)tasks.size()&&tasks[i].arrival_time==t) { intask.push_back(tasks[i]); i++; }
}
inline void sim_complete(int t, int m, vector<vector<Task>>&proc, CostResult &cost) {
    for (int k=1;k<=m;k++) {
        vector<Task> kept;
        for (auto &tk:proc[k]) { if(tk.end_time==t){cost.task_completed(tk);continue;} kept.push_back(tk); }
        proc[k]=kept;
    }
}
inline void sim_energy(int t, int m, vector<vector<double>>&util, CostResult &cost) {
    for (int j=1;j<=m;j++) {
        double u=util[j][t-1];
        // Per-BS heterogeneous power model:
        // P_j(u) = p_idle_j + alpha_j * u^3
        // Normalized by BS's own p_max: P_j(u)/p_max_j
        double r, alpha_norm;
        if (bs_power.count(j)) {
            r = bs_power[j].p_idle / bs_power[j].p_max;
            alpha_norm = 1.0 - r;
        } else {
            r = P_IDLE / P_MAX;
            alpha_norm = 1.0 - r;
        }
        cost.util_cost += r + alpha_norm * u * u * u;
        cost.total_util_sum+=u; cost.total_util_samples++;
        cost.bs_util_sum[j]+=u; cost.bs_util_cnt[j]++;
    }
}

// Call after scheduling is done: print stats if --stats flag
inline void print_stats_if_needed(const string &name, CostResult &cost) {
    cost.print(name);
    if (STATS_OUTPUT) {
        cost.print_bs_load(name);
        cost.print_timing(name);
        // Energy breakdown: idle vs dynamic
        double r = P_IDLE / P_MAX;
        int T = cost.total_util_samples / cost.m;  // timesteps
        double idle_total = r * cost.m * T;
        double dynamic_total = cost.util_cost - idle_total;
        cerr << "\n=== [" << name << "] Energy Breakdown ===\n"
             << "  P_idle=" << P_IDLE << "W  P_max=" << P_MAX << "W  ratio=" 
             << fixed << setprecision(3) << r << "\n"
             << "  Idle power (all " << cost.m << " BSs × " << T << " steps × " 
             << r << "): " << idle_total << "\n"
             << "  Dynamic power (alpha×u³ summed): " << dynamic_total << "\n"
             << "  Total energy cost: " << cost.util_cost 
             << "  (idle=" << setprecision(1) << (idle_total/cost.util_cost*100) 
             << "% dynamic=" << (dynamic_total/cost.util_cost*100) << "%)\n"
             << "  Ce = " << setprecision(4) << cost.cost_energy() << "\n";

        // Machine-parseable box plot data on stdout (after CSV line)
        // DIST:<name>:<comma-separated per-task distances>
        cout << "DIST:" << name << ":";
        for (size_t i = 0; i < cost.task_distances.size(); i++) {
            if (i) cout << ",";
            cout << fixed << setprecision(4) << cost.task_distances[i];
        }
        cout << "\n";
        // UTIL:<name>:<comma-separated per-BS average utilizations>
        cout << "UTIL:" << name << ":";
        for (int j = 1; j <= cost.m; j++) {
            if (j > 1) cout << ",";
            double avg = (cost.bs_util_cnt[j] > 0) ?
                cost.bs_util_sum[j] / cost.bs_util_cnt[j] : 0.0;
            cout << fixed << setprecision(4) << avg;
        }
        cout << "\n";
    }
}

#endif
