// gen_data: Workload generator that reads parameters from config.txt
//
// Usage: ./gen_data <outfile> [config_file]
//   If config_file not specified, reads ./config.txt
//   All parameters come from config. No hardcoded values.
//
// Or legacy mode: ./gen_data <num_av> <num_bs> <sim_ms> <outfile> [config_file]

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <iomanip>
using namespace std;

mt19937 rng(42);
int randI(int lo, int hi) { return uniform_int_distribution<int>(lo,hi)(rng); }
double randD(double lo, double hi) { return uniform_real_distribution<double>(lo,hi)(rng); }
double randG(double m, double s) { return normal_distribution<double>(m,s)(rng); }

// Read config file into key-value map
map<string,string> read_config(const string &path) {
    map<string,string> cfg;
    ifstream f(path);
    if (!f.is_open()) { cerr << "Warning: cannot open " << path << ", using defaults\n"; return cfg; }
    string line;
    while (getline(f, line)) {
        // Strip comments
        auto pos = line.find('#');
        if (pos != string::npos) line = line.substr(0, pos);
        // Find KEY = VALUE
        pos = line.find('=');
        if (pos == string::npos) continue;
        string key = line.substr(0, pos);
        string val = line.substr(pos + 1);
        // Trim whitespace
        auto trim = [](string &s) {
            while (!s.empty() && isspace(s.front())) s.erase(s.begin());
            while (!s.empty() && isspace(s.back())) s.pop_back();
        };
        trim(key); trim(val);
        if (!key.empty() && !val.empty()) cfg[key] = val;
    }
    return cfg;
}

int cfg_int(map<string,string> &c, const string &k, int def) {
    return c.count(k) ? stoi(c[k]) : def;
}
double cfg_dbl(map<string,string> &c, const string &k, double def) {
    return c.count(k) ? stod(c[k]) : def;
}

int main(int argc, char *argv[]) {
    // Parse arguments: support both old and new format
    string outfile, config_path = "config.txt";
    int cli_av = -1, cli_bs = -1, cli_ms = -1;

    if (argc >= 5 && atoi(argv[1]) > 0) {
        // Legacy: ./gen_data <num_av> <num_bs> <sim_ms> <outfile> [config]
        cli_av = atoi(argv[1]);
        cli_bs = atoi(argv[2]);
        cli_ms = atoi(argv[3]);
        outfile = argv[4];
        if (argc >= 6) config_path = argv[5];
    } else if (argc >= 2) {
        // New: ./gen_data <outfile> [config]
        outfile = argv[1];
        if (argc >= 3) config_path = argv[2];
    } else {
        cerr << "Usage: ./gen_data <outfile> [config.txt]\n"
             << "   or: ./gen_data <num_av> <num_bs> <sim_ms> <outfile> [config.txt]\n";
        return 1;
    }

    // Read config
    auto cfg = read_config(config_path);

    // Parameters: CLI overrides config
    int num_av     = (cli_av > 0) ? cli_av : cfg_int(cfg, "NUM_AV", 80);
    int num_bs     = (cli_bs > 0) ? cli_bs : cfg_int(cfg, "NUM_BS", 10);
    int sim_ms     = (cli_ms > 0) ? cli_ms : cfg_int(cfg, "SIM_MS", 10000);
    int grid       = cfg_int(cfg, "GRID_SIZE", 200);

    int HARD_DEADLINE    = cfg_int(cfg, "HARD_DEADLINE", 150);
    int SOFT_DEADLINE    = cfg_int(cfg, "SOFT_DEADLINE", 3000);
    double hard_prob_def = cfg_dbl(cfg, "HARD_PROB", -1);  // -1 = auto from ratio
    int hard_interval    = cfg_int(cfg, "HARD_INTERVAL", 15);

    // Soft task intervals: fixed reasonable values (not user-facing)
    int soft_nav_lo = 1000, soft_nav_hi = 3000;
    int soft_info_lo = 3000, soft_info_hi = 10000;
    int soft_nav_pt_lo   = cfg_int(cfg, "SOFT_NAV_PT_LO", 30);
    int soft_nav_pt_hi   = cfg_int(cfg, "SOFT_NAV_PT_HI", 80);
    int soft_info_pt_lo  = cfg_int(cfg, "SOFT_INFO_PT_LO", 80);
    int soft_info_pt_hi  = cfg_int(cfg, "SOFT_INFO_PT_HI", 200);

    double hotspot_prob    = cfg_dbl(cfg, "HOTSPOT_PROB", 0.50);
    int hotspot_radius     = cfg_int(cfg, "HOTSPOT_RADIUS", 30);
    double hotspot_cluster = cfg_dbl(cfg, "HOTSPOT_CLUSTER", 0.30);

    // Auto-compute HARD_PROB from HARD_SOFT_RATIO if not explicitly set
    double target_ratio  = cfg_dbl(cfg, "HARD_SOFT_RATIO", 29);
    if (hard_prob_def < 0) {
        double mean_nav = (soft_nav_lo + soft_nav_hi) / 2.0;
        double mean_info = (soft_info_lo + soft_info_hi) / 2.0;
        double e_soft_per_av = (double)sim_ms / mean_nav + (double)sim_ms / mean_info;
        double c = hotspot_cluster;
        double rate_factor = (double)sim_ms / hard_interval;

        // First try: solve assuming hotspot_prob stays fixed
        double needed = target_ratio * e_soft_per_av / rate_factor - c * hotspot_prob;
        hard_prob_def = needed / (1.0 - c);

        if (hard_prob_def < 0.01) {
            // Hotspot alone exceeds target ratio.
            // Scale down BOTH hard_prob and hotspot_prob proportionally.
            // Let hotspot_prob = k * hard_prob (keep ratio fixed)
            // ratio = [(1-c)*prob + c*k*prob] * rate_factor / e_soft_per_av
            // prob = target_ratio * e_soft_per_av / (rate_factor * [(1-c) + c*k])
            double k = hotspot_prob / 0.21;  // original ratio between hotspot and normal
            hard_prob_def = target_ratio * e_soft_per_av / (rate_factor * ((1-c) + c*k));
            hotspot_prob = hard_prob_def * k;
        }
        hard_prob_def = max(0.005, min(hard_prob_def, 0.95));
        hotspot_prob = max(0.005, min(hotspot_prob, 0.95));
    }

    // ── Generate BS locations ──
    ofstream fout(outfile);
    fout << num_bs << "\n";

    set<pair<int,int>> used;
    vector<pair<int,int>> bs_locs;
    for (int i = 0; i < num_bs; i++) {
        int x, y;
        do { x=randI(1,grid); y=randI(1,grid); } while (used.count({x,y}));
        used.insert({x,y}); bs_locs.push_back({x,y});
        fout << x << " " << y << "\n";
    }

    // ── Hotspot setup ──
    int hotspot_bs = min(num_bs-1, 8);
    int hx = bs_locs[hotspot_bs].first, hy = bs_locs[hotspot_bs].second;
    int num_clustered = (int)(num_av * hotspot_cluster);

    // ── AV locations ──
    vector<pair<int,int>> av_locs;
    for (int i = 0; i < num_clustered; i++) {
        int x, y;
        do {
            x = hx + randI(-hotspot_radius, hotspot_radius);
            y = hy + randI(-hotspot_radius, hotspot_radius);
            x = max(1, min(grid, x)); y = max(1, min(grid, y));
        } while (used.count({x,y}));
        used.insert({x,y}); av_locs.push_back({x,y});
    }
    for (int i = num_clustered; i < num_av; i++) {
        int x, y;
        do { x=randI(1,grid); y=randI(1,grid); } while (used.count({x,y}));
        used.insert({x,y}); av_locs.push_back({x,y});
    }

    // ── YOLO model distribution ──
    struct Model { int pt_lo, pt_hi; };
    Model models[] = {{3,6}, {6,12}, {12,25}, {25,45}};
    int model_cdf[] = {20, 50, 80, 100};

    struct RawTask { int arrival, proc, deadline, flag, x, y; };
    vector<RawTask> tasks;

    for (int av = 0; av < num_av; av++) {
        int ax = av_locs[av].first, ay = av_locs[av].second;

        int r = randI(1, 100);
        int mi = 0; for (int k = 0; k < 4; k++) { if (r <= model_cdf[k]) { mi = k; break; } }

        double dx = ax - hx, dy = ay - hy;
        bool in_hotspot = (dx*dx + dy*dy) <= hotspot_radius * hotspot_radius;
        double hp = in_hotspot ? hotspot_prob : hard_prob_def;

        // HARD TASKS
        for (int t = 0; t < sim_ms; t += hard_interval) {
            if (randD(0,1) < hp) {
                double comm = max(8.0, randG(12.5, 1.25));
                int eff_dl = t + (int)(HARD_DEADLINE - comm - 1.0);
                int pt = randI(models[mi].pt_lo, models[mi].pt_hi);
                if (eff_dl <= t + pt + 1 || eff_dl > sim_ms) continue;
                tasks.push_back({t, pt, eff_dl, 0, ax, ay});
            }
        }

        // SOFT TASKS: navigation
        int next_nav = randI(0, 2000);
        while (next_nav < sim_ms) {
            double comm = max(8.0, randG(12.5, 1.25));
            int eff_dl = next_nav + (int)(SOFT_DEADLINE - comm - 1.0);
            if (eff_dl > sim_ms) eff_dl = sim_ms;
            int spt = randI(soft_nav_pt_lo, soft_nav_pt_hi);
            if (eff_dl > next_nav + spt + 1)
                tasks.push_back({next_nav, spt, eff_dl, 1, ax, ay});
            next_nav += randI(soft_nav_lo, soft_nav_hi);
        }

        // SOFT TASKS: infotainment
        int next_info = randI(0, 5000);
        while (next_info < sim_ms) {
            double comm = max(8.0, randG(12.5, 1.25));
            int eff_dl = next_info + (int)(SOFT_DEADLINE - comm - 1.0);
            if (eff_dl > sim_ms) eff_dl = sim_ms;
            int spt = randI(soft_info_pt_lo, soft_info_pt_hi);
            if (eff_dl > next_info + spt + 1)
                tasks.push_back({next_info, spt, eff_dl, 1, ax, ay});
            next_info += randI(soft_info_lo, soft_info_hi);
        }
    }

    sort(tasks.begin(), tasks.end(), [](const RawTask &a, const RawTask &b) {
        return a.arrival < b.arrival;
    });

    int tid = 1;
    for (auto &t : tasks)
        fout << t.arrival<<" "<<t.proc<<" "<<t.deadline<<" "
             << t.flag<<" "<<t.x<<" "<<t.y<<" "<<tid++<<"\n";
    fout.close();

    // Stats
    int hard=0, soft=0; double hpt=0, spt=0;
    int hard_hotspot=0;
    for (auto &t : tasks) {
        if (t.flag==0) { hard++; hpt+=t.proc;
            double ddx=t.x-hx, ddy=t.y-hy;
            if (ddx*ddx+ddy*ddy <= hotspot_radius*hotspot_radius) hard_hotspot++;
        } else { soft++; spt+=t.proc; }
    }
    cerr << "Generated " << outfile << "\n"
         << "  Config: " << config_path << "\n"
         << "  Target ratio: " << target_ratio << ":1  → HARD_PROB=" << fixed << setprecision(3) << hard_prob_def << "\n"
         << "  Total tasks: " << tasks.size() << " (" << hard << " hard, " << soft << " soft)\n"
         << "  Actual ratio: " << (soft>0 ? hard/soft : hard) << ":1\n"
         << "  Avg hard proc_time: " << setprecision(1) << (hard>0?hpt/hard:0) << "\n"
         << "  Avg soft proc_time: " << (soft>0?spt/soft:0) << "\n"
         << "  HOTSPOT: BS" << hotspot_bs+1 << " at (" << hx << "," << hy
         << ") radius=" << hotspot_radius << " p=" << hotspot_prob << "\n"
         << "    " << num_clustered << "/" << num_av << " AVs clustered nearby\n"
         << "    " << hard_hotspot << " hard tasks from hotspot ("
         << (hard>0 ? 100*hard_hotspot/hard : 0) << "%)\n";
    return 0;
}
