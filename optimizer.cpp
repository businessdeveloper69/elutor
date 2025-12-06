#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <iomanip>
#include <numeric>
#include <chrono>

using namespace std;

constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

const double TX[NV] = {0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075,
                       -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125};
const double TY[NV] = {0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2,
                       -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5};

struct Poly {
    vector<double> px, py;
    double xmin, ymin, xmax, ymax;
    
    Poly() : px(NV), py(NV) {}
    
    void set(double ox, double oy, double deg) {
        double rad = deg * PI / 180.0;
        double c = cos(rad), s = sin(rad);
        xmin = ymin = 1e18; xmax = ymax = -1e18;
        for (int i = 0; i < NV; i++) {
            px[i] = TX[i] * c - TY[i] * s + ox;
            py[i] = TX[i] * s + TY[i] * c + oy;
            xmin = min(xmin, px[i]); ymin = min(ymin, py[i]);
            xmax = max(xmax, px[i]); ymax = max(ymax, py[i]);
        }
    }
};

double cross(double ox, double oy, double ax, double ay, double bx, double by) {
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

bool point_in_poly(double x, double y, const Poly& P) {
    int cnt = 0;
    for (int i = 0; i < NV; i++) {
        int j = (i + 1) % NV;
        if ((P.py[i] <= y && P.py[j] > y) || (P.py[j] <= y && P.py[i] > y)) {
            double xi = P.px[i] + (y - P.py[i]) / (P.py[j] - P.py[i]) * (P.px[j] - P.px[i]);
            if (x < xi) cnt++;
        }
    }
    return cnt % 2 == 1;
}

bool seg_intersect(double ax, double ay, double bx, double by,
                   double cx, double cy, double dx, double dy) {
    double d1 = cross(ax, ay, bx, by, cx, cy);
    double d2 = cross(ax, ay, bx, by, dx, dy);
    double d3 = cross(cx, cy, dx, dy, ax, ay);
    double d4 = cross(cx, cy, dx, dy, bx, by);
    if (((d1 > 1e-12) != (d2 > 1e-12)) && ((d3 > 1e-12) != (d4 > 1e-12))) return true;
    return false;
}

bool overlaps(const Poly& A, const Poly& B) {
    double margin = 1e-9;
    if (A.xmax <= B.xmin + margin || B.xmax <= A.xmin + margin ||
        A.ymax <= B.ymin + margin || B.ymax <= A.ymin + margin) return false;
    
    for (int i = 0; i < NV; i++) {
        if (point_in_poly(A.px[i], A.py[i], B)) return true;
        if (point_in_poly(B.px[i], B.py[i], A)) return true;
    }
    
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (seg_intersect(A.px[i], A.py[i], A.px[ni], A.py[ni],
                              B.px[j], B.py[j], B.px[nj], B.py[nj]))
                return true;
        }
    }
    return false;
}

struct Tree {
    double x, y, a;
    Tree() : x(0), y(0), a(0) {}
    Tree(double x, double y, double a) : x(x), y(y), a(a) {}
};

struct Config {
    vector<Tree> trees;
    vector<Poly> polys;
    
    int n() const { return trees.size(); }
    
    void rebuild() {
        polys.resize(trees.size());
        for (size_t i = 0; i < trees.size(); i++)
            polys[i].set(trees[i].x, trees[i].y, trees[i].a);
    }
    
    void rebuild_one(int i) {
        polys[i].set(trees[i].x, trees[i].y, trees[i].a);
    }
    
    bool check_one(int i) {
        for (int j = 0; j < n(); j++)
            if (j != i && overlaps(polys[i], polys[j])) return false;
        return true;
    }
    
    bool check_all() {
        for (int i = 0; i < n(); i++)
            for (int j = i+1; j < n(); j++)
                if (overlaps(polys[i], polys[j])) return false;
        return true;
    }
    
    double score() {
        if (n() == 0) return 1e18;
        double x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
        for (auto& P : polys) {
            x0 = min(x0, P.xmin); y0 = min(y0, P.ymin);
            x1 = max(x1, P.xmax); y1 = max(y1, P.ymax);
        }
        double side = max(x1 - x0, y1 - y0);
        return side * side / n();
    }
    
    void center() {
        double x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18;
        for (auto& P : polys) {
            x0 = min(x0, P.xmin); y0 = min(y0, P.ymin);
            x1 = max(x1, P.xmax); y1 = max(y1, P.ymax);
        }
        double cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0;
        for (auto& t : trees) { t.x -= cx; t.y -= cy; }
        rebuild();
    }
};

void run_sa(Config& cfg, mt19937& rng, int iters, double T_init) {
    int N = cfg.n();
    if (N == 0) return;
    
    double best_score = cfg.score();
    Config best = cfg;
    double cur_score = best_score;
    double T = T_init;
    double cool = pow(1e-8 / T, 1.0 / iters);
    
    uniform_real_distribution<double> ud(0.0, 1.0);
    
    for (int it = 0; it < iters; it++) {
        int i = rng() % N;
        Tree old_t = cfg.trees[i];
        Poly old_p = cfg.polys[i];
        
        double r = ud(rng);
        double scale = T + 0.001;
        
        if (r < 0.65) {
            cfg.trees[i].a += (ud(rng) - 0.5) * scale * 60.0;
        } else if (r < 0.90) {
            cfg.trees[i].x += (ud(rng) - 0.5) * scale;
            cfg.trees[i].y += (ud(rng) - 0.5) * scale;
        } else if (N > 1) {
            int j = rng() % N;
            if (i != j) {
                swap(cfg.trees[i], cfg.trees[j]);
                cfg.rebuild_one(j);
            }
        }
        
        cfg.rebuild_one(i);
        if (cfg.check_one(i)) {
            double new_score = cfg.score();
            double delta = new_score - cur_score;
            if (delta < 0 || ud(rng) < exp(-delta / (T + 0.001))) {
                cur_score = new_score;
                if (new_score < best_score - 1e-12) {
                    best = cfg;
                    best_score = new_score;
                }
            } else {
                cfg.trees[i] = old_t;
                cfg.polys[i] = old_p;
            }
        } else {
            cfg.trees[i] = old_t;
            cfg.polys[i] = old_p;
        }
        T *= cool;
    }
    cfg = best;
    cfg.center();
}

void opt_angles(Config& cfg) {
    for (int i = 0; i < cfg.n(); i++) {
        double orig = cfg.trees[i].a;
        double best = cfg.score();
        double best_a = orig;
        for (int da = -45; da <= 45; da += 5) {
            cfg.trees[i].a = orig + da;
            cfg.rebuild_one(i);
            if (cfg.check_one(i)) {
                double s = cfg.score();
                if (s < best - 1e-12) { best = s; best_a = orig + da; }
            }
        }
        cfg.trees[i].a = best_a;
        cfg.rebuild_one(i);
    }
    cfg.center();
}

void compact(Config& cfg, mt19937& rng, int iters) {
    int N = cfg.n();
    Config best = cfg;
    double best_score = cfg.score();
    uniform_real_distribution<double> ud(0.0, 1.0);
    
    for (int it = 0; it < iters; it++) {
        double cx = 0, cy = 0;
        for (auto& t : cfg.trees) { cx += t.x; cy += t.y; }
        cx /= N; cy /= N;
        int i = rng() % N;
        Tree old_t = cfg.trees[i];
        Poly old_p = cfg.polys[i];
        double dx = cx - cfg.trees[i].x, dy = cy - cfg.trees[i].y;
        double d = sqrt(dx*dx + dy*dy);
        if (d > 0.001) {
            double step = 0.001 + ud(rng) * 0.02;
            cfg.trees[i].x += step * dx / d;
            cfg.trees[i].y += step * dy / d;
            cfg.rebuild_one(i);
            if (cfg.check_one(i)) {
                double s = cfg.score();
                if (s < best_score - 1e-12) { best = cfg; best_score = s; }
            } else {
                cfg.trees[i] = old_t;
                cfg.polys[i] = old_p;
            }
        }
    }
    cfg = best;
    cfg.center();
}

unordered_map<int, Config> load(const string& path) {
    unordered_map<int, Config> res;
    unordered_map<int, vector<Tree>> grp;
    ifstream f(path);
    if (!f) { cerr << "Cannot open " << path << endl; return res; }
    string line;
    getline(f, line);
    
    int lines = 0;
    while (getline(f, line)) {
        if (line.empty()) continue;
        lines++;
        
        size_t c1 = line.find(','), c2 = line.find(',', c1+1), c3 = line.find(',', c2+1);
        if (c1 == string::npos || c2 == string::npos || c3 == string::npos) continue;
        
        string id_str = line.substr(0, c1);
        size_t underscore = id_str.find('_');
        if (underscore == string::npos) continue;
        
        int n_val = stoi(id_str.substr(0, underscore));
        
        string xs = line.substr(c1+1, c2-c1-1);
        string ys = line.substr(c2+1, c3-c2-1);
        string as = line.substr(c3+1);
        
        if (!xs.empty() && xs[0] == 's') xs = xs.substr(1);
        if (!ys.empty() && ys[0] == 's') ys = ys.substr(1);
        if (!as.empty() && as[0] == 's') as = as.substr(1);
        
        try {
            grp[n_val].push_back(Tree(stod(xs), stod(ys), stod(as)));
        } catch (...) {}
    }
    
    cerr << "Loaded " << lines << " lines into " << grp.size() << " configs\n";
    
    for (auto& kv : grp) {
        Config c;
        c.trees = kv.second;
        c.rebuild();
        res[kv.first] = c;
    }
    return res;
}

void save(const unordered_map<int, Config>& cfgs, const string& path) {
    ofstream f(path);
    f << "id,x,y,deg\n" << fixed << setprecision(15);
    for (int n = 1; n <= 200; n++) {
        auto it = cfgs.find(n);
        if (it == cfgs.end()) continue;
        for (int i = 0; i < it->second.n(); i++) {
            auto& t = it->second.trees[i];
            f << setfill('0') << setw(3) << n << "_" << setw(2) << i
              << ",s" << t.x << ",s" << t.y << ",s" << t.a << "\n";
        }
    }
}

double total(unordered_map<int, Config>& cfgs) {
    double s = 0;
    for (auto& kv : cfgs) { kv.second.rebuild(); s += kv.second.score(); }
    return s;
}

int main(int argc, char** argv) {
    string input = argc > 1 ? argv[1] : "submission.csv";
    string output = argc > 2 ? argv[2] : "optimized.csv";
    int sa_iters = argc > 3 ? atoi(argv[3]) : 80000;
    int gens = argc > 4 ? atoi(argv[4]) : 2;
    
    cout << fixed << setprecision(6);
    cout << "=== POLYGON PACKING OPTIMIZER (C++) ===" << endl;
    cout << "Input: " << input << ", SA iters: " << sa_iters << ", Generations: " << gens << endl << endl;
    
    auto start = chrono::high_resolution_clock::now();
    auto cfgs = load(input);
    cout << "Loaded " << cfgs.size() << " configurations\n";
    double init = total(cfgs);
    cout << "Initial score: " << init << endl << endl;
    
    for (int g = 0; g < gens; g++) {
        auto gs = chrono::high_resolution_clock::now();
        cout << "=== GENERATION " << g+1 << " ===" << endl;
        int improved = 0;
        double delta = 0;
        int skipped = 0;
        
        for (int n = 200; n >= 1; n--) {
            if (cfgs.find(n) == cfgs.end()) continue;
            Config& cur = cfgs[n];
            cur.rebuild();
            
            // Skip if already invalid (happens with greedy init for large n)
            if (!cur.check_all()) {
                skipped++;
                continue;
            }
            
            double old_s = cur.score();
            Config best = cur;
            double best_s = old_s;
            mt19937 rng(n * 54321 + g * 999);
            
            // Angle optimization
            opt_angles(best);
            if (best.check_all()) best_s = best.score();
            
            // Compaction
            compact(best, rng, 600 + n*6);
            if (best.check_all()) best_s = best.score();
            
            // Tuned SA parameters
            double t_init = (n <= 10) ? 3.0 : min(6.0, 30.0/n);
            int iters = min(sa_iters, max(5000, sa_iters / (1 + n/40)));
            run_sa(best, rng, iters, t_init);
            if (best.check_all()) best_s = best.score();
            
            opt_angles(best);
            if (best.check_all()) best_s = best.score();
            
            if (best_s < old_s - 1e-12) {
                cfgs[n] = best;
                improved++;
                delta += old_s - best_s;
                if (n <= 15 || n % 50 == 0) 
                    cout << "n=" << n << ": " << old_s << " -> " << best_s << "\n";
            }
        }
        
        double new_total = total(cfgs);
        auto ge = chrono::high_resolution_clock::now();
        auto gtime = chrono::duration_cast<chrono::seconds>(ge - gs).count();
        cout << "Gen " << g+1 << ": improved=" << improved << " skipped=" << skipped 
             << " delta=" << delta << " total=" << new_total << " time=" << gtime << "s\n\n";
        save(cfgs, output);
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::seconds>(end - start).count();
    double fin = total(cfgs);
    cout << "=== OPTIMIZATION COMPLETE ===" << endl;
    cout << "Initial: " << init << "\nFinal: " << fin 
         << "\nImprovement: " << init - fin << " (" << 100*(init-fin)/init << "%)"
         << "\nTotal Time: " << elapsed << "s\n";
    save(cfgs, output);
    return 0;
}
