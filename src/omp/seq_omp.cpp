#include<iostream>
#include<vector>
#include<array>
#include<cmath>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<limits>
#include<chrono>
#include<queue>
#include<omp.h>

using namespace std;
using namespace std::chrono;

struct Point {
    int x, y, z;
    int intensity;
    int original_index;
};

struct Result{
    int x, y, z;
    int remapped_intensity;
};


void runKNN(int n, int k, const vector<Point>& points){
    vector<Result> results(n);

    auto cmp = [&points](pair<int,int> a, pair<int,int> b){
        if(a.first != b.first) return a.first < b.first;
        const Point& pa = points[a.second];
        const Point& pb = points[b.second];
        if(pa.x != pb.x) return pa.x < pb.x;
        if(pa.y != pb.y) return pa.y < pb.y;
        return pa.z < pb.z;
    };

    #pragma omp parallel for schedule(dynamic, 64)
    for(int i = 0; i < n; i++){
        int xi = points[i].x, yi = points[i].y, zi = points[i].z;

        priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> heap(cmp);

        for(int j = 0; j < n; j++){

            if(j == i) continue;

            int dx = xi - points[j].x;
            int dy = yi - points[j].y;
            int dz = zi - points[j].z;
            int d2 = dx*dx + dy*dy + dz*dz;

            if((int)heap.size() < k){
                heap.push({d2, j});
            } else {
                auto [top_d, top_j] = heap.top();
                bool replace = (d2 < top_d);
                if(!replace && d2 == top_d){
                    const Point& pj = points[j];
                    const Point& pt = points[top_j];
                    if(pj.x != pt.x) replace = (pj.x < pt.x);
                    else if(pj.y != pt.y) replace = (pj.y < pt.y);
                    else replace = (pj.z < pt.z);
                }
                if(replace){ heap.pop(); heap.push({d2, j}); }
            }
        }

        int m = (int)heap.size();
        int hist[256] = {};
        while(!heap.empty()){
            hist[points[heap.top().second].intensity]++;
            heap.pop();
        }
        //piazza @26_f20: include point itself in histogram, m = k+1
        hist[points[i].intensity]++;
        m++;

        for(int v = 1; v < 256; v++) hist[v] += hist[v-1];

        int c_min = 0;
        for(int v = 0; v < 256; v++){
            if(hist[v] > 0){ c_min = hist[v]; break; }
        }

        int cur = points[i].intensity;
        if(m == c_min){
            results[i] = {xi, yi, zi, cur};
            continue;
        }

        float num = (float)(hist[cur] - c_min);
        float den = (float)(m - c_min);
        results[i] = {xi, yi, zi, max(0, (int)floor((num / den) * 255.0f))};
    }

    ofstream out("knn_n.txt");
    for(const auto& r : results){
        out << r.x << " " << r.y << " " << r.z << " " << r.remapped_intensity << "\n";
    }
    out.close();
}


void runApproxKNN(int n, int k, const vector<Point>& points){
    vector<Result> results(n);
    int k_eff = min(k, n);

    const int G       = max(3, (int)cbrt((double)n / 100.0));  // ~10 for n=100k
    const int NCELLS  = G * G * G;
    const int CMIN    = -10000;
    const int CELL_SZ = (20001 + G - 1) / G;  // = 2001 for G=10

    auto gcell = [&](int v) -> int {
        return min(G - 1, max(0, (v - CMIN) / CELL_SZ));
    };

    // --- Build grid (O(n)) ---
    vector<int> cid(n);
    vector<int> cnt(NCELLS, 0);
    for(int i = 0; i < n; i++){
        int gx = gcell(points[i].x), gy = gcell(points[i].y), gz = gcell(points[i].z);
        cid[i] = (gx * G + gy) * G + gz;
        cnt[cid[i]]++;
    }

    vector<int> cell_start(NCELLS + 1, 0);
    for(int c = 0; c < NCELLS; c++) cell_start[c + 1] = cell_start[c] + cnt[c];

    // Counting sort: stable order within each cell
    vector<int> sorted_idx(n);
    {
        vector<int> tmp(NCELLS, 0);
        for(int i = 0; i < n; i++)
            sorted_idx[cell_start[cid[i]] + tmp[cid[i]]++] = i;
    }

    auto cmp = [&points](pair<int,int> a, pair<int,int> b){
        if(a.first != b.first) return a.first < b.first;
        const Point& pa = points[a.second], &pb = points[b.second];
        if(pa.x != pb.x) return pa.x < pb.x;
        if(pa.y != pb.y) return pa.y < pb.y;
        return pa.z < pb.z;
    };

    // --- Per-query grid search (parallel) ---
    #pragma omp parallel for schedule(dynamic, 32)
    for(int i = 0; i < n; i++){
        int xi = points[i].x, yi = points[i].y, zi = points[i].z;
        int qgx = gcell(xi), qgy = gcell(yi), qgz = gcell(zi);

        priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> heap(cmp);

        // Expand search shell R=1,2,... until heap has k_eff entries.
        // For typical interior points, R=1 (27 cells) is always sufficient.
        for(int R = 1; R <= G && (int)heap.size() < k_eff; R++){
            for(int dx = -R; dx <= R; dx++){
            for(int dy = -R; dy <= R; dy++){
            for(int dz = -R; dz <= R; dz++){
                // Only process the new outer shell to avoid re-visiting inner cells
                if(R > 1 && abs(dx) < R && abs(dy) < R && abs(dz) < R) continue;
                int cgx = qgx+dx, cgy = qgy+dy, cgz = qgz+dz;
                if(cgx<0||cgx>=G||cgy<0||cgy>=G||cgz<0||cgz>=G) continue;
                int c = (cgx * G + cgy) * G + cgz;
                for(int p = cell_start[c]; p < cell_start[c + 1]; p++){
                    int j = sorted_idx[p];
                    if(j == i) continue;
                    int ddx = xi - points[j].x;
                    int ddy = yi - points[j].y;
                    int ddz = zi - points[j].z;
                    int d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                    if((int)heap.size() < k_eff){
                        heap.push({d2, j});
                    } else if(cmp({d2, j}, heap.top())){
                        heap.pop();
                        heap.push({d2, j});
                    }
                }
            }}}
        }

        int m = (int)heap.size();
        int hist[256] = {};
        while(!heap.empty()){
            hist[points[heap.top().second].intensity]++;
            heap.pop();
        }
        //piazza @26_f20: include point itself in histogram, m = k+1
        hist[points[i].intensity]++;
        m++;

        for(int v = 1; v < 256; v++) hist[v] += hist[v-1];

        int c_min = 0;
        for(int v = 0; v < 256; v++){
            if(hist[v] > 0){ c_min = hist[v]; break; }
        }

        int cur = points[i].intensity;
        if(m == c_min){
            results[i] = {xi, yi, zi, cur};
            continue;
        }
        float num = (float)(hist[cur] - c_min);
        float den = (float)(m - c_min);
        results[i] = {xi, yi, zi, max(0, (int)floor((num / den) * 255.0f))};
    }

    ofstream out("approx_knn_n.txt");
    for(const auto& r : results)
        out << r.x << " " << r.y << " " << r.z << " " << r.remapped_intensity << "\n";
    out.close();
}


void runKmeans(int n, int k, int t_max, const vector<Point>& points){

    struct Centroid{ float x, y, z; };
    vector<Centroid> centroids(k);
    for(int i = 0; i < k; i++){
        centroids[i] = {(float)points[i].x, (float)points[i].y, (float)points[i].z};
    }

    vector<int> assignments(n, -1);

    for(int iter = 0; iter < t_max; iter++){
        bool changed = false;

        #pragma omp parallel for schedule(static) reduction(||:changed)
        for(int i = 0; i < n; i++){
            float min_dist = numeric_limits<float>::max();
            int best = 0;
            for(int j = 0; j < k; j++){
                float dx = points[i].x - centroids[j].x;
                float dy = points[i].y - centroids[j].y;
                float dz = points[i].z - centroids[j].z;
                float d2 = dx*dx + dy*dy + dz*dz;
                //piazza @26_f19: tie -> assign to cluster with lexically smaller centroid
                if(d2 < min_dist || (d2 == min_dist &&
                    (centroids[j].x < centroids[best].x ||
                    (centroids[j].x == centroids[best].x && centroids[j].y < centroids[best].y) ||
                    (centroids[j].x == centroids[best].x && centroids[j].y == centroids[best].y && centroids[j].z < centroids[best].z)))){
                    min_dist = d2; best = j;
                }
            }
            if(assignments[i] != best){
                assignments[i] = best;
                changed = true;
            }
        }

        vector<int> sx(k,0), sy(k,0), sz(k,0);
        vector<int> count(k, 0);

        #pragma omp parallel
        {
            vector<int> lsx(k,0), lsy(k,0), lsz(k,0);
            vector<int> lcount(k,0);

            #pragma omp for schedule(static)
            for(int i = 0; i < n; i++){
                int c = assignments[i];
                lsx[c] += points[i].x;
                lsy[c] += points[i].y;
                lsz[c] += points[i].z;
                lcount[c]++;
            }

            #pragma omp critical
            for(int j = 0; j < k; j++){
                sx[j] += lsx[j];
                sy[j] += lsy[j];
                sz[j] += lsz[j];
                count[j] += lcount[j];
            }
        }

        for(int j = 0; j < k; j++){
            if(count[j] > 0){
                //piazza @26_f19: integer division for centroids
                centroids[j] = {(float)(sx[j]/count[j]), (float)(sy[j]/count[j]), (float)(sz[j]/count[j])};
            }
        }

        if(!changed){
            cout << "Converged at iteration " << iter << endl;
            break;
        }
    }

    vector<array<int,256>> cluster_hist(k);
    for(auto& h : cluster_hist) h.fill(0);
    vector<int> cluster_size(k, 0);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int c = assignments[i];
        #pragma omp atomic
        cluster_hist[c][points[i].intensity]++;
        #pragma omp atomic
        cluster_size[c]++;
    }

    for(int c = 0; c < k; c++){
        for(int v = 1; v < 256; v++){
            cluster_hist[c][v] += cluster_hist[c][v-1];
        }
    }

    vector<int> cluster_cmin(k, 0);
    for(int c = 0; c < k; c++){
        for(int v = 0; v < 256; v++){
            if(cluster_hist[c][v] > 0){ cluster_cmin[c] = cluster_hist[c][v]; break; }
        }
    }

    vector<Result> results(n);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; i++){
        int c    = assignments[i];
        int m    = cluster_size[c];
        int cmin = cluster_cmin[c];
        int cur  = points[i].intensity;

        if(m == cmin){
            results[i] = {points[i].x, points[i].y, points[i].z, cur};
            continue;
        }

        float num = (float)(cluster_hist[c][cur] - cmin);
        float den = (float)(m - cmin);
        results[i] = {points[i].x, points[i].y, points[i].z,
                      max(0, (int)floor((num / den) * 255.0f))};
    }

    ofstream out("kmeans_n.txt");
    for(const auto& r : results){
        out << r.x << " " << r.y << " " << r.z << " " << r.remapped_intensity << "\n";
    }
    out.close();
}


int main(int argc, char* argv[]){
    const char* fname = (argc >= 2) ? argv[1] : "input.txt";

    ifstream in(fname);
    if(!in){ cerr << "Failed to open " << fname << endl; return 1; }

    int n, k, t;
    if(!(in >> n >> k >> t)) return 0;

    vector<Point> points(n);
    for(int i = 0; i < n; i++){
        in >> points[i].x >> points[i].y >> points[i].z >> points[i].intensity;
        points[i].original_index = i;
    }
    in.close();

    auto t0 = high_resolution_clock::now();
    runKNN(n, k, points);
    auto t1 = high_resolution_clock::now();

    runApproxKNN(n, k, points);
    auto t2 = high_resolution_clock::now();

    runKmeans(n, k, t, points);
    auto t3 = high_resolution_clock::now();

    long long knn_ms    = duration_cast<milliseconds>(t1-t0).count();
    long long approx_ms = duration_cast<milliseconds>(t2-t1).count();
    long long kmeans_ms = duration_cast<milliseconds>(t3-t2).count();
    long long total_ms  = duration_cast<milliseconds>(t3-t0).count();

    cout << "Sequential Timing Summary ............." << endl;
    cout << left
         << setw(25) << "Algorithm"
         << setw(15) << "Time (ms)"
         << "Output file\n";
    cout << string(55, '-') << "\n";
    cout << setw(25) << "Exact KNN"       << setw(15) << knn_ms    << "knn.txt\n";
    cout << setw(25) << "Approximate KNN" << setw(15) << approx_ms << "approx_knn.txt\n";
    cout << setw(25) << "K-Means"         << setw(15) << kmeans_ms << "kmeans.txt\n";
    cout << string(55, '-') << "\n";
    cout << setw(25) << "Total wall time" << setw(15) << total_ms  << "\n";

    return 0;
}
