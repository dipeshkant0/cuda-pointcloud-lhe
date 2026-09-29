#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <queue>
#include <omp.h>

using namespace std;

struct cloud_point {
    int x;
    int y;
    int z;
    uint8_t I; 
    int original_index; 
};

struct FlatVoxelGrid {
    int min_x, min_y, min_z;
    float voxel_size;
    int hash_size;

    vector<int> head; 
    vector<int> next; 

    FlatVoxelGrid(int n, int desired_hash_size) {
        hash_size = desired_hash_size;
        head.assign(hash_size, -1);
        next.assign(n, -1);
    }

    void get_voxel_index(float x, float y, float z, int& ix, int& iy, int& iz) {
        ix = floor((x - min_x) / voxel_size);
        iy = floor((y - min_y) / voxel_size);
        iz = floor((z - min_z) / voxel_size);
    }

    int get_hash(int ix, int iy, int iz) {  
        const int p1 = 73856093;
        const int p2 = 19349663;
        const int p3 = 83492791;
        int h = ((ix * p1) ^ (iy * p2) ^ (iz * p3)) % hash_size;
        if (h < 0) h += hash_size;
        return h;
    }
};

struct centroid {
    int x, y, z;
    uint8_t I;

    long long sum_x = 0, sum_y = 0, sum_z = 0;
    int point_count = 0;

    void reset() {
        sum_x = sum_y = sum_z = 0;
        point_count = 0;
    }
};


void KNN_local_histogram_equalization(vector<cloud_point>& cloud_points, int n, int k, int t) {
    vector<cloud_point> new_cloud_points = cloud_points;


    auto tie_breaker = [&cloud_points](const pair<int, int>& a, const pair<int, int>& b) {
        if (a.first == b.first) {
            int idx_a = a.second;
            int idx_b = b.second;
            if (cloud_points[idx_a].x != cloud_points[idx_b].x) return cloud_points[idx_a].x < cloud_points[idx_b].x;
            if (cloud_points[idx_a].y != cloud_points[idx_b].y) return cloud_points[idx_a].y < cloud_points[idx_b].y;
            if (cloud_points[idx_a].z != cloud_points[idx_b].z) return cloud_points[idx_a].z < cloud_points[idx_b].z;
            return cloud_points[idx_a].original_index < cloud_points[idx_b].original_index;
        }
        return a.first < b.first; 
    };

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(tie_breaker)> heap(tie_breaker);

        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            long long dx = cloud_points[i].x - cloud_points[j].x;
            long long dy = cloud_points[i].y - cloud_points[j].y;
            long long dz = cloud_points[i].z - cloud_points[j].z;
            long long dist = dx*dx + dy*dy + dz*dz;

            if ((int)heap.size() < k) {
                heap.push({dist, j});
            } else if (tie_breaker({dist, j}, heap.top())) {
                heap.pop();
                heap.push({dist, j});
            }
        }

        int actual_k = heap.size()+1;
        vector<int> hist(256, 0);
        hist[cloud_points[i].I]++;
        while (!heap.empty()) {
            hist[cloud_points[heap.top().second].I]++;
            heap.pop();
        }

        vector<int> cdf(256, 0);
        cdf[0] = hist[0];
        int c_min = (cdf[0] > 0) ? cdf[0] : 0;

        for (int intensity = 1; intensity < 256; intensity++) {
            cdf[intensity] = cdf[intensity - 1] + hist[intensity];
            if (cdf[intensity] > 0 && c_min == 0) c_min = cdf[intensity]; 
        }

        uint8_t original_I = cloud_points[i].I;
        if (actual_k == c_min || actual_k == 0){
            new_cloud_points[i].I = original_I;
        } else {
            float mapped_val = ((float)(cdf[original_I] - c_min) / (float)(actual_k - c_min)) * 255.0f;
            new_cloud_points[i].I = (uint8_t)floorf(fmaxf(0.0f, fminf(255.0f, mapped_val)));
        }   
    }

    ofstream outfile("knn.txt");
    for (const auto& cp : new_cloud_points) {
        outfile << fixed << setprecision(2) << cp.x << " " << cp.y << " " << cp.z << " " << static_cast<int>(cp.I) << endl;
    }
    outfile.close();
}


void approximate_KNN_local_histogram_equalization(vector<cloud_point>& cloud_points, int n, int k, int t) {
    if (n == 0) return;

    int min_x = cloud_points[0].x, max_x = cloud_points[0].x;
    int min_y = cloud_points[0].y, max_y = cloud_points[0].y;
    int min_z = cloud_points[0].z, max_z = cloud_points[0].z;

    #pragma omp parallel for reduction(min:min_x,min_y,min_z) reduction(max:max_x,max_y,max_z)
    for (int i = 1; i < n; i++) {
        min_x = min(min_x, cloud_points[i].x); max_x = max(max_x, cloud_points[i].x);
        min_y = min(min_y, cloud_points[i].y); max_y = max(max_y, cloud_points[i].y);
        min_z = min(min_z, cloud_points[i].z); max_z = max(max_z, cloud_points[i].z);
    }

    float dx_span = (float)max_x - (float)min_x + 1.0f;
    float dy_span = (float)max_y - (float)min_y + 1.0f;
    float dz_span = (float)max_z - (float)min_z + 1.0f;
    float total_volume = dx_span * dy_span * dz_span;
    
    float ratio = (float)k / (float)n;
    float target_points_per_voxel;
    if (ratio < 0.001f) target_points_per_voxel = 8.0f;
    else if (ratio < 0.01f) target_points_per_voxel = 50.0f;
    else if (ratio < 0.1f) target_points_per_voxel = 200.0f;
    else target_points_per_voxel = 500.0f;
    
    float ideal_voxel_volume = target_points_per_voxel * (total_volume / (float)n);
    float voxel_size = cbrtf(ideal_voxel_volume);
    if (voxel_size < 1.0f) voxel_size = 1.0f;

    FlatVoxelGrid v_grid(n, 200003);
    v_grid.min_x = min_x - 1; 
    v_grid.min_y = min_y - 1; 
    v_grid.min_z = min_z - 1;
    v_grid.voxel_size = voxel_size;

    for (int i = 0; i < n; i++) {
        int ix, iy, iz;
        v_grid.get_voxel_index(cloud_points[i].x, cloud_points[i].y, cloud_points[i].z, ix, iy, iz);
        int h = v_grid.get_hash(ix, iy, iz);   
        v_grid.next[i] = v_grid.head[h];
        v_grid.head[h] = i;
    }

    vector<cloud_point> new_points = cloud_points;
    int k_eff = min(k, n);

    auto tie_breaker = [&cloud_points](const pair<int, int>& a, const pair<int, int>& b) {
        if (a.first == b.first) {
            int idx_a = a.second, idx_b = b.second;
            if (cloud_points[idx_a].x != cloud_points[idx_b].x) return cloud_points[idx_a].x < cloud_points[idx_b].x;
            if (cloud_points[idx_a].y != cloud_points[idx_b].y) return cloud_points[idx_a].y < cloud_points[idx_b].y;
            if (cloud_points[idx_a].z != cloud_points[idx_b].z) return cloud_points[idx_a].z < cloud_points[idx_b].z;
            return cloud_points[idx_a].original_index < cloud_points[idx_b].original_index;
        }
        return a.first < b.first; 
    };

    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; i++) {
        int ix, iy, iz;
        v_grid.get_voxel_index(cloud_points[i].x, cloud_points[i].y, cloud_points[i].z, ix, iy, iz);
        
        priority_queue<pair<int, int>, vector<pair<int, int>>, decltype(tie_breaker)> heap(tie_breaker);
        
        int R = 1;
        while ((int)heap.size() < k_eff && R <= 10) {
            for (int dx = -R; dx <= R; dx++) {
                for (int dy = -R; dy <= R; dy++) {
                    for (int dz = -R; dz <= R; dz++) {
                        // Skip inner cells if we are expanding beyond R=1
                        if (R > 1 && abs(dx) < R && abs(dy) < R && abs(dz) < R) continue;

                        int h = v_grid.get_hash(ix + dx, iy + dy, iz + dz);
                        int p_idx = v_grid.head[h];
                        
                        while (p_idx != -1) {
                            if (i != p_idx) {
                                int p_ix, p_iy, p_iz;
                                v_grid.get_voxel_index(cloud_points[p_idx].x, cloud_points[p_idx].y, cloud_points[p_idx].z, p_ix, p_iy, p_iz);
                                
                                if (p_ix == (ix+dx) && p_iy == (iy+dy) && p_iz == (iz+dz)) {
                                    long long d_x = cloud_points[i].x - cloud_points[p_idx].x;
                                    long long d_y = cloud_points[i].y - cloud_points[p_idx].y;
                                    long long d_z = cloud_points[i].z - cloud_points[p_idx].z;
                                    long long dist = d_x*d_x + d_y*d_y + d_z*d_z;
                                    
                                    if ((int)heap.size() < k_eff) {
                                        heap.push({dist, p_idx});
                                    } else if (tie_breaker({dist, p_idx}, heap.top())) {
                                        heap.pop();
                                        heap.push({dist, p_idx});
                                    }
                                }
                            }
                            p_idx = v_grid.next[p_idx];
                        }
                    }
                }
            }
            R++;
        }

        int actual_k = heap.size()+1;
        
        if (actual_k > 0) {
            vector<int> hist(256, 0);
            hist[cloud_points[i].I]++;
            while(!heap.empty()) {
                hist[cloud_points[heap.top().second].I]++;
                heap.pop();
            }

            vector<int> cdf(256, 0);
            cdf[0] = hist[0];
            int c_min = (cdf[0] > 0) ? cdf[0] : 0;
            for (int v = 1; v < 256; v++) {
                cdf[v] = cdf[v - 1] + hist[v];
                if (cdf[v] > 0 && c_min == 0) c_min = cdf[v];
            }

            uint8_t original_I = cloud_points[i].I;
            if (actual_k == c_min) {
                new_points[i].I = original_I;
            } else {
                float mapped = ((float)(cdf[original_I] - c_min) / (actual_k - c_min)) * 255.0f;
                new_points[i].I = (uint8_t)floorf(fmaxf(0.0f, fminf(255.0f, mapped)));
            }
        }
    }

    ofstream outfile("approx_knn.txt");
    for (const auto& cp : new_points) {
        outfile << fixed << setprecision(2) << cp.x << " " << cp.y << " " << cp.z << " " << (int)cp.I << endl;
    }
    outfile.close();
}

void k_mean_local_histogram_equalization(vector<cloud_point>& cloud_points, int n, int k, int t) {
    if(n <= 0 || k <= 0 || t <= 0) return;
    if(k > n) k = n;

    vector<centroid> centroids(k);

    for(int i = 0; i < k; i++){
        centroids[i].x = cloud_points[i].x;
        centroids[i].y = cloud_points[i].y;
        centroids[i].z = cloud_points[i].z;
        centroids[i].I = cloud_points[i].I;
        centroids[i].point_count = 0;
    }

    bool converged = false;
    int iterations = 0;
    vector<int> points_cluster_id(n, -1);
    vector<cloud_point> new_cloud_points = cloud_points;

    auto is_better = [&](long long d_new, int c_new, long long d_best, int c_best) {
        if (d_new != d_best) return d_new < d_best;
        if (centroids[c_new].x != centroids[c_best].x) return centroids[c_new].x < centroids[c_best].x;
        if (centroids[c_new].y != centroids[c_best].y) return centroids[c_new].y < centroids[c_best].y;
        return centroids[c_new].z < centroids[c_best].z;
    };

    while(!converged && iterations < t) {
        converged = true;
        
        for(int c = 0; c < k; c++){
            centroids[c].reset();
        }

        int points_changed = 0;

        
        #pragma omp parallel
        {
            vector<long long> local_sum_x(k, 0), local_sum_y(k, 0), local_sum_z(k, 0);
            vector<int> local_count(k, 0);
            int local_changed = 0;

            #pragma omp for
            for(int i = 0; i < n; i++){
                long long min_dist = 9223372036854775807LL; 
                int closest_centroid = -1;

                for(int c = 0; c < k; c++){
                    long long dx = cloud_points[i].x - centroids[c].x;
                    long long dy = cloud_points[i].y - centroids[c].y;
                    long long dz = cloud_points[i].z - centroids[c].z;
                    long long dist = dx*dx + dy*dy + dz*dz;

                    if(dist < min_dist || (dist == min_dist && is_better(dist, c, min_dist, closest_centroid))){
                        min_dist = dist;
                        closest_centroid = c;
                    }
                }

                if(points_cluster_id[i] != closest_centroid){
                    local_changed++;
                    points_cluster_id[i] = closest_centroid;
                }
                
                
                local_sum_x[closest_centroid] += cloud_points[i].x;
                local_sum_y[closest_centroid] += cloud_points[i].y;
                local_sum_z[closest_centroid] += cloud_points[i].z;
                local_count[closest_centroid]++;
            }

           
            #pragma omp critical
            {
                points_changed += local_changed;
                for(int c = 0; c < k; c++) {
                    centroids[c].sum_x += local_sum_x[c];
                    centroids[c].sum_y += local_sum_y[c];
                    centroids[c].sum_z += local_sum_z[c];
                    centroids[c].point_count += local_count[c];
                }
            }
        }

        if (points_changed > 0) converged = false;

        for(int c = 0; c < k; c++){
            if(centroids[c].point_count > 0){
                centroids[c].x = centroids[c].sum_x / centroids[c].point_count;
                centroids[c].y = centroids[c].sum_y / centroids[c].point_count;
                centroids[c].z = centroids[c].sum_z / centroids[c].point_count;
            }
        }
        iterations++;
    }

    vector<vector<int>> cluster_hist(k, vector<int>(256, 0));
    vector<vector<int>> cluster_cdf(k, vector<int>(256, 0));
    vector<int> cluster_cmin(k, 0);

    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        int c_id = points_cluster_id[i];
        if(c_id != -1){
            #pragma omp atomic
            cluster_hist[c_id][cloud_points[i].I]++;
        }
    }

    for(int c = 0; c < k; c++){
        if(centroids[c].point_count > 0){
            cluster_cdf[c][0] = cluster_hist[c][0];
            int c_min = (cluster_cdf[c][0] > 0) ? cluster_cdf[c][0] : 0;

            for(int intensity = 1; intensity < 256; intensity++){
                cluster_cdf[c][intensity] = cluster_cdf[c][intensity - 1] + cluster_hist[c][intensity];
                if(cluster_cdf[c][intensity] > 0 && c_min == 0) c_min = cluster_cdf[c][intensity];
            }
            cluster_cmin[c] = c_min;
        }
    }

    #pragma omp parallel for
    for(int i = 0; i < n; i++){
        int c_id = points_cluster_id[i];
        if(c_id != -1 && centroids[c_id].point_count > 0){
            uint8_t original_I = cloud_points[i].I;
            int m = centroids[c_id].point_count;
            int cdf_val = cluster_cdf[c_id][original_I];
            int c_min = cluster_cmin[c_id];

            if(m == c_min){
                new_cloud_points[i].I = original_I;
            } else {
                double mapped_val = ((double)(cdf_val - c_min) / (m - c_min)) * 255.0;
                new_cloud_points[i].I = static_cast<uint8_t>(floor(max(0.0, min(255.0, mapped_val))));
            }
        }
    }

    ofstream outfile("kmeans.txt");
    for(const auto& cp : new_cloud_points){
        outfile << fixed << setprecision(2) << cp.x << " " << cp.y << " " << cp.z << " " << static_cast<int>(cp.I) << endl;
    }
    outfile.close();
}

int main(int argc, char* argv[]) {
    vector<cloud_point> cloud_points;
    
    
    const char* fname = (argc >= 2) ? argv[1] : "input.txt";
    ifstream infile(fname);
    
    if (!infile) {
        cerr << "Error opening file: " << fname << endl;
        return 1;
    }

    int n, k, t;
    infile >> n >> k >> t; 

    cloud_points.reserve(n);

    for(int i = 0; i < n; i++){
        cloud_point cp;
        int intensity;
        infile >> cp.x >> cp.y >> cp.z >> intensity;
        cp.I = static_cast<uint8_t>(intensity);
        cp.original_index = i; // Save the unique ID!
        cloud_points.push_back(cp);
    }
    infile.close();

    auto start = chrono::high_resolution_clock::now();
    KNN_local_histogram_equalization(cloud_points, n, k, t);
    auto end = chrono::high_resolution_clock::now();
    cout << "Exact KNN took " << chrono::duration_cast<chrono::milliseconds>(end - start).count()/1000.0 << " s.\n";

    start = chrono::high_resolution_clock::now();
    approximate_KNN_local_histogram_equalization(cloud_points, n, k, t);
    end = chrono::high_resolution_clock::now();
    cout << "Approximate KNN took " << chrono::duration_cast<chrono::milliseconds>(end - start).count()/1000.0 << " s.\n";

    start = chrono::high_resolution_clock::now();
    k_mean_local_histogram_equalization(cloud_points, n, k, t);
    end = chrono::high_resolution_clock::now();
    cout << "K-Mean took " << chrono::duration_cast<chrono::milliseconds>(end - start).count()/1000.0 << " s.\n";

    return 0;
}