#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

using namespace std;

struct CloudNode {
    int coord_x;
    int coord_y;
    int coord_z;
    uint8_t intensity_val;
    int true_id;
    int bucket_id;
};

__device__ inline int compute_spatial_hash(const int grid_x, const int grid_y, const int grid_z, const int max_size) {
    const int magic1 = 73856093;
    const int magic2 = 19349663;
    const int magic3 = 83492791;
    
    int raw_hash = ((grid_x * magic1) ^ (grid_y * magic2) ^ (grid_z * magic3)) % max_size;
    return (raw_hash < 0) ? (raw_hash + max_size) : raw_hash;
}

__device__ inline void determine_cell(const int px, const int py, const int pz, const int bound_x, const int bound_y, const int bound_z, const float cell_dim, int& out_x, int& out_y, int& out_z) {
    out_x = floorf((px - bound_x) / cell_dim);
    out_y = floorf((py - bound_y) / cell_dim);
    out_z = floorf((pz - bound_z) / cell_dim);
}

__device__ inline bool evaluate_priority(const int dist_a, const int idx_a, const int dist_b, const int idx_b, const int* arr_x, const int* arr_y, const int* arr_z, const int* arr_orig) {
    if (dist_a != dist_b) {
        return dist_a > dist_b;
    }
    if (idx_a == 2147483647) return true;
    if (idx_b == 2147483647) return false;

    int val_xa = arr_x[idx_a];
    int val_xb = arr_x[idx_b];
    if (val_xa != val_xb) return val_xa > val_xb;

    int val_ya = arr_y[idx_a];
    int val_yb = arr_y[idx_b];
    if (val_ya != val_yb) return val_ya > val_yb;

    int val_za = arr_z[idx_a];
    int val_zb = arr_z[idx_b];
    if (val_za != val_zb) return val_za > val_zb;

    return arr_orig[idx_a] > arr_orig[idx_b];
}

__global__ void gpu_fast_knn_search(const int* dev_x, const int* dev_y, const int* dev_z, const uint8_t* dev_i, const int* dev_orig, 
                                    uint8_t* dev_result, const int* map_start, const int* map_end, 
                                    int total_pts, int target_k, int base_x, int base_y, int base_z, float dim_size, int map_cap) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_pts) return;

    int safe_k = (target_k < 128) ? target_k : 128;

    int pq_dist[128];
    int pq_idx[128];

    #pragma unroll
    for(int pos = 0; pos < 128; pos++) { 
        pq_dist[pos] = 2147483647; 
        pq_idx[pos] = 2147483647; 
    }

    int cx = dev_x[tid];
    int cy = dev_y[tid];
    int cz = dev_z[tid];
    uint8_t c_val = dev_i[tid];
    int original_loc = dev_orig[tid];

    int cell_x, cell_y, cell_z;
    determine_cell(cx, cy, cz, base_x, base_y, base_z, dim_size, cell_x, cell_y, cell_z);

    int worst_dist = 2147483647;

    #pragma unroll
    for (int offset_x = -1; offset_x <= 1; offset_x++) {
        #pragma unroll
        for (int offset_y = -1; offset_y <= 1; offset_y++) {
            #pragma unroll
            for (int offset_z = -1; offset_z <= 1; offset_z++) {
                
                int search_x = cell_x + offset_x;
                int search_y = cell_y + offset_y;
                int search_z = cell_z + offset_z;
                int bucket = compute_spatial_hash(search_x, search_y, search_z, map_cap);

                int read_start = map_start[bucket];
                if (read_start != -1) {
                    int read_end = map_end[bucket];
                    
                    for (int ptr = read_start; ptr < read_end; ptr++) {
                        if (tid == ptr) continue;
                        
                        int match_x, match_y, match_z;
                        determine_cell(dev_x[ptr], dev_y[ptr], dev_z[ptr], base_x, base_y, base_z, dim_size, match_x, match_y, match_z);
                        
                        if (match_x == search_x && match_y == search_y && match_z == search_z) {
                            int diff_x = cx - dev_x[ptr];
                            int diff_y = cy - dev_y[ptr];
                            int diff_z = cz - dev_z[ptr];
                            int sq_dist = (diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z);

                            if (sq_dist < worst_dist || (sq_dist == worst_dist && evaluate_priority(worst_dist, pq_idx[0], sq_dist, ptr, dev_x, dev_y, dev_z, dev_orig))) {
                                pq_dist[0] = sq_dist;
                                pq_idx[0] = ptr;
                                
                                int node = 0;
                                while (true) {
                                    int l_child = 2 * node + 1;
                                    int r_child = 2 * node + 2;
                                    int max_node = node;

                                    if (l_child < safe_k && evaluate_priority(pq_dist[l_child], pq_idx[l_child], pq_dist[max_node], pq_idx[max_node], dev_x, dev_y, dev_z, dev_orig)) {
                                        max_node = l_child;
                                    }
                                    if (r_child < safe_k && evaluate_priority(pq_dist[r_child], pq_idx[r_child], pq_dist[max_node], pq_idx[max_node], dev_x, dev_y, dev_z, dev_orig)) {
                                        max_node = r_child;
                                    }

                                    if (max_node != node) {
                                        int swap_d = pq_dist[node]; 
                                        pq_dist[node] = pq_dist[max_node]; 
                                        pq_dist[max_node] = swap_d;
                                        
                                        int swap_i = pq_idx[node]; 
                                        pq_idx[node] = pq_idx[max_node]; 
                                        pq_idx[max_node] = swap_i;
                                        
                                        node = max_node; 
                                    } else {
                                        break; 
                                    }
                                }
                                worst_dist = pq_dist[0];
                            }
                        }
                    }
                }
            }
        }
    }

    int found_neighbors = 1;
    int rank_cdf = 1;
    uint8_t lowest_i = c_val;

    for (int step = 0; step < safe_k; step++) {
        if (pq_idx[step] != 2147483647) {
            found_neighbors++;
            uint8_t fetched_i = dev_i[pq_idx[step]];
            
            if (fetched_i <= c_val) rank_cdf++;
            if (fetched_i < lowest_i) lowest_i = fetched_i;
        }
    }

    int baseline_count = 0;
    if (c_val == lowest_i) baseline_count++;

    for (int step = 0; step < safe_k; step++) {
        if (pq_idx[step] != 2147483647) {
            if (dev_i[pq_idx[step]] == lowest_i) {
                baseline_count++;
            }
        }
    }

    if (found_neighbors == baseline_count) {
        dev_result[original_loc] = c_val;
    } else {
        float new_intensity = ((float)(rank_cdf - baseline_count) / (found_neighbors - baseline_count)) * 255.0f;
        dev_result[original_loc] = (uint8_t)fminf(255.0f, fmaxf(0.0f, floorf(new_intensity)));
    }
}

void approx_knn_histogram_equalization_cuda(const int* in_x, const int* in_y, const int* in_z, const uint8_t* in_val, int num_points, int k_val, uint8_t* out_val) {
    if (num_points == 0) return;

    int min_b_x = in_x[0], max_b_x = in_x[0];
    int min_b_y = in_y[0], max_b_y = in_y[0];
    int min_b_z = in_z[0], max_b_z = in_z[0];

    #pragma omp parallel for reduction(min:min_b_x,min_b_y,min_b_z) reduction(max:max_b_x,max_b_y,max_b_z)
    for (int p = 1; p < num_points; p++) {
        min_b_x = min(min_b_x, in_x[p]);
        max_b_x = max(max_b_x, in_x[p]);
        min_b_y = min(min_b_y, in_y[p]);
        max_b_y = max(max_b_y, in_y[p]);
        min_b_z = min(min_b_z, in_z[p]); 
        max_b_z = max(max_b_z, in_z[p]);
    }

    float len_x = (float)max_b_x - (float)min_b_x + 1.0f;
    float len_y = (float)max_b_y - (float)min_b_y + 1.0f;
    float len_z = (float)max_b_z - (float)min_b_z + 1.0f;
    float space_volume = len_x * len_y * len_z;

    float point_ratio = (float)k_val / (float)num_points;
    float density_target;
    
    if (point_ratio < 0.001f) {
        density_target = 10.0f;
    } else if (point_ratio < 0.01f) {
        density_target = 50.0f;
    } else if (point_ratio < 0.1f) {
        density_target = 200.0f;
    } else {
        density_target = 500.0f;
    }

    float ideal_vol = density_target * (space_volume / (float)num_points);
    float grid_dim = cbrtf(ideal_vol);
    if (grid_dim < 1.0f) grid_dim = 1.0f;

    min_b_x -= 1; 
    min_b_y -= 1; 
    min_b_z -= 1;
    int max_buckets = 200003;

    vector<CloudNode> data_nodes(num_points);
    #pragma omp parallel for
    for(int j = 0; j < num_points; j++) {
        data_nodes[j].coord_x = in_x[j];
        data_nodes[j].coord_y = in_y[j];
        data_nodes[j].coord_z = in_z[j];
        data_nodes[j].intensity_val = in_val[j];
        data_nodes[j].true_id = j;
        
        int gx = floorf((in_x[j] - min_b_x) / grid_dim);
        int gy = floorf((in_y[j] - min_b_y) / grid_dim);
        int gz = floorf((in_z[j] - min_b_z) / grid_dim);
        
        const int pm1 = 73856093; 
        const int pm2 = 19349663; 
        const int pm3 = 83492791;
        
        int b_hash = ((gx * pm1) ^ (gy * pm2) ^ (gz * pm3)) % max_buckets;
        if (b_hash < 0) b_hash += max_buckets;
        data_nodes[j].bucket_id = b_hash;
    }

    std::sort(data_nodes.begin(), data_nodes.end(), [](const CloudNode& first, const CloudNode& second) {
        return first.bucket_id < second.bucket_id;
    });

    vector<int> flat_x(num_points), flat_y(num_points), flat_z(num_points), flat_orig(num_points);
    vector<uint8_t> flat_val(num_points);
    vector<int> start_ptr(max_buckets, -1);
    vector<int> end_ptr(max_buckets, -1);

    for (int pos = 0; pos < num_points; pos++) {
        flat_x[pos] = data_nodes[pos].coord_x;
        flat_y[pos] = data_nodes[pos].coord_y;
        flat_z[pos] = data_nodes[pos].coord_z;
        flat_val[pos] = data_nodes[pos].intensity_val;
        flat_orig[pos] = data_nodes[pos].true_id;
        
        int curr_hash = data_nodes[pos].bucket_id;
        if (pos == 0 || data_nodes[pos - 1].bucket_id != curr_hash) {
            start_ptr[curr_hash] = pos;
        }
        if (pos == num_points - 1 || data_nodes[pos + 1].bucket_id != curr_hash) {
            end_ptr[curr_hash] = pos + 1;
        }
    }

    int *gpu_x, *gpu_y, *gpu_z, *gpu_orig, *gpu_start, *gpu_end;
    uint8_t *gpu_in_val, *gpu_out_val;

    cudaMalloc(&gpu_start, max_buckets * sizeof(int));
    cudaMalloc(&gpu_end, max_buckets * sizeof(int));
    cudaMalloc(&gpu_x, num_points * sizeof(int));
    cudaMalloc(&gpu_y, num_points * sizeof(int));
    cudaMalloc(&gpu_z, num_points * sizeof(int));
    cudaMalloc(&gpu_orig, num_points * sizeof(int));
    cudaMalloc(&gpu_in_val, num_points * sizeof(uint8_t));
    cudaMalloc(&gpu_out_val, num_points * sizeof(uint8_t));

    // Transfer Data
    cudaMemcpy(gpu_start, start_ptr.data(), max_buckets * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_end, end_ptr.data(), max_buckets * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_x, flat_x.data(), num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, flat_y.data(), num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_z, flat_z.data(), num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_orig, flat_orig.data(), num_points * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_in_val, flat_val.data(), num_points * sizeof(uint8_t), cudaMemcpyHostToDevice);

    int t_per_block = 32;
    int total_blocks = (num_points + t_per_block - 1) / t_per_block;

    gpu_fast_knn_search<<<total_blocks, t_per_block>>>(
        gpu_x, gpu_y, gpu_z, gpu_in_val, gpu_orig, gpu_out_val, gpu_start, gpu_end, 
        num_points, k_val, min_b_x, min_b_y, min_b_z, grid_dim, max_buckets
    );

    cudaDeviceSynchronize();

    cudaMemcpy(out_val, gpu_out_val, num_points * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(gpu_start); 
    cudaFree(gpu_end);
    cudaFree(gpu_in_val); 
    cudaFree(gpu_out_val);
    cudaFree(gpu_x); 
    cudaFree(gpu_y); 
    cudaFree(gpu_z); 
    cudaFree(gpu_orig);
}