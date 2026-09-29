#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>


//Point Assignment & Shared Memory Reduction
__global__ void evaluate_and_assign_points(const int* p_x, const int* p_y, const int* p_z, 
                                           const int* c_x, const int* c_y, const int* c_z,
                                           int* cluster_mapping, int* acc_x, int* acc_y, int* acc_z, 
                                           int* cluster_sizes, int num_pts, int num_clusters, int* convergence_status) {
    

    extern __shared__ int shared_data[]; 
    int* local_sum_x = (int*)shared_data;           
    int* local_sum_y = (int*)&shared_data[num_clusters];        
    int* local_sum_z = (int*)&shared_data[2 * num_clusters];   
    int* local_counts = (int*)&shared_data[3 * num_clusters];      

    for (int idx = threadIdx.x; idx < num_clusters; idx += blockDim.x) {
        local_sum_x[idx] = 0; 
        local_sum_y[idx] = 0; 
        local_sum_z[idx] = 0; 
        local_counts[idx] = 0;
    }
    __syncthreads();

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_tid < num_pts) {
        int my_px = p_x[global_tid];
        int my_py = p_y[global_tid];
        int my_pz = p_z[global_tid];
        
        int lowest_dist = 2147483647; 
        int best_c_id = -1;

        // iterate through all centroids to find the closest one
        for (int c_idx = 0; c_idx < num_clusters; ++c_idx) {
            int diff_x = my_px - c_x[c_idx];
            int diff_y = my_py - c_y[c_idx];
            int diff_z = my_pz - c_z[c_idx];
            int current_dist = (diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z); 

            bool is_new_best = false;
            
            if (best_c_id == -1 || current_dist < lowest_dist) {
                is_new_best = true;
            } else if (current_dist == lowest_dist) {
                bool tie_x = (c_x[c_idx] == c_x[best_c_id]);
                bool tie_y = (c_y[c_idx] == c_y[best_c_id]);
                
                if (c_x[c_idx] < c_x[best_c_id]) {
                    is_new_best = true;
                } else if (tie_x && c_y[c_idx] < c_y[best_c_id]) {
                    is_new_best = true;
                } else if (tie_x && tie_y && c_z[c_idx] < c_z[best_c_id]) {
                    is_new_best = true;
                }
            }

            if (is_new_best) {
                lowest_dist = current_dist;
                best_c_id = c_idx;
            }
        }

        if (cluster_mapping[global_tid] != best_c_id) {
            cluster_mapping[global_tid] = best_c_id;
            atomicExch(convergence_status, 1);
        }

        atomicAdd(&local_sum_x[best_c_id], my_px);
        atomicAdd(&local_sum_y[best_c_id], my_py);
        atomicAdd(&local_sum_z[best_c_id], my_pz);
        atomicAdd(&local_counts[best_c_id], 1);
    }
    
    __syncthreads(); 

    for (int idx = threadIdx.x; idx < num_clusters; idx += blockDim.x) {
        if (local_counts[idx] > 0) {
            atomicAdd(&acc_x[idx], local_sum_x[idx]);
            atomicAdd(&acc_y[idx], local_sum_y[idx]);
            atomicAdd(&acc_z[idx], local_sum_z[idx]);
            atomicAdd(&cluster_sizes[idx], local_counts[idx]);
        }
    }
}

__global__ void recalculate_cluster_centers(int* c_x, int* c_y, int* c_z, 
                                            int* acc_x, int* acc_y, int* acc_z, 
                                            int* cluster_sizes, int num_clusters) {
    
    int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (t_idx >= num_clusters) return;

    int total_points = cluster_sizes[t_idx];
    
    if (total_points > 0) {
        c_x[t_idx] = acc_x[t_idx] / total_points;
        c_y[t_idx] = acc_y[t_idx] / total_points;
        c_z[t_idx] = acc_z[t_idx] / total_points;
    }

    acc_x[t_idx] = 0;
    acc_y[t_idx] = 0;
    acc_z[t_idx] = 0;
    cluster_sizes[t_idx] = 0;
}

//create histogram
__global__ void construct_intensity_histograms(const uint8_t* raw_I, const int* mappings, int* hist_array, int num_pts) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pts) {
        int target_cluster = mappings[idx];
        if (target_cluster != -1) {
            atomicAdd(&hist_array[target_cluster * 256 + raw_I[idx]], 1);
        }
    }
}

//calculate cdf
__global__ void calculate_cumulative_distribution(const int* hist_array, int* cdf_array, int* min_c_vals, int num_clusters) {

    int c_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_id >= num_clusters) return;
    
    int running_sum = 0;
    int first_valid_cmin = 0;
    
    for (int intensity_val = 0; intensity_val < 256; ++intensity_val) {
        running_sum += hist_array[c_id * 256 + intensity_val];
        cdf_array[c_id * 256 + intensity_val] = running_sum;
        
        if (running_sum > 0 && first_valid_cmin == 0) {
            first_valid_cmin = running_sum;
        }
    }
    
    min_c_vals[c_id] = first_valid_cmin;
}

//equilize intensities
__global__ void finalize_intensity_mapping(const uint8_t* raw_I, const int* mappings, 
                                           const int* cdf_array, const int* min_c_vals, 
                                           uint8_t* final_I, int num_pts) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_pts) return;
    
    int my_cluster = mappings[thread_id];
    
    if (my_cluster != -1) {
        int cluster_population = cdf_array[my_cluster * 256 + 255]; 
        
        if (cluster_population > 0) {
            int c_minimum = min_c_vals[my_cluster];
            uint8_t original_intensity = raw_I[thread_id];
            
            if (cluster_population == c_minimum) {
                final_I[thread_id] = original_intensity;
            } else {
                float math_mapping = ((float)(cdf_array[my_cluster * 256 + original_intensity] - c_minimum) / (cluster_population - c_minimum)) * 255.0f;
                final_I[thread_id] = (uint8_t)fminf(255.0f, fmaxf(0.0f, floorf(math_mapping)));
            }
            return;
        }
    }
    
    final_I[thread_id] = raw_I[thread_id];
}

void k_mean_local_histogram_equalization_cuda(const int* host_x, const int* host_y, const int* host_z, const uint8_t* host_I, int num_elements, int num_k, int max_iters, uint8_t* host_out_I) {
    
    if (num_elements <= 0 || num_k <= 0 || max_iters <= 0) return;
    if (num_k > num_elements) num_k = num_elements;

    int *gpu_x, *gpu_y, *gpu_z;
    int *gpu_cent_x, *gpu_cent_y, *gpu_cent_z;
    int *gpu_cluster_map, *gpu_population, *gpu_change_flag;
    int *gpu_sum_x, *gpu_sum_y, *gpu_sum_z;
    uint8_t *gpu_in_I, *gpu_final_I;

    cudaMalloc(&gpu_cluster_map, num_elements * sizeof(int));
    cudaMalloc(&gpu_x, num_elements * sizeof(int));
    cudaMalloc(&gpu_y, num_elements * sizeof(int));
    cudaMalloc(&gpu_z, num_elements * sizeof(int));
    cudaMalloc(&gpu_in_I, num_elements * sizeof(uint8_t));
    cudaMalloc(&gpu_final_I, num_elements * sizeof(uint8_t));
    
    cudaMalloc(&gpu_sum_x, num_k * sizeof(int)); 
    cudaMalloc(&gpu_sum_y, num_k * sizeof(int)); 
    cudaMalloc(&gpu_sum_z, num_k * sizeof(int));
    cudaMalloc(&gpu_cent_x, num_k * sizeof(int)); 
    cudaMalloc(&gpu_cent_y, num_k * sizeof(int));
    cudaMalloc(&gpu_cent_z, num_k * sizeof(int));
    
    cudaMalloc(&gpu_population, num_k * sizeof(int));
    cudaMalloc(&gpu_change_flag, sizeof(int));

    cudaMemset(gpu_cluster_map, -1, num_elements * sizeof(int));
    cudaMemcpy(gpu_x, host_x, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_y, host_y, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_z, host_z, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_in_I, host_I, num_elements * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_cent_x, gpu_x, num_k * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_cent_y, gpu_y, num_k * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpu_cent_z, gpu_z, num_k * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaMemset(gpu_sum_x, 0, num_k * sizeof(int));
    cudaMemset(gpu_sum_y, 0, num_k * sizeof(int));
    cudaMemset(gpu_sum_z, 0, num_k * sizeof(int));

    int block_size = 256;
    int grid_elements = (num_elements + block_size - 1) / block_size;
    int grid_clusters = (num_k + block_size - 1) / block_size;
    int dynamic_smem_size = 4 * num_k * sizeof(int); 

    int loop_status_flag = 1;
    int iteration_counter = 0;

    while (loop_status_flag == 1 && iteration_counter < max_iters) {
        cudaMemset(gpu_change_flag, 0, sizeof(int));

        evaluate_and_assign_points<<<grid_elements, block_size, dynamic_smem_size>>>( gpu_x, gpu_y, gpu_z, gpu_cent_x, gpu_cent_y, gpu_cent_z,
            gpu_cluster_map, gpu_sum_x, gpu_sum_y, gpu_sum_z, 
            gpu_population, num_elements, num_k, gpu_change_flag
        );

        recalculate_cluster_centers<<<grid_clusters, block_size>>>( gpu_cent_x, gpu_cent_y, gpu_cent_z, gpu_sum_x, gpu_sum_y, gpu_sum_z, gpu_population, num_k);

        cudaMemcpy(&loop_status_flag, gpu_change_flag, sizeof(int), cudaMemcpyDeviceToHost);
        iteration_counter++;
    }

    int *gpu_hist_data, *gpu_cdf_data, *gpu_min_c_data;
    cudaMalloc(&gpu_hist_data, num_k * 256 * sizeof(int));
    cudaMalloc(&gpu_cdf_data, num_k * 256 * sizeof(int));
    cudaMalloc(&gpu_min_c_data, num_k * sizeof(int));
    cudaMemset(gpu_hist_data, 0, num_k * 256 * sizeof(int));

    construct_intensity_histograms<<<grid_elements, block_size>>>(gpu_in_I, gpu_cluster_map, gpu_hist_data, num_elements);
    calculate_cumulative_distribution<<<grid_clusters, block_size>>>(gpu_hist_data, gpu_cdf_data, gpu_min_c_data, num_k);
    finalize_intensity_mapping<<<grid_elements, block_size>>>(gpu_in_I, gpu_cluster_map, gpu_cdf_data, gpu_min_c_data, gpu_final_I, num_elements);

    cudaDeviceSynchronize();
    
    cudaMemcpy(host_out_I, gpu_final_I, num_elements * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(gpu_hist_data);
    cudaFree(gpu_cdf_data);
    cudaFree(gpu_min_c_data);
    cudaFree(gpu_sum_x);
    cudaFree(gpu_sum_y);
    cudaFree(gpu_sum_z);
    cudaFree(gpu_change_flag);
    cudaFree(gpu_population);
    cudaFree(gpu_cluster_map);
    cudaFree(gpu_cent_x);
    cudaFree(gpu_cent_y);
    cudaFree(gpu_cent_z);
    cudaFree(gpu_final_I);
    cudaFree(gpu_in_I);
    cudaFree(gpu_z);
    cudaFree(gpu_y);
    cudaFree(gpu_x);
}