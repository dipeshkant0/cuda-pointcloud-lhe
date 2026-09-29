#include "kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>

__device__ inline bool tie_breaker_logic(int d1, int p1, int d2, int p2, const int* x, const int* y, const int* z) {
    if (d1 != d2) return d1 > d2;
    if (p1 == 2147483647) return true;
    if (p2 == 2147483647) return false;

    if (x[p1] != x[p2]) return x[p1] > x[p2];
    if (y[p1] != y[p2]) return y[p1] > y[p2];
    if (z[p1] != z[p2]) return z[p1] > z[p2];

    return p1 > p2;
}

__global__ void exact_knn(const int* dx, const int* dy, const int* dz, const uint8_t* dI, uint8_t* out, int n, int k) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int tx[128];
    __shared__ int ty[128];
    __shared__ int tz[128];
    __shared__ uint8_t ti[128];
    
    int sk = (k < 128) ? k : 128;

    int dist_h[128];
    int idx_h[128];
    uint8_t val_h[128];

    #pragma unroll
    for(int p = 0; p < 128; p++) { 
        dist_h[p] = 2147483647; idx_h[p] = 2147483647; val_h[p] = 0; 
    }

    int ax = 0, ay = 0, az = 0;
    if (tid < n) {
        ax = dx[tid]; ay = dy[tid]; az = dz[tid];
    }

    int max_d_reg = 2147483647;

    int total_tiles = (n + blockDim.x - 1) / blockDim.x;

    for (int t = 0; t < total_tiles; t++) {
        int load_idx = t * blockDim.x + threadIdx.x;

        if (load_idx < n) {
            tx[threadIdx.x] = dx[load_idx];
            ty[threadIdx.x] = dy[load_idx];
            tz[threadIdx.x] = dz[load_idx];
            ti[threadIdx.x] = dI[load_idx];
        }
        __syncthreads(); 

        if (tid < n) {
            int items = (n - t * blockDim.x);
            if (blockDim.x < items) items = blockDim.x;

            #pragma unroll 16
            for (int j = 0; j < items; j++) {
                int global_j = t * blockDim.x + j;
                if (tid == global_j) continue;

                int rx = ax - tx[j];
                int ry = ay - ty[j];
                int rz = az - tz[j];
                int d2 = (rx * rx) + (ry * ry) + (rz * rz);

                if (d2 < max_d_reg || (d2 == max_d_reg && tie_breaker_logic(max_d_reg, idx_h[0], d2, global_j, dx, dy, dz))) {
                    
                    dist_h[0] = d2; 
                    idx_h[0] = global_j; 
                    val_h[0] = ti[j];
                    
                    int root = 0;
                    while (true) {
                        int L = 2 * root + 1, R = 2 * root + 2, largest = root;
                        if (L < sk && tie_breaker_logic(dist_h[L], idx_h[L], dist_h[largest], idx_h[largest], dx, dy, dz)) largest = L;
                        if (R < sk && tie_breaker_logic(dist_h[R], idx_h[R], dist_h[largest], idx_h[largest], dx, dy, dz)) largest = R;

                        if (largest != root) {
                            int d_tmp = dist_h[root];
                            dist_h[root] = dist_h[largest];
                            dist_h[largest] = d_tmp;

                            int i_tmp = idx_h[root];
                            idx_h[root] = idx_h[largest];
                            idx_h[largest] = i_tmp;

                            uint8_t v_tmp = val_h[root];
                            val_h[root] = val_h[largest];
                            val_h[largest] = v_tmp;

                            root = largest; 
                        } else break;
                    }
                    max_d_reg = dist_h[0];
                }
            }
        }
        __syncthreads(); 
    }

    if (tid < n) {
        uint8_t self_i = dI[tid]; 
        int total_k = 1, rank = 1;          
        uint8_t min_v = self_i;

        for(int p = 0; p < sk; p++) {
            if (dist_h[p] != 2147483647) {
                total_k++;
                if (val_h[p] <= self_i) rank++;
                if (val_h[p] < min_v) min_v = val_h[p];
            }
        }

        int matches = (self_i == min_v) ? 1 : 0;
        for(int p = 0; p < sk; p++) {
            if (dist_h[p] != 2147483647 && val_h[p] == min_v) matches++;
        }

        if (total_k == matches) out[tid] = self_i;
        else {
            float mapped = ((float)(rank - matches) / (total_k - matches)) * 255.0f;
            out[tid] = (uint8_t)fminf(255.0f, fmaxf(0.0f, floorf(mapped)));
        }
    }
}

void KNN_local_histogram_equalization_cuda(const int* hx, const int* hy, const int* hz, const uint8_t* hI, int n, int k, uint8_t* hout) {
    int *px, *py, *pz;
    uint8_t *pi, *po;

    cudaMalloc(&px, n * sizeof(int)); cudaMalloc(&py, n * sizeof(int)); cudaMalloc(&pz, n * sizeof(int));
    cudaMalloc(&pi, n * sizeof(uint8_t)); cudaMalloc(&po, n * sizeof(uint8_t));

    cudaMemcpy(px, hx, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(py, hy, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pz, hz, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pi, hI, n * sizeof(uint8_t), cudaMemcpyHostToDevice);

    exact_knn<<< (n + 128 - 1) / 128, 128 >>>(px, py, pz, pi, po, n, k);
    
    cudaDeviceSynchronize();
    cudaMemcpy(hout, po, n * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(px);
    cudaFree(py);
    cudaFree(pz);
    cudaFree(pi);
    cudaFree(po);
}