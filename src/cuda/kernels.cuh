#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <stdint.h>

void KNN_local_histogram_equalization_cuda(const int* h_x, const int* h_y, const int* h_z, const uint8_t* h_I, int n, int k, uint8_t* h_output_I);
void approx_knn_histogram_equalization_cuda(const int* h_x, const int* h_y, const int* h_z, const uint8_t* h_I, int n, int k, uint8_t* h_output_I);
void k_mean_local_histogram_equalization_cuda(const int* h_x, const int* h_y, const int* h_z, const uint8_t* h_I, int n, int k,int t_max, uint8_t* h_output_I);

#endif