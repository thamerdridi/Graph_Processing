#include <cuda_runtime.h>
#include <stdio.h>

#define cudaCheckError(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
    } \
}

__global__ void label_propagation_kernel(
    const int* row_offsets,
    const int* col_indices,
    const int* old_labels,
    int* new_labels,
    int* changed,
    int num_nodes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        int start = row_offsets[node];
        int end = row_offsets[node + 1];
        
        if (start == end) {
            new_labels[node] = old_labels[node];
            return;
        }
        
        int max_label = -1;
        int max_count = 0;
        
        for (int i = start; i < end; i++) {
            int neighbor = col_indices[i];
            int label = old_labels[neighbor];
            
            int count = 0;
            for (int j = start; j < end; j++) {
                if (old_labels[col_indices[j]] == label) count++;
            }
            
            if (count > max_count || (count == max_count && label < max_label)) {
                max_count = count;
                max_label = label;
            }
        }
        
        new_labels[node] = max_label;
        
        if (new_labels[node] != old_labels[node]) {
            *changed = 1;
        }
    }
}

extern "C" {
    void cuda_label_propagation(
        const int* h_row_offsets,
        const int* h_col_indices,
        int num_nodes,
        int num_edges,
        int* h_labels,
        int max_iter
    ) {
        int *d_row_offsets, *d_col_indices;
        int *d_old_labels, *d_new_labels, *d_changed;
        
        cudaCheckError(cudaMalloc(&d_row_offsets, (num_nodes + 1) * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_col_indices, num_edges * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_old_labels, num_nodes * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_new_labels, num_nodes * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_changed, sizeof(int)));
        
        cudaCheckError(cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_col_indices, h_col_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_old_labels, h_labels, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
        
        int threads = 256;
        int blocks = (num_nodes + threads - 1) / threads;
        
        for (int iter = 0; iter < max_iter; iter++) {
            int h_changed = 0;
            cudaCheckError(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
            
            label_propagation_kernel<<<blocks, threads>>>(
                d_row_offsets, d_col_indices, d_old_labels,
                d_new_labels, d_changed, num_nodes
            );
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());
            
            cudaCheckError(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            
            int* temp = d_old_labels;
            d_old_labels = d_new_labels;
            d_new_labels = temp;
            
            if (h_changed == 0) break;
        }
        
        cudaCheckError(cudaMemcpy(h_labels, d_old_labels, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_old_labels);
        cudaFree(d_new_labels);
        cudaFree(d_changed);
    }
}
