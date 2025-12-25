#include <cuda_runtime.h>
#include <stdio.h>

#define cudaCheckError(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
    } \
}

// Push-based kernel: each thread pushes its rank contribution to neighbors
__global__ void pagerank_push_kernel(
    const int* row_offsets,
    const int* col_indices,
    const float* old_ranks,
    float* new_ranks,
    int num_nodes
) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (src < num_nodes) {
        int start = row_offsets[src];
        int end = row_offsets[src + 1];
        int out_degree = end - start;
        
        if (out_degree > 0) {
            float contribution = old_ranks[src] / (float)out_degree;
            
            for (int i = start; i < end; i++) {
                int dst = col_indices[i];
                atomicAdd(&new_ranks[dst], contribution);
            }
        }
    }
}

// Apply damping factor
__global__ void pagerank_damping_kernel(
    float* ranks,
    int num_nodes,
    float damping
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        ranks[node] = (1.0f - damping) / (float)num_nodes + damping * ranks[node];
    }
}

extern "C" {
    void cuda_pagerank(
        const int* h_row_offsets,
        const int* h_col_indices,
        int num_nodes,
        int num_edges,
        float* h_ranks,
        int max_iter,
        float tol,
        float damp
    ) {
        int *d_row_offsets, *d_col_indices;
        float *d_old_ranks, *d_new_ranks;
        
        cudaCheckError(cudaMalloc(&d_row_offsets, (num_nodes + 1) * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_col_indices, num_edges * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_old_ranks, num_nodes * sizeof(float)));
        cudaCheckError(cudaMalloc(&d_new_ranks, num_nodes * sizeof(float)));
        
        cudaCheckError(cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_col_indices, h_col_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice));
        
        // Initialize ranks to 1/N
        float init_rank = 1.0f / num_nodes;
        float* h_init = (float*)malloc(num_nodes * sizeof(float));
        for (int i = 0; i < num_nodes; i++) h_init[i] = init_rank;
        cudaCheckError(cudaMemcpy(d_old_ranks, h_init, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
        free(h_init);
        
        int threads = 256;
        int blocks = (num_nodes + threads - 1) / threads;
        
        for (int iter = 0; iter < max_iter; iter++) {
            // Reset new_ranks to 0
            cudaCheckError(cudaMemset(d_new_ranks, 0, num_nodes * sizeof(float)));
            
            // Push contributions
            pagerank_push_kernel<<<blocks, threads>>>(
                d_row_offsets, d_col_indices, d_old_ranks, d_new_ranks, num_nodes
            );
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());
            
            // Apply damping
            pagerank_damping_kernel<<<blocks, threads>>>(d_new_ranks, num_nodes, damp);
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());
            
            // Swap
            float* temp = d_old_ranks;
            d_old_ranks = d_new_ranks;
            d_new_ranks = temp;
        }
        
        cudaCheckError(cudaMemcpy(h_ranks, d_old_ranks, num_nodes * sizeof(float), cudaMemcpyDeviceToHost));
        
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_old_ranks);
        cudaFree(d_new_ranks);
    }
}
