#include <cuda_runtime.h>

__global__ void pagerank_kernel(
    const int* row_offsets,
    const int* col_indices,
    const float* old_ranks,
    float* new_ranks,
    const int* out_degrees,
    int num_nodes,
    float damping
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (node < num_nodes) {
        float sum = 0.0f;
        
        for (int src = 0; src < num_nodes; src++) {
            int start = row_offsets[src];
            int end = row_offsets[src + 1];
            
            for (int i = start; i < end; i++) {
                if (col_indices[i] == node) {
                    int deg = out_degrees[src];
                    if (deg > 0) sum += old_ranks[src] / (float)deg;
                    break;
                }
            }
        }
        
        new_ranks[node] = (1.0f - damping) / (float)num_nodes + damping * sum;
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
        int *d_row_offsets, *d_col_indices, *d_out_degrees;
        float *d_old_ranks, *d_new_ranks;
        
        cudaMalloc(&d_row_offsets, (num_nodes + 1) * sizeof(int));
        cudaMalloc(&d_col_indices, num_edges * sizeof(int));
        cudaMalloc(&d_out_degrees, num_nodes * sizeof(int));
        cudaMalloc(&d_old_ranks, num_nodes * sizeof(float));
        cudaMalloc(&d_new_ranks, num_nodes * sizeof(float));
        
        cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, h_col_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
        
        int* h_out_degrees = (int*)malloc(num_nodes * sizeof(int));
        for (int i = 0; i < num_nodes; i++) {
            h_out_degrees[i] = h_row_offsets[i + 1] - h_row_offsets[i];
        }
        cudaMemcpy(d_out_degrees, h_out_degrees, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
        free(h_out_degrees);
        
        float init_rank = 1.0f / num_nodes;
        float* h_init = (float*)malloc(num_nodes * sizeof(float));
        for (int i = 0; i < num_nodes; i++) h_init[i] = init_rank;
        cudaMemcpy(d_old_ranks, h_init, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
        free(h_init);
        
        int threads = 256;
        int blocks = (num_nodes + threads - 1) / threads;
        
        for (int iter = 0; iter < max_iter; iter++) {
            pagerank_kernel<<<blocks, threads>>>(
                d_row_offsets, d_col_indices, d_old_ranks, d_new_ranks,
                d_out_degrees, num_nodes, damp
            );
            cudaDeviceSynchronize();
            
            float* temp = d_old_ranks;
            d_old_ranks = d_new_ranks;
            d_new_ranks = temp;
        }
        
        cudaMemcpy(h_ranks, d_old_ranks, num_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_out_degrees);
        cudaFree(d_old_ranks);
        cudaFree(d_new_ranks);
    }
}