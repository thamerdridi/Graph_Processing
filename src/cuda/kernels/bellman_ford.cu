#include <cuda_runtime.h>

#define INF 2147483647

__global__ void bellman_ford_kernel(
    const int* row_offsets,
    const int* col_indices,
    const int* edge_weights,
    int* distances,
    int* updated,
    int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int node = tid; node < num_nodes; node += stride) {
        int current_dist = distances[node];
        if (current_dist == INF) continue;
        
        int start = row_offsets[node];
        int end = row_offsets[node + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = col_indices[i];
            int new_dist = current_dist + edge_weights[i];
            int old_dist = atomicMin(&distances[neighbor], new_dist);
            if (new_dist < old_dist) *updated = 1;
        }
    }
}

extern "C" {
    int cuda_bellman_ford(
        const int* h_row_offsets,
        const int* h_col_indices,
        const int* h_edge_weights,
        int num_nodes,
        int num_edges,
        int source_node,
        int* h_distances
    ) {
        int *d_row_offsets, *d_col_indices, *d_edge_weights, *d_distances, *d_updated;
        
        cudaMalloc(&d_row_offsets, (num_nodes + 1) * sizeof(int));
        cudaMalloc(&d_col_indices, num_edges * sizeof(int));
        cudaMalloc(&d_edge_weights, num_edges * sizeof(int));
        cudaMalloc(&d_distances, num_nodes * sizeof(int));
        cudaMalloc(&d_updated, sizeof(int));
        
        cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, h_col_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_edge_weights, h_edge_weights, num_edges * sizeof(int), cudaMemcpyHostToDevice);
        
        int* h_inf = (int*)malloc(num_nodes * sizeof(int));
        for (int i = 0; i < num_nodes; i++) h_inf[i] = INF;
        h_inf[source_node] = 0;
        cudaMemcpy(d_distances, h_inf, num_nodes * sizeof(int), cudaMemcpyHostToDevice);
        free(h_inf);
        
        int threads = 256;
        int blocks = (num_nodes + threads - 1) / threads;
        
        for (int iter = 0; iter < num_nodes - 1; iter++) {
            int h_updated = 0;
            cudaMemcpy(d_updated, &h_updated, sizeof(int), cudaMemcpyHostToDevice);
            
            bellman_ford_kernel<<<blocks, threads>>>(
                d_row_offsets, d_col_indices, d_edge_weights,
                d_distances, d_updated, num_nodes
            );
            
            cudaDeviceSynchronize();
            cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_updated == 0) break;
        }
        
        cudaMemcpy(h_distances, d_distances, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_edge_weights);
        cudaFree(d_distances);
        cudaFree(d_updated);
        
        return 0;
    }
}