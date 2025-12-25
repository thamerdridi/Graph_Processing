#include <cuda_runtime.h>
#include <stdio.h>

#define cudaCheckError(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
    } \
}

__global__ void bfs_kernel(
    const int* row_offsets,
    const int* col_indices,
    int* distances,
    int* current_frontier,
    int* next_frontier,
    int* frontier_size,
    int current_frontier_size,
    int current_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < current_frontier_size) {
        int node = current_frontier[tid];
        int start = row_offsets[node];
        int end = row_offsets[node + 1];
        
        for (int i = start; i < end; i++) {
            int neighbor = col_indices[i];
            if (atomicCAS(&distances[neighbor], -1, current_level + 1) == -1) {
                int pos = atomicAdd(frontier_size, 1);
                next_frontier[pos] = neighbor;
            }
        }
    }
}

extern "C" {
    void cuda_bfs(
        const int* h_row_offsets,
        const int* h_col_indices,
        int num_nodes,
        int num_edges,
        int source_node,
        int* h_distances
    ) {
        int *d_row_offsets, *d_col_indices, *d_distances;
        int *d_current_frontier, *d_next_frontier, *d_frontier_size;
        
        cudaCheckError(cudaMalloc(&d_row_offsets, (num_nodes + 1) * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_col_indices, num_edges * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_distances, num_nodes * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_current_frontier, num_nodes * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_next_frontier, num_nodes * sizeof(int)));
        cudaCheckError(cudaMalloc(&d_frontier_size, sizeof(int)));
        
        cudaCheckError(cudaMemcpy(d_row_offsets, h_row_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_col_indices, h_col_indices, num_edges * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemset(d_distances, -1, num_nodes * sizeof(int)));
        
        int zero = 0;
        cudaCheckError(cudaMemcpy(&d_distances[source_node], &zero, sizeof(int), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_current_frontier, &source_node, sizeof(int), cudaMemcpyHostToDevice));
        
        int current_frontier_size = 1;
        int current_level = 0;
        int threads = 256;
        
        while (current_frontier_size > 0) {
            int blocks = (current_frontier_size + threads - 1) / threads;
            cudaCheckError(cudaMemset(d_frontier_size, 0, sizeof(int)));
            
            bfs_kernel<<<blocks, threads>>>(
                d_row_offsets, d_col_indices, d_distances,
                d_current_frontier, d_next_frontier, d_frontier_size,
                current_frontier_size, current_level
            );
            cudaCheckError(cudaGetLastError());
            cudaCheckError(cudaDeviceSynchronize());
            
            cudaCheckError(cudaMemcpy(&current_frontier_size, d_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
            
            int* temp = d_current_frontier;
            d_current_frontier = d_next_frontier;
            d_next_frontier = temp;
            current_level++;
        }
        
        cudaCheckError(cudaMemcpy(h_distances, d_distances, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_distances);
        cudaFree(d_current_frontier);
        cudaFree(d_next_frontier);
        cudaFree(d_frontier_size);
    }
}
