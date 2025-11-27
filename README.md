# CUDA-Accelerated Graph Processing in Rust

A high-performance graph processing library that implements popular graph algorithms with both CPU and GPU (CUDA) execution, enabling direct performance comparison between traditional and GPU-accelerated approaches.

## Overview

This project demonstrates the implementation of four fundamental graph algorithms:
- **BFS (Breadth-First Search)** - Graph traversal for finding shortest paths in unweighted graphs
- **Bellman-Ford** - Single-source shortest paths algorithm supporting negative edge weights
- **PageRank** - Iterative algorithm for ranking nodes based on link structure
- **Label Propagation** - Community detection through neighbor voting

Each algorithm is implemented twice: once in pure Rust for CPU execution, and once as a CUDA kernel for GPU execution. The program runs both versions automatically, allowing for direct performance comparison.

## Architecture

### Project Structure

```
Graph_Processing/
├── Cargo.toml              # Rust dependencies (petgraph)
├── build.rs                # CUDA build script (compiles .cu files)
├── src/
│   ├── main.rs            # Entry point, FFI declarations, user interface
│   ├── graph.rs           # Graph structures (CSR format for GPU)
│   ├── algorithms/        # CPU implementations
│   │   ├── bfs.rs         # BFS using adjacency list
│   │   ├── bellman_ford.rs
│   │   ├── page_rank.rs
│   │   └── label_propagation.rs
│   └── cuda/
│       ├── mod.rs         # Module declaration
│       └── kernels/       # CUDA implementations
│           ├── bfs.cu     # Level-synchronous BFS
│           ├── bellman_ford.cu
│           ├── pagerank.cu
│           └── label_propagation.cu
└── target/                # Build output
```

### Key Components

**1. Build System (`build.rs`)**
- Compiles CUDA kernels using `nvcc` compiler
- Links compiled objects into static library
- Configured for sm_86 architecture (Ampere GPUs)

**2. CPU Algorithms (`src/algorithms/`)**
- Pure Rust implementations using `petgraph` library
- Use standard graph data structures (adjacency lists, hash maps)
- Optimized for small to medium graphs

**3. GPU Kernels (`src/cuda/kernels/`)**
- Written in CUDA C++
- Use Compressed Sparse Row (CSR) format for efficient GPU memory access
- Implement parallel algorithms with atomic operations for synchronization

**4. FFI Bridge (`src/main.rs`)**
- Rust `extern "C"` declarations for CUDA functions
- Unsafe calls to GPU kernels with proper memory management
- Converts between Rust graph structures and CSR format

## Requirements

- **NVIDIA GPU** with compute capability >= 5.0 (Tested: RTX 3050 Ti)
- **CUDA Toolkit 12.0+**
- **Rust toolchain** (latest stable)
- **Linux** (Ubuntu 24.04 tested)

## Setup & Build

### 1. Install CUDA Toolkit

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

Verify:
```bash
nvcc --version
nvidia-smi
```

### 2. Build the Project

```bash
cargo build --release
```

This will:
1. Compile CUDA kernels to object files (.o)
2. Archive them into `libcuda_kernels.a`
3. Link with Rust code
4. Produce optimized executable at `./target/release/Graph_Processing`

## How It Works

### Execution Flow

When you run the program, it follows this simple flow:

1. **Graph Input**: Enter edges in format `NodeA-NodeB` or `NodeA-NodeB:Weight`
2. **Algorithm Selection**: Choose from BFS, Bellman-Ford, PageRank, Label Propagation, or All
3. **Dual Execution**: Program automatically runs the algorithm on both CPU and GPU
4. **Results Display**: Shows results and execution times for comparison

### Example Flow

```bash
$ ./target/release/Graph_Processing

Enter graph direction (d for Directed, u for Undirected):
d

Enter your graph as an edge list (format: 'A-B' or 'A-B:weight'). End with empty line:
0-1
0-2
1-3
2-3
[press Enter on empty line]

Select algorithm to run:
  1) BFS (Breadth-First Search)
  2) Bellman–Ford (Shortest Paths)
  3) PageRank
  4) Label Propagation (Community Detection)
  5) All algorithms
> 1

=== CPU BFS ===
Starting from node: 0
Distance to 0: 0
Distance to 1: 1
Distance to 2: 1
Distance to 3: 2
Max distance: 2
CPU Time: 5.23 µs

=== CUDA BFS ===
Starting from node: 0
Distance to 0: 0
Distance to 1: 1
Distance to 2: 1
Distance to 3: 2
Max distance: 2
GPU Time: 1.847 s
```

**Note**: GPU shows higher time for small graphs due to memory transfer overhead. GPU becomes faster with graphs of 10,000+ nodes.

### Data Flow

```
User Input (edges)
        ↓
[Graph Construction]
  - CPU: petgraph adjacency list
  - GPU: CSR format conversion
        ↓
[Algorithm Execution]
  CPU Thread          GPU Thread
      ↓                   ↓
  Rust impl         CUDA kernel
  (algorithms/)     (cuda/kernels/)
      ↓                   ↓
[Results Collection]
        ↓
[Display & Compare]
```

### CSR Graph Format (GPU)

The GPU uses **Compressed Sparse Row (CSR)** format for efficient parallel access:

```
Graph: 0→1, 0→2, 1→3, 2→3

CSR Representation:
  row_offsets: [0, 2, 3, 4, 4]  (where each node's edges start)
  col_indices: [1, 2, 3, 3]     (destination nodes)
  edge_weights: [1, 1, 1, 1]    (optional weights)
```

Benefits:
- Coalesced memory access (threads read adjacent memory)
- O(1) neighbor lookup
- Minimal memory overhead

## Implementation Details

### CUDA Kernels

Each algorithm is optimized for GPU execution:

1. **BFS (`bfs.cu`)**: Level-synchronous with frontier queues, atomic operations for distance updates
2. **Bellman-Ford (`bellman_ford.cu`)**: Edge relaxation with atomicMin, early termination detection
3. **PageRank (`pagerank.cu`)**: SpMV pattern with CSC format, shared memory for convergence
4. **Label Propagation (`label_propagation.cu`)**: Neighbor voting with atomic label updates

### Build System (`build.rs`)

**Essential component** - cannot be removed:
- Compiles CUDA kernels (`.cu` files) with `nvcc` compiler
- Creates static library `libcuda_kernels.a` from compiled objects
- Links with CUDA runtime (`cudart`)
- Configured for sm_86 architecture (RTX 30 series)

Without `build.rs`, CUDA kernels won't compile and the project won't build.

### FFI Bridge

Located in `src/main.rs`:
```rust
extern "C" {
    fn cuda_bfs(...) -> i32;
    fn cuda_bellman_ford(...) -> i32;
    fn cuda_pagerank(...) -> i32;
    fn cuda_label_propagation(...) -> i32;
}
```

Direct FFI approach:
- No wrapper layers for simplicity
- Unsafe blocks handle raw pointers
- Automatic conversion between Rust graphs and CSR format

## Performance Notes

### GPU vs CPU Trade-offs

**GPU is slower for small graphs** (< 10,000 nodes):
- Kernel launch overhead: ~1-2ms
- Memory transfer overhead dominates computation time
- CPU direct memory access is faster

**GPU becomes faster for large graphs** (> 100,000 nodes):
- Parallel processing outweighs overhead
- Memory bandwidth advantage on large datasets
- Especially beneficial for iterative algorithms (PageRank, Label Propagation)

Example timing on RTX 3050 Ti:
- 4 nodes: CPU ~5µs, GPU ~1.8s (CPU wins)
- 1,000 nodes: CPU ~2ms, GPU ~1.8s (CPU wins)
- 10,000 nodes: CPU ~40ms, GPU starts to compete
- 1,000,000+ nodes: GPU significantly faster

## Troubleshooting

**Build Error**: `nvcc not found`
```bash
sudo apt install nvidia-cuda-toolkit
```

**Build Error**: Wrong GPU architecture
Edit `build.rs` and change architecture flag:
- RTX 20 series: `-arch=sm_75`
- RTX 30 series: `-arch=sm_86`
- RTX 40 series: `-arch=sm_89`

**Runtime Error**: CUDA out of memory
- Reduce graph size or use smaller batch

## License

MIT
