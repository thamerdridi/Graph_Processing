# Graph Processing: CPU vs GPU

University project comparing CPU and GPU performance for graph algorithms using Rust and CUDA.

## Overview

Implements 4 graph algorithms with both CPU (Rust) and GPU (CUDA) versions:
- **BFS** - Breadth-First Search
- **Bellman-Ford** - Shortest paths with negative weights
- **PageRank** - Node ranking algorithm
- **Label Propagation** - Community detection

## Requirements

- Rust (latest stable)
- CUDA Toolkit 12.0+
- NVIDIA GPU (compute capability 5.0+)
- GCC/G++

## Build & Run

```bash
cargo build --release
./target/release/Graph_Processing
```

## Architecture

**Graph Representation:**
- CSR (Compressed Sparse Row) format
- Unified data structure for both CPU and GPU
- Efficient memory layout for parallel processing

**Components:**
- **CPU Algorithms**: Rust implementations using CSR
- **GPU Algorithms**: CUDA kernels with parallel execution
- **FFI Interface**: Direct C interface between Rust and CUDA
- **Build System**: Automatic CUDA compilation via build.rs

## Project Structure

```
Graph_Processing/
├── Cargo.toml          # Dependencies (petgraph, rand, libc)
├── build.rs            # CUDA kernel compilation
└── src/
    ├── main.rs         # Entry point, random generation
    ├── graph.rs        # CSR graph structure
    ├── algorithms/     # CPU implementations (Rust)
    └── cuda/kernels/   # GPU implementations (CUDA)
```

## How It Works

1. **Input**: User enters number of nodes
2. **Generation**: Random directed graph created (~15 edges/node)
3. **Conversion**: Graph converted to CSR format
4. **Execution**: Algorithm runs on both CPU and GPU
5. **Results**: Shows timing and speedup comparison

## Flow Example

```
$ ./target/release/Graph_Processing

Enter number of nodes:
> 5000

Select algorithm:
1) BFS
2) Bellman-Ford
3) PageRank
4) Label Propagation
5) All algorithms
> 3

Generating random graph with 5000 nodes...
Graph created: 5000 nodes, 74977 edges

=== CPU Results ===
PageRank:
  CPU time: 13.23s
  Top 3 nodes:
    3139 -> 0.000380
    2696 -> 0.000369
    782 -> 0.000364

=== GPU Results ===
PageRank:
  GPU time: 2.32s
  Top 3 nodes:
    3139 -> 0.000380
    2696 -> 0.000369
    782 -> 0.000364

=== Performance Summary ===
Total CPU time: 13.25s
Total GPU time: 2.33s
GPU Speedup: 5.70x faster
```
