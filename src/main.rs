mod graph;
mod algorithms;

use graph::CSRGraph;
use petgraph::graph::Graph;
use petgraph::EdgeType;
use std::collections::HashMap;
use std::io::{self, Write};
use std::time::Instant;

// FFI declarations for CUDA kernels
extern "C" {
    fn cuda_bfs(
        row_offsets: *const i32,
        col_indices: *const i32,
        num_nodes: i32,
        num_edges: i32,
        source_node: i32,
        distances: *mut i32,
    );

    fn cuda_bellman_ford(
        row_offsets: *const i32,
        col_indices: *const i32,
        edge_weights: *const i32,
        num_nodes: i32,
        num_edges: i32,
        source_node: i32,
        distances: *mut i32,
    ) -> i32;

    fn cuda_pagerank(
        row_offsets: *const i32,
        col_indices: *const i32,
        num_nodes: i32,
        num_edges: i32,
        ranks: *mut f32,
        max_iterations: i32,
        tolerance: f32,
        damping_factor: f32,
    );

    fn cuda_label_propagation(
        row_offsets: *const i32,
        col_indices: *const i32,
        num_nodes: i32,
        num_edges: i32,
        labels: *mut i32,
        max_iterations: i32,
    );
}

// CPU algorithm execution using CSR
fn run_algorithms<Ty>(g: &Graph<String, i32, Ty>, choice: u8) -> (Option<String>, Option<String>)
where
    Ty: EdgeType,
{
    // Convert to CSR for CPU algorithms
    let csr_graph = CSRGraph::from_graph(g);

    let start_node = "0".to_string();
    let bfs_start_node;
    let bf_start_node;

    // BFS
    if choice == 1 || choice == 5 {
        bfs_start_node = Some(start_node.clone());

        let t0 = Instant::now();
        let (bfs_distances, _) = algorithms::bfs::bfs(&csr_graph, &start_node);
        let cpu_time = t0.elapsed();
        
        let reachable = bfs_distances.len();
        let max_dist = bfs_distances.values().max().unwrap_or(&0);
        
        println!("\nBFS:");
        println!("  CPU time: {:?}", cpu_time);
        println!("  Reachable: {} nodes, Max distance: {}", reachable, max_dist);
    } else {
        bfs_start_node = None;
    }

    // Bellman-Ford
    if choice == 2 || choice == 5 {
        bf_start_node = Some(start_node.clone());

        let t1 = Instant::now();
        let bf_outcome = algorithms::bellman_ford::bellman_ford(&csr_graph, &start_node);
        let cpu_time = t1.elapsed();

        if let Some((bf_distances, _)) = bf_outcome {
            let reachable = bf_distances.iter().filter(|(_, &d)| d != i32::MAX).count();
            println!("\nBellman-Ford:");
            println!("  CPU time: {:?}", cpu_time);
            println!("  Reachable: {} nodes", reachable);
        } else {
            println!("\nBellman-Ford: Failed (negative cycle)");
        }
    } else {
        bf_start_node = None;
    }

    // PageRank
    if choice == 3 || choice == 5 {
        let t2 = Instant::now();
        let pr = algorithms::page_rank::page_rank(&csr_graph);
        let cpu_time = t2.elapsed();

        let mut pr_sorted: Vec<_> = pr.iter().collect();
        pr_sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        println!("\nPageRank:");
        println!("  CPU time: {:?}", cpu_time);
        println!("  Top 3 nodes:");
        for (node, score) in pr_sorted.iter().take(3) {
            println!("    {} -> {:.6}", node, score);
        }
    }

    // Label Propagation
    if choice == 4 || choice == 5 {
        let t3 = Instant::now();
        let communities = algorithms::label_propagation::label_propagation(&csr_graph);
        let cpu_time = t3.elapsed();

        let mut community_groups: HashMap<String, Vec<String>> = HashMap::new();
        for (node, community) in communities.iter() {
            community_groups
                .entry(community.clone())
                .or_default()
                .push(node.clone());
        }
        
        println!("\nLabel Propagation:");
        println!("  CPU time: {:?}", cpu_time);
        println!("  Communities: {}", community_groups.len());
    }

    (bfs_start_node, bf_start_node)
}

// GPU algorithm execution
fn run_cuda_algorithms<Ty>(
    g: &Graph<String, i32, Ty>,
    choice: u8,
    bfs_start: Option<String>,
    bf_start: Option<String>,
) where
    Ty: EdgeType,
{
    println!("\n=== GPU Results ===");
    let csr_graph = CSRGraph::from_graph(g);

    // BFS
    if choice == 1 || choice == 5 {
        if let Some(start_node) = bfs_start {
            if let Some(source_idx) = csr_graph.get_node_idx(&start_node) {
                let mut distances = vec![-1i32; csr_graph.num_nodes];
                
                let t0 = Instant::now();
                unsafe {
                    cuda_bfs(
                        csr_graph.row_offsets.as_ptr(),
                        csr_graph.col_indices.as_ptr(),
                        csr_graph.num_nodes as i32,
                        csr_graph.num_edges as i32,
                        source_idx as i32,
                        distances.as_mut_ptr(),
                    );
                }
                let gpu_time = t0.elapsed();

                let reachable = distances.iter().filter(|&&d| d >= 0).count();
                let max_dist = distances.iter().filter(|&&d| d >= 0).max().unwrap_or(&0);
                
                println!("\nBFS:");
                println!("  GPU time: {:?}", gpu_time);
                println!("  Reachable: {} nodes, Max distance: {}", reachable, max_dist);
            }
        }
    }

    // Bellman-Ford
    if choice == 2 || choice == 5 {
        if let Some(start_node) = bf_start {
            if let Some(source_idx) = csr_graph.get_node_idx(&start_node) {
                let mut distances = vec![i32::MAX; csr_graph.num_nodes];
                
                let t1 = Instant::now();
                let result = unsafe {
                    cuda_bellman_ford(
                        csr_graph.row_offsets.as_ptr(),
                        csr_graph.col_indices.as_ptr(),
                        csr_graph.edge_weights.as_ptr(),
                        csr_graph.num_nodes as i32,
                        csr_graph.num_edges as i32,
                        source_idx as i32,
                        distances.as_mut_ptr(),
                    )
                };
                let gpu_time = t1.elapsed();

                println!("\nBellman-Ford:");
                println!("  GPU time: {:?}", gpu_time);
                if result == -1 {
                    println!("  Negative cycle detected");
                } else {
                    let reachable = distances.iter().filter(|&&d| d != i32::MAX).count();
                    println!("  Reachable: {} nodes", reachable);
                }
            }
        }
    }

    // PageRank
    if choice == 3 || choice == 5 {
        let mut ranks = vec![0.0f32; csr_graph.num_nodes];
        
        let t2 = Instant::now();
        unsafe {
            cuda_pagerank(
                csr_graph.row_offsets.as_ptr(),
                csr_graph.col_indices.as_ptr(),
                csr_graph.num_nodes as i32,
                csr_graph.num_edges as i32,
                ranks.as_mut_ptr(),
                100,
                1e-6,
                0.85,
            );
        }
        let gpu_time = t2.elapsed();

        let mut pr_with_labels: Vec<_> = ranks.iter().enumerate()
            .map(|(i, &r)| (csr_graph.get_node_label(i).unwrap(), r))
            .collect();
        pr_with_labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("\nPageRank:");
        println!("  GPU time: {:?}", gpu_time);
        println!("  Top 3 nodes:");
        for (node, score) in pr_with_labels.iter().take(3) {
            println!("    {} -> {:.6}", node, score);
        }
    }

    // Label Propagation
    if choice == 4 || choice == 5 {
        let mut labels: Vec<i32> = (0..csr_graph.num_nodes as i32).collect();
        
        let t3 = Instant::now();
        unsafe {
            cuda_label_propagation(
                csr_graph.row_offsets.as_ptr(),
                csr_graph.col_indices.as_ptr(),
                csr_graph.num_nodes as i32,
                csr_graph.num_edges as i32,
                labels.as_mut_ptr(),
                100,
            );
        }
        let gpu_time = t3.elapsed();

        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        
        println!("\nLabel Propagation:");
        println!("  GPU time: {:?}", gpu_time);
        println!("  Communities: {}", unique_labels.len());
    }
}

fn generate_random_graph(num_nodes: usize) -> Graph<String, i32, petgraph::Directed> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut g = Graph::new();
    
    // Add nodes
    let nodes: Vec<_> = (0..num_nodes)
        .map(|i| g.add_node(i.to_string()))
        .collect();
    
    // Add random edges (average degree ~15)
    let avg_degree = 15;
    let num_edges = (num_nodes * avg_degree).min(num_nodes * num_nodes / 10);
    
    for _ in 0..num_edges {
        let from = rng.gen_range(0..num_nodes);
        let to = rng.gen_range(0..num_nodes);
        if from != to {
            let weight = rng.gen_range(1..11);
            g.add_edge(nodes[from], nodes[to], weight);
        }
    }
    
    g
}

fn main() {
    println!("=== Graph Processing: CPU vs GPU ===\n");
    
    // Get number of nodes
    println!("Enter number of nodes:");
    print!("> ");
    io::stdout().flush().expect("Failed to flush");
    let mut nodes_str = String::new();
    io::stdin().read_line(&mut nodes_str).expect("Failed to read");
    let num_nodes = nodes_str.trim().parse::<usize>().unwrap_or(1000);
    
    // Generate random directed graph
    println!("\nGenerating random graph with {} nodes...", num_nodes);
    let g = generate_random_graph(num_nodes);
    println!("Graph created: {} nodes, {} edges\n", g.node_count(), g.edge_count());
    
    // Select algorithm
    println!("Select algorithm:");
    println!("1) BFS");
    println!("2) Bellman-Ford");
    println!("3) PageRank");
    println!("4) Label Propagation");
    println!("5) All algorithms");
    print!("> ");
    io::stdout().flush().unwrap();
    let mut choice_str = String::new();
    io::stdin().read_line(&mut choice_str).unwrap();
    let choice = choice_str.trim().parse::<u8>().unwrap_or(5);
    
    // Run algorithms on CPU
    println!("\n=== CPU Results ===");
    let cpu_start = std::time::Instant::now();
    let (bfs_start, bf_start) = run_algorithms(&g, choice);
    let cpu_total = cpu_start.elapsed();
    
    // Run algorithms on GPU
    let gpu_start = std::time::Instant::now();
    run_cuda_algorithms(&g, choice, bfs_start, bf_start);
    let gpu_total = gpu_start.elapsed();
    
    // Performance Summary
    println!("\n=== Performance Summary ===");
    println!("Total CPU time: {:?}", cpu_total);
    println!("Total GPU time: {:?}", gpu_total);
    if gpu_total.as_secs_f64() > 0.0 {
        let speedup = cpu_total.as_secs_f64() / gpu_total.as_secs_f64();
        if speedup > 1.0 {
            println!("GPU Speedup: {:.2}x faster", speedup);
        }
    }
}
