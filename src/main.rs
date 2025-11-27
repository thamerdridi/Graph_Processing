mod graph;
mod algorithms;
mod cuda;

use graph::{build_graph, GraphDirection, MyGraph, CSRGraph};
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

// CPU algorithm execution: returns starting nodes for BFS and Bellman-Ford
fn run_algorithms<Ty>(g: &Graph<String, i32, Ty>, choice: u8) -> (Option<String>, Option<String>)
where
    Ty: EdgeType,
{
    println!("\nGraph created.");
    println!("Number of nodes: {}", g.node_count());
    println!("Number of edges: {}", g.edge_count());

    let mut bfs_start_node = None;
    let mut bf_start_node = None;

    // BFS
    if choice == 1 || choice == 5 {
        println!("\nBFS - Enter starting node:");
        let mut bfs_start = String::new();
        io::stdin().read_line(&mut bfs_start).expect("Failed to read");
        let bfs_start = bfs_start.trim().to_string();
        bfs_start_node = Some(bfs_start.clone());

        let adj_list = algorithms::bfs::convert_to_adj_list(g);
        let t0 = Instant::now();
        let (bfs_distances, _) = algorithms::bfs::bfs(&adj_list, &bfs_start);
        let cpu_time = t0.elapsed();
        
        let reachable = bfs_distances.len();
        let max_dist = bfs_distances.values().max().unwrap_or(&0);
        
        println!("CPU time: {:?}", cpu_time);
        println!("Reachable nodes: {}, Max distance: {}", reachable, max_dist);
    }

    // Bellman-Ford
    if choice == 2 || choice == 5 {
        println!("\nBellman-Ford - Enter starting node:");
        let mut bf_start = String::new();
        io::stdin().read_line(&mut bf_start).expect("Failed to read");
        let bf_start = bf_start.trim().to_string();
        bf_start_node = Some(bf_start.clone());

        let t1 = Instant::now();
        let bf_outcome = algorithms::bellman_ford::bellman_ford(g, &bf_start);
        let cpu_time = t1.elapsed();

        if let Some((bf_distances, _)) = bf_outcome {
            let reachable = bf_distances.iter().filter(|(_, &d)| d != i32::MAX).count();
            println!("CPU time: {:?}", cpu_time);
            println!("Reachable nodes: {}", reachable);
        } else {
            println!("Failed: negative cycle or node not found");
        }
    }

    // PageRank
    if choice == 3 || choice == 5 {
        println!("\nPageRank:");
        let t2 = Instant::now();
        let pr = algorithms::page_rank::page_rank(g);
        let cpu_time = t2.elapsed();
        println!("CPU time: {:?}", cpu_time);

        let mut pr_sorted: Vec<_> = pr.iter().collect();
        pr_sorted.sort_by_key(|&(node, _)| node);
        println!("Top 3 nodes:");
        for (node, score) in pr_sorted.iter().take(3) {
            println!("  {} -> {:.6}", node, score);
        }
    }

    // Label Propagation
    if choice == 4 || choice == 5 {
        println!("\nLabel Propagation:");
        let t3 = Instant::now();
        let communities = algorithms::label_propagation::label_propagation(g);
        let cpu_time = t3.elapsed();
        println!("CPU time: {:?}", cpu_time);

        let mut community_groups: HashMap<String, Vec<String>> = HashMap::new();
        for (node, community) in communities.iter() {
            community_groups
                .entry(community.clone())
                .or_default()
                .push(node.clone());
        }
        println!("Communities detected: {}", community_groups.len());
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
    println!("\n=== GPU ===");
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
                println!("GPU time: {:?}", gpu_time);
                println!("Reachable nodes: {}, Max distance: {}", reachable, max_dist);
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
                println!("GPU time: {:?}", gpu_time);

                if result == -1 {
                    println!("Negative cycle detected!");
                } else {
                    let reachable = distances.iter().filter(|&&d| d != i32::MAX).count();
                    println!("Reachable nodes: {}", reachable);
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
        println!("GPU time: {:?}", gpu_time);

        let mut pr_with_labels: Vec<_> = ranks.iter().enumerate()
            .map(|(i, &r)| (csr_graph.get_node_label(i).unwrap(), r))
            .collect();
        pr_with_labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("Top 3 nodes:");
        for (node, score) in pr_with_labels.iter().take(3) {
            println!("  {} -> {:.6}", node, score);
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
        println!("GPU time: {:?}", gpu_time);

        let unique_labels: std::collections::HashSet<_> = labels.iter().collect();
        println!("Communities detected: {}", unique_labels.len());
    }
}

fn main() {
    println!("Graph direction (d/u):");
    let mut direction_input = String::new();
    io::stdin().read_line(&mut direction_input).expect("Failed to read");
    let direction = if direction_input.trim().eq_ignore_ascii_case("d") {
        GraphDirection::Directed
    } else {
        GraphDirection::Undirected
    };

    println!("\nEnter edges (NodeA-NodeB or NodeA-NodeB:Weight):");
    println!("Empty line to finish:\n");

    let mut input_lines = Vec::<String>::new();
    loop {
        print!("> ");
        io::stdout().flush().expect("Failed to flush");
        let mut line = String::new();
        io::stdin().read_line(&mut line).expect("Failed to read");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        input_lines.push(trimmed.to_string());
    }
    let input = input_lines.join("\n");

    let my_graph = build_graph(&input, direction);

    println!("\nAlgorithm: 1)BFS 2)Bellman-Ford 3)PageRank 4)LabelProp 5)All");
    print!("> ");
    io::stdout().flush().unwrap();
    let mut choice_str = String::new();
    io::stdin().read_line(&mut choice_str).unwrap();
    let choice = choice_str.trim().parse::<u8>().unwrap_or(5);

    // Execute on both CPU and GPU
    match &my_graph {
        MyGraph::Directed(g) => {
            println!("\n=== CPU ===");
            let (bfs_start, bf_start) = run_algorithms(g, choice);
            run_cuda_algorithms(g, choice, bfs_start, bf_start);
        }
        MyGraph::Undirected(g) => {
            println!("\n=== CPU ===");
            let (bfs_start, bf_start) = run_algorithms(g, choice);
            run_cuda_algorithms(g, choice, bfs_start, bf_start);
        }
    }
}
