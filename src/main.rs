mod graph;
mod algorithms;

use graph::{build_graph, GraphDirection, MyGraph};
use petgraph::graph::Graph;
use petgraph::EdgeType;
use std::collections::HashMap;
use std::io::{self, Write};

/// Reconstructs the shortest path from the predecessors map.
fn reconstruct_path(
    predecessors: &HashMap<String, String>,
    start: &str,
    target: &str,
) -> Option<Vec<String>> {
    let mut path = Vec::new();
    let mut current = target.to_string();
    path.push(current.clone());
    while current != start {
        if let Some(prev) = predecessors.get(&current) {
            current = prev.clone();
            path.push(current.clone());
        } else {
            return None;
        }
    }
    path.reverse();
    Some(path)
}

/// Generic helper to run both BFS and Bellman–Ford on the given graph.
fn run_algorithms<Ty>(g: &Graph<String, i32, Ty>)
where
    Ty: EdgeType,
{
    println!("\nGraph created.");
    println!("Number of nodes: {}", g.node_count());
    println!("Number of edges: {}", g.edge_count());

    // --- BFS Section ---
    println!("\nBFS:");
    println!("Enter the starting node label for BFS:");
    let mut bfs_start = String::new();
    io::stdin()
        .read_line(&mut bfs_start)
        .expect("Failed to read starting node");
    let bfs_start = bfs_start.trim();

    println!("Enter the target node label for BFS:");
    let mut bfs_target = String::new();
    io::stdin()
        .read_line(&mut bfs_target)
        .expect("Failed to read target node");
    let bfs_target = bfs_target.trim();

    // Convert the petgraph graph into a simple adjacency list.
    let adj_list = algorithms::bfs::convert_to_adj_list(g);
    let (bfs_distances, bfs_predecessors) = algorithms::bfs::bfs(&adj_list, bfs_start);

    if let Some(distance) = bfs_distances.get(bfs_target) {
        println!(
            "\nBFS: Shortest distance from '{}' to '{}' is: {}",
            bfs_start, bfs_target, distance
        );
        if let Some(path) = reconstruct_path(&bfs_predecessors, bfs_start, bfs_target) {
            println!("BFS: Shortest path: {:?}", path);
        } else {
            println!("BFS: Unable to reconstruct path.");
        }
    } else {
        println!(
            "BFS: Target node '{}' is not reachable from '{}'.",
            bfs_target, bfs_start
        );
    }
    if let Some(max_distance) = bfs_distances.values().max() {
        println!("BFS: Maximum number of edges crossed (BFS depth): {}", max_distance);
    }

    // --- Bellman–Ford Section ---
    println!("\nBellman–Ford:");
    println!("Enter the starting node label for Bellman–Ford:");
    let mut bf_start = String::new();
    io::stdin()
        .read_line(&mut bf_start)
        .expect("Failed to read starting node");
    let bf_start = bf_start.trim();

    println!("Enter the target node label for Bellman–Ford:");
    let mut bf_target = String::new();
    io::stdin()
        .read_line(&mut bf_target)
        .expect("Failed to read target node");
    let bf_target = bf_target.trim();

    if let Some((bf_distances, bf_predecessors)) = algorithms::bellman_ford::bellman_ford(g, bf_start)
    {
        if let Some(distance) = bf_distances.get(bf_target) {
            println!(
                "\nBellman–Ford: Shortest distance from '{}' to '{}' is: {}",
                bf_start, bf_target, distance
            );
            if let Some(path) = reconstruct_path(&bf_predecessors, bf_start, bf_target) {
                println!("Bellman–Ford: Shortest path: {:?}", path);
            } else {
                println!("Bellman–Ford: Unable to reconstruct path.");
            }
        } else {
            println!(
                "Bellman–Ford: Target node '{}' is not reachable from '{}'.",
                bf_target, bf_start
            );
        }
    } else {
        println!("Bellman–Ford failed (start node not found or negative cycle detected).");
    }
}

fn main() {
    // --- Graph Input Section ---
    println!("Enter graph direction (d for Directed, u for Undirected):");
    let mut direction_input = String::new();
    io::stdin()
        .read_line(&mut direction_input)
        .expect("Failed to read direction");

    let direction = if direction_input.trim().eq_ignore_ascii_case("d") {
        GraphDirection::Directed
    } else {
        GraphDirection::Undirected
    };

    println!("\nEnter your graph as an edge list. Each line should be in the format:");
    println!("  NodeA-NodeB       (for default weight of 1)");
    println!("  NodeA-NodeB:Weight (for an explicit weight)");
    println!("Enter an empty line when done:\n");

    let mut input_lines = Vec::new();
    loop {
        print!("> ");
        io::stdout().flush().expect("Failed to flush stdout");
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .expect("Failed to read line");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        input_lines.push(trimmed.to_string());
    }
    let input = input_lines.join("\n");

    println!("\nBuilding a {:?} graph from input:", direction);
    println!("{}", input);

    // --- Build the Graph ---
    let my_graph = build_graph(&input, direction);

    match &my_graph {
        MyGraph::Directed(g) => run_algorithms(g),
        MyGraph::Undirected(g) => run_algorithms(g),
    }
}
