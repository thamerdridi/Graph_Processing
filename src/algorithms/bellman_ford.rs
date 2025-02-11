use petgraph::graph::{Graph, NodeIndex};
use petgraph::EdgeType;
use std::collections::HashMap;
use petgraph::visit::EdgeRef;
use std::i32;

/// Runs the Bellman–Ford algorithm on the given graph starting from the node with label `start`.
pub fn bellman_ford<Ty>(
    graph: &Graph<String, i32, Ty>,
    start: &str,
) -> Option<(HashMap<String, i32>, HashMap<String, String>)>
where
    Ty: EdgeType,
{
    // Initialize distances: set all to i32::MAX (infinity).
    let mut distances: HashMap<NodeIndex, i32> = HashMap::new();
    let mut predecessors: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    for node in graph.node_indices() {
        distances.insert(node, i32::MAX);
    }
    
    // Find the start node by label.
    let start_node = graph.node_indices().find(|&node| graph[node] == start)?;
    distances.insert(start_node, 0);

    let num_nodes = graph.node_count();

    // Relax edges (num_nodes - 1) times.
    for _ in 0..num_nodes - 1 {
        for edge in graph.edge_references() {
            let u = edge.source();
            let v = edge.target();
            let weight = *edge.weight();
            if graph.is_directed() {
                if distances[&u] != i32::MAX && distances[&u] + weight < distances[&v] {
                    distances.insert(v, distances[&u] + weight);
                    predecessors.insert(v, u);
                }
            } else {
                // For undirected, relax in both directions.
                if distances[&u] != i32::MAX && distances[&u] + weight < distances[&v] {
                    distances.insert(v, distances[&u] + weight);
                    predecessors.insert(v, u);
                }
                if distances[&v] != i32::MAX && distances[&v] + weight < distances[&u] {
                    distances.insert(u, distances[&v] + weight);
                    predecessors.insert(u, v);
                }
            }
        }
    }

    // Check for negative weight cycles.
    for edge in graph.edge_references() {
        let u = edge.source();
        let v = edge.target();
        let weight = *edge.weight();
        if graph.is_directed() {
            if distances[&u] != i32::MAX && distances[&u] + weight < distances[&v] {
                println!("Graph contains a negative weight cycle.");
                return None;
            }
        } else {
            if distances[&u] != i32::MAX && distances[&u] + weight < distances[&v] {
                println!("Graph contains a negative weight cycle.");
                return None;
            }
            if distances[&v] != i32::MAX && distances[&v] + weight < distances[&u] {
                println!("Graph contains a negative weight cycle.");
                return None;
            }
        }
    }

    // Convert the results from NodeIndex to node labels.
    let mut result_distances: HashMap<String, i32> = HashMap::new();
    let mut result_predecessors: HashMap<String, String> = HashMap::new();
    for node in graph.node_indices() {
        result_distances.insert(graph[node].clone(), *distances.get(&node).unwrap());
        if let Some(&pred) = predecessors.get(&node) {
            result_predecessors.insert(graph[node].clone(), graph[pred].clone());
        }
    }

    Some((result_distances, result_predecessors))
}