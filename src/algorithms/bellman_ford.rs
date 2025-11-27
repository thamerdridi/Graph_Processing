use crate::graph::CSRGraph;
use std::collections::HashMap;

// Bellman-Ford using CSR: returns (distances, predecessors) or None if negative cycle detected
pub fn bellman_ford(csr: &CSRGraph, start: &str) -> Option<(HashMap<String, i32>, HashMap<String, String>)> {
    let start_idx = csr.get_node_idx(start)?;
    
    let mut distances = vec![i32::MAX; csr.num_nodes];
    let mut predecessors = vec![-1i32; csr.num_nodes];
    distances[start_idx] = 0;

    // Relax edges (num_nodes - 1) times
    for _ in 0..csr.num_nodes - 1 {
        let mut updated = false;
        for u in 0..csr.num_nodes {
            if distances[u] == i32::MAX {
                continue;
            }
            let start = csr.row_offsets[u] as usize;
            let end = csr.row_offsets[u + 1] as usize;
            
            for i in start..end {
                let v = csr.col_indices[i] as usize;
                let weight = csr.edge_weights[i];
                if distances[u] + weight < distances[v] {
                    distances[v] = distances[u] + weight;
                    predecessors[v] = u as i32;
                    updated = true;
                }
            }
        }
        if !updated {
            break;
        }
    }

    // Check for negative cycles
    for u in 0..csr.num_nodes {
        if distances[u] == i32::MAX {
            continue;
        }
        let start = csr.row_offsets[u] as usize;
        let end = csr.row_offsets[u + 1] as usize;
        
        for i in start..end {
            let v = csr.col_indices[i] as usize;
            let weight = csr.edge_weights[i];
            if distances[u] + weight < distances[v] {
                println!("Graph contains a negative weight cycle.");
                return None;
            }
        }
    }

    // Convert to HashMap with labels
    let mut result_distances: HashMap<String, i32> = HashMap::new();
    let mut result_predecessors: HashMap<String, String> = HashMap::new();
    
    for i in 0..csr.num_nodes {
        let label = csr.get_node_label(i).unwrap().clone();
        result_distances.insert(label.clone(), distances[i]);
        if predecessors[i] >= 0 {
            let pred_label = csr.get_node_label(predecessors[i] as usize).unwrap().clone();
            result_predecessors.insert(label, pred_label);
        }
    }

    Some((result_distances, result_predecessors))
}