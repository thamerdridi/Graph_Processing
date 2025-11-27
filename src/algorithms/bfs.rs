use crate::graph::CSRGraph;
use std::collections::{HashMap, VecDeque};

// BFS using CSR format: returns (distances, predecessors)
pub fn bfs(csr: &CSRGraph, start: &str) -> (HashMap<String, i32>, HashMap<String, String>) {
    let start_idx = match csr.get_node_idx(start) {
        Some(idx) => idx,
        None => return (HashMap::new(), HashMap::new()),
    };

    let mut distances = vec![-1i32; csr.num_nodes];
    let mut predecessor = vec![-1i32; csr.num_nodes];
    let mut queue: VecDeque<usize> = VecDeque::new();

    distances[start_idx] = 0;
    queue.push_back(start_idx);

    while let Some(current) = queue.pop_front() {
        let current_distance = distances[current];
        let start = csr.row_offsets[current] as usize;
        let end = csr.row_offsets[current + 1] as usize;

        for i in start..end {
            let neighbor = csr.col_indices[i] as usize;
            if distances[neighbor] == -1 {
                distances[neighbor] = current_distance + 1;
                predecessor[neighbor] = current as i32;
                queue.push_back(neighbor);
            }
        }
    }

    // Convert to HashMap with labels
    let mut result_distances: HashMap<String, i32> = HashMap::new();
    let mut result_predecessors: HashMap<String, String> = HashMap::new();
    
    for i in 0..csr.num_nodes {
        if distances[i] >= 0 {
            let label = csr.get_node_label(i).unwrap().clone();
            result_distances.insert(label.clone(), distances[i]);
            if predecessor[i] >= 0 {
                let pred_label = csr.get_node_label(predecessor[i] as usize).unwrap().clone();
                result_predecessors.insert(label, pred_label);
            }
        }
    }

    (result_distances, result_predecessors)
}