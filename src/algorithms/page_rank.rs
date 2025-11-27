use crate::graph::CSRGraph;
use std::collections::HashMap;

// PageRank using CSR: iterative algorithm with damping factor 0.85
pub fn page_rank(csr: &CSRGraph) -> HashMap<String, f64> {
    if csr.num_nodes == 0 {
        return HashMap::new();
    }

    let d = 0.85;
    let init_rank = 1.0 / csr.num_nodes as f64;
    let mut ranks = vec![init_rank; csr.num_nodes];
    let mut new_ranks = vec![0.0; csr.num_nodes];
    
    // Compute out-degrees
    let mut out_degrees = vec![0; csr.num_nodes];
    for i in 0..csr.num_nodes {
        out_degrees[i] = (csr.row_offsets[i + 1] - csr.row_offsets[i]) as usize;
    }

    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        let mut diff = 0.0;

        for node in 0..csr.num_nodes {
            let mut rank_sum = 0.0;
            
            // Find incoming edges (nodes that point to this node)
            for src in 0..csr.num_nodes {
                let start = csr.row_offsets[src] as usize;
                let end = csr.row_offsets[src + 1] as usize;
                
                for i in start..end {
                    if csr.col_indices[i] as usize == node {
                        if out_degrees[src] > 0 {
                            rank_sum += ranks[src] / out_degrees[src] as f64;
                        }
                        break;
                    }
                }
            }

            new_ranks[node] = (1.0 - d) / (csr.num_nodes as f64) + d * rank_sum;
            diff += (new_ranks[node] - ranks[node]).abs();
        }

        ranks.copy_from_slice(&new_ranks);
        
        if diff < tol {
            break;
        }
    }

    let mut result = HashMap::new();
    for i in 0..csr.num_nodes {
        let label = csr.get_node_label(i).unwrap().clone();
        result.insert(label, ranks[i]);
    }
    result
}
