use crate::graph::CSRGraph;
use std::collections::HashMap;

// PageRank using CSR: push-based algorithm with damping factor 0.85
pub fn page_rank(csr: &CSRGraph) -> HashMap<String, f64> {
    if csr.num_nodes == 0 {
        return HashMap::new();
    }

    let d = 0.85;
    let init_rank = 1.0 / csr.num_nodes as f64;
    let mut ranks = vec![init_rank; csr.num_nodes];
    let mut new_ranks = vec![0.0; csr.num_nodes];

    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        // Reset new_ranks
        new_ranks.fill(0.0);

        // Push contributions: each node pushes rank/out_degree to neighbors
        for src in 0..csr.num_nodes {
            let start = csr.row_offsets[src] as usize;
            let end = csr.row_offsets[src + 1] as usize;
            let out_degree = end - start;

            if out_degree > 0 {
                let contribution = ranks[src] / out_degree as f64;
                for i in start..end {
                    let dst = csr.col_indices[i] as usize;
                    new_ranks[dst] += contribution;
                }
            }
        }

        // Apply damping and compute diff
        let mut diff = 0.0;
        for node in 0..csr.num_nodes {
            new_ranks[node] = (1.0 - d) / (csr.num_nodes as f64) + d * new_ranks[node];
            diff += (new_ranks[node] - ranks[node]).abs();
        }

        std::mem::swap(&mut ranks, &mut new_ranks);

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