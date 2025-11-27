use crate::graph::CSRGraph;
use std::collections::HashMap;

// Label Propagation using CSR: community detection via neighbor voting
pub fn label_propagation(csr: &CSRGraph) -> HashMap<String, String> {
    // Initialize labels: each node gets its own index as label
    let mut labels: Vec<usize> = (0..csr.num_nodes).collect();
    let mut new_labels = vec![0; csr.num_nodes];

    let max_iter = 100;
    for _ in 0..max_iter {
        let mut changed = false;
        
        for node in 0..csr.num_nodes {
            let start = csr.row_offsets[node] as usize;
            let end = csr.row_offsets[node + 1] as usize;
            
            if start == end {
                new_labels[node] = labels[node];
                continue;
            }

            // Count label frequencies among neighbors
            let mut freq: HashMap<usize, usize> = HashMap::new();
            for i in start..end {
                let neighbor = csr.col_indices[i] as usize;
                *freq.entry(labels[neighbor]).or_insert(0) += 1;
            }

            // Find most frequent label
            let mut max_count = 0;
            let mut best_label = labels[node];
            for (&lbl, &count) in &freq {
                if count > max_count || (count == max_count && lbl < best_label) {
                    max_count = count;
                    best_label = lbl;
                }
            }

            new_labels[node] = best_label;
            if new_labels[node] != labels[node] {
                changed = true;
            }
        }

        labels.copy_from_slice(&new_labels);
        
        if !changed {
            break;
        }
    }

    // Convert to HashMap with string labels
    let mut result: HashMap<String, String> = HashMap::new();
    for i in 0..csr.num_nodes {
        let node_label = csr.get_node_label(i).unwrap().clone();
        let community_label = csr.get_node_label(labels[i]).unwrap().clone();
        result.insert(node_label, community_label);
    }
    result
}
