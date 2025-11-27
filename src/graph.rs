use petgraph::graph::Graph;
use petgraph::EdgeType;
use std::collections::HashMap;

// Compressed Sparse Row format for GPU operations
#[derive(Debug, Clone)]
pub struct CSRGraph {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub row_offsets: Vec<i32>,    // Start index of each node's edges
    pub col_indices: Vec<i32>,    // Destination nodes
    pub edge_weights: Vec<i32>,   // Edge weights
    pub label_to_idx: HashMap<String, usize>,
    pub idx_to_label: Vec<String>,
}

impl CSRGraph {
    // Convert petgraph to CSR format
    pub fn from_graph<Ty>(graph: &Graph<String, i32, Ty>) -> Self
    where
        Ty: EdgeType,
    {
        use petgraph::visit::EdgeRef;

        let num_nodes = graph.node_count();
        let mut label_to_idx = HashMap::new();
        let mut idx_to_label = Vec::new();

        for (idx, node) in graph.node_indices().enumerate() {
            let label = graph[node].clone();
            label_to_idx.insert(label.clone(), idx);
            idx_to_label.push(label);
        }

        let mut row_offsets = vec![0i32; num_nodes + 1];
        for node in graph.node_indices() {
            let node_idx = label_to_idx[&graph[node]];
            let out_degree = graph.edges(node).count() as i32;
            row_offsets[node_idx + 1] = out_degree;
        }

        for i in 0..num_nodes {
            row_offsets[i + 1] += row_offsets[i];
        }

        let num_edges = row_offsets[num_nodes] as usize;
        let mut col_indices = vec![0i32; num_edges];
        let mut edge_weights = vec![0i32; num_edges];
        let mut edge_count = vec![0; num_nodes];

        for node in graph.node_indices() {
            let src_idx = label_to_idx[&graph[node]];
            for edge in graph.edges(node) {
                let dst_idx = label_to_idx[&graph[edge.target()]];
                let pos = row_offsets[src_idx] as usize + edge_count[src_idx];
                col_indices[pos] = dst_idx as i32;
                edge_weights[pos] = *edge.weight();
                edge_count[src_idx] += 1;
            }
        }

        CSRGraph {
            num_nodes,
            num_edges,
            row_offsets,
            col_indices,
            edge_weights,
            label_to_idx,
            idx_to_label,
        }
    }

    pub fn get_node_idx(&self, label: &str) -> Option<usize> {
        self.label_to_idx.get(label).copied()
    }

    pub fn get_node_label(&self, idx: usize) -> Option<&String> {
        self.idx_to_label.get(idx)
    }
}
