use petgraph::graph::Graph;
use petgraph::EdgeType;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

// Label Propagation: community detection via neighbor voting
pub fn label_propagation<Ty>(graph: &Graph<String, i32, Ty>) -> HashMap<String, String>
where
    Ty: EdgeType,
{
    let mut local_adj: HashMap<String, Vec<String>> = HashMap::new();
    for node in graph.node_indices() {
        let label = graph[node].clone();
        local_adj.insert(label.clone(), Vec::new());
    }
    for edge in graph.edge_references() {
        let source = graph[edge.source()].clone();
        let target = graph[edge.target()].clone();
        local_adj.get_mut(&source).unwrap().push(target.clone());
        local_adj.get_mut(&target).unwrap().push(source.clone());
    }

    let mut labels: HashMap<String, String> = HashMap::new();
    for node in local_adj.keys() {
        labels.insert(node.clone(), node.clone());
    }

    let max_iter = 100;
    for _ in 0..max_iter {
        let mut changed = false;
        for node in local_adj.keys() {
            let neighbors = &local_adj[node];
            let mut freq: HashMap<String, usize> = HashMap::new();
            for neighbor in neighbors {
                if let Some(nlabel) = labels.get(neighbor) {
                    *freq.entry(nlabel.clone()).or_insert(0) += 1;
                }
            }
            let mut max_count = 0;
            let mut best_label = labels[node].clone();
            for (lbl, count) in freq {
                if count > max_count {
                    max_count = count;
                    best_label = lbl;
                }
            }
            if best_label != labels[node] {
                labels.insert(node.clone(), best_label);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    labels
}
