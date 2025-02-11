use petgraph::graph::Graph;
use petgraph::EdgeType;
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Computes PageRank for all nodes in the graph using an iterative algorithm.
pub fn page_rank<Ty>(graph: &Graph<String, i32, Ty>) -> HashMap<String, f64>
where
    Ty: EdgeType,
{
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }
    let d = 0.85;
    let init_rank = 1.0 / n as f64;
    let mut ranks: HashMap<_, f64> = graph
        .node_indices()
        .map(|node| (node, init_rank))
        .collect();

    let max_iter = 100;
    let tol = 1e-6;

    for _ in 0..max_iter {
        let mut new_ranks = HashMap::new();
        let mut diff = 0.0;

        for node in graph.node_indices() {
            let mut rank_sum = 0.0;
            // For directed graphs, sum contributions from incoming edges;
            // for undirected graphs, treat every edge as bidirectional.
            let incoming = if graph.is_directed() {
                graph.edges_directed(node, petgraph::Direction::Incoming)
            } else {
                graph.edges(node)
            };

            for edge in incoming {
                let source = if graph.is_directed() {
                    edge.source()
                } else {
                    // For undirected graphs, determine the neighbor (the node that is not the current one)
                    let (s, t) = (edge.source(), edge.target());
                    if s == node { t } else { s }
                };
                // Compute out-degree for the source node.
                let out_degree = if graph.is_directed() {
                    graph.edges_directed(source, petgraph::Direction::Outgoing).count()
                } else {
                    graph.edges(source).count()
                };
                if out_degree > 0 {
                    rank_sum += ranks[&source] / out_degree as f64;
                }
            }

            let new_rank = (1.0 - d) / (n as f64) + d * rank_sum;
            new_ranks.insert(node, new_rank);
            diff += (new_rank - ranks[&node]).abs();
        }
        ranks = new_ranks;
        if diff < tol {
            break;
        }
    }

    // Convert from NodeIndex to node label.
    let mut result = HashMap::new();
    for node in graph.node_indices() {
        result.insert(graph[node].clone(), ranks[&node]);
    }
    result
}
