use petgraph::graph::{Graph, NodeIndex};
use petgraph::{Directed, Undirected};
use std::collections::HashMap;

/// Specifies whether the graph should be directed or undirected.
#[derive(Debug, Clone, Copy)]
pub enum GraphDirection {
    Directed,
    Undirected,
}

/// Our simplified graph type. We always use a weighted graph,
/// where each edge has a weight of type `i32`.
#[derive(Debug)]
pub enum MyGraph {
    Directed(Graph<String, i32, Directed>),
    Undirected(Graph<String, i32, Undirected>),
}

/// Builds a weighted graph from an input edge list string.
///
/// Each line represents an edge in one of the following formats:
///  - "NodeA-NodeB" (no weight specified; defaults to 1)
///  - "NodeA-NodeB:Weight" (explicit weight provided)
///
/// # Arguments
///
/// * `input` - A multiline string where each line is an edge.
/// * `direction` - Specifies whether the graph should be directed or undirected.
///
/// # Returns
///
/// A [`MyGraph`] instance containing the constructed graph.
pub fn build_graph(input: &str, direction: GraphDirection) -> MyGraph {
    match direction {
        GraphDirection::Directed => {
            let mut graph: Graph<String, i32, Directed> = Graph::new();
            let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

            for line in input.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                // Split the line on '-' to separate the source from the target (and possibly weight)
                let parts: Vec<&str> = line.split('-').collect();
                if parts.len() != 2 {
                    eprintln!("Invalid edge format: {}", line);
                    continue;
                }

                let source = parts[0].trim().to_string();
                // Split the target portion on ':' to check for an optional weight.
                let target_parts: Vec<&str> = parts[1].split(':').collect();
                let target = target_parts[0].trim().to_string();
                let weight: i32 = if target_parts.len() > 1 {
                    target_parts[1].trim().parse().unwrap_or(1)
                } else {
                    1
                };

                // Insert nodes if they don't exist already.
                let source_index = *node_map.entry(source.clone())
                    .or_insert_with(|| graph.add_node(source));
                let target_index = *node_map.entry(target.clone())
                    .or_insert_with(|| graph.add_node(target));
                graph.add_edge(source_index, target_index, weight);
            }
            MyGraph::Directed(graph)
        },
        GraphDirection::Undirected => {
            let mut graph: Graph<String, i32, Undirected> = Graph::new_undirected();
            let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

            for line in input.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let parts: Vec<&str> = line.split('-').collect();
                if parts.len() != 2 {
                    eprintln!("Invalid edge format: {}", line);
                    continue;
                }

                let source = parts[0].trim().to_string();
                let target_parts: Vec<&str> = parts[1].split(':').collect();
                let target = target_parts[0].trim().to_string();
                let weight: i32 = if target_parts.len() > 1 {
                    target_parts[1].trim().parse().unwrap_or(1)
                } else {
                    1
                };

                let source_index = *node_map.entry(source.clone())
                    .or_insert_with(|| graph.add_node(source));
                let target_index = *node_map.entry(target.clone())
                    .or_insert_with(|| graph.add_node(target));
                graph.add_edge(source_index, target_index, weight);
            }
            MyGraph::Undirected(graph)
        },
    }
}
