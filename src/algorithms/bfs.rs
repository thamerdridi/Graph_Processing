use petgraph::graph::Graph;
use petgraph::EdgeType;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, VecDeque};

/// Converts a petgraph::Graph into a simple adjacency list.
pub fn convert_to_adj_list<Ty>(
    graph: &Graph<String, i32, Ty>,
) -> HashMap<String, Vec<String>>
where
    Ty: EdgeType,
{
    let mut adj_list: HashMap<String, Vec<String>> = HashMap::new();
    for node in graph.node_indices() {
        let label = graph[node].clone();
        adj_list.insert(label, Vec::new());
    }
    for edge in graph.edge_references() {
        let source = graph[edge.source()].clone();
        let target = graph[edge.target()].clone();
        adj_list.get_mut(&source).unwrap().push(target.clone());
        if !graph.is_directed() {
            adj_list.get_mut(&target).unwrap().push(source.clone());
        }
    }
    adj_list
}

/// Performs a standard BFS on the given adjacency list starting from `start`.
/// Returns a tuple with:
///   - A HashMap mapping each reachable node label to its distance (number of edges) from start.
///   - A predecessor map (node label -> previous node label on the shortest path).
pub fn bfs(adj_list: &HashMap<String, Vec<String>>, start: &str) -> (HashMap<String, i32>, HashMap<String, String>) {
    let mut distances: HashMap<String, i32> = HashMap::new();
    let mut predecessor: HashMap<String, String> = HashMap::new();
    let mut queue: VecDeque<String> = VecDeque::new();

    distances.insert(start.to_string(), 0);
    queue.push_back(start.to_string());

    while let Some(current) = queue.pop_front() {
        let current_distance = distances[&current];
        if let Some(neighbors) = adj_list.get(&current) {
            for neighbor in neighbors {
                if !distances.contains_key(neighbor) {
                    distances.insert(neighbor.clone(), current_distance + 1);
                    predecessor.insert(neighbor.clone(), current.clone());
                    queue.push_back(neighbor.clone());
                }
            }
        }
    }
    (distances, predecessor)
}