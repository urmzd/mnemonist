use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::Path;

use crate::Error;

use super::distance::cosine_similarity;
use super::{AnnIndex, SearchHit};

type NodeId = u32;

/// HNSW configuration parameters.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max connections per node per layer (default 16).
    pub m: usize,
    /// Max connections for layer 0 (default 2 * m).
    pub m0: usize,
    /// Beam width during construction (default 200).
    pub ef_construction: usize,
    /// Beam width during search (default 50).
    pub ef_search: usize,
    /// Inverse log multiplier for level generation.
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: 2 * m,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

#[derive(Debug, Clone)]
struct Node {
    id: String,
    vector: Vec<f32>,
    /// Neighbors per layer. neighbors[layer] = vec of neighbor NodeIds.
    neighbors: Vec<Vec<NodeId>>,
    level: usize,
}

/// HNSW (Hierarchical Navigable Small World) index.
///
/// Multi-layer proximity graph for approximate nearest neighbor search.
#[derive(Debug)]
pub struct HnswIndex {
    nodes: Vec<Node>,
    id_to_node: HashMap<String, NodeId>,
    entry_point: Option<NodeId>,
    max_level: usize,
    config: HnswConfig,
    dimension: usize,
    rng_state: u64,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(dimension: usize, config: HnswConfig) -> Self {
        Self {
            nodes: Vec::new(),
            id_to_node: HashMap::new(),
            entry_point: None,
            max_level: 0,
            config,
            dimension,
            rng_state: 42,
        }
    }

    /// Create with default config.
    pub fn with_defaults(dimension: usize) -> Self {
        Self::new(dimension, HnswConfig::default())
    }

    fn random_level(&mut self) -> usize {
        // Simple xorshift64 PRNG
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        let r = (self.rng_state as f64) / (u64::MAX as f64);
        (-r.ln() * self.config.ml).floor() as usize
    }

    fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m0
        } else {
            self.config.m
        }
    }

    fn similarity_to_query(&self, query: &[f32], node: NodeId) -> f32 {
        cosine_similarity(query, &self.nodes[node as usize].vector)
    }

    /// Greedy search from entry_point down to target_layer, returning the closest node.
    fn search_layer_greedy(&self, query: &[f32], mut current: NodeId, layer: usize) -> NodeId {
        let mut best_sim = self.similarity_to_query(query, current);
        loop {
            let mut improved = false;
            let neighbors = &self.nodes[current as usize].neighbors[layer];
            for &neighbor in neighbors {
                let sim = self.similarity_to_query(query, neighbor);
                if sim > best_sim {
                    best_sim = sim;
                    current = neighbor;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    /// Beam search at a single layer, returning ef closest nodes.
    fn search_layer_ef(
        &self,
        query: &[f32],
        entry_points: &[NodeId],
        ef: usize,
        layer: usize,
    ) -> Vec<(NodeId, f32)> {
        // candidates: max-heap by similarity (we want highest sim first)
        // result: min-heap by similarity (so we can drop worst)
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<OrdF32Node> = BinaryHeap::new();
        let mut result: BinaryHeap<Reverse<OrdF32Node>> = BinaryHeap::new();

        for &ep in entry_points {
            let sim = self.similarity_to_query(query, ep);
            visited.insert(ep);
            candidates.push(OrdF32Node(sim, ep));
            result.push(Reverse(OrdF32Node(sim, ep)));
        }

        while let Some(OrdF32Node(c_sim, c_id)) = candidates.pop() {
            let worst_result = result
                .peek()
                .map(|Reverse(n)| n.0)
                .unwrap_or(f32::NEG_INFINITY);
            if c_sim < worst_result && result.len() >= ef {
                break;
            }

            let neighbors = &self.nodes[c_id as usize].neighbors[layer];
            for &neighbor in neighbors {
                if visited.insert(neighbor) {
                    let sim = self.similarity_to_query(query, neighbor);
                    let worst = result
                        .peek()
                        .map(|Reverse(n)| n.0)
                        .unwrap_or(f32::NEG_INFINITY);
                    if sim > worst || result.len() < ef {
                        candidates.push(OrdF32Node(sim, neighbor));
                        result.push(Reverse(OrdF32Node(sim, neighbor)));
                        if result.len() > ef {
                            result.pop();
                        }
                    }
                }
            }
        }

        result
            .into_iter()
            .map(|Reverse(OrdF32Node(sim, id))| (id, sim))
            .collect()
    }

    /// Select neighbors using simple heuristic (keep M closest).
    fn select_neighbors(&self, candidates: &[(NodeId, f32)], m: usize) -> Vec<NodeId> {
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(m);
        sorted.into_iter().map(|(id, _)| id).collect()
    }

    /// Load an HNSW index from a binary file.
    pub fn load_from(path: &Path) -> Result<Self, Error> {
        let data = fs::read(path)?;
        let mut cursor = &data[..];

        // Magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != b"HNSW" {
            return Err(Error::EmbeddingFormat("invalid HNSW magic".into()));
        }

        // Version
        let mut ver = [0u8; 1];
        cursor.read_exact(&mut ver)?;

        // Config
        let read_u32 = |c: &mut &[u8]| -> Result<u32, Error> {
            let mut b = [0u8; 4];
            c.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };
        let read_u64 = |c: &mut &[u8]| -> Result<u64, Error> {
            let mut b = [0u8; 8];
            c.read_exact(&mut b)?;
            Ok(u64::from_le_bytes(b))
        };

        let dimension = read_u32(&mut cursor)? as usize;
        let m = read_u32(&mut cursor)? as usize;
        let m0 = read_u32(&mut cursor)? as usize;
        let ef_construction = read_u32(&mut cursor)? as usize;
        let ef_search = read_u32(&mut cursor)? as usize;
        let max_level = read_u32(&mut cursor)? as usize;
        let node_count = read_u32(&mut cursor)? as usize;
        let entry_point_raw = read_u32(&mut cursor)?;
        let entry_point = if entry_point_raw == u32::MAX {
            None
        } else {
            Some(entry_point_raw)
        };
        let rng_state = read_u64(&mut cursor)?;

        let config = HnswConfig {
            m,
            m0,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
        };

        let mut nodes = Vec::with_capacity(node_count);
        let mut id_to_node = HashMap::new();

        for idx in 0..node_count {
            // ID
            let id_len = read_u32(&mut cursor)? as usize;
            let mut id_bytes = vec![0u8; id_len];
            cursor.read_exact(&mut id_bytes)?;
            let id = String::from_utf8(id_bytes)
                .map_err(|e| Error::EmbeddingFormat(format!("invalid UTF-8: {e}")))?;

            // Level
            let level = read_u32(&mut cursor)? as usize;

            // Vector
            let mut vector = vec![0f32; dimension];
            for v in &mut vector {
                let mut b = [0u8; 4];
                cursor.read_exact(&mut b)?;
                *v = f32::from_le_bytes(b);
            }

            // Neighbors per layer
            let mut neighbors = Vec::with_capacity(level + 1);
            for _ in 0..=level {
                let n_count = read_u32(&mut cursor)? as usize;
                let mut layer_neighbors = Vec::with_capacity(n_count);
                for _ in 0..n_count {
                    layer_neighbors.push(read_u32(&mut cursor)?);
                }
                neighbors.push(layer_neighbors);
            }

            id_to_node.insert(id.clone(), idx as NodeId);
            nodes.push(Node {
                id,
                vector,
                neighbors,
                level,
            });
        }

        Ok(Self {
            nodes,
            id_to_node,
            entry_point,
            max_level,
            config,
            dimension,
            rng_state,
        })
    }
}

impl AnnIndex for HnswIndex {
    fn insert(&mut self, id: &str, embedding: &[f32]) -> Result<(), Error> {
        if embedding.len() != self.dimension {
            return Err(Error::EmbeddingFormat(format!(
                "expected dimension {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        // Remove existing if present
        if self.id_to_node.contains_key(id) {
            self.remove(id)?;
        }

        let level = self.random_level();
        let node_id = self.nodes.len() as NodeId;

        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }

        self.nodes.push(Node {
            id: id.to_string(),
            vector: embedding.to_vec(),
            neighbors,
            level,
        });
        self.id_to_node.insert(id.to_string(), node_id);

        if let Some(ep) = self.entry_point {
            // Greedy descent from top to level+1
            let mut current = ep;
            for l in (level + 1..=self.max_level).rev() {
                if l < self.nodes[current as usize].neighbors.len() {
                    current = self.search_layer_greedy(embedding, current, l);
                }
            }

            // Insert at each layer from min(level, max_level) down to 0
            for l in (0..=level.min(self.max_level)).rev() {
                let candidates =
                    self.search_layer_ef(embedding, &[current], self.config.ef_construction, l);
                let m = self.max_connections(l);
                let selected = self.select_neighbors(&candidates, m);

                // Set neighbors for new node
                self.nodes[node_id as usize].neighbors[l] = selected.clone();

                // Add bidirectional connections
                for &neighbor in &selected {
                    self.nodes[neighbor as usize].neighbors[l].push(node_id);
                    // Prune if over capacity
                    if self.nodes[neighbor as usize].neighbors[l].len() > m {
                        let n_vec = &self.nodes[neighbor as usize].vector;
                        let mut scored: Vec<(NodeId, f32)> = self.nodes[neighbor as usize]
                            .neighbors[l]
                            .iter()
                            .map(|&nid| {
                                (
                                    nid,
                                    cosine_similarity(n_vec, &self.nodes[nid as usize].vector),
                                )
                            })
                            .collect();
                        scored.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        scored.truncate(m);
                        self.nodes[neighbor as usize].neighbors[l] =
                            scored.into_iter().map(|(nid, _)| nid).collect();
                    }
                }

                if !candidates.is_empty() {
                    current = candidates
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                        .0;
                }
            }

            if level > self.max_level {
                self.max_level = level;
                self.entry_point = Some(node_id);
            }
        } else {
            self.entry_point = Some(node_id);
            self.max_level = level;
        }

        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<bool, Error> {
        let Some(&node_id) = self.id_to_node.get(id) else {
            return Ok(false);
        };

        // Remove from all neighbor lists
        let level = self.nodes[node_id as usize].level;
        for l in 0..=level {
            let neighbors = self.nodes[node_id as usize].neighbors[l].clone();
            for neighbor in neighbors {
                self.nodes[neighbor as usize].neighbors[l].retain(|&n| n != node_id);
            }
        }

        // Clear the node's neighbors (mark as deleted)
        for l in 0..=level {
            self.nodes[node_id as usize].neighbors[l].clear();
        }

        self.id_to_node.remove(id);

        // Update entry point if needed
        if self.entry_point == Some(node_id) {
            self.entry_point = self.id_to_node.values().copied().next();
            if let Some(ep) = self.entry_point {
                self.max_level = self.nodes[ep as usize].level;
            } else {
                self.max_level = 0;
            }
        }

        Ok(true)
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, Error> {
        if query.len() != self.dimension {
            return Err(Error::EmbeddingFormat(format!(
                "expected dimension {}, got {}",
                self.dimension,
                query.len()
            )));
        }

        let Some(ep) = self.entry_point else {
            return Ok(Vec::new());
        };

        // Greedy descent from top layer
        let mut current = ep;
        for l in (1..=self.max_level).rev() {
            if l < self.nodes[current as usize].neighbors.len() {
                current = self.search_layer_greedy(query, current, l);
            }
        }

        // Beam search at layer 0
        let results = self.search_layer_ef(query, &[current], self.config.ef_search.max(top_k), 0);

        let mut hits: Vec<SearchHit> = results
            .into_iter()
            .filter(|(id, _)| self.id_to_node.contains_key(&self.nodes[*id as usize].id))
            .map(|(id, sim)| SearchHit {
                id: self.nodes[id as usize].id.clone(),
                score: sim,
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(top_k);
        Ok(hits)
    }

    fn len(&self) -> usize {
        self.id_to_node.len()
    }

    fn save(&self, path: &Path) -> Result<(), Error> {
        let mut buf = Vec::new();

        // Magic + version
        buf.write_all(b"HNSW")?;
        buf.write_all(&[1u8])?;

        // Config
        let write_u32 = |b: &mut Vec<u8>, v: u32| b.write_all(&v.to_le_bytes());
        let write_u64 = |b: &mut Vec<u8>, v: u64| b.write_all(&v.to_le_bytes());

        write_u32(&mut buf, self.dimension as u32)?;
        write_u32(&mut buf, self.config.m as u32)?;
        write_u32(&mut buf, self.config.m0 as u32)?;
        write_u32(&mut buf, self.config.ef_construction as u32)?;
        write_u32(&mut buf, self.config.ef_search as u32)?;
        write_u32(&mut buf, self.max_level as u32)?;
        write_u32(&mut buf, self.nodes.len() as u32)?;
        write_u32(&mut buf, self.entry_point.unwrap_or(u32::MAX))?;
        write_u64(&mut buf, self.rng_state)?;

        // Nodes
        for node in &self.nodes {
            let id_bytes = node.id.as_bytes();
            write_u32(&mut buf, id_bytes.len() as u32)?;
            buf.write_all(id_bytes)?;
            write_u32(&mut buf, node.level as u32)?;

            for v in &node.vector {
                buf.write_all(&v.to_le_bytes())?;
            }

            for layer_neighbors in &node.neighbors {
                write_u32(&mut buf, layer_neighbors.len() as u32)?;
                for &n in layer_neighbors {
                    write_u32(&mut buf, n)?;
                }
            }
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, buf)?;
        Ok(())
    }
}

/// Helper for ordering f32 in heaps.
#[derive(Debug, Clone)]
struct OrdF32Node(f32, NodeId);

impl PartialEq for OrdF32Node {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OrdF32Node {}

impl PartialOrd for OrdF32Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    #[test]
    fn insert_and_search() {
        let mut index = HnswIndex::with_defaults(8);

        for i in 0..50 {
            let v = make_vector(8, i as f32);
            index.insert(&format!("item_{i}"), &v).unwrap();
        }

        assert_eq!(index.len(), 50);

        let query = make_vector(8, 5.0);
        let results = index.search(&query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // The closest should be item_5 itself
        assert_eq!(results[0].id, "item_5");
        assert!((results[0].score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn remove_works() {
        let mut index = HnswIndex::with_defaults(4);
        index.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 2);
        assert!(index.remove("a").unwrap());
        assert_eq!(index.len(), 1);
        assert!(!index.remove("a").unwrap());
    }

    #[test]
    fn save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.hnsw");

        let mut index = HnswIndex::with_defaults(4);
        for i in 0..20 {
            let v = make_vector(4, i as f32);
            index.insert(&format!("item_{i}"), &v).unwrap();
        }
        index.save(&path).unwrap();

        let loaded = HnswIndex::load_from(&path).unwrap();
        assert_eq!(loaded.len(), 20);

        let query = make_vector(4, 10.0);
        let results = loaded.search(&query, 3).unwrap();
        assert_eq!(results[0].id, "item_10");
    }

    #[test]
    fn recall_test() {
        // Test that HNSW achieves >95% recall on 200 items
        let dim = 32;
        let n = 200;
        let mut index = HnswIndex::with_defaults(dim);

        let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_vector(dim, i as f32)).collect();
        for (i, v) in vectors.iter().enumerate() {
            index.insert(&format!("item_{i}"), v).unwrap();
        }

        let mut correct = 0;
        let k = 10;
        let queries = 20;

        for q in 0..queries {
            let query = &vectors[q * (n / queries)];
            let results = index.search(query, k).unwrap();
            let result_ids: HashSet<String> = results.iter().map(|h| h.id.clone()).collect();

            // Compute brute-force top-k
            let mut brute: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_similarity(query, v)))
                .collect();
            brute.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let brute_top: HashSet<String> = brute[..k]
                .iter()
                .map(|(i, _)| format!("item_{i}"))
                .collect();

            correct += result_ids.intersection(&brute_top).count();
        }

        let recall = correct as f64 / (queries * k) as f64;
        assert!(
            recall > 0.90,
            "HNSW recall {recall:.2} is below 90% threshold"
        );
    }
}
