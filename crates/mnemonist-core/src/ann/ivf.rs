use std::collections::HashMap;
use std::fs;
use std::io::{Read as _, Write as _};
use std::path::Path;

use crate::Error;

use super::distance::cosine_similarity;
use super::{AnnIndex, SearchHit};

/// IVF-Flat configuration.
#[derive(Debug, Clone)]
pub struct IvfConfig {
    /// Number of clusters. Default: sqrt(n) at build time.
    pub n_lists: usize,
    /// Number of clusters to probe during search. Default: 10.
    pub n_probe: usize,
    /// K-means iterations. Default: 20.
    pub kmeans_iters: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_lists: 16,
            n_probe: 10,
            kmeans_iters: 20,
        }
    }
}

#[derive(Debug, Clone)]
struct VectorEntry {
    id: String,
    vector: Vec<f32>,
}

/// IVF-Flat (Inverted File Index with flat search).
///
/// Partitions vectors into clusters via k-means, then searches
/// only the nearest clusters for each query.
#[derive(Debug)]
pub struct IvfFlatIndex {
    centroids: Vec<Vec<f32>>,
    cells: Vec<Vec<VectorEntry>>,
    id_to_cell: HashMap<String, usize>,
    config: IvfConfig,
    dimension: usize,
    trained: bool,
}

impl IvfFlatIndex {
    /// Create a new IVF-Flat index.
    pub fn new(dimension: usize, config: IvfConfig) -> Self {
        Self {
            centroids: Vec::new(),
            cells: Vec::new(),
            id_to_cell: HashMap::new(),
            config,
            dimension,
            trained: false,
        }
    }

    /// Create with default config.
    pub fn with_defaults(dimension: usize) -> Self {
        Self::new(dimension, IvfConfig::default())
    }

    /// Train centroids using k-means on the given vectors.
    pub fn train(&mut self, vectors: &[Vec<f32>]) {
        let k = self.config.n_lists.min(vectors.len());
        if k == 0 {
            return;
        }

        // Initialize centroids from first k vectors
        let mut centroids: Vec<Vec<f32>> = vectors[..k].to_vec();

        for _ in 0..self.config.kmeans_iters {
            // Assign vectors to nearest centroid
            let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); k];
            for (i, v) in vectors.iter().enumerate() {
                let nearest = nearest_centroid(v, &centroids);
                assignments[nearest].push(i);
            }

            // Recompute centroids
            for (c, assigned) in assignments.iter().enumerate() {
                if assigned.is_empty() {
                    continue;
                }
                let mut new_centroid = vec![0.0f32; self.dimension];
                for &idx in assigned {
                    for (j, val) in vectors[idx].iter().enumerate() {
                        new_centroid[j] += val;
                    }
                }
                let n = assigned.len() as f32;
                for val in &mut new_centroid {
                    *val /= n;
                }
                centroids[c] = new_centroid;
            }
        }

        self.centroids = centroids;
        self.cells = vec![Vec::new(); k];
        self.trained = true;

        // Re-assign any existing entries
        let entries: Vec<_> = self.cells.iter().flat_map(|c| c.iter().cloned()).collect();
        self.id_to_cell.clear();
        self.cells = vec![Vec::new(); k];
        for entry in entries {
            let cell = nearest_centroid(&entry.vector, &self.centroids);
            self.id_to_cell.insert(entry.id.clone(), cell);
            self.cells[cell].push(entry);
        }
    }

    /// Find the nearest centroid for a single query.
    fn nearest_centroids(&self, query: &[f32], n: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine_similarity(query, c)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Load from a binary file.
    pub fn load_from(path: &Path) -> Result<Self, Error> {
        let data = fs::read(path)?;
        let mut cursor = &data[..];

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != b"IVFF" {
            return Err(Error::EmbeddingFormat("invalid IVF magic".into()));
        }

        let mut ver = [0u8; 1];
        cursor.read_exact(&mut ver)?;

        let read_u32 = |c: &mut &[u8]| -> Result<u32, Error> {
            let mut b = [0u8; 4];
            c.read_exact(&mut b)?;
            Ok(u32::from_le_bytes(b))
        };

        let dimension = read_u32(&mut cursor)? as usize;
        let n_lists = read_u32(&mut cursor)? as usize;
        let n_probe = read_u32(&mut cursor)? as usize;
        let kmeans_iters = read_u32(&mut cursor)? as usize;

        // Read centroids
        let mut centroids = Vec::with_capacity(n_lists);
        for _ in 0..n_lists {
            let mut c = vec![0f32; dimension];
            for v in &mut c {
                let mut b = [0u8; 4];
                cursor.read_exact(&mut b)?;
                *v = f32::from_le_bytes(b);
            }
            centroids.push(c);
        }

        // Read cells
        let mut cells = Vec::with_capacity(n_lists);
        let mut id_to_cell = HashMap::new();
        for cell_idx in 0..n_lists {
            let count = read_u32(&mut cursor)? as usize;
            let mut cell = Vec::with_capacity(count);
            for _ in 0..count {
                let id_len = read_u32(&mut cursor)? as usize;
                let mut id_bytes = vec![0u8; id_len];
                cursor.read_exact(&mut id_bytes)?;
                let id = String::from_utf8(id_bytes)
                    .map_err(|e| Error::EmbeddingFormat(format!("invalid UTF-8: {e}")))?;

                let mut vector = vec![0f32; dimension];
                for v in &mut vector {
                    let mut b = [0u8; 4];
                    cursor.read_exact(&mut b)?;
                    *v = f32::from_le_bytes(b);
                }

                id_to_cell.insert(id.clone(), cell_idx);
                cell.push(VectorEntry { id, vector });
            }
            cells.push(cell);
        }

        Ok(Self {
            centroids,
            cells,
            id_to_cell,
            config: IvfConfig {
                n_lists,
                n_probe,
                kmeans_iters,
            },
            dimension,
            trained: true,
        })
    }
}

impl AnnIndex for IvfFlatIndex {
    fn insert(&mut self, id: &str, embedding: &[f32]) -> Result<(), Error> {
        if embedding.len() != self.dimension {
            return Err(Error::EmbeddingFormat(format!(
                "expected dimension {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        // Remove existing
        if self.id_to_cell.contains_key(id) {
            self.remove(id)?;
        }

        if !self.trained || self.centroids.is_empty() {
            // Not yet trained — put in cell 0 as fallback
            if self.cells.is_empty() {
                self.cells.push(Vec::new());
                self.centroids.push(embedding.to_vec());
                self.trained = true;
            }
            let cell = nearest_centroid(embedding, &self.centroids);
            self.id_to_cell.insert(id.to_string(), cell);
            self.cells[cell].push(VectorEntry {
                id: id.to_string(),
                vector: embedding.to_vec(),
            });
        } else {
            let cell = nearest_centroid(embedding, &self.centroids);
            self.id_to_cell.insert(id.to_string(), cell);
            self.cells[cell].push(VectorEntry {
                id: id.to_string(),
                vector: embedding.to_vec(),
            });
        }

        Ok(())
    }

    fn remove(&mut self, id: &str) -> Result<bool, Error> {
        let Some(cell) = self.id_to_cell.remove(id) else {
            return Ok(false);
        };
        self.cells[cell].retain(|e| e.id != id);
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

        if self.centroids.is_empty() {
            return Ok(Vec::new());
        }

        let probe_cells = self.nearest_centroids(query, self.config.n_probe);

        let mut candidates: Vec<SearchHit> = Vec::new();
        for cell_idx in probe_cells {
            for entry in &self.cells[cell_idx] {
                let score = cosine_similarity(query, &entry.vector);
                candidates.push(SearchHit {
                    id: entry.id.clone(),
                    score,
                });
            }
        }

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(top_k);
        Ok(candidates)
    }

    fn len(&self) -> usize {
        self.id_to_cell.len()
    }

    fn save(&self, path: &Path) -> Result<(), Error> {
        let mut buf = Vec::new();

        buf.write_all(b"IVFF")?;
        buf.write_all(&[1u8])?;

        let write_u32 = |b: &mut Vec<u8>, v: u32| b.write_all(&v.to_le_bytes());

        write_u32(&mut buf, self.dimension as u32)?;
        write_u32(&mut buf, self.config.n_lists as u32)?;
        write_u32(&mut buf, self.config.n_probe as u32)?;
        write_u32(&mut buf, self.config.kmeans_iters as u32)?;

        // Centroids
        for c in &self.centroids {
            for v in c {
                buf.write_all(&v.to_le_bytes())?;
            }
        }

        // Cells
        for cell in &self.cells {
            write_u32(&mut buf, cell.len() as u32)?;
            for entry in cell {
                let id_bytes = entry.id.as_bytes();
                write_u32(&mut buf, id_bytes.len() as u32)?;
                buf.write_all(id_bytes)?;
                for v in &entry.vector {
                    buf.write_all(&v.to_le_bytes())?;
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

fn nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity(vector, c)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vector(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    #[test]
    fn insert_and_search() {
        let dim = 8;
        let mut index = IvfFlatIndex::new(
            dim,
            IvfConfig {
                n_lists: 4,
                n_probe: 4,
                kmeans_iters: 10,
            },
        );

        let vectors: Vec<Vec<f32>> = (0..30).map(|i| make_vector(dim, i as f32)).collect();

        // Train on all vectors
        index.train(&vectors);

        // Insert all
        for (i, v) in vectors.iter().enumerate() {
            index.insert(&format!("item_{i}"), v).unwrap();
        }

        assert_eq!(index.len(), 30);

        let query = make_vector(dim, 5.0);
        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "item_5");
    }

    #[test]
    fn remove_works() {
        let mut index = IvfFlatIndex::with_defaults(4);
        index.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 2);
        assert!(index.remove("a").unwrap());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test.ivf");

        let dim = 4;
        let mut index = IvfFlatIndex::new(
            dim,
            IvfConfig {
                n_lists: 3,
                n_probe: 3,
                kmeans_iters: 5,
            },
        );

        let vectors: Vec<Vec<f32>> = (0..15).map(|i| make_vector(dim, i as f32)).collect();
        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(&format!("item_{i}"), v).unwrap();
        }

        index.save(&path).unwrap();
        let loaded = IvfFlatIndex::load_from(&path).unwrap();

        assert_eq!(loaded.len(), 15);
        let query = make_vector(dim, 7.0);
        let results = loaded.search(&query, 3).unwrap();
        assert_eq!(results[0].id, "item_7");
    }

    #[test]
    fn recall_test() {
        let dim = 32;
        let n = 200;
        let mut index = IvfFlatIndex::new(
            dim,
            IvfConfig {
                n_lists: 14, // ~sqrt(200)
                n_probe: 10,
                kmeans_iters: 20,
            },
        );

        let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_vector(dim, i as f32)).collect();
        index.train(&vectors);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(&format!("item_{i}"), v).unwrap();
        }

        let mut correct = 0;
        let k = 10;
        let queries = 20;

        for q in 0..queries {
            let query = &vectors[q * (n / queries)];
            let results = index.search(query, k).unwrap();
            let result_ids: std::collections::HashSet<String> =
                results.iter().map(|h| h.id.clone()).collect();

            let mut brute: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, cosine_similarity(query, v)))
                .collect();
            brute.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let brute_top: std::collections::HashSet<String> = brute[..k]
                .iter()
                .map(|(i, _)| format!("item_{i}"))
                .collect();

            correct += result_ids.intersection(&brute_top).count();
        }

        let recall = correct as f64 / (queries * k) as f64;
        assert!(
            recall > 0.85,
            "IVF recall {recall:.2} is below 85% threshold"
        );
    }
}
