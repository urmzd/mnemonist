//! Embedding quality metrics for evaluating how well embeddings
//! distribute across the vector space.

use mnemonist_core::distance::{cosine_similarity, normalize};
use serde::{Deserialize, Serialize};

/// Aggregated embedding quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetrics {
    pub anisotropy: f32,
    pub similarity_range: f32,
    pub discrimination_gap: Option<f32>,
    pub intrinsic_dimensionality: f32,
    pub sample_size: usize,
}

/// Anisotropy: average pairwise cosine similarity across all embeddings.
/// Lower is better — 0.0 means uniformly distributed, 1.0 means all identical.
/// Target: < 0.3
pub fn anisotropy(embeddings: &[Vec<f32>]) -> f32 {
    let n = embeddings.len();
    if n < 2 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut count = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += cosine_similarity(&embeddings[i], &embeddings[j]) as f64;
            count += 1;
        }
    }

    (sum / count as f64) as f32
}

/// Similarity range: max - min off-diagonal pairwise cosine similarity.
/// Higher is better — embeddings use more of the similarity space.
/// Target: > 0.3
pub fn similarity_range(embeddings: &[Vec<f32>]) -> f32 {
    let n = embeddings.len();
    if n < 2 {
        return 0.0;
    }

    let mut min_sim = f32::MAX;
    let mut max_sim = f32::MIN;
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            min_sim = min_sim.min(sim);
            max_sim = max_sim.max(sim);
        }
    }

    max_sim - min_sim
}

/// Discrimination gap: average intra-group similarity minus average inter-group
/// similarity. Higher is better — related items cluster tighter than unrelated.
/// Target: > 0.05
///
/// `groups` maps each embedding index to a group label.
pub fn discrimination_gap(embeddings: &[Vec<f32>], groups: &[usize]) -> f32 {
    assert_eq!(embeddings.len(), groups.len());
    let n = embeddings.len();
    if n < 2 {
        return 0.0;
    }

    let mut intra_sum = 0.0f64;
    let mut intra_count = 0u64;
    let mut inter_sum = 0.0f64;
    let mut inter_count = 0u64;

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]) as f64;
            if groups[i] == groups[j] {
                intra_sum += sim;
                intra_count += 1;
            } else {
                inter_sum += sim;
                inter_count += 1;
            }
        }
    }

    let intra_avg = if intra_count > 0 {
        intra_sum / intra_count as f64
    } else {
        0.0
    };
    let inter_avg = if inter_count > 0 {
        inter_sum / inter_count as f64
    } else {
        0.0
    };

    (intra_avg - inter_avg) as f32
}

/// Mean-center embeddings and re-normalize to unit length.
/// Reduces anisotropy by shifting the centroid to the origin.
pub fn mean_center(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    let n = embeddings.len() as f32;

    let mut mean = vec![0.0f32; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            mean[i] += v;
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    embeddings
        .iter()
        .map(|emb| {
            let mut centered: Vec<f32> = emb.iter().zip(&mean).map(|(v, m)| v - m).collect();
            normalize(&mut centered);
            centered
        })
        .collect()
}

/// Intrinsic dimensionality estimate via the participation ratio.
///
/// Computes the covariance matrix eigenvalue spectrum and returns
/// d_eff = (sum(λ))² / sum(λ²). Values range from 1 (all variance on one axis)
/// to the embedding dimension (uniform spread).
pub fn intrinsic_dimensionality(embeddings: &[Vec<f32>]) -> f32 {
    if embeddings.len() < 2 {
        return 1.0;
    }

    let n = embeddings.len();
    let dim = embeddings[0].len();

    // Compute mean
    let mut mean = vec![0.0f64; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            mean[i] += v as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // Build covariance matrix (dim x dim) in row-major order
    let mut cov = vec![0.0f64; dim * dim];
    for emb in embeddings {
        for i in 0..dim {
            let xi = emb[i] as f64 - mean[i];
            for j in i..dim {
                let xj = emb[j] as f64 - mean[j];
                let val = xi * xj;
                cov[i * dim + j] += val;
                if i != j {
                    cov[j * dim + i] += val;
                }
            }
        }
    }
    let scale = 1.0 / (n - 1) as f64;
    for v in &mut cov {
        *v *= scale;
    }

    // Compute eigenvalues via power iteration on the covariance matrix.
    // We use deflation to extract the top eigenvalues until we've captured
    // enough variance for the participation ratio to converge.
    let max_iters = 100;
    let tol = 1e-8;
    let n_eigenvalues = dim.min(n - 1).min(64); // cap for performance

    let mut eigenvalues = Vec::with_capacity(n_eigenvalues);
    let mut deflated_cov = cov;

    for _ in 0..n_eigenvalues {
        // Power iteration
        let mut v = vec![1.0f64; dim];
        let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= v_norm;
        }

        let mut eigenvalue = 0.0f64;
        for _ in 0..max_iters {
            // w = cov * v
            let mut w = vec![0.0f64; dim];
            for i in 0..dim {
                let mut s = 0.0f64;
                for j in 0..dim {
                    s += deflated_cov[i * dim + j] * v[j];
                }
                w[i] = s;
            }

            let new_eigenvalue: f64 = w.iter().zip(&v).map(|(a, b)| a * b).sum();
            let w_norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if w_norm < 1e-15 {
                break;
            }
            for x in &mut w {
                *x /= w_norm;
            }
            v = w;

            if (new_eigenvalue - eigenvalue).abs() < tol {
                eigenvalue = new_eigenvalue;
                break;
            }
            eigenvalue = new_eigenvalue;
        }

        if eigenvalue < 1e-15 {
            break;
        }
        eigenvalues.push(eigenvalue);

        // Deflate: cov = cov - λ * v * v^T
        for i in 0..dim {
            for j in 0..dim {
                deflated_cov[i * dim + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    if eigenvalues.is_empty() {
        return 1.0;
    }

    // Participation ratio: (Σλ)² / Σλ²
    let sum: f64 = eigenvalues.iter().sum();
    let sum_sq: f64 = eigenvalues.iter().map(|l| l * l).sum();
    if sum_sq < 1e-30 {
        return 1.0;
    }

    ((sum * sum) / sum_sq) as f32
}

/// Compute all embedding quality metrics at once.
pub fn evaluate_embeddings(embeddings: &[Vec<f32>], groups: Option<&[usize]>) -> EmbeddingMetrics {
    EmbeddingMetrics {
        anisotropy: anisotropy(embeddings),
        similarity_range: similarity_range(embeddings),
        discrimination_gap: groups.map(|g| discrimination_gap(embeddings, g)),
        intrinsic_dimensionality: intrinsic_dimensionality(embeddings),
        sample_size: embeddings.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orthogonal_basis(dim: usize, count: usize) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = 1.0;
                v
            })
            .collect()
    }

    #[test]
    fn orthogonal_vectors_low_anisotropy() {
        let vecs = orthogonal_basis(8, 8);
        let a = anisotropy(&vecs);
        assert!(a.abs() < 0.01, "expected ~0, got {a}");
    }

    #[test]
    fn identical_vectors_high_anisotropy() {
        let v = vec![0.5f32; 4];
        let vecs = vec![v.clone(), v.clone(), v.clone()];
        let a = anisotropy(&vecs);
        assert!((a - 1.0).abs() < 0.01, "expected ~1, got {a}");
    }

    #[test]
    fn mean_centering_reduces_anisotropy() {
        let base = vec![1.0f32, 0.0, 0.0, 0.0];
        let vecs: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut v = base.clone();
                v[1] = (i as f32) * 0.05;
                normalize(&mut v);
                v
            })
            .collect();

        let before = anisotropy(&vecs);
        let centered = mean_center(&vecs);
        let after = anisotropy(&centered);
        assert!(after < before, "{before} -> {after}");
    }

    #[test]
    fn discrimination_gap_positive_for_clusters() {
        let mut embeddings = Vec::new();
        let mut groups = Vec::new();

        for i in 0..5 {
            let mut v = vec![1.0, 0.0, 0.0, 0.0];
            v[1] = (i as f32) * 0.01;
            normalize(&mut v);
            embeddings.push(v);
            groups.push(0);
        }

        for i in 0..5 {
            let mut v = vec![0.0, 0.0, 1.0, 0.0];
            v[3] = (i as f32) * 0.01;
            normalize(&mut v);
            embeddings.push(v);
            groups.push(1);
        }

        let gap = discrimination_gap(&embeddings, &groups);
        assert!(gap > 0.05, "got {gap}");
    }

    #[test]
    fn similarity_range_basic() {
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.7071, 0.7071, 0.0],
        ];
        let range = similarity_range(&vecs);
        assert!(range > 0.5, "got {range}");
    }

    #[test]
    fn intrinsic_dim_single_axis() {
        // All variance on one axis → id ≈ 1
        let vecs: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 * 0.1, 0.0, 0.0, 0.0])
            .collect();
        let id = intrinsic_dimensionality(&vecs);
        assert!(id < 1.5, "expected ~1, got {id}");
    }

    #[test]
    fn intrinsic_dim_uniform_spread() {
        // Random unit vectors in high-dim space spread across many axes
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, StandardNormal};

        let dim = 16;
        let mut rng = StdRng::seed_from_u64(42);
        let vecs: Vec<Vec<f32>> = (0..50)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect();

        let id = intrinsic_dimensionality(&vecs);
        // Random vectors in 16d should have high intrinsic dimensionality
        assert!(id > 5.0, "expected high id, got {id}");
    }

    #[test]
    fn evaluate_embeddings_combines() {
        let vecs = orthogonal_basis(4, 4);
        let m = evaluate_embeddings(&vecs, None);
        assert!(m.anisotropy.abs() < 0.01);
        assert!(m.discrimination_gap.is_none());
        assert_eq!(m.sample_size, 4);
    }

    #[test]
    fn single_embedding_edge_cases() {
        let single = vec![vec![1.0, 0.0]];
        assert_eq!(anisotropy(&single), 0.0);
        assert_eq!(similarity_range(&single), 0.0);
        assert!((intrinsic_dimensionality(&single) - 1.0).abs() < 1e-6);
    }
}
