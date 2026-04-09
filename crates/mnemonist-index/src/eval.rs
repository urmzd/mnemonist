use mnemonist_core::distance::{cosine_similarity, normalize};

/// Embedding quality metrics for evaluating how well embeddings
/// distribute across the vector space.
///
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

    // Compute mean vector
    let mut mean = vec![0.0f32; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            mean[i] += v;
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    // Subtract mean and re-normalize
    embeddings
        .iter()
        .map(|emb| {
            let mut centered: Vec<f32> = emb.iter().zip(&mean).map(|(v, m)| v - m).collect();
            normalize(&mut centered);
            centered
        })
        .collect()
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
        assert!(
            a.abs() < 0.01,
            "orthogonal anisotropy should be ~0, got {a}"
        );
    }

    #[test]
    fn identical_vectors_high_anisotropy() {
        let v = vec![0.5f32; 4];
        let vecs = vec![v.clone(), v.clone(), v.clone()];
        let a = anisotropy(&vecs);
        assert!(
            (a - 1.0).abs() < 0.01,
            "identical anisotropy should be ~1, got {a}"
        );
    }

    #[test]
    fn mean_centering_reduces_anisotropy() {
        // Create clustered vectors (high anisotropy)
        let base = vec![1.0f32, 0.0, 0.0, 0.0];
        let vecs: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let mut v = base.clone();
                v[1] = (i as f32) * 0.05; // small perturbation
                let mut normalized = v;
                normalize(&mut normalized);
                normalized
            })
            .collect();

        let before = anisotropy(&vecs);
        let centered = mean_center(&vecs);
        let after = anisotropy(&centered);

        assert!(
            after < before,
            "mean centering should reduce anisotropy: {before} -> {after}"
        );
    }

    #[test]
    fn discrimination_gap_positive_for_clusters() {
        // Two well-separated clusters
        let mut embeddings = Vec::new();
        let mut groups = Vec::new();

        // Cluster 0: near [1, 0, 0, 0]
        for i in 0..5 {
            let mut v = vec![1.0, 0.0, 0.0, 0.0];
            v[1] = (i as f32) * 0.01;
            normalize(&mut v);
            embeddings.push(v);
            groups.push(0);
        }

        // Cluster 1: near [0, 0, 1, 0]
        for i in 0..5 {
            let mut v = vec![0.0, 0.0, 1.0, 0.0];
            v[3] = (i as f32) * 0.01;
            normalize(&mut v);
            embeddings.push(v);
            groups.push(1);
        }

        let gap = discrimination_gap(&embeddings, &groups);
        assert!(gap > 0.05, "clusters should have positive gap, got {gap}");
    }

    #[test]
    fn similarity_range_basic() {
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
                0.0,
            ], // 45 degrees from both
        ];

        let range = similarity_range(&vecs);
        // Min sim is between [1,0,0] and [0,1,0] = 0.0
        // Max sim is between [1,0,0] and [0.7,0.7,0] ≈ 0.707
        assert!(range > 0.5, "expected range > 0.5, got {range}");
    }

    #[test]
    fn single_embedding_edge_cases() {
        let single = vec![vec![1.0, 0.0]];
        assert_eq!(anisotropy(&single), 0.0);
        assert_eq!(similarity_range(&single), 0.0);
    }
}
