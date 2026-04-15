//! Search quality metrics for evaluating retrieval results.
//!
//! All functions operate on ranked result lists compared against ground truth.
//! They are pure — no dependency on any specific index implementation.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// A single query evaluation case.
pub struct QueryEval {
    pub query_id: String,
    /// Ranked list of retrieved document IDs (best first).
    pub retrieved: Vec<String>,
    /// Ground truth: map from doc ID to relevance grade (0 = not relevant).
    pub judgments: HashMap<String, u32>,
}

/// Aggregated search quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub mrr: f64,
    pub ndcg_at_k: f64,
    pub k: usize,
    pub n_queries: usize,
}

/// Precision at k: fraction of top-k results that are relevant.
pub fn precision_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let top_k = retrieved.iter().take(k);
    let hits = top_k.filter(|id| relevant.contains(id.as_str())).count();
    hits as f64 / k as f64
}

/// Recall at k: fraction of relevant docs found in top-k results.
pub fn recall_at_k(retrieved: &[String], relevant: &HashSet<String>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let top_k: HashSet<&str> = retrieved.iter().take(k).map(|s| s.as_str()).collect();
    let hits = relevant
        .iter()
        .filter(|id| top_k.contains(id.as_str()))
        .count();
    hits as f64 / relevant.len() as f64
}

/// Mean Reciprocal Rank: 1/rank of first relevant result.
pub fn mrr(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    for (i, id) in retrieved.iter().enumerate() {
        if relevant.contains(id.as_str()) {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Discounted Cumulative Gain at k with graded relevance.
fn dcg_at_k(retrieved: &[String], judgments: &HashMap<String, u32>, k: usize) -> f64 {
    retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let grade = *judgments.get(id.as_str()).unwrap_or(&0) as f64;
            // Using the standard formula: (2^rel - 1) / log2(i + 2)
            (grade.exp2() - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum()
}

/// Normalized Discounted Cumulative Gain at k.
///
/// Compares actual DCG against ideal DCG (perfect ranking).
/// Handles graded relevance — grade 0 = irrelevant, higher = more relevant.
pub fn ndcg_at_k(retrieved: &[String], judgments: &HashMap<String, u32>, k: usize) -> f64 {
    let actual = dcg_at_k(retrieved, judgments, k);

    // Ideal ranking: sort all judged docs by grade descending
    let mut ideal_grades: Vec<u32> = judgments.values().copied().collect();
    ideal_grades.sort_unstable_by(|a, b| b.cmp(a));

    let ideal: f64 = ideal_grades
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &grade)| {
            let g = grade as f64;
            (g.exp2() - 1.0) / (i as f64 + 2.0).log2()
        })
        .sum();

    if ideal == 0.0 { 0.0 } else { actual / ideal }
}

/// Recall-any at k: fraction of queries where **at least one** gold document appears in top-k.
///
/// This is the metric MemPalace headlines as "recall@5" — a lenient measure
/// that only requires one relevant hit per query.
pub fn recall_any_at_k(queries: &[QueryEval], k: usize) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }
    let hits = queries
        .iter()
        .filter(|q| {
            let relevant: HashSet<&str> = q
                .judgments
                .iter()
                .filter(|(_, g)| **g > 0)
                .map(|(id, _)| id.as_str())
                .collect();
            let top_k: HashSet<&str> = q.retrieved.iter().take(k).map(|s| s.as_str()).collect();
            top_k.iter().any(|id| relevant.contains(id))
        })
        .count();
    hits as f64 / queries.len() as f64
}

/// Recall-all at k: fraction of queries where **all** gold documents appear in top-k.
///
/// A strict measure — queries with multiple gold sessions must have *every* one
/// in top-k. MemPalace conspicuously doesn't headline this metric.
pub fn recall_all_at_k(queries: &[QueryEval], k: usize) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }
    let hits = queries
        .iter()
        .filter(|q| {
            let relevant: HashSet<&str> = q
                .judgments
                .iter()
                .filter(|(_, g)| **g > 0)
                .map(|(id, _)| id.as_str())
                .collect();
            if relevant.is_empty() {
                return false;
            }
            let top_k: HashSet<&str> = q.retrieved.iter().take(k).map(|s| s.as_str()).collect();
            relevant.iter().all(|id| top_k.contains(id))
        })
        .count();
    hits as f64 / queries.len() as f64
}

/// Compute all search metrics for a batch of queries at the given k.
pub fn evaluate_search(queries: &[QueryEval], k: usize) -> SearchMetrics {
    if queries.is_empty() {
        return SearchMetrics {
            precision_at_k: 0.0,
            recall_at_k: 0.0,
            mrr: 0.0,
            ndcg_at_k: 0.0,
            k,
            n_queries: 0,
        };
    }

    let n = queries.len() as f64;
    let mut sum_p = 0.0;
    let mut sum_r = 0.0;
    let mut sum_mrr = 0.0;
    let mut sum_ndcg = 0.0;

    for q in queries {
        let relevant: HashSet<String> = q
            .judgments
            .iter()
            .filter(|(_, g)| **g > 0)
            .map(|(id, _)| id.clone())
            .collect();

        sum_p += precision_at_k(&q.retrieved, &relevant, k);
        sum_r += recall_at_k(&q.retrieved, &relevant, k);
        sum_mrr += mrr(&q.retrieved, &relevant);
        sum_ndcg += ndcg_at_k(&q.retrieved, &q.judgments, k);
    }

    SearchMetrics {
        precision_at_k: sum_p / n,
        recall_at_k: sum_r / n,
        mrr: sum_mrr / n,
        ndcg_at_k: sum_ndcg / n,
        k,
        n_queries: queries.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relevant_set(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    fn retrieved(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn precision_perfect() {
        let r = retrieved(&["a", "b", "c"]);
        let rel = relevant_set(&["a", "b", "c"]);
        assert!((precision_at_k(&r, &rel, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn precision_half() {
        let r = retrieved(&["a", "x", "b", "y"]);
        let rel = relevant_set(&["a", "b"]);
        assert!((precision_at_k(&r, &rel, 4) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn recall_partial() {
        let r = retrieved(&["a", "x"]);
        let rel = relevant_set(&["a", "b", "c"]);
        let rc = recall_at_k(&r, &rel, 2);
        assert!((rc - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn mrr_first() {
        let r = retrieved(&["a", "b"]);
        let rel = relevant_set(&["a"]);
        assert!((mrr(&r, &rel) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mrr_second() {
        let r = retrieved(&["x", "a"]);
        let rel = relevant_set(&["a"]);
        assert!((mrr(&r, &rel) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn mrr_none() {
        let r = retrieved(&["x", "y"]);
        let rel = relevant_set(&["a"]);
        assert_eq!(mrr(&r, &rel), 0.0);
    }

    #[test]
    fn ndcg_perfect_binary() {
        let r = retrieved(&["a", "b"]);
        let mut j = HashMap::new();
        j.insert("a".to_string(), 1);
        j.insert("b".to_string(), 1);
        let score = ndcg_at_k(&r, &j, 2);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ndcg_reversed_graded() {
        // Ideal order: b(3), a(1). Actual: a(1), b(3). Should be < 1.0
        let r = retrieved(&["a", "b"]);
        let mut j = HashMap::new();
        j.insert("a".to_string(), 1);
        j.insert("b".to_string(), 3);
        let score = ndcg_at_k(&r, &j, 2);
        assert!(score < 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn evaluate_search_aggregates() {
        let queries = vec![
            QueryEval {
                query_id: "q1".to_string(),
                retrieved: retrieved(&["a", "b"]),
                judgments: HashMap::from([("a".to_string(), 1), ("b".to_string(), 1)]),
            },
            QueryEval {
                query_id: "q2".to_string(),
                retrieved: retrieved(&["x", "y"]),
                judgments: HashMap::from([("a".to_string(), 1)]),
            },
        ];

        let m = evaluate_search(&queries, 2);
        assert_eq!(m.n_queries, 2);
        assert_eq!(m.k, 2);
        // q1: p=1.0, q2: p=0.0 → avg=0.5
        assert!((m.precision_at_k - 0.5).abs() < 1e-10);
    }

    #[test]
    fn empty_queries() {
        let m = evaluate_search(&[], 5);
        assert_eq!(m.n_queries, 0);
        assert_eq!(m.precision_at_k, 0.0);
    }

    #[test]
    fn recall_empty_relevant() {
        let r = retrieved(&["a"]);
        let rel = relevant_set(&[]);
        assert_eq!(recall_at_k(&r, &rel, 1), 0.0);
    }

    #[test]
    fn recall_any_finds_one() {
        let queries = vec![
            QueryEval {
                query_id: "q1".to_string(),
                retrieved: retrieved(&["a", "x", "y", "z", "w"]),
                judgments: HashMap::from([("a".to_string(), 1), ("b".to_string(), 1)]),
            },
            QueryEval {
                query_id: "q2".to_string(),
                retrieved: retrieved(&["x", "y", "z", "w", "v"]),
                judgments: HashMap::from([("a".to_string(), 1)]),
            },
        ];
        // q1: "a" is in top-5 → hit. q2: "a" not in top-5 → miss. → 0.5
        assert!((recall_any_at_k(&queries, 5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn recall_all_requires_every_gold() {
        let queries = vec![QueryEval {
            query_id: "q1".to_string(),
            retrieved: retrieved(&["a", "x", "b", "y", "z"]),
            judgments: HashMap::from([
                ("a".to_string(), 1),
                ("b".to_string(), 1),
                ("c".to_string(), 1), // c not in top-5
            ]),
        }];
        // "a" and "b" present but "c" missing → 0.0
        assert_eq!(recall_all_at_k(&queries, 5), 0.0);
    }

    #[test]
    fn recall_all_perfect() {
        let queries = vec![QueryEval {
            query_id: "q1".to_string(),
            retrieved: retrieved(&["a", "b", "c", "x", "y"]),
            judgments: HashMap::from([
                ("a".to_string(), 1),
                ("b".to_string(), 1),
                ("c".to_string(), 1),
            ]),
        }];
        assert!((recall_all_at_k(&queries, 5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn recall_any_empty() {
        assert_eq!(recall_any_at_k(&[], 5), 0.0);
    }

    #[test]
    fn recall_all_empty() {
        assert_eq!(recall_all_at_k(&[], 5), 0.0);
    }
}
