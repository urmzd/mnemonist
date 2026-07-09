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

/// Wilson score interval for a binomial proportion.
///
/// Returns `(lo, hi)` bounds for the true success rate given `successes` out of
/// `n` trials at critical value `z` (1.96 for 95%). Unlike the normal
/// approximation it behaves sensibly near 0%/100% and at small n. With `n == 0`
/// there is no information, so the interval is the full `(0, 1)`.
pub fn wilson_interval(successes: usize, n: usize, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n_f = n as f64;
    let p = successes as f64 / n_f;
    let z2 = z * z;
    let denom = 1.0 + z2 / n_f;
    let center = (p + z2 / (2.0 * n_f)) / denom;
    let margin = (z / denom) * (p * (1.0 - p) / n_f + z2 / (4.0 * n_f * n_f)).sqrt();
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

/// 95% Wilson interval as a `[lo, hi]` pair for embedding in serialized reports.
pub fn wilson95(successes: usize, n: usize) -> [f64; 2] {
    let (lo, hi) = wilson_interval(successes, n, 1.96);
    [lo, hi]
}

/// Exact McNemar test on paired binary outcomes.
///
/// `b` and `c` are the discordant pair counts (arm A hit where arm B missed,
/// and vice versa). Concordant pairs carry no information about the difference.
/// Returns the two-sided p-value from the exact binomial reference
/// distribution `min(b, c) ~ Binomial(b + c, 0.5)` — no chi-square
/// approximation, so it is valid at any count. `b + c == 0` returns 1.0.
pub fn mcnemar_exact_p(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    // P(X <= k) for X ~ Binomial(n, 0.5), accumulating pmf iteratively:
    // pmf(0) = 0.5^n, pmf(i+1) = pmf(i) * (n - i) / (i + 1).
    let mut pmf = 0.5f64.powi(n as i32);
    let mut cdf = pmf;
    for i in 0..k {
        pmf *= (n - i) as f64 / (i + 1) as f64;
        cdf += pmf;
    }
    (2.0 * cdf).min(1.0)
}

/// Count discordant pairs between two paired hit vectors.
///
/// Returns `(b, c)`: `b` = hits in `a` that miss in `other`, `c` = the reverse.
/// Panics if the vectors are not the same length — they must be paired
/// per-query outcomes over the identical query population.
pub fn discordant_pairs(a: &[bool], other: &[bool]) -> (usize, usize) {
    assert_eq!(
        a.len(),
        other.len(),
        "paired hit vectors must cover the same query population"
    );
    let mut b = 0;
    let mut c = 0;
    for (&x, &y) in a.iter().zip(other) {
        match (x, y) {
            (true, false) => b += 1,
            (false, true) => c += 1,
            _ => {}
        }
    }
    (b, c)
}

/// Recall-any at k: fraction of queries where **at least one** gold document appears in top-k.
///
/// A lenient measure that only requires one relevant hit per query — it
/// saturates quickly on easy constructs and should not be headlined alone.
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
/// in top-k. This is the discriminating metric when recall-any saturates.
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

    #[test]
    fn wilson_known_value() {
        // 3/30 at 95%: the textbook Wilson interval is [0.0346, 0.2562].
        let (lo, hi) = wilson_interval(3, 30, 1.96);
        assert!((lo - 0.0346).abs() < 1e-3, "lo = {lo}");
        assert!((hi - 0.2562).abs() < 1e-3, "hi = {hi}");
    }

    #[test]
    fn wilson_bounds_contain_proportion() {
        for &(s, n) in &[(0usize, 10usize), (5, 10), (10, 10), (186, 500), (1, 2)] {
            let (lo, hi) = wilson_interval(s, n, 1.96);
            let p = s as f64 / n as f64;
            assert!((0.0..=1.0).contains(&lo));
            assert!((0.0..=1.0).contains(&hi));
            assert!(lo <= p && p <= hi, "({s}, {n}): [{lo}, {hi}] vs {p}");
        }
    }

    #[test]
    fn wilson_zero_successes_starts_at_zero() {
        let (lo, hi) = wilson_interval(0, 20, 1.96);
        assert_eq!(lo, 0.0);
        assert!(hi > 0.0 && hi < 0.25);
    }

    #[test]
    fn wilson_all_successes_ends_at_one() {
        let (lo, hi) = wilson_interval(20, 20, 1.96);
        assert_eq!(hi, 1.0);
        assert!(lo > 0.75 && lo < 1.0);
    }

    #[test]
    fn wilson_empty_sample_is_uninformative() {
        assert_eq!(wilson_interval(0, 0, 1.96), (0.0, 1.0));
    }

    #[test]
    fn wilson_narrows_with_n() {
        let (lo_small, hi_small) = wilson_interval(10, 100, 1.96);
        let (lo_big, hi_big) = wilson_interval(100, 1000, 1.96);
        assert!(hi_big - lo_big < hi_small - lo_small);
    }

    #[test]
    fn mcnemar_no_discordance_is_one() {
        assert_eq!(mcnemar_exact_p(0, 0), 1.0);
    }

    #[test]
    fn mcnemar_balanced_discordance_is_one() {
        assert_eq!(mcnemar_exact_p(7, 7), 1.0);
    }

    #[test]
    fn mcnemar_one_sided_extreme() {
        // 10 vs 0 discordant pairs: p = 2 * 0.5^10 = 0.001953125 exactly.
        let p = mcnemar_exact_p(10, 0);
        assert!((p - 0.001953125).abs() < 1e-12, "p = {p}");
    }

    #[test]
    fn mcnemar_symmetric_in_args() {
        assert_eq!(mcnemar_exact_p(3, 9), mcnemar_exact_p(9, 3));
    }

    #[test]
    fn mcnemar_known_value() {
        // b=2, c=8: p = 2 * P(X <= 2), X ~ Bin(10, 0.5)
        //         = 2 * (1 + 10 + 45) / 1024 = 0.109375 exactly.
        let p = mcnemar_exact_p(2, 8);
        assert!((p - 0.109375).abs() < 1e-12, "p = {p}");
    }

    #[test]
    fn discordant_pairs_counts() {
        let a = [true, true, false, false, true];
        let b = [true, false, true, false, false];
        assert_eq!(discordant_pairs(&a, &b), (2, 1));
    }
}
