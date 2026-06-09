//! Offline smoke tests for the benchmark harness.
//!
//! Runs every experiment against a committed ~10-question LongMemEval fixture
//! with a deterministic hash-based embedder — no models, no network. The point
//! is mechanical regression-proofing of the measurement layer itself:
//! denominators (eval-slice populations), output shapes, and the JSON report.
//!
//! The fixture is engineered so the expected recalls are exact: queries q0–q7
//! share most tokens with their gold session (hits), while q8/q9 share three
//! tokens with six distractor sessions and only one with gold, pinning gold
//! outside top-5 but inside top-10. Eval-slice (q5–q9) recall@5 is therefore
//! 3/5, while all-query recall@5 is 8/10 — so any experiment that silently
//! scores an arm on the wrong query population produces a different number
//! and fails here.

#![cfg(feature = "evals")]

use std::path::PathBuf;

use mnemonist_core::Error;
use mnemonist_core::embed::Embedder;
use mnemonist_core::evals::bench::{
    self, latency_scaling, longmemeval_qa, mempalace_comparison, storage_footprint,
    temporal_retrieval, vector_retrieval,
};
use mnemonist_core::evals::longmemeval::{LongMemEvalDataset, load_dataset};
use mnemonist_core::evals::qa;

const DIM: usize = 256;

/// Deterministic bag-of-words embedder: each whitespace token is FNV-1a-hashed
/// into one of `DIM` buckets, counts are L2-normalized. Token overlap maps to
/// cosine similarity, so retrieval quality on the fixture is fully predictable.
struct HashEmbedder;

fn fnv1a(token: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in token.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let mut v = vec![0.0f32; DIM];
        for token in text.split_whitespace() {
            v[(fnv1a(token) % DIM as u64) as usize] += 1.0;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        Ok(v)
    }

    fn dimension(&self) -> Result<usize, Error> {
        Ok(DIM)
    }
}

fn fixture() -> LongMemEvalDataset {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/longmemeval_tiny.json");
    load_dataset(&path).expect("fixture must parse")
}

fn assert_ci_contains(ci: [f64; 2], value: f64) {
    assert!(
        ci[0] <= value && value <= ci[1] && ci[0] >= 0.0 && ci[1] <= 1.0,
        "CI {ci:?} must contain {value}"
    );
}

#[test]
fn fixture_shape() {
    let ds = fixture();
    assert_eq!(ds.queries.len(), 10);
    assert_eq!(ds.sessions.len(), 16);
    // One abstention question (no gold answer), per LongMemEval convention.
    let abstentions = ds.queries.iter().filter(|q| q.answer.is_none()).count();
    assert_eq!(abstentions, 1);
}

#[test]
fn vector_retrieval_denominators_and_recall() {
    let ds = fixture();
    let r = vector_retrieval::run(&ds, &HashEmbedder).unwrap();

    assert_eq!(r.n_queries, 10);
    assert_eq!(r.n_sessions, 16);
    // 8 of 10 queries have gold in top-5; all 10 within top-10 (haystacks <= 8 docs).
    assert!(
        (r.recall_any_at_5 - 0.8).abs() < 1e-9,
        "{}",
        r.recall_any_at_5
    );
    assert!((r.recall_any_at_10 - 1.0).abs() < 1e-9);
    assert_ci_contains(r.recall_any_at_5_ci95, r.recall_any_at_5);
    assert_ci_contains(r.recall_all_at_5_ci95, r.recall_all_at_5);
    assert_ci_contains(r.recall_any_at_10_ci95, r.recall_any_at_10);
    assert_ci_contains(r.recall_all_at_10_ci95, r.recall_all_at_10);
}

#[test]
fn latency_scaling_repeats_and_deterministic_subsets() {
    let ds = fixture();
    let l = latency_scaling::run(&ds, &HashEmbedder, &[8, 100]).unwrap();

    assert_eq!(l.scale_points.len(), 2);
    assert_eq!(l.scale_points[0].n_documents, 8);
    // Requested 100 but the corpus has 16 documents.
    assert_eq!(l.scale_points[1].n_documents, 16);
    for p in &l.scale_points {
        assert_eq!(p.n_queries, 10);
        assert_eq!(p.n_runs, 5);
        assert_eq!(p.n_warmup_queries, 10);
        assert!(p.query_latency_p50_us <= p.query_latency_p95_us);
        assert!(p.query_latency_p95_us <= p.query_latency_p99_us);
    }
}

#[test]
fn storage_footprint_baseline_and_paired_tests() {
    let ds = fixture();
    let s = storage_footprint::run(&ds, &HashEmbedder, &[1, 4], 42).unwrap();

    assert_eq!(s.n_vectors, 16);
    assert_eq!(s.n_queries, 10);
    assert_eq!(s.dimension, DIM);
    // Global brute-force baseline: same 8/10 + 10/10 structure as Exp 1.
    assert!((s.raw_recall_any_at_5 - 0.8).abs() < 1e-9);
    assert!((s.raw_recall_any_at_10 - 1.0).abs() < 1e-9);
    assert_eq!(s.quantized.len(), 2);
    for q in &s.quantized {
        assert!(q.compression_ratio > 1.0);
        assert!((0.0..=1.0).contains(&q.recall_any_at_5));
        assert!(q.mcnemar_p_vs_raw_at_5 > 0.0 && q.mcnemar_p_vs_raw_at_5 <= 1.0);
        assert!(q.mcnemar_p_vs_raw_at_10 > 0.0 && q.mcnemar_p_vs_raw_at_10 <= 1.0);
    }
}

/// The denominator test that would have caught the published +9.4pp artifact:
/// the baseline must be scored on the held-out eval slice (recall 3/5), not on
/// all queries (recall 8/10). Both arms share that denominator.
#[test]
fn temporal_baseline_uses_eval_slice_denominator() {
    let ds = fixture();
    let t = temporal_retrieval::run(&ds, &HashEmbedder, 2).unwrap();

    assert_eq!(t.n_documents, 16);
    assert_eq!(t.n_queries_per_cycle, 5);
    assert_eq!(t.n_eval_queries, 5);
    assert!(
        (t.baseline_recall_any_at_5 - 0.6).abs() < 1e-9,
        "baseline {} must be eval-slice recall (3/5), not all-query recall (8/10)",
        t.baseline_recall_any_at_5
    );
    // Zero-reinforcement control through the rerank path matches the baseline.
    assert!((t.control_recall_any_at_5 - t.baseline_recall_any_at_5).abs() < 1e-9);
    // Reinforcing s0–s4 cannot flip held-out outcomes in this fixture, so the
    // arms agree on every query and the paired test is a clean null.
    assert!((t.reinforced_recall_any_at_5 - t.control_recall_any_at_5).abs() < 1e-9);
    assert_eq!(t.n_discordant_reinforced_vs_control, [0, 0]);
    assert!((t.mcnemar_p_reinforced_vs_control - 1.0).abs() < 1e-12);
    // The delta isolates the reinforcement signal: reinforced minus control,
    // not minus the pure-cosine baseline (which would credit the reranker).
    assert!(
        (t.recall_delta - (t.reinforced_recall_any_at_5 - t.control_recall_any_at_5)).abs() < 1e-12
    );
}

#[test]
fn mempalace_comparison_denominators() {
    let ds = fixture();
    let m = mempalace_comparison::run(&ds, &HashEmbedder).unwrap();

    assert_eq!(m.n_queries, 10);
    assert_eq!(m.n_sessions, 16);
    assert!((m.mnemonist_recall_any_at_5 - 0.8).abs() < 1e-9);
}

#[test]
fn qa_retrieval_emits_one_record_per_question() {
    let ds = fixture();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("context.jsonl");

    let config = longmemeval_qa::QaExperimentConfig {
        top_k: 5,
        output_path: Some(out.clone()),
        answers_path: None,
    };
    let q = longmemeval_qa::run(&ds, &HashEmbedder, &config).unwrap();

    assert_eq!(q.mode, "retrieval");
    assert_eq!(q.n_questions, Some(10));
    assert!((q.retrieval_recall_any_at_k.unwrap() - 0.8).abs() < 1e-9);

    let content = std::fs::read_to_string(&out).unwrap();
    let records: Vec<qa::QaContextRecord> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.len(), 10);
    let hits = records.iter().filter(|r| r.retrieval_hit).count();
    assert_eq!(hits, 8);
    for r in &records {
        assert!(r.retrieved_session_ids.len() <= 5);
        assert_eq!(r.retrieved_context.len(), r.retrieved_session_ids.len());
    }
}

#[test]
fn qa_scoring_mode_counts_correct_answers() {
    let dir = tempfile::tempdir().unwrap();
    let answers = dir.path().join("answers.jsonl");
    let records = [
        serde_json::json!({
            "question_id": "q0", "question_type": "single-session-user",
            "question": "apple orchard cider", "gold_answer": "cider",
            "model_answer": "It was cider."
        }),
        serde_json::json!({
            "question_id": "q1", "question_type": "multi-session",
            "question": "rocket orbital telemetry", "gold_answer": "telemetry",
            "model_answer": "No idea."
        }),
        serde_json::json!({
            "question_id": "q7_abs", "question_type": "single-session-user",
            "question": "aquarium coral salinity", "gold_answer": null,
            "model_answer": "I don't know."
        }),
    ];
    let jsonl: Vec<String> = records.iter().map(|r| r.to_string()).collect();
    std::fs::write(&answers, jsonl.join("\n")).unwrap();

    let ds = fixture();
    let config = longmemeval_qa::QaExperimentConfig {
        top_k: 5,
        output_path: None,
        answers_path: Some(answers),
    };
    let q = longmemeval_qa::run(&ds, &HashEmbedder, &config).unwrap();

    assert_eq!(q.mode, "scoring");
    assert_eq!(q.n_questions, Some(3));
    let acc = q.overall_accuracy.unwrap();
    assert!((acc - 2.0 / 3.0).abs() < 1e-9);
    let ci = q.overall_accuracy_ci95.unwrap();
    assert!(ci[0] <= acc && acc <= ci[1]);
}

#[test]
fn full_report_serializes_to_json() {
    let ds = fixture();
    let report = bench::BenchReport {
        timestamp: "0s".to_string(),
        retrieval: Some(vector_retrieval::run(&ds, &HashEmbedder).unwrap()),
        latency: Some(latency_scaling::run(&ds, &HashEmbedder, &[8]).unwrap()),
        storage: Some(storage_footprint::run(&ds, &HashEmbedder, &[2], 42).unwrap()),
        temporal: Some(temporal_retrieval::run(&ds, &HashEmbedder, 2).unwrap()),
        mempalace: Some(mempalace_comparison::run(&ds, &HashEmbedder).unwrap()),
        qa: None,
    };

    let json: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
    assert_eq!(json["retrieval"]["n_queries"], 10);
    assert_eq!(json["temporal"]["n_eval_queries"], 5);
    assert!(json["temporal"]["mcnemar_p_reinforced_vs_control"].is_number());
    assert!(json["storage"]["quantized"][0]["mcnemar_p_vs_raw_at_5"].is_number());
    assert!(json["retrieval"]["recall_any_at_5_ci95"].is_array());
    assert!(!report.to_summary().is_empty());
}
