//! mnemonist-bench: Benchmark suite comparing mnemonist retrieval infrastructure.
//!
//! Runs up to 6 experiments against a LongMemEval dataset:
//!   1. Vector retrieval (recall_any@5, recall_all@5) — raw session retrieval, NOT QA
//!   2. Latency scaling (index build + p50/p95/p99 query latency at 100–10k docs)
//!   3. Storage footprint (raw vs TurboQuant at 1–4 bits, with recall degradation)
//!   4. Temporal retrieval (Hebbian reinforcement vs static baseline)
//!   5. MemPalace comparison — apples-to-apples retrieval parity (NOT a LongMemEval QA score)
//!   6. LongMemEval QA — real end-to-end evaluation (see qa.rs)
//!
//! Usage:
//!   cargo run --bin mnemonist-bench --features bench-cli -- --dataset data/longmemeval_s_cleaned.json
//!   cargo run --bin mnemonist-bench --features bench-cli -- --dataset data.json --experiments 1,2 --format json

use std::path::PathBuf;

use clap::Parser;
use mnemonist_core::embed::Embedder;
use mnemonist_core::evals::bench;
use mnemonist_core::evals::longmemeval;

#[derive(Parser)]
#[command(
    name = "mnemonist-bench",
    about = "Benchmark suite for mnemonist — 6 experiments comparing retrieval infrastructure"
)]
struct Cli {
    /// Path to LongMemEval JSON dataset.
    #[arg(long)]
    dataset: PathBuf,

    /// Which experiments to run (comma-separated: 1,2,3,4,5,6 or "all").
    #[arg(long, default_value = "all")]
    experiments: String,

    /// Output format: "text" for human-readable, "json" for structured output.
    #[arg(long, default_value = "text")]
    format: String,

    /// Save results to this directory (writes both .json and .txt).
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Scale sizes for Experiment 2 (comma-separated document counts).
    #[arg(long, default_value = "100,500,1000,5000,10000")]
    scale_sizes: String,

    /// Quantization bit widths for Experiment 3 (comma-separated).
    #[arg(long, default_value = "1,2,3,4")]
    quant_bits: String,

    /// Number of consolidation cycles for Experiment 4.
    #[arg(long, default_value_t = 5)]
    temporal_cycles: usize,

    /// [Exp 6] Number of context sessions to retrieve per question.
    #[arg(long, default_value_t = 5)]
    qa_top_k: usize,

    /// [Exp 6] Output path for QA context JSONL (default: stdout).
    #[arg(long)]
    qa_output: Option<PathBuf>,

    /// [Exp 6] Path to LLM-generated answers JSONL for scoring.
    /// If provided, scores answers instead of running retrieval.
    #[arg(long)]
    qa_answers: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let experiments: Vec<usize> = if cli.experiments == "all" {
        vec![1, 2, 3, 4, 5]
    } else {
        cli.experiments
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    };

    let scale_sizes: Vec<usize> = cli
        .scale_sizes
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let quant_bits: Vec<u8> = cli
        .quant_bits
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // ── Load dataset ─────────────────────────────────────────────────────
    eprintln!("Loading dataset from {:?}...", cli.dataset);
    let dataset = longmemeval::load_dataset(&cli.dataset)?;
    eprintln!(
        "  {} sessions, {} queries",
        dataset.sessions.len(),
        dataset.queries.len()
    );

    // ── Initialize embedder ──────────────────────────────────────────────
    eprintln!("Initializing embedder (all-MiniLM-L6-v2, candle)...");
    let embedder = mnemonist_core::embed::CandleEmbedder::default_model()?;

    // ── Pre-embed everything (shared across experiments 2–4) ─────────────
    let total_sessions = dataset.sessions.len();
    eprintln!("Embedding {total_sessions} sessions...");
    let session_ids: Vec<String> = dataset.sessions.keys().cloned().collect();
    let session_texts: Vec<&str> = session_ids
        .iter()
        .map(|id| dataset.sessions[id].as_str())
        .collect();

    const BATCH_SIZE: usize = 256;
    let session_start = std::time::Instant::now();
    let mut session_embeddings: Vec<Vec<f32>> = Vec::with_capacity(total_sessions);
    for (i, chunk) in session_texts.chunks(BATCH_SIZE).enumerate() {
        let done = i * BATCH_SIZE;
        let t0 = std::time::Instant::now();
        let batch = embedder.embed_batch(chunk)?;
        eprintln!(
            "  [{done}/{total_sessions}] ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
        session_embeddings.extend(batch);
    }
    eprintln!(
        "  [{total_sessions}/{total_sessions}] done ({:.1}s total)",
        session_start.elapsed().as_secs_f64()
    );

    let embeddings: Vec<(String, Vec<f32>)> =
        session_ids.into_iter().zip(session_embeddings).collect();

    let total_queries = dataset.queries.len();
    eprintln!("Embedding {total_queries} queries...");
    let query_texts: Vec<&str> = dataset
        .queries
        .iter()
        .map(|q| q.question.as_str())
        .collect();
    let query_start = std::time::Instant::now();
    let mut query_embeddings: Vec<Vec<f32>> = Vec::with_capacity(total_queries);
    for (i, chunk) in query_texts.chunks(BATCH_SIZE).enumerate() {
        let done = i * BATCH_SIZE;
        let t0 = std::time::Instant::now();
        let batch = embedder.embed_batch(chunk)?;
        eprintln!(
            "  [{done}/{total_queries}] ({:.1}s)",
            t0.elapsed().as_secs_f64()
        );
        query_embeddings.extend(batch);
    }
    eprintln!(
        "  [{total_queries}/{total_queries}] done ({:.1}s total)",
        query_start.elapsed().as_secs_f64()
    );
    let query_gold: Vec<Vec<String>> = dataset
        .queries
        .iter()
        .map(|q| q.gold_session_ids.clone())
        .collect();

    eprintln!("Embeddings ready. Running experiments...\n");

    // ── Run experiments ──────────────────────────────────────────────────
    let mut report = bench::BenchReport {
        timestamp: format!(
            "{}s",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ),
        retrieval: None,
        latency: None,
        #[cfg(feature = "quant")]
        storage: None,
        temporal: None,
        mempalace: None,
    };

    if experiments.contains(&1) {
        eprintln!("Running Experiment 1: Vector Retrieval (NOT LongMemEval QA)...");
        match bench::vector_retrieval::run(&dataset, &embedder) {
            Ok(r) => {
                eprintln!(
                    "  recall_any@5={:.1}%  recall_all@5={:.1}%",
                    r.recall_any_at_5 * 100.0,
                    r.recall_all_at_5 * 100.0
                );
                report.retrieval = Some(r);
            }
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }

    if experiments.contains(&2) {
        eprintln!("Running Experiment 2: Latency Scaling...");
        match bench::latency_scaling::run(&embeddings, &query_embeddings, &scale_sizes) {
            Ok(l) => {
                for p in &l.scale_points {
                    eprintln!(
                        "  n={}: build={}ms p50={:.0}µs p99={:.0}µs",
                        p.n_documents,
                        p.index_build_time_ms,
                        p.query_latency_p50_us,
                        p.query_latency_p99_us
                    );
                }
                report.latency = Some(l);
            }
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }

    #[cfg(feature = "quant")]
    if experiments.contains(&3) {
        eprintln!("Running Experiment 3: Storage Footprint...");
        match bench::storage_footprint::run(
            &embeddings,
            &query_embeddings,
            &query_gold,
            &quant_bits,
            42,
        ) {
            Ok(s) => {
                for q in &s.quantized {
                    eprintln!(
                        "  {}bit: {:.1}x compression  recall@5={:.1}%",
                        q.bits,
                        q.compression_ratio,
                        q.recall_any_at_5 * 100.0
                    );
                }
                report.storage = Some(s);
            }
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }

    if experiments.contains(&4) {
        eprintln!("Running Experiment 4: Temporal Retrieval...");
        match bench::temporal_retrieval::run(
            &embeddings,
            &query_embeddings,
            &query_gold,
            cli.temporal_cycles,
        ) {
            Ok(t) => {
                eprintln!(
                    "  baseline={:.1}%  reinforced={:.1}%  delta={:+.1}%",
                    t.baseline_recall_any_at_5 * 100.0,
                    t.reinforced_recall_any_at_5 * 100.0,
                    t.recall_delta * 100.0
                );
                report.temporal = Some(t);
            }
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }

    if experiments.contains(&5) {
        eprintln!("Running Experiment 5: MemPalace Comparison (Retrieval Only)...");
        match bench::mempalace_comparison::run(&dataset, &embedder) {
            Ok(m) => {
                eprintln!(
                    "  mnemonist recall_any@5={:.1}%  recall_all@5={:.1}%",
                    m.mnemonist_recall_any_at_5 * 100.0,
                    m.mnemonist_recall_all_at_5 * 100.0
                );
                eprintln!("  note: {}", m.note);
                report.mempalace = Some(m);
            }
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }

    if experiments.contains(&6) {
        use mnemonist_core::evals::qa;

        if let Some(ref answers_path) = cli.qa_answers {
            // Scoring mode: read LLM-generated answers and score them
            eprintln!("Running Experiment 6: LongMemEval QA — Scoring...");
            let content = std::fs::read_to_string(answers_path)?;
            let records: Vec<qa::QaAnswerRecord> = content
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| serde_json::from_str(l))
                .collect::<Result<Vec<_>, _>>()?;

            let qa_report = qa::score_answers(&records);
            eprintln!(
                "  overall accuracy: {:.1}% ({}/{})",
                qa_report.overall_accuracy * 100.0,
                (qa_report.overall_accuracy * qa_report.n_questions as f64) as usize,
                qa_report.n_questions
            );
            for t in &qa_report.per_type {
                eprintln!(
                    "    {}: {:.1}% ({} questions)",
                    t.question_type,
                    t.accuracy * 100.0,
                    t.count
                );
            }
        } else {
            // Retrieval mode: ingest sessions, retrieve context, output JSONL
            eprintln!("Running Experiment 6: LongMemEval QA — Retrieval...");
            let qa_config = qa::QaConfig {
                top_k: cli.qa_top_k,
            };

            match qa::run_qa_retrieval(&dataset, &embedder, &qa_config) {
                Ok((records, summary)) => {
                    eprintln!(
                        "  retrieval recall_any@{}: {:.1}% ({:.0}ms avg/question)",
                        cli.qa_top_k,
                        summary.retrieval_recall_any_at_k * 100.0,
                        summary.avg_time_per_question_ms
                    );
                    for t in &summary.per_type_retrieval {
                        eprintln!(
                            "    {}: {:.1}% ({} questions)",
                            t.question_type,
                            t.retrieval_recall * 100.0,
                            t.count
                        );
                    }

                    // Write JSONL output
                    let jsonl: String = records
                        .iter()
                        .map(|r| serde_json::to_string(r).unwrap())
                        .collect::<Vec<_>>()
                        .join("\n");

                    if let Some(ref output_path) = cli.qa_output {
                        std::fs::write(output_path, &jsonl)?;
                        eprintln!("  wrote {} records to {:?}", records.len(), output_path);
                    } else {
                        println!("{jsonl}");
                    }
                }
                Err(e) => eprintln!("  ERROR: {e}"),
            }
        }
    }

    // ── Output (experiments 1-5) ────────────────────────────────────────
    if !experiments.contains(&6) || cli.qa_output.is_some() {
        eprintln!();
        let summary = report.to_summary();
        let json = report.to_json();

        match cli.format.as_str() {
            "json" => println!("{json}"),
            _ => println!("{summary}"),
        }

        // Save to output directory if specified
        if let Some(ref dir) = cli.output_dir {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("results.json"), &json)?;
            std::fs::write(dir.join("results.txt"), &summary)?;
            eprintln!("Results saved to {:?}", dir);
        }
    }

    Ok(())
}
