//! Experiment 6: LongMemEval QA — End-to-End Evaluation
//!
//! Real end-to-end question answering: ingest sessions, retrieve context,
//! output JSONL for LLM scoring, or score pre-generated LLM answers.
//!
//! Two modes:
//! - **Retrieval:** builds per-question HNSW, retrieves context, emits JSONL
//! - **Scoring:** reads LLM-generated answers JSONL, computes accuracy

use std::path::PathBuf;

use serde::Serialize;

use crate::embed::Embedder;
use crate::evals::EvalError;
use crate::evals::longmemeval::LongMemEvalDataset;
use crate::evals::qa;

/// Configuration for the QA experiment.
pub struct QaExperimentConfig {
    /// Number of context sessions to retrieve per question.
    pub top_k: usize,
    /// If set, write JSONL to this path instead of stdout.
    pub output_path: Option<PathBuf>,
    /// If set, score answers from this JSONL file instead of running retrieval.
    pub answers_path: Option<PathBuf>,
}

/// Results from the QA experiment.
#[derive(Debug, Clone, Serialize)]
pub struct QaExperimentResult {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retrieval_recall_any_at_k: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_time_per_question_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overall_accuracy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_questions: Option<usize>,
}

/// Run Experiment 6: LongMemEval QA.
pub fn run(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
    config: &QaExperimentConfig,
) -> Result<QaExperimentResult, EvalError> {
    if let Some(ref answers_path) = config.answers_path {
        run_scoring(answers_path)
    } else {
        run_retrieval(dataset, embedder, config)
    }
}

fn run_scoring(answers_path: &PathBuf) -> Result<QaExperimentResult, EvalError> {
    let content = std::fs::read_to_string(answers_path)
        .map_err(|e| EvalError::Other(e.to_string()))?;
    let records: Vec<qa::QaAnswerRecord> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| EvalError::Other(e.to_string()))?;

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

    Ok(QaExperimentResult {
        mode: "scoring".to_string(),
        retrieval_recall_any_at_k: None,
        avg_time_per_question_ms: None,
        overall_accuracy: Some(qa_report.overall_accuracy),
        n_questions: Some(qa_report.n_questions),
    })
}

fn run_retrieval(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
    config: &QaExperimentConfig,
) -> Result<QaExperimentResult, EvalError> {
    let qa_config = qa::QaConfig {
        top_k: config.top_k,
    };

    let (records, summary) = qa::run_qa_retrieval(dataset, embedder, &qa_config)
        .map_err(|e| EvalError::Other(e.to_string()))?;

    eprintln!(
        "  retrieval recall_any@{}: {:.1}% ({:.0}ms avg/question)",
        config.top_k,
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

    if let Some(ref output_path) = config.output_path {
        std::fs::write(output_path, &jsonl)
            .map_err(|e| EvalError::Other(e.to_string()))?;
        eprintln!("  wrote {} records to {:?}", records.len(), output_path);
    } else {
        println!("{jsonl}");
    }

    Ok(QaExperimentResult {
        mode: "retrieval".to_string(),
        retrieval_recall_any_at_k: Some(summary.retrieval_recall_any_at_k),
        avg_time_per_question_ms: Some(summary.avg_time_per_question_ms),
        overall_accuracy: None,
        n_questions: Some(records.len()),
    })
}
