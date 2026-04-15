//! Experiment 6: End-to-end LongMemEval QA evaluation.
//!
//! This is the **real** LongMemEval benchmark — not raw vector retrieval (Experiments 1/5),
//! but actual question-answering accuracy.
//!
//! Two-phase design:
//! - **Phase A (this module):** For each question, ingest its haystack sessions,
//!   build an HNSW index, retrieve top-k context, and output JSONL.
//! - **Phase B (external scripts):** Feed context to an LLM, generate answers,
//!   and score with a judge (GPT-4o or string match).

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::embed::Embedder;
use crate::ann::AnnIndex;
use crate::ann::hnsw::{HnswConfig, HnswIndex};

use super::EvalError;
use super::longmemeval::LongMemEvalDataset;

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the QA experiment.
pub struct QaConfig {
    /// Number of context sessions to retrieve per question (default 5).
    pub top_k: usize,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self { top_k: 5 }
    }
}

// ── Output types ───────────────────────────────────────────────────────────

/// One record per question — output of Phase A (retrieval).
///
/// Written as JSONL for consumption by external LLM scripts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaContextRecord {
    pub question_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub question_type: Option<String>,
    pub question: String,
    pub retrieved_context: Vec<String>,
    pub retrieved_session_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gold_answer: Option<String>,
    pub gold_session_ids: Vec<String>,
    /// Whether any gold session was retrieved in top-k.
    pub retrieval_hit: bool,
}

/// One record per question — input for scoring (Phase B output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaAnswerRecord {
    pub question_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub question_type: Option<String>,
    pub question: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gold_answer: Option<String>,
    pub model_answer: String,
}

// ── Report types ───────────────────────────────────────────────────────────

/// Full QA evaluation report.
#[derive(Debug, Clone, Serialize)]
pub struct QaReport {
    /// Overall answer accuracy (fraction correct).
    pub overall_accuracy: f64,
    /// Retrieval recall: fraction of questions where any gold session was in top-k.
    pub retrieval_recall_any_at_k: f64,
    /// Per question-type breakdown.
    pub per_type: Vec<QuestionTypeResult>,
    pub n_questions: usize,
    pub scoring_method: String,
}

/// Per question-type accuracy breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct QuestionTypeResult {
    pub question_type: String,
    pub accuracy: f64,
    pub retrieval_recall: f64,
    pub count: usize,
}

/// Summary of the retrieval phase (before LLM scoring).
#[derive(Debug, Clone, Serialize)]
pub struct QaRetrievalSummary {
    pub n_questions: usize,
    pub retrieval_recall_any_at_k: f64,
    pub per_type_retrieval: Vec<QuestionTypeRetrieval>,
    pub total_time_ms: u64,
    pub avg_time_per_question_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuestionTypeRetrieval {
    pub question_type: String,
    pub retrieval_recall: f64,
    pub count: usize,
}

// ── Phase A: Retrieval ─────────────────────────────────────────────────────

/// Run the QA retrieval phase: for each question, build an index from its
/// haystack sessions and retrieve top-k context.
///
/// Returns a list of QaContextRecords (one per question) plus a summary.
pub fn run_qa_retrieval(
    dataset: &LongMemEvalDataset,
    embedder: &dyn Embedder,
    config: &QaConfig,
) -> Result<(Vec<QaContextRecord>, QaRetrievalSummary), EvalError> {
    let total = dataset.queries.len();
    if total == 0 {
        return Err(EvalError::InsufficientData { min: 1, got: 0 });
    }

    let overall_start = Instant::now();
    let mut records = Vec::with_capacity(total);
    let mut hits = 0usize;

    for (i, query) in dataset.queries.iter().enumerate() {
        let t0 = Instant::now();

        // Collect this question's haystack sessions
        let (session_ids, session_texts) = if !query.haystack_session_ids.is_empty() {
            // Use per-question haystack (native format)
            let ids: Vec<&str> = query
                .haystack_session_ids
                .iter()
                .map(|s| s.as_str())
                .collect();
            let texts: Vec<&str> = ids
                .iter()
                .filter_map(|id| dataset.sessions.get(*id).map(|s| s.as_str()))
                .collect();
            let valid_ids: Vec<&str> = ids
                .iter()
                .filter(|id| dataset.sessions.contains_key(**id))
                .copied()
                .collect();
            (valid_ids, texts)
        } else {
            // Fallback: use all sessions (split format)
            let ids: Vec<&str> = dataset.sessions.keys().map(|s| s.as_str()).collect();
            let texts: Vec<&str> = ids
                .iter()
                .map(|id| dataset.sessions[*id].as_str())
                .collect();
            (ids, texts)
        };

        if session_texts.is_empty() {
            records.push(QaContextRecord {
                question_id: query
                    .question_id
                    .clone()
                    .unwrap_or_else(|| format!("q_{i}")),
                question_type: query.question_type.clone(),
                question: query.question.clone(),
                retrieved_context: Vec::new(),
                retrieved_session_ids: Vec::new(),
                gold_answer: query.answer.clone(),
                gold_session_ids: query.gold_session_ids.clone(),
                retrieval_hit: false,
            });
            continue;
        }

        // Embed sessions
        let session_embeddings = embedder
            .embed_batch(&session_texts)
            .map_err(|e| EvalError::Other(format!("embed sessions: {e}")))?;

        let dim = session_embeddings.first().map(|v| v.len()).unwrap_or(384);

        // Build HNSW index for this question's haystack
        let mut index = HnswIndex::new(
            dim,
            HnswConfig {
                ef_search: 100,
                ..HnswConfig::default()
            },
        );

        for (j, emb) in session_embeddings.iter().enumerate() {
            index
                .insert(session_ids[j], emb)
                .map_err(|e| EvalError::Other(format!("insert: {e}")))?;
        }

        // Embed question and retrieve
        let q_embedding = embedder
            .embed(&query.question)
            .map_err(|e| EvalError::Other(format!("embed query: {e}")))?;

        let search_results = index
            .search(&q_embedding, config.top_k)
            .map_err(|e| EvalError::Other(format!("search: {e}")))?;

        let retrieved_ids: Vec<String> = search_results.iter().map(|h| h.id.clone()).collect();
        let retrieved_context: Vec<String> = retrieved_ids
            .iter()
            .filter_map(|id| dataset.sessions.get(id).cloned())
            .collect();

        let retrieval_hit = query
            .gold_session_ids
            .iter()
            .any(|g| retrieved_ids.iter().any(|r| r == g));

        if retrieval_hit {
            hits += 1;
        }

        let qid = query
            .question_id
            .clone()
            .unwrap_or_else(|| format!("q_{i}"));
        let qtype_label = query.question_type.as_deref().unwrap_or("unknown");

        eprintln!(
            "  [{}/{total}] {qid} ({qtype_label}) — hit={retrieval_hit} ({:.1}s)",
            i + 1,
            t0.elapsed().as_secs_f64()
        );

        records.push(QaContextRecord {
            question_id: qid,
            question_type: query.question_type.clone(),
            question: query.question.clone(),
            retrieved_context,
            retrieved_session_ids: retrieved_ids,
            gold_answer: query.answer.clone(),
            gold_session_ids: query.gold_session_ids.clone(),
            retrieval_hit,
        });
    }

    // Build per-type retrieval summary
    let mut type_hits: HashMap<String, (usize, usize)> = HashMap::new();
    for rec in &records {
        let qtype = rec
            .question_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let entry = type_hits.entry(qtype).or_insert((0, 0));
        entry.1 += 1; // total
        if rec.retrieval_hit {
            entry.0 += 1; // hits
        }
    }

    let mut per_type_retrieval: Vec<QuestionTypeRetrieval> = type_hits
        .into_iter()
        .map(|(qtype, (h, t))| QuestionTypeRetrieval {
            question_type: qtype,
            retrieval_recall: if t == 0 { 0.0 } else { h as f64 / t as f64 },
            count: t,
        })
        .collect();
    per_type_retrieval.sort_by(|a, b| a.question_type.cmp(&b.question_type));

    let elapsed = overall_start.elapsed();

    let summary = QaRetrievalSummary {
        n_questions: total,
        retrieval_recall_any_at_k: hits as f64 / total as f64,
        per_type_retrieval,
        total_time_ms: elapsed.as_millis() as u64,
        avg_time_per_question_ms: elapsed.as_millis() as f64 / total as f64,
    };

    Ok((records, summary))
}

// ── Phase B: Scoring ───────────────────────────────────────────────────────

/// Score answers using case-insensitive substring matching.
///
/// For each answer record, checks if the gold answer is contained within the
/// model's response. This is a baseline scorer — for proper LongMemEval scoring,
/// use the external GPT-4o judge script.
pub fn score_answers(records: &[QaAnswerRecord]) -> QaReport {
    let mut type_stats: HashMap<String, (usize, usize)> = HashMap::new();
    let mut total_correct = 0;

    for rec in records {
        let correct = match &rec.gold_answer {
            Some(gold) => {
                let gold_lower = gold.to_lowercase();
                let answer_lower = rec.model_answer.to_lowercase();
                answer_lower.contains(&gold_lower)
            }
            None => {
                // Abstention: correct if model declines to answer
                let lower = rec.model_answer.to_lowercase();
                lower.contains("don't know")
                    || lower.contains("not mentioned")
                    || lower.contains("no information")
                    || lower.contains("cannot answer")
                    || lower.contains("i don't have")
            }
        };

        if correct {
            total_correct += 1;
        }

        let qtype = rec
            .question_type
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let entry = type_stats.entry(qtype).or_insert((0, 0));
        entry.1 += 1;
        if correct {
            entry.0 += 1;
        }
    }

    let mut per_type: Vec<QuestionTypeResult> = type_stats
        .into_iter()
        .map(|(qtype, (correct, total))| QuestionTypeResult {
            question_type: qtype,
            accuracy: if total == 0 {
                0.0
            } else {
                correct as f64 / total as f64
            },
            retrieval_recall: 0.0, // Not available in answer-only scoring
            count: total,
        })
        .collect();
    per_type.sort_by(|a, b| a.question_type.cmp(&b.question_type));

    QaReport {
        overall_accuracy: if records.is_empty() {
            0.0
        } else {
            total_correct as f64 / records.len() as f64
        },
        retrieval_recall_any_at_k: 0.0, // Not available in answer-only scoring
        per_type,
        n_questions: records.len(),
        scoring_method: "string_match".to_string(),
    }
}
