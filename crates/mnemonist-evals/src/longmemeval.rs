//! LongMemEval dataset loader.
//!
//! Supports two JSON formats:
//!
//! **Format A — native LongMemEval** (`longmemeval_s_cleaned.json`):
//! ```json
//! [
//!   {
//!     "question_id": "q_001",
//!     "question": "What restaurant did I mention?",
//!     "answer": "Olive Garden",
//!     "answer_session_ids": ["s1"],
//!     "haystack_session_ids": ["s1", "s2", "s3"],
//!     "haystack_sessions": [
//!       [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
//!       [{"role": "user", "content": "..."}],
//!       [{"role": "user", "content": "..."}]
//!     ]
//!   }
//! ]
//! ```
//!
//! **Format B — simplified** (for custom datasets):
//! ```json
//! {
//!   "sessions": {"s1": [{"role": "user", "content": "..."}]},
//!   "queries": [{"question": "...", "gold_session_ids": ["s1"]}]
//! }
//! ```
//!
//! In both cases, only **user turns** are concatenated to form the document
//! for each session (matching the MemPalace methodology).

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::EvalError;

/// A single conversation turn.
#[derive(Debug, Clone, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

/// Parsed LongMemEval dataset ready for benchmarking.
#[derive(Debug, Clone)]
pub struct LongMemEvalDataset {
    /// All sessions keyed by ID. Value is the concatenated user-turn text.
    pub sessions: HashMap<String, String>,
    /// Queries with ground-truth gold session IDs.
    pub queries: Vec<LongMemEvalQuery>,
}

/// A single evaluation query.
#[derive(Debug, Clone)]
pub struct LongMemEvalQuery {
    pub question_id: Option<String>,
    pub question_type: Option<String>,
    pub question_date: Option<String>,
    pub question: String,
    pub answer: Option<String>,
    pub gold_session_ids: Vec<String>,
    /// All session IDs in the haystack (preserves ordering for sequential ingestion).
    pub haystack_session_ids: Vec<String>,
    /// Dates corresponding to each haystack session (parallel with haystack_session_ids).
    pub haystack_dates: Option<Vec<String>>,
}

// ── Internal deserialization types ───────────────────────────────────────

/// Native LongMemEval format (per-entry with parallel arrays).
#[derive(Deserialize)]
struct NativeEntry {
    #[serde(default)]
    question_id: Option<String>,
    #[serde(default)]
    question_type: Option<String>,
    #[serde(default)]
    question_date: Option<String>,
    question: String,
    /// Can be a string, number, or array in the dataset — we capture as Value.
    #[serde(default)]
    answer: Option<serde_json::Value>,
    /// Gold session IDs containing the answer.
    answer_session_ids: Vec<String>,
    /// Session IDs in the haystack (parallel with `haystack_sessions`).
    haystack_session_ids: Vec<String>,
    /// Sessions as a list of turn-lists (parallel with `haystack_session_ids`).
    haystack_sessions: Vec<Vec<Turn>>,
    /// Dates corresponding to each haystack session.
    #[serde(default)]
    haystack_dates: Option<Vec<String>>,
}

/// Simplified split format.
#[derive(Deserialize)]
struct SplitFormat {
    sessions: HashMap<String, Vec<Turn>>,
    queries: Vec<RawQuery>,
}

#[derive(Deserialize)]
struct RawQuery {
    question: String,
    #[serde(default)]
    answer: Option<String>,
    #[serde(alias = "evidence_session_ids", alias = "answer_session_ids")]
    gold_session_ids: Vec<String>,
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Concatenate user turns from a session into a single document string.
fn user_turns_to_document(turns: &[Turn]) -> String {
    turns
        .iter()
        .filter(|t| t.role == "user")
        .map(|t| t.content.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Convert a serde_json::Value answer to an Option<String>.
fn answer_to_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(arr) => {
            let strs: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
            if strs.is_empty() {
                None
            } else {
                Some(strs.join(", "))
            }
        }
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Null => None,
        other => Some(other.to_string()),
    }
}

// ── Public API ───────────────────────────────────────────────────────────

/// Load a LongMemEval dataset from a JSON file.
///
/// Auto-detects format based on top-level JSON structure. For the native
/// LongMemEval format, sessions are deduplicated across entries.
pub fn load_dataset(path: &Path) -> Result<LongMemEvalDataset, EvalError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| EvalError::Other(format!("failed to read dataset: {e}")))?;
    parse_dataset(&content)
}

/// Parse a LongMemEval dataset from a JSON string.
pub fn parse_dataset(json: &str) -> Result<LongMemEvalDataset, EvalError> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| EvalError::Other(format!("invalid JSON: {e}")))?;

    if value.is_array() {
        // Check if it's native format (has answer_session_ids) or generic entry format
        let arr = value.as_array().unwrap();
        if let Some(first) = arr.first()
            && first.get("haystack_sessions").is_some()
        {
            return parse_native_format(value);
        }
        parse_generic_entry_format(value)
    } else {
        parse_split_format(value)
    }
}

/// Parse the native LongMemEval format (parallel arrays for sessions).
fn parse_native_format(value: serde_json::Value) -> Result<LongMemEvalDataset, EvalError> {
    let entries: Vec<NativeEntry> = serde_json::from_value(value)
        .map_err(|e| EvalError::Other(format!("invalid native LongMemEval format: {e}")))?;

    let mut sessions: HashMap<String, String> = HashMap::new();
    let mut queries = Vec::with_capacity(entries.len());

    for entry in &entries {
        if entry.haystack_session_ids.len() != entry.haystack_sessions.len() {
            return Err(EvalError::Other(format!(
                "haystack_session_ids ({}) and haystack_sessions ({}) length mismatch for question {:?}",
                entry.haystack_session_ids.len(),
                entry.haystack_sessions.len(),
                entry.question_id,
            )));
        }

        // Build session map from parallel arrays
        for (sid, turns) in entry
            .haystack_session_ids
            .iter()
            .zip(entry.haystack_sessions.iter())
        {
            sessions
                .entry(sid.clone())
                .or_insert_with(|| user_turns_to_document(turns));
        }

        let answer = entry.answer.as_ref().and_then(answer_to_string);

        queries.push(LongMemEvalQuery {
            question_id: entry.question_id.clone(),
            question_type: entry.question_type.clone(),
            question_date: entry.question_date.clone(),
            question: entry.question.clone(),
            answer,
            gold_session_ids: entry.answer_session_ids.clone(),
            haystack_session_ids: entry.haystack_session_ids.clone(),
            haystack_dates: entry.haystack_dates.clone(),
        });
    }

    Ok(LongMemEvalDataset { sessions, queries })
}

/// Parse a generic per-entry format (dict-based sessions).
fn parse_generic_entry_format(value: serde_json::Value) -> Result<LongMemEvalDataset, EvalError> {
    #[derive(Deserialize)]
    struct GenericEntry {
        question: String,
        #[serde(default)]
        answer: Option<String>,
        #[serde(alias = "evidence_session_ids", alias = "answer_session_ids")]
        gold_session_ids: Vec<String>,
        #[serde(default)]
        sessions: HashMap<String, Vec<Turn>>,
    }

    let entries: Vec<GenericEntry> = serde_json::from_value(value)
        .map_err(|e| EvalError::Other(format!("invalid entry format: {e}")))?;

    let mut sessions = HashMap::new();
    let mut queries = Vec::new();

    for entry in entries {
        for (sid, turns) in &entry.sessions {
            sessions
                .entry(sid.clone())
                .or_insert_with(|| user_turns_to_document(turns));
        }
        queries.push(LongMemEvalQuery {
            question_id: None,
            question_type: None,
            question_date: None,
            question: entry.question,
            answer: entry.answer,
            gold_session_ids: entry.gold_session_ids,
            haystack_session_ids: Vec::new(),
            haystack_dates: None,
        });
    }

    Ok(LongMemEvalDataset { sessions, queries })
}

/// Parse the simplified split format.
fn parse_split_format(value: serde_json::Value) -> Result<LongMemEvalDataset, EvalError> {
    let split: SplitFormat = serde_json::from_value(value)
        .map_err(|e| EvalError::Other(format!("invalid split format: {e}")))?;

    let sessions = split
        .sessions
        .into_iter()
        .map(|(sid, turns)| (sid, user_turns_to_document(&turns)))
        .collect();

    let queries = split
        .queries
        .into_iter()
        .map(|q| LongMemEvalQuery {
            question_id: None,
            question_type: None,
            question_date: None,
            question: q.question,
            answer: q.answer,
            gold_session_ids: q.gold_session_ids,
            haystack_session_ids: Vec::new(),
            haystack_dates: None,
        })
        .collect();

    Ok(LongMemEvalDataset { sessions, queries })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_native_longmemeval_format() {
        let json = r#"[
            {
                "question_id": "q_001",
                "question_type": "single-session",
                "question": "What restaurant did I mention?",
                "question_date": "2024/01/15",
                "answer": "Olive Garden",
                "answer_session_ids": ["s1"],
                "haystack_dates": ["2024/01/10", "2024/01/11"],
                "haystack_session_ids": ["s1", "s2"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "I ate at Olive Garden"},
                        {"role": "assistant", "content": "How was it?"},
                        {"role": "user", "content": "Great pasta"}
                    ],
                    [
                        {"role": "user", "content": "My cat Luna is playful"}
                    ]
                ]
            }
        ]"#;

        let ds = parse_dataset(json).unwrap();
        assert_eq!(ds.sessions.len(), 2);
        assert_eq!(ds.queries.len(), 1);
        assert_eq!(ds.sessions["s1"], "I ate at Olive Garden Great pasta");
        assert_eq!(ds.sessions["s2"], "My cat Luna is playful");
        assert_eq!(ds.queries[0].gold_session_ids, vec!["s1"]);
        assert_eq!(ds.queries[0].answer.as_deref(), Some("Olive Garden"));
        assert_eq!(ds.queries[0].question_id.as_deref(), Some("q_001"));
        assert_eq!(
            ds.queries[0].question_type.as_deref(),
            Some("single-session")
        );
        assert_eq!(ds.queries[0].question_date.as_deref(), Some("2024/01/15"));
        assert_eq!(ds.queries[0].haystack_session_ids, vec!["s1", "s2"]);
        assert_eq!(
            ds.queries[0].haystack_dates.as_deref(),
            Some(vec!["2024/01/10".to_string(), "2024/01/11".to_string()]).as_deref()
        );
    }

    #[test]
    fn parse_native_deduplicates_sessions() {
        let json = r#"[
            {
                "question_id": "q_001",
                "question": "Q1?",
                "answer": "A1",
                "answer_session_ids": ["s1"],
                "haystack_session_ids": ["s1", "s2"],
                "haystack_sessions": [
                    [{"role": "user", "content": "session one"}],
                    [{"role": "user", "content": "session two"}]
                ]
            },
            {
                "question_id": "q_002",
                "question": "Q2?",
                "answer": "A2",
                "answer_session_ids": ["s2"],
                "haystack_session_ids": ["s1", "s2", "s3"],
                "haystack_sessions": [
                    [{"role": "user", "content": "session one"}],
                    [{"role": "user", "content": "session two"}],
                    [{"role": "user", "content": "session three"}]
                ]
            }
        ]"#;

        let ds = parse_dataset(json).unwrap();
        assert_eq!(ds.sessions.len(), 3); // s1, s2, s3 deduplicated
        assert_eq!(ds.queries.len(), 2);
    }

    #[test]
    fn parse_split_format() {
        let json = r#"{
            "sessions": {
                "s1": [
                    {"role": "user", "content": "I ate at Olive Garden"},
                    {"role": "assistant", "content": "How was it?"},
                    {"role": "user", "content": "Great pasta"}
                ],
                "s2": [
                    {"role": "user", "content": "My cat Luna is playful"}
                ]
            },
            "queries": [
                {"question": "What restaurant?", "gold_session_ids": ["s1"]},
                {"question": "Pet name?", "gold_session_ids": ["s2"]}
            ]
        }"#;

        let ds = parse_dataset(json).unwrap();
        assert_eq!(ds.sessions.len(), 2);
        assert_eq!(ds.queries.len(), 2);
        assert_eq!(ds.sessions["s1"], "I ate at Olive Garden Great pasta");
    }

    #[test]
    fn parse_answer_mixed_types() {
        // answer can be string, array, or number in the wild
        let json = r#"[
            {
                "question_id": "q_001",
                "question": "Q?",
                "answer": ["ans1", "ans2"],
                "answer_session_ids": ["s1"],
                "haystack_session_ids": ["s1"],
                "haystack_sessions": [[{"role": "user", "content": "hi"}]]
            }
        ]"#;

        let ds = parse_dataset(json).unwrap();
        assert_eq!(ds.queries[0].answer.as_deref(), Some("ans1, ans2"));
    }

    #[test]
    fn user_turns_only() {
        let turns = vec![
            Turn {
                role: "user".into(),
                content: "hello".into(),
            },
            Turn {
                role: "assistant".into(),
                content: "hi there".into(),
            },
            Turn {
                role: "user".into(),
                content: "world".into(),
            },
        ];
        assert_eq!(user_turns_to_document(&turns), "hello world");
    }
}
