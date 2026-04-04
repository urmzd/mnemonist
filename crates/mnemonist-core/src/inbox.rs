use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::Error;

/// A single item in working memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboxItem {
    /// Unique identifier (slugified content prefix, or chunk id for learn items).
    pub id: String,
    /// The content of this observation.
    pub content: String,
    /// How this item entered the inbox: "note" or "learn".
    pub source: String,
    /// Attention score (0.0-1.0). Higher = more likely to survive consolidation.
    pub attention_score: f32,
    /// When this item was created (ISO 8601).
    pub created_at: String,
    /// Optional source file metadata (for learn/experience items).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_source: Option<FileSource>,
}

/// Source location for items ingested from code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSource {
    pub file: String,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub kind: String,
}

/// Working memory inbox -- a capacity-limited staging area.
///
/// Items from `note` and `learn` land here before consolidation
/// promotes them to long-term memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inbox {
    /// Maximum number of items (default 7, like working memory capacity).
    pub capacity: usize,
    /// Current items, ordered by attention score descending.
    pub items: Vec<InboxItem>,
    /// When this inbox was last modified (ISO 8601).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated: Option<String>,
}

impl Inbox {
    /// Create a new empty inbox with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::new(),
            last_updated: None,
        }
    }

    /// Load inbox from `.inbox.json` in the given directory.
    /// Returns a new empty inbox if the file doesn't exist.
    pub fn load(dir: &Path, default_capacity: usize) -> Result<Self, Error> {
        let path = dir.join(".inbox.json");
        if !path.exists() {
            return Ok(Self::new(default_capacity));
        }
        let content = fs::read_to_string(&path)?;
        let inbox: Self =
            serde_json::from_str(&content).map_err(|e| Error::ConfigFormat(e.to_string()))?;
        Ok(inbox)
    }

    /// Save inbox to `.inbox.json` in the given directory.
    pub fn save(&self, dir: &Path) -> Result<(), Error> {
        let path = dir.join(".inbox.json");
        let content =
            serde_json::to_string_pretty(self).map_err(|e| Error::ConfigFormat(e.to_string()))?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Push an item into the inbox.
    ///
    /// If at capacity, the lowest-scored item is evicted to make room.
    /// Items are kept sorted by attention_score descending.
    pub fn push(&mut self, item: InboxItem) {
        self.items.push(item);
        self.items
            .sort_by(|a, b| b.attention_score.partial_cmp(&a.attention_score).unwrap());
        self.items.truncate(self.capacity);
    }

    /// Drain all items from the inbox, returning them and leaving it empty.
    pub fn drain(&mut self) -> Vec<InboxItem> {
        std::mem::take(&mut self.items)
    }

    /// Number of items currently in the inbox.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the inbox is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: &str, score: f32) -> InboxItem {
        InboxItem {
            id: id.to_string(),
            content: format!("content for {id}"),
            source: "note".to_string(),
            attention_score: score,
            created_at: "2026-03-27T00:00:00Z".to_string(),
            file_source: None,
        }
    }

    #[test]
    fn push_respects_capacity() {
        let mut inbox = Inbox::new(3);
        inbox.push(make_item("a", 0.5));
        inbox.push(make_item("b", 0.9));
        inbox.push(make_item("c", 0.3));
        inbox.push(make_item("d", 0.7));

        assert_eq!(inbox.len(), 3);
        assert_eq!(inbox.items[0].id, "b"); // highest score
        assert_eq!(inbox.items[1].id, "d");
        assert_eq!(inbox.items[2].id, "a");
        // "c" (0.3) was evicted
    }

    #[test]
    fn drain_empties() {
        let mut inbox = Inbox::new(5);
        inbox.push(make_item("a", 0.5));
        inbox.push(make_item("b", 0.9));

        let drained = inbox.drain();
        assert_eq!(drained.len(), 2);
        assert!(inbox.is_empty());
    }

    #[test]
    fn roundtrip_json() {
        let tmp = tempfile::tempdir().unwrap();
        let mut inbox = Inbox::new(7);
        inbox.push(make_item("test", 0.8));
        inbox.last_updated = Some("2026-03-27T12:00:00Z".to_string());
        inbox.save(tmp.path()).unwrap();

        let loaded = Inbox::load(tmp.path(), 7).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.items[0].id, "test");
        assert_eq!(loaded.capacity, 7);
    }

    #[test]
    fn load_missing_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let inbox = Inbox::load(tmp.path(), 7).unwrap();
        assert!(inbox.is_empty());
        assert_eq!(inbox.capacity, 7);
    }

    #[test]
    fn snapshot_inbox_json() {
        let mut inbox = Inbox::new(7);
        inbox.push(InboxItem {
            id: "prefer-rust".to_string(),
            content: "Always use Rust for CLI tools".to_string(),
            source: "note".to_string(),
            attention_score: 0.9,
            created_at: "2026-03-29T10:00:00Z".to_string(),
            file_source: None,
        });
        inbox.push(InboxItem {
            id: "api-handler".to_string(),
            content: "fn handle_request() in src/server.rs".to_string(),
            source: "learn".to_string(),
            attention_score: 0.6,
            created_at: "2026-03-29T11:00:00Z".to_string(),
            file_source: Some(FileSource {
                file: "src/server.rs".to_string(),
                start_line: Some(42),
                end_line: Some(80),
                kind: "function".to_string(),
            }),
        });
        inbox.last_updated = Some("2026-03-29T11:00:00Z".to_string());
        insta::assert_json_snapshot!(inbox);
    }

    #[test]
    fn snapshot_inbox_after_eviction() {
        let mut inbox = Inbox::new(2);
        inbox.push(make_item("a", 0.3));
        inbox.push(make_item("b", 0.9));
        inbox.push(make_item("c", 0.6));
        // "a" should be evicted (lowest score)
        insta::assert_json_snapshot!(inbox);
    }
}
