use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

use crate::Error;

/// Which level a memory belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryLevel {
    Project,
    Global,
}

/// Memory type classification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
    #[default]
    User,
    Feedback,
    Project,
    Reference,
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Feedback => write!(f, "feedback"),
            Self::Project => write!(f, "project"),
            Self::Reference => write!(f, "reference"),
        }
    }
}

impl std::str::FromStr for MemoryType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "user" => Ok(Self::User),
            "feedback" => Ok(Self::Feedback),
            "project" => Ok(Self::Project),
            "reference" => Ok(Self::Reference),
            _ => Err(format!("unknown memory type: {s}")),
        }
    }
}

/// YAML frontmatter of a memory file.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Frontmatter {
    pub name: String,
    pub description: String,
    #[serde(rename = "type")]
    pub memory_type: MemoryType,

    /// When this memory was first created (ISO 8601).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    /// When this memory was last accessed/retrieved (ISO 8601).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_accessed: Option<String>,

    /// Number of times this memory has been retrieved.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub access_count: u32,

    /// Consolidation strength: increases when memory survives consolidation.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub strength: f32,

    /// Original memory filenames if this was created via consolidation merge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consolidated_from: Option<Vec<String>>,

    /// How this memory was created: "note", "memorize", "learn", "consolidation".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Inter-layer edges: references to code chunk IDs ("file:start:end")
    /// or other memory filenames. Connects this memory to adjacent graph layers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refs: Vec<String>,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

/// A parsed memory file.
#[derive(Debug, Clone)]
pub struct MemoryFile {
    pub frontmatter: Frontmatter,
    pub body: String,
}

impl MemoryFile {
    /// Parse a memory file from its string content.
    pub fn parse(content: &str, path: &str) -> Result<Self, Error> {
        let content = content.trim();
        if !content.starts_with("---") {
            return Err(Error::MissingFrontmatter {
                path: path.to_string(),
            });
        }

        let after_first = &content[3..];
        let end = after_first
            .find("---")
            .ok_or_else(|| Error::MissingFrontmatter {
                path: path.to_string(),
            })?;

        let yaml = &after_first[..end];
        let body = after_first[end + 3..].trim().to_string();
        let frontmatter: Frontmatter = serde_yaml::from_str(yaml)?;

        Ok(Self { frontmatter, body })
    }

    /// Read and parse a memory file from disk.
    pub fn read(path: &Path) -> Result<Self, Error> {
        let content = fs::read_to_string(path)?;
        Self::parse(&content, &path.display().to_string())
    }

    /// Serialize this memory file to its markdown representation.
    pub fn to_markdown(&self) -> String {
        let yaml = serde_yaml::to_string(&self.frontmatter).unwrap_or_default();
        format!("---\n{}---\n\n{}\n", yaml, self.body)
    }

    /// Write this memory file to disk.
    pub fn write(&self, path: &Path) -> Result<(), Error> {
        fs::write(path, self.to_markdown())?;
        Ok(())
    }

    /// Generate the canonical filename for this memory.
    pub fn filename(&self) -> String {
        format!(
            "{}_{}.md",
            self.frontmatter.memory_type, self.frontmatter.name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_memory() {
        let content = r#"---
name: test-memory
description: A test memory
type: feedback
---

Use open standards.

**Why:** Portability.

**How to apply:** Always reference specs.
"#;
        let mem = MemoryFile::parse(content, "test.md").unwrap();
        assert_eq!(mem.frontmatter.name, "test-memory");
        assert_eq!(mem.frontmatter.memory_type, MemoryType::Feedback);
        assert!(mem.body.contains("open standards"));
    }

    #[test]
    fn roundtrip() {
        let mem = MemoryFile {
            frontmatter: Frontmatter {
                name: "roundtrip".to_string(),
                description: "Test roundtrip".to_string(),
                memory_type: MemoryType::User,
                ..Default::default()
            },
            body: "Some content.".to_string(),
        };
        let md = mem.to_markdown();
        let parsed = MemoryFile::parse(&md, "test.md").unwrap();
        assert_eq!(parsed.frontmatter.name, "roundtrip");
        assert_eq!(parsed.body, "Some content.");
    }

    #[test]
    fn missing_frontmatter() {
        let result = MemoryFile::parse("no frontmatter here", "bad.md");
        assert!(result.is_err());
    }

    #[test]
    fn snapshot_memory_file_markdown() {
        let mem = MemoryFile {
            frontmatter: Frontmatter {
                name: "prefer-rust".to_string(),
                description: "Use Rust for CLI tools".to_string(),
                memory_type: MemoryType::Feedback,
                created_at: Some("2026-03-01T00:00:00Z".to_string()),
                last_accessed: Some("2026-03-29T12:00:00Z".to_string()),
                access_count: 5,
                strength: 1.5,
                consolidated_from: None,
                source: Some("memorize".to_string()),
                refs: Vec::new(),
            },
            body: "Always prefer Rust for CLI tools.\n\n**Why:** Performance and safety.\n\n**How to apply:** Default to Rust for new CLIs.".to_string(),
        };
        insta::assert_snapshot!(mem.to_markdown());
    }

    #[test]
    fn snapshot_memory_file_filename() {
        let mem = MemoryFile {
            frontmatter: Frontmatter {
                name: "api-endpoint".to_string(),
                description: "REST API docs".to_string(),
                memory_type: MemoryType::Reference,
                ..Default::default()
            },
            body: String::new(),
        };
        insta::assert_snapshot!(mem.filename(), @"reference_api-endpoint.md");
    }

    #[test]
    fn snapshot_frontmatter_yaml() {
        let fm = Frontmatter {
            name: "merge-freeze".to_string(),
            description: "Merge freeze for mobile release".to_string(),
            memory_type: MemoryType::Project,
            created_at: Some("2026-03-25T00:00:00Z".to_string()),
            strength: 1.0,
            source: Some("note".to_string()),
            ..Default::default()
        };
        insta::assert_yaml_snapshot!(fm);
    }
}
