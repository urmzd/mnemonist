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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryType {
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frontmatter {
    pub name: String,
    pub description: String,
    #[serde(rename = "type")]
    pub memory_type: MemoryType,
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
}
