mod self_update;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use mnemonist_core::{
    Config, Embedder, EmbeddingStore, Frontmatter, Inbox, InboxItem, IndexEntry, MemoryFile,
    MemoryIndex, MemoryType, global_dir, project_dir,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, IsTerminal, Read as _};
use std::path::PathBuf;

// ── TUI ────────────────────────────────────────────────────────────────────

fn is_tty() -> bool {
    std::env::var("MNEMONIST_TUI").is_ok() || io::stderr().is_terminal()
}

mod tui {
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const RESET: &str = "\x1b[0m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const CYAN: &str = "\x1b[36m";
    pub const MAGENTA: &str = "\x1b[35m";
    #[allow(dead_code)]
    pub const RED: &str = "\x1b[31m";
    pub const BAR: &str = "\x1b[38;5;75m";

    /// Render a horizontal bar of width `filled` out of `total`.
    pub fn bar(filled: usize, total: usize, width: usize) -> String {
        let f = if total > 0 {
            (filled * width) / total
        } else {
            0
        }
        .max(if filled > 0 { 1 } else { 0 })
        .min(width);
        let e = width - f;
        format!("{BAR}{}{DIM}{}{RESET}", "█".repeat(f), "░".repeat(e),)
    }

    /// Format a metric with color based on whether it meets target.
    pub fn metric(name: &str, value: f32, target: &str, good: bool) -> String {
        let color = if good { GREEN } else { YELLOW };
        format!("  {DIM}{name:<20}{RESET} {color}{BOLD}{value:.4}{RESET}  {DIM}{target}{RESET}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
enum OutputFormat {
    Human,
    Json,
}

#[derive(Parser)]
#[command(
    name = "mnemonist",
    about = "Memory for AI agents — learn, remember, consolidate"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Suppress elapsed-time output on stderr
    #[arg(long, short = 'q', global = true)]
    quiet: bool,

    /// Output format
    #[arg(long, global = true, value_enum, default_value = "human")]
    format: OutputFormat,
}

#[derive(Subcommand)]
enum Command {
    /// Initialize a memory directory
    Init {
        /// Initialize global memory (~/.mnemonist/global/) instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Deliberately encode a point into long-term memory
    Memorize {
        /// What to memorize (free text)
        point: String,

        /// Memory type: user, feedback, project, reference
        #[arg(long, short = 't', value_parser = parse_memory_type)]
        memory_type: Option<MemoryType>,

        /// Short kebab-case name (auto-generated if omitted)
        #[arg(long, short)]
        name: Option<String>,

        /// Store in global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Read structured input from stdin (JSON)
        #[arg(long)]
        stdin: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Jot a quick note into working memory (inbox)
    Note {
        /// What to note down
        point: String,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,

        /// Store in global inbox instead of project
        #[arg(long, short)]
        global: bool,
    },

    /// Recall memories by cue — "what do I know about X?"
    Remember {
        /// What to remember (search cue)
        ask: String,

        /// Max output chars (approximate token budget)
        #[arg(long, default_value = "2000")]
        budget: usize,

        /// Search level: project, global, or both
        #[arg(long, short, default_value = "both")]
        level: String,

        /// Read query from stdin JSON ({"query": "..."})
        #[arg(long)]
        stdin: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Ingest a codebase — sensory experience
    Learn {
        /// Path to ingest (defaults to current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Only attend to files matching this glob pattern
        #[arg(long, short)]
        attend: Option<String>,

        /// Max observations to promote to inbox
        #[arg(long)]
        capacity: Option<usize>,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Reorganize memories — merge, prune, strengthen
    Consolidate {
        /// Show what would change without applying
        #[arg(long)]
        dry_run: bool,

        /// Consolidate global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Introspect — review what you know
    Reflect {
        /// Show global memories instead of project
        #[arg(long, short)]
        global: bool,

        /// Show both project and global memories
        #[arg(long, short)]
        all: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Deliberately forget a memory
    Forget {
        /// Filename to forget (e.g., feedback_old-approach.md)
        file: String,

        /// Forget from global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },

    /// Manage mnemonist configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Self-update mnemonist to the latest release
    Update,
    /// Print the current version
    Version,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Get a config value by key (dot-notation, e.g. embedding.model)
    Get {
        /// Config key in dot-notation
        key: String,
    },
    /// Set a config value by key
    Set {
        /// Config key in dot-notation
        key: String,
        /// New value
        value: String,
    },
    /// Create default config file
    Init,
    /// Print config file path
    Path,
}

fn parse_memory_type(s: &str) -> Result<MemoryType, String> {
    s.parse()
}

fn resolve_root(root: &std::path::Path) -> PathBuf {
    std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf())
}

fn resolve_dir(global: bool, root: &std::path::Path) -> Result<PathBuf> {
    if global {
        global_dir().context("could not determine config directory")
    } else {
        Ok(project_dir(&resolve_root(root)))
    }
}

/// Collect memory directories for a given level string.
fn level_dirs(level: &str, root: &std::path::Path) -> Result<Vec<PathBuf>> {
    let resolved = resolve_root(root);
    match level {
        "project" => Ok(vec![project_dir(&resolved)]),
        "global" => Ok(vec![
            global_dir().context("could not determine config directory")?,
        ]),
        _ => {
            let mut v = vec![project_dir(&resolved)];
            if let Some(g) = global_dir() {
                v.push(g);
            }
            Ok(v)
        }
    }
}

// -- JSON output helpers --

fn output_ok(data: Value) {
    let out = json!({"ok": true, "data": data});
    println!("{}", serde_json::to_string(&out).unwrap());
}

fn output_err(msg: &str) -> ! {
    let out = json!({"ok": false, "error": msg});
    println!("{}", serde_json::to_string(&out).unwrap());
    std::process::exit(1);
}

/// Print to stderr for human UX (ignored when piped).
macro_rules! info {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

// -- Elapsed-time reporting --

struct CommandTimer {
    start: std::time::Instant,
    #[allow(dead_code)]
    label: String,
}

impl CommandTimer {
    fn start(label: impl Into<String>) -> Self {
        Self {
            start: std::time::Instant::now(),
            label: label.into(),
        }
    }
}

impl Drop for CommandTimer {
    fn drop(&mut self) {
        if is_tty() {
            let elapsed = self.start.elapsed().as_secs_f64();
            eprintln!(
                "  {DIM}{:.2}s{RESET}",
                elapsed,
                DIM = tui::DIM,
                RESET = tui::RESET,
            );
        }
    }
}

// -- Stdin JSON types --

#[derive(Deserialize)]
struct MemorizeInput {
    #[serde(rename = "type")]
    memory_type: String,
    name: String,
    description: String,
    #[serde(default)]
    body: String,
    #[serde(default = "default_level")]
    level: String,
    /// Inter-layer edges: code chunk IDs or memory filenames.
    #[serde(default)]
    refs: Vec<String>,
}

#[derive(Deserialize)]
struct RememberInput {
    query: String,
}

fn default_level() -> String {
    "project".to_string()
}

// -- Serializable output types --

#[derive(Serialize)]
struct ReflectEntry {
    title: String,
    file: String,
    summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    strength: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    access_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_accessed: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
}

#[derive(Serialize)]
struct MemoryOut {
    file: String,
    #[serde(rename = "type")]
    memory_type: String,
    name: String,
    description: String,
    body: String,
    score: f32,
}

/// Auto-generate a kebab-case name from free text.
fn slugify(text: &str, max_words: usize) -> String {
    text.split_whitespace()
        .take(max_words)
        .collect::<Vec<_>>()
        .join("-")
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .collect::<String>()
}

/// Get current timestamp as ISO 8601 string.
fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn main() {
    let cli = Cli::parse();

    let quiet = cli.quiet || Config::load().output.quiet;

    let cmd_label = match &cli.command {
        Command::Init { .. } => "init",
        Command::Memorize { .. } => "memorize",
        Command::Note { .. } => "note",
        Command::Remember { .. } => "remember",
        Command::Learn { .. } => "learn",
        Command::Consolidate { .. } => "consolidate",
        Command::Reflect { .. } => "reflect",
        Command::Forget { .. } => "forget",
        Command::Config { .. } => "config",
        Command::Update => "update",
        Command::Version => "version",
    };

    let _timer = if quiet {
        None
    } else {
        Some(CommandTimer::start(cmd_label))
    };

    if let Err(e) = run(cli) {
        drop(_timer);
        output_err(&format!("{e:#}"));
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        // -- Infrastructure --
        Command::Init { global, root } => {
            let dir = resolve_dir(global, &root)?;
            MemoryIndex::init(&dir)?;
            let level = if global { "global" } else { "project" };
            info!("initialized {level} memory at {}", dir.display());
            output_ok(json!({
                "level": level,
                "path": dir.display().to_string(),
            }));
        }

        // -- memorize: deliberate encoding into long-term memory --
        Command::Memorize {
            point,
            memory_type,
            name,
            global,
            stdin,
            root,
        } => {
            let (mt, n, desc, body, is_global, refs) = if stdin {
                let mut buf = String::new();
                io::stdin()
                    .read_to_string(&mut buf)
                    .context("failed to read stdin")?;
                let input: MemorizeInput =
                    serde_json::from_str(&buf).context("invalid JSON on stdin")?;
                let mt: MemoryType = input
                    .memory_type
                    .parse()
                    .map_err(|e: String| anyhow::anyhow!(e))?;
                let is_global = input.level == "global";
                (
                    mt,
                    input.name,
                    input.description,
                    input.body,
                    is_global,
                    input.refs,
                )
            } else {
                let mt = memory_type.unwrap_or(MemoryType::Feedback);
                let n = name.unwrap_or_else(|| slugify(&point, 4));
                (mt, n, point.clone(), point, global, Vec::new())
            };

            let dir = resolve_dir(is_global, &root)?;
            let mut index = MemoryIndex::load(&dir)?;

            let mem = MemoryFile {
                frontmatter: Frontmatter {
                    name: n.clone(),
                    description: desc.clone(),
                    memory_type: mt,
                    created_at: Some(now_iso()),
                    source: Some("memorize".to_string()),
                    strength: 1.0,
                    refs,
                    ..Default::default()
                },
                body,
            };

            let filename = mem.filename();
            let entry = IndexEntry {
                title: n.replace('-', " "),
                file: filename.clone(),
                summary: desc,
            };

            let action = index.upsert(entry);
            mem.write(&dir.join(&filename))?;
            index.save()?;

            // Embed and rebuild memory HNSW
            let embed_result = try_embed_single(&dir, &filename);
            let _ = build_memory_hnsw(&dir);

            if is_tty() {
                use tui::*;
                let ac = if action == "created" { GREEN } else { YELLOW };
                let ec = if embed_result.is_ok() {
                    format!("{GREEN}embedded{RESET}")
                } else {
                    format!("{DIM}no embed{RESET}")
                };
                eprintln!("  {ac}{action}{RESET} {BOLD}{filename}{RESET}  {ec}");
            } else {
                info!("{action} {filename}");
            }
            output_ok(json!({
                "file": filename,
                "action": action,
                "embedded": embed_result.is_ok(),
            }));
        }

        // -- note: quick capture into working memory inbox --
        Command::Note {
            point,
            root,
            global,
        } => {
            let config = Config::load();
            let dir = resolve_dir(global, &root)?;
            std::fs::create_dir_all(&dir)?;

            let mut inbox = Inbox::load(&dir, config.inbox.capacity)?;

            let item = InboxItem {
                id: slugify(&point, 4),
                content: point,
                source: "note".to_string(),
                attention_score: 0.95, // user intent > auto-learned heuristics
                created_at: now_iso(),
                file_source: None,
            };

            inbox.push(item);
            inbox.last_updated = Some(now_iso());
            inbox.save(&dir)?;

            info!("noted ({}/{})", inbox.len(), inbox.capacity);
            output_ok(json!({
                "inbox_size": inbox.len(),
                "capacity": inbox.capacity,
            }));
        }

        // -- remember: cue-based retrieval --
        Command::Remember {
            ask,
            budget,
            level,
            stdin,
            root,
        } => {
            let q = if stdin {
                let mut buf = String::new();
                io::stdin()
                    .read_to_string(&mut buf)
                    .context("failed to read stdin")?;
                let input: RememberInput = serde_json::from_str(&buf)
                    .context("stdin must be JSON with a \"query\" field")?;
                input.query
            } else {
                ask
            };

            if q.trim().is_empty() {
                anyhow::bail!("query must not be empty");
            }

            {
                let dirs = level_dirs(&level, &root)?;
                let mut memories: Vec<MemoryOut> = Vec::new();
                let mut total_chars = 0usize;

                let recall_cfg = Config::load();
                let min_results = recall_cfg.recall.min_results;

                let mut matched = semantic_search(&q, &dirs, &root);
                if matched.is_empty() {
                    matched = text_search(&q, &dirs);
                }

                // Build candidates with all available signals for the reranker
                let now_utc = chrono::Utc::now();
                let candidates: Vec<mnemonist_core::Candidate> = matched
                    .iter()
                    .map(|(dir, entry, cosine)| {
                        let parts: Vec<&str> = entry.file.rsplitn(3, ':').collect();
                        let is_code = parts.len() == 3
                            && parts[1].parse::<usize>().is_ok()
                            && parts[0].parse::<usize>().is_ok();

                        let (memory_signals, source_file) = if is_code {
                            (None, parts[2].to_string())
                        } else {
                            let path = dir.join(&entry.file);
                            let signals = MemoryFile::read(&path).ok().map(|mem| {
                                let fm = &mem.frontmatter;
                                let parse_dt =
                                    |s: &Option<String>| -> Option<chrono::DateTime<chrono::Utc>> {
                                        s.as_ref()?
                                            .parse::<chrono::DateTime<chrono::FixedOffset>>()
                                            .ok()
                                            .map(|dt| dt.to_utc())
                                    };
                                let created = parse_dt(&fm.created_at);
                                let accessed = parse_dt(&fm.last_accessed).or(created);
                                let age_days = created
                                    .map(|c| (now_utc - c).num_seconds() as f64 / 86400.0)
                                    .unwrap_or(1.0);
                                let recency_days = accessed
                                    .map(|a| (now_utc - a).num_seconds() as f64 / 86400.0)
                                    .unwrap_or(age_days);

                                mnemonist_core::MemorySignals {
                                    memory_type: fm.memory_type,
                                    access_count: fm.access_count,
                                    strength: fm.strength,
                                    recency_days,
                                    age_days,
                                    source: fm.source.clone(),
                                    ref_count: fm.refs.len(),
                                }
                            });
                            (signals, entry.file.clone())
                        };

                        mnemonist_core::Candidate {
                            id: entry.file.clone(),
                            cosine_score: *cosine,
                            memory_signals,
                            source_file,
                        }
                    })
                    .collect();

                // Load project-specific recall profile (or use uncalibrated defaults)
                let profile = dirs
                    .iter()
                    .find_map(|d| mnemonist_core::RecallProfile::load(d))
                    .unwrap_or_else(mnemonist_core::RecallProfile::uncalibrated);

                // Rerank with multi-signal scoring + dynamic threshold
                let ranked = mnemonist_core::rerank(&candidates, &profile);

                // Resolve ranked results into memory output with budget control
                // Build a lookup from entry.file -> (dir, entry) for efficient access
                let entry_map: std::collections::HashMap<String, (&PathBuf, &IndexEntry)> = matched
                    .iter()
                    .map(|(d, e, _)| (e.file.clone(), (d, e)))
                    .collect();

                for result in &ranked {
                    if total_chars >= budget && memories.len() >= min_results {
                        break;
                    }

                    let Some((dir, entry)) = entry_map.get(&result.id) else {
                        continue;
                    };

                    // Code chunk entries have file format "path:start:end"
                    let parts: Vec<&str> = entry.file.rsplitn(3, ':').collect();
                    if parts.len() == 3 {
                        if let (Ok(start), Ok(end)) =
                            (parts[1].parse::<usize>(), parts[0].parse::<usize>())
                        {
                            let source_path = root.join(parts[2]);
                            if let Ok(content) = std::fs::read_to_string(&source_path) {
                                let lines: Vec<&str> = content.lines().collect();
                                let s = start.saturating_sub(1).min(lines.len());
                                let e = end.min(lines.len());
                                let body = lines[s..e].join("\n");
                                let body_chars = body.len();
                                if total_chars + body_chars > budget
                                    && memories.len() >= min_results
                                {
                                    break;
                                }
                                total_chars += body_chars;
                                memories.push(MemoryOut {
                                    file: entry.file.clone(),
                                    memory_type: "code".to_string(),
                                    name: entry.title.clone(),
                                    description: entry.summary.clone(),
                                    body,
                                    score: result.final_score,
                                });
                            }
                        }
                        continue;
                    }

                    // Memory file entries
                    let path = dir.join(&entry.file);
                    if let Ok(mut mem) = MemoryFile::read(&path) {
                        let body_chars = mem.body.len();
                        if total_chars + body_chars > budget && memories.len() >= min_results {
                            break;
                        }
                        total_chars += body_chars;

                        mem.frontmatter.access_count += 1;
                        mem.frontmatter.last_accessed = Some(now_iso());
                        let _ = mem.write(&path);

                        memories.push(MemoryOut {
                            file: entry.file.clone(),
                            memory_type: mem.frontmatter.memory_type.to_string(),
                            name: mem.frontmatter.name.clone(),
                            description: mem.frontmatter.description.clone(),
                            body: mem.body.clone(),
                            score: result.final_score,
                        });

                        // Edge expansion: follow refs to code chunks and related memories
                        if recall_cfg.recall.expand_refs && !mem.frontmatter.refs.is_empty() {
                            let max_exp = recall_cfg.recall.max_ref_expansions;
                            for ref_id in mem.frontmatter.refs.iter().take(max_exp) {
                                if total_chars >= budget {
                                    break;
                                }
                                if ref_id.ends_with(".md") {
                                    let ref_path = dir.join(ref_id);
                                    if let Ok(ref_mem) = MemoryFile::read(&ref_path) {
                                        total_chars += ref_mem.body.len();
                                        memories.push(MemoryOut {
                                            file: ref_id.clone(),
                                            memory_type: ref_mem
                                                .frontmatter
                                                .memory_type
                                                .to_string(),
                                            name: ref_mem.frontmatter.name.clone(),
                                            description: format!("via {}", entry.file),
                                            body: ref_mem.body.clone(),
                                            score: 0.0,
                                        });
                                    }
                                } else {
                                    let ref_parts: Vec<&str> = ref_id.rsplitn(3, ':').collect();
                                    if ref_parts.len() == 3
                                        && let (Ok(s), Ok(e)) = (
                                            ref_parts[1].parse::<usize>(),
                                            ref_parts[0].parse::<usize>(),
                                        )
                                    {
                                        let src = root.join(ref_parts[2]);
                                        if let Ok(content) = std::fs::read_to_string(&src) {
                                            let lines: Vec<&str> = content.lines().collect();
                                            let start = s.saturating_sub(1).min(lines.len());
                                            let end = e.min(lines.len());
                                            let full = lines[start..end].join("\n");
                                            let max_c =
                                                recall_cfg.consolidation.max_memory_tokens * 4;
                                            let body = if full.len() > max_c {
                                                format!("{}...", &full[..max_c])
                                            } else {
                                                full
                                            };
                                            total_chars += body.len();
                                            memories.push(MemoryOut {
                                                file: ref_id.clone(),
                                                memory_type: "ref".to_string(),
                                                name: entry.title.clone(),
                                                description: format!("via {}", entry.file),
                                                body,
                                                score: 0.0,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // TUI on stderr (human), JSON on stdout (pipe)
                if is_tty() {
                    use tui::*;
                    eprintln!();
                    eprintln!("  {BOLD}{CYAN}mnemonist remember{RESET}  {DIM}\"{q}\"{RESET}");
                    eprintln!("  {DIM}────────────────────────────────────────{RESET}");
                    for (i, m) in memories.iter().enumerate() {
                        let tc = match m.memory_type.as_str() {
                            "feedback" => GREEN,
                            "user" => CYAN,
                            "project" => YELLOW,
                            "reference" | "ref" => MAGENTA,
                            "code" => BAR,
                            _ => RESET,
                        };
                        let preview: String = m
                            .body
                            .chars()
                            .take(72)
                            .collect::<String>()
                            .replace('\n', " ");
                        eprintln!(
                            "  {DIM}{:>2}.{RESET} {tc}{:<10}{RESET} {BOLD}{}{RESET}  {DIM}{:.3}{RESET}",
                            i + 1,
                            m.memory_type,
                            m.file,
                            m.score
                        );
                        eprintln!("      {DIM}{preview}{RESET}");
                    }
                    eprintln!();
                    eprintln!(
                        "  {DIM}~{} tokens across {} results{RESET}",
                        total_chars / 4,
                        memories.len()
                    );
                    eprintln!();
                }

                output_ok(json!({
                    "memories": memories,
                    "token_estimate": total_chars / 4,
                }));
            }
        }

        // -- learn: ingest a codebase as sensory experience --
        Command::Learn {
            path,
            attend: _attend,
            capacity,
            root,
        } => {
            let root = std::fs::canonicalize(&root).context("could not resolve project root")?;
            let ingest_path =
                std::fs::canonicalize(&path).context("could not resolve ingest path")?;

            let config = Config::load();
            let cap = capacity.unwrap_or(config.inbox.capacity);

            // Phase 1: chunk files
            let strategy = mnemonist_core::ann::code::ParagraphChunking::default();
            let mut code_index = mnemonist_core::ann::code::CodeIndex::new(&ingest_path, &strategy);
            let chunk_count = code_index.index(&config.code.exclude_patterns)?;

            let file_count = code_index
                .chunks()
                .iter()
                .map(|c| &c.file)
                .collect::<std::collections::HashSet<_>>()
                .len();

            // Save code manifest (backward compat)
            let mem_dir = project_dir(&root);
            std::fs::create_dir_all(&mem_dir)?;
            let manifest_path = mem_dir.join(".code-index.json");

            let manifest: Vec<Value> = code_index
                .chunks()
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id(),
                        "file": c.file,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                    })
                })
                .collect();

            let now = now_iso();
            let manifest_doc = json!({
                "root": ingest_path.display().to_string(),
                "indexed_at": now,
                "chunks": manifest,
            });
            std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest_doc)?)?;

            // Phase 2: embed chunks and build HNSW index
            let mut embedded_count = 0usize;
            let mut eval_json = json!(null);
            match mnemonist_core::CandleEmbedder::default_model() {
                Ok(embedder) => {
                    let dim = embedder.dimension()?;
                    let mut hnsw = mnemonist_core::ann::hnsw::HnswIndex::with_defaults(dim);
                    match code_index.build_ann(&embedder, &mut hnsw) {
                        Ok(n) => {
                            embedded_count = n;
                            let hnsw_path = mem_dir.join(".code-index.hnsw");
                            if let Err(e) = mnemonist_core::ann::AnnIndex::save(&hnsw, &hnsw_path) {
                                eprintln!("warning: failed to save code HNSW index: {e}");
                            }

                            // Eval: sample up to 100 chunks for quality metrics
                            let sample_size = chunk_count.min(100);
                            let step = if sample_size > 0 {
                                chunk_count / sample_size
                            } else {
                                1
                            }
                            .max(1);
                            let sample_texts: Vec<&str> = code_index
                                .chunks()
                                .iter()
                                .step_by(step)
                                .take(sample_size)
                                .map(|c| c.content.as_str())
                                .collect();

                            if let Ok(sample_embeddings) = embedder.embed_batch(&sample_texts) {
                                let aniso = mnemonist_core::ann::eval::anisotropy(&sample_embeddings);
                                let range =
                                    mnemonist_core::ann::eval::similarity_range(&sample_embeddings);
                                eval_json = json!({
                                    "sample_size": sample_embeddings.len(),
                                    "anisotropy": (aniso * 10000.0).round() / 10000.0,
                                    "similarity_range": (range * 10000.0).round() / 10000.0,
                                });

                                // Auto-calibrate recall profile from embedding quality
                                let profile = mnemonist_core::RecallProfile::calibrate(
                                    aniso,
                                    range,
                                    sample_embeddings.len(),
                                );
                                if let Err(e) = profile.save(&mem_dir) {
                                    eprintln!("warning: failed to save recall profile: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("warning: embedding failed: {e}");
                        }
                    }
                }
                Err(e) => {
                    eprintln!("warning: could not load embedding model: {e}");
                }
            }

            // Phase 3: attention scoring — promote top chunks to inbox
            let mut inbox = Inbox::load(&mem_dir, cap)?;
            inbox.capacity = cap;

            // Score chunks by simple content heuristics:
            // - public items (pub, export) score higher
            // - longer chunks score slightly higher (more context)
            for chunk in code_index.chunks() {
                let base_score = 0.5;
                let visibility_bonus =
                    if chunk.content.contains("pub ") || chunk.content.contains("export ") {
                        0.2
                    } else {
                        0.0
                    };
                let size_bonus = ((chunk.end_line - chunk.start_line) as f32 / 100.0).min(0.2);

                let item = InboxItem {
                    id: chunk.id(),
                    content: chunk.content.clone(),
                    source: "learn".to_string(),
                    attention_score: base_score + visibility_bonus + size_bonus,
                    created_at: now.clone(),
                    file_source: Some(mnemonist_core::FileSource {
                        file: chunk.file.clone(),
                        start_line: Some(chunk.start_line),
                        end_line: Some(chunk.end_line),
                        kind: "block".to_string(),
                    }),
                };

                inbox.push(item);
            }

            inbox.last_updated = Some(now.clone());
            inbox.save(&mem_dir)?;

            if is_tty() {
                use tui::*;
                eprintln!();
                eprintln!("  {BOLD}{CYAN}mnemonist learn{RESET}");
                eprintln!("  {DIM}────────────────────────────────────────{RESET}");
                eprintln!("  {DIM}files{RESET}     {BOLD}{file_count}{RESET}");
                eprintln!("  {DIM}chunks{RESET}    {BOLD}{chunk_count}{RESET}");
                eprintln!(
                    "  {DIM}embedded{RESET}  {BOLD}{embedded_count}{RESET}  {}",
                    bar(embedded_count, chunk_count, 20)
                );
                eprintln!(
                    "  {DIM}inbox{RESET}     {BOLD}{}{RESET}/{}",
                    inbox.len(),
                    cap
                );
                if let Some(eval) = eval_json.as_object() {
                    let aniso = eval
                        .get("anisotropy")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;
                    let range = eval
                        .get("similarity_range")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;
                    let sample = eval
                        .get("sample_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    eprintln!();
                    eprintln!("  {DIM}eval (n={sample}){RESET}");
                    eprintln!("{}", metric("anisotropy", aniso, "< 0.3", aniso < 0.3));
                    eprintln!(
                        "{}",
                        metric("similarity range", range, "> 0.3", range > 0.3)
                    );
                }
                eprintln!();
            } else {
                info!(
                    "learned {} chunks from {} files, {} embedded, {} in inbox",
                    chunk_count,
                    file_count,
                    embedded_count,
                    inbox.len()
                );
            }
            output_ok(json!({
                "chunks": chunk_count,
                "files": file_count,
                "embedded": embedded_count,
                "inbox_size": inbox.len(),
                "indexed_at": now,
                "eval": eval_json,
            }));
        }

        // -- consolidate: sleep consolidation --
        Command::Consolidate {
            dry_run,
            global,
            root,
        } => {
            let config = Config::load();
            let dir = resolve_dir(global, &root)?;

            if !dir.exists() {
                output_err("memory directory does not exist — run `mnemonist init` first");
            }

            let mut index = MemoryIndex::load(&dir)?;
            let mut actions: Vec<Value> = Vec::new();

            // Phase 1: Promote inbox items
            let mut inbox = Inbox::load(&dir, config.inbox.capacity)?;
            let inbox_items = inbox.drain();
            for item in &inbox_items {
                let name = slugify(&item.id, 4);
                let mt = if item.source == "learn" {
                    MemoryType::Reference
                } else {
                    MemoryType::Feedback
                };

                // Preserve file_source as inter-layer edge refs
                let refs = item
                    .file_source
                    .as_ref()
                    .map(|fs| {
                        vec![format!(
                            "{}:{}:{}",
                            fs.file,
                            fs.start_line.unwrap_or(0),
                            fs.end_line.unwrap_or(0)
                        )]
                    })
                    .unwrap_or_default();

                // Memory body is a cue, not a copy.
                // Code: extract signatures. Notes: truncate. Full content via refs.
                let max_chars = config.consolidation.max_memory_tokens * 4;
                let body = if item.content.len() <= max_chars {
                    item.content.clone()
                } else if item.source == "learn" || item.file_source.is_some() {
                    extract_code_cue(&item.content, max_chars)
                } else {
                    let truncated: String = item.content.chars().take(max_chars).collect();
                    format!("{truncated}...")
                };

                let mem = MemoryFile {
                    frontmatter: Frontmatter {
                        name: name.clone(),
                        description: item.content.chars().take(120).collect::<String>(),
                        memory_type: mt,
                        created_at: Some(item.created_at.clone()),
                        source: Some(item.source.clone()),
                        strength: item.attention_score,
                        refs,
                        ..Default::default()
                    },
                    body,
                };

                let filename = mem.filename();
                actions.push(json!({
                    "action": "promote",
                    "file": filename,
                    "source": item.source,
                }));

                if !dry_run {
                    let entry = IndexEntry {
                        title: name.replace('-', " "),
                        file: filename.clone(),
                        summary: mem.frontmatter.description.clone(),
                    };
                    index.upsert(entry);
                    mem.write(&dir.join(&filename))?;
                }
            }

            // Phase 1.5: Associative linking — connect similar memories
            if !dry_run {
                let store_path = dir.join(".embeddings.bin");
                if store_path.exists()
                    && let Ok(store) = EmbeddingStore::load(&store_path)
                {
                    let threshold = config.consolidation.merge_threshold;
                    let entries = &store.entries;

                    // Build temp HNSW for O(n log n) neighbor search
                    let mut temp_hnsw =
                        mnemonist_core::ann::hnsw::HnswIndex::with_defaults(store.dimension);
                    for entry in entries {
                        let _ = mnemonist_core::ann::AnnIndex::insert(
                            &mut temp_hnsw,
                            &entry.file,
                            &entry.embedding,
                        );
                    }

                    // Per-entry neighbor search instead of O(n²) pairwise
                    let k = 10;
                    let mut links: Vec<(String, String)> = Vec::new();
                    for entry in entries {
                        if let Ok(hits) =
                            mnemonist_core::ann::AnnIndex::search(&temp_hnsw, &entry.embedding, k + 1)
                        {
                            for hit in hits {
                                if hit.id != entry.file && hit.score >= threshold {
                                    // Canonical ordering to deduplicate (a,b) == (b,a)
                                    let pair = if entry.file < hit.id {
                                        (entry.file.clone(), hit.id.clone())
                                    } else {
                                        (hit.id.clone(), entry.file.clone())
                                    };
                                    links.push(pair);
                                }
                            }
                        }
                    }
                    links.sort();
                    links.dedup();
                    // Apply bidirectional refs
                    for (a, b) in &links {
                        let path_a = dir.join(a);
                        if let Ok(mut mem_a) = MemoryFile::read(&path_a)
                            && !mem_a.frontmatter.refs.contains(b)
                        {
                            mem_a.frontmatter.refs.push(b.clone());
                            let _ = mem_a.write(&path_a);
                        }
                        let path_b = dir.join(b);
                        if let Ok(mut mem_b) = MemoryFile::read(&path_b)
                            && !mem_b.frontmatter.refs.contains(a)
                        {
                            mem_b.frontmatter.refs.push(a.clone());
                            let _ = mem_b.write(&path_b);
                        }
                    }
                    if !links.is_empty() {
                        actions.push(json!({
                            "action": "associate",
                            "links": links.len(),
                        }));
                    }
                }
            }

            // Phase 2: Decay / Prune stale memories
            let now = chrono::Utc::now();
            let mut to_remove = Vec::new();

            for entry in &index.entries {
                let path = dir.join(&entry.file);
                if let Ok(mem) = MemoryFile::read(&path) {
                    let fm = &mem.frontmatter;
                    let days_since_access = fm
                        .last_accessed
                        .as_ref()
                        .or(fm.created_at.as_ref())
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| (now - dt.to_utc()).num_days().max(0) as u64)
                        .unwrap_or(u64::MAX);

                    let decay_threshold = if fm.memory_type == MemoryType::Feedback {
                        config.consolidation.decay_days * 2
                    } else {
                        config.consolidation.decay_days
                    };

                    if days_since_access > decay_threshold
                        && fm.access_count < config.consolidation.protected_access_count
                        && fm.strength < 1.0
                    {
                        to_remove.push(entry.file.clone());
                        actions.push(json!({
                            "action": "decay",
                            "file": entry.file,
                            "days_since_access": days_since_access,
                            "access_count": fm.access_count,
                            "strength": fm.strength,
                        }));
                    }
                }
            }

            if !dry_run {
                for file in &to_remove {
                    index.remove(file);
                    let path = dir.join(file);
                    if path.exists() {
                        std::fs::remove_file(&path)?;
                    }
                }
            }

            // Phase 3: Re-embed and save
            if !dry_run {
                inbox.save(&dir)?;
                index.save()?;

                // Re-embed all memories and rebuild memory HNSW
                let embed_result = try_embed_all(&dir);
                if let Ok((synced, total)) = embed_result {
                    actions.push(json!({
                        "action": "embed",
                        "synced": synced,
                        "total": total,
                    }));
                }
                if let Ok(n) = build_memory_hnsw(&dir) {
                    actions.push(json!({
                        "action": "index",
                        "indexed": n,
                    }));
                }
            }

            let promoted = inbox_items.len();
            let decayed = to_remove.len();

            if is_tty() {
                use tui::*;
                let links = actions
                    .iter()
                    .find(|a| a.get("action").and_then(|v| v.as_str()) == Some("associate"))
                    .and_then(|a| a.get("links").and_then(|v| v.as_u64()))
                    .unwrap_or(0);
                let indexed = actions
                    .iter()
                    .find(|a| a.get("action").and_then(|v| v.as_str()) == Some("index"))
                    .and_then(|a| a.get("indexed").and_then(|v| v.as_u64()))
                    .unwrap_or(0);
                eprintln!();
                eprintln!(
                    "  {BOLD}{CYAN}mnemonist consolidate{RESET}{}",
                    if dry_run {
                        format!("  {DIM}(dry run){RESET}")
                    } else {
                        String::new()
                    }
                );
                eprintln!("  {DIM}────────────────────────────────────────{RESET}");
                eprintln!("  {DIM}promoted{RESET}    {GREEN}{BOLD}{promoted}{RESET}");
                eprintln!(
                    "  {DIM}decayed{RESET}     {}{BOLD}{decayed}{RESET}",
                    if decayed > 0 { YELLOW } else { DIM }
                );
                if links > 0 {
                    eprintln!(
                        "  {DIM}associated{RESET}  {MAGENTA}{BOLD}{links}{RESET} {DIM}links{RESET}"
                    );
                }
                if indexed > 0 {
                    eprintln!(
                        "  {DIM}indexed{RESET}     {BOLD}{indexed}{RESET} {DIM}memories{RESET}"
                    );
                }
                eprintln!();
            } else {
                info!(
                    "consolidated: {} promoted, {} decayed{}",
                    promoted,
                    decayed,
                    if dry_run { " (dry run)" } else { "" }
                );
            }
            output_ok(json!({
                "promoted": promoted,
                "decayed": decayed,
                "dry_run": dry_run,
                "actions": actions,
            }));
        }

        // -- reflect: introspection --
        Command::Reflect { global, all, root } => {
            let dirs = if all {
                let mut v = vec![project_dir(&root)];
                if let Some(g) = global_dir() {
                    v.push(g);
                }
                v
            } else {
                vec![resolve_dir(global, &root)?]
            };

            let mut entries = Vec::new();
            for dir in dirs {
                if let Ok(index) = MemoryIndex::load(&dir) {
                    for entry in &index.entries {
                        // Load full memory to get cognitive metadata
                        let path = dir.join(&entry.file);
                        let (strength, access_count, last_accessed, source) =
                            if let Ok(mem) = MemoryFile::read(&path) {
                                let fm = &mem.frontmatter;
                                (
                                    if fm.strength > 0.0 {
                                        Some(fm.strength)
                                    } else {
                                        None
                                    },
                                    if fm.access_count > 0 {
                                        Some(fm.access_count)
                                    } else {
                                        None
                                    },
                                    fm.last_accessed.clone(),
                                    fm.source.clone(),
                                )
                            } else {
                                (None, None, None, None)
                            };

                        entries.push(ReflectEntry {
                            title: entry.title.clone(),
                            file: entry.file.clone(),
                            summary: entry.summary.clone(),
                            strength,
                            access_count,
                            last_accessed,
                            source,
                        });
                    }
                }
            }

            // Also show inbox contents
            let inbox_dir = resolve_dir(false, &root).unwrap_or_default();
            let inbox = Inbox::load(&inbox_dir, 7).unwrap_or_else(|_| Inbox::new(7));

            output_ok(json!({
                "memories": entries,
                "inbox": {
                    "size": inbox.len(),
                    "capacity": inbox.capacity,
                    "items": inbox.items.iter().map(|i| json!({
                        "id": i.id,
                        "source": i.source,
                        "attention_score": i.attention_score,
                        "created_at": i.created_at,
                    })).collect::<Vec<_>>(),
                },
            }));
        }

        // -- forget: deliberate forgetting --
        Command::Forget { file, global, root } => {
            let dir = resolve_dir(global, &root)?;
            let mut index = MemoryIndex::load(&dir)?;

            // Resolve partial names: exact → +.md → *_{name}.md → contains
            let resolved = if index.entries.iter().any(|e| e.file == file) {
                file.clone()
            } else if index.entries.iter().any(|e| e.file == format!("{file}.md")) {
                format!("{file}.md")
            } else {
                let suffix = format!("_{file}.md");
                let candidates: Vec<&str> = index
                    .entries
                    .iter()
                    .filter(|e| e.file.ends_with(&suffix))
                    .map(|e| e.file.as_str())
                    .collect();
                match candidates.len() {
                    1 => candidates[0].to_string(),
                    0 => file.clone(), // fall through to "not found"
                    _ => {
                        anyhow::bail!("ambiguous: {} matches {:?}", file, candidates);
                    }
                }
            };

            if index.remove(&resolved) {
                let path = dir.join(&resolved);
                if path.exists() {
                    std::fs::remove_file(&path)?;
                }

                // Also remove from embedding store
                let store_path = dir.join(".embeddings.bin");
                if store_path.exists()
                    && let Ok(mut store) = EmbeddingStore::load(&store_path)
                {
                    store.remove(&resolved);
                    let _ = store.save(&store_path);
                }

                index.save()?;
                info!("forgot {resolved}");
                output_ok(json!({
                    "file": resolved,
                    "action": "forgotten",
                }));
            } else {
                output_err(&format!("not found: {file}"));
            }
        }

        // -- update: self-update --
        Command::Update => {
            eprintln!("current version: {}", env!("CARGO_PKG_VERSION"));
            match self_update::self_update(
                "urmzd/mnemonist",
                env!("CARGO_PKG_VERSION"),
                "mnemonist",
            )? {
                self_update::UpdateResult::AlreadyUpToDate => eprintln!("already up to date"),
                self_update::UpdateResult::Updated { from, to } => {
                    eprintln!("updated: {from} → {to}")
                }
            }
        }

        // -- version: print version --
        Command::Version => {
            println!("mnemonist v{}", env!("CARGO_PKG_VERSION"));
        }

        // -- config: manage configuration --
        Command::Config { action } => match action {
            ConfigAction::Show => {
                let config = mnemonist_core::Config::load();
                let toml_str =
                    toml::to_string_pretty(&config).context("failed to serialize config")?;
                output_ok(json!({
                    "config": toml_str,
                    "path": config.path().display().to_string(),
                }));
            }
            ConfigAction::Get { key } => {
                let config = mnemonist_core::Config::load();
                match config.get(&key) {
                    Some(value) => output_ok(json!({ "key": key, "value": value })),
                    None => output_err(&format!("unknown config key: {key}")),
                }
            }
            ConfigAction::Set { key, value } => {
                let mut config = mnemonist_core::Config::load();
                config.set(&key, &value)?;
                config.save()?;
                info!("set {key} = {value}");
                output_ok(json!({ "key": key, "value": value }));
            }
            ConfigAction::Init => {
                let config = mnemonist_core::Config::default();
                config.save()?;
                info!("created {}", config.path().display());
                output_ok(json!({
                    "path": config.path().display().to_string(),
                    "action": "created",
                }));
            }
            ConfigAction::Path => {
                let config = mnemonist_core::Config::load();
                output_ok(json!({
                    "path": config.path().display().to_string(),
                }));
            }
        },
    }

    Ok(())
}

// -- Helper functions --

/// Map memory type prefix in filename to priority (lower = higher priority).
#[allow(dead_code)]
fn type_priority_from_filename(filename: &str) -> u8 {
    if filename.starts_with("feedback_") {
        0
    } else if filename.starts_with("project_") {
        1
    } else if filename.starts_with("user_") {
        2
    } else {
        3
    }
}

/// Text-based search across directories (fallback when no embeddings).
fn text_search(query: &str, dirs: &[PathBuf]) -> Vec<(PathBuf, IndexEntry, f32)> {
    let mut matched = Vec::new();
    for dir in dirs {
        if let Ok(index) = MemoryIndex::load(dir) {
            for entry in index.search(query) {
                matched.push((dir.clone(), entry.clone(), 0.5));
            }
        }
    }
    matched
}

/// Semantic search using HNSW index + embeddings.
/// Returns (dir, entry, cosine_score) tuples.
fn semantic_search(
    query: &str,
    dirs: &[PathBuf],
    _root: &std::path::Path,
) -> Vec<(PathBuf, IndexEntry, f32)> {
    let embedder = match mnemonist_core::CandleEmbedder::default_model() {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let query_embedding = match embedder.embed(query) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut memory_hits: Vec<(PathBuf, IndexEntry, f32)> = Vec::new();
    let mut code_hits: Vec<(PathBuf, IndexEntry, f32)> = Vec::new();

    for dir in dirs {
        // Memory layer HNSW
        let mem_hnsw_path = dir.join(".memory-index.hnsw");
        if mem_hnsw_path.exists()
            && let Ok(hnsw) = mnemonist_core::ann::hnsw::HnswIndex::load_from(&mem_hnsw_path)
            && let Ok(hits) = mnemonist_core::ann::AnnIndex::search(&hnsw, &query_embedding, 10)
            && let Ok(mem_index) = MemoryIndex::load(dir)
        {
            for hit in hits {
                if let Some(entry) = mem_index.entries.iter().find(|e| e.file == hit.id) {
                    memory_hits.push((dir.clone(), entry.clone(), hit.score));
                }
            }
        }

        // Code layer HNSW
        let code_hnsw_path = dir.join(".code-index.hnsw");
        if code_hnsw_path.exists()
            && let Ok(hnsw) = mnemonist_core::ann::hnsw::HnswIndex::load_from(&code_hnsw_path)
            && let Ok(hits) = mnemonist_core::ann::AnnIndex::search(&hnsw, &query_embedding, 10)
        {
            for hit in hits {
                let parts: Vec<&str> = hit.id.rsplitn(3, ':').collect();
                if parts.len() == 3 {
                    let file = parts[2].to_string();
                    let start = parts[1];
                    let end = parts[0];
                    code_hits.push((
                        dir.clone(),
                        IndexEntry {
                            title: file.clone(),
                            file: format!("{}:{}:{}", file, start, end),
                            summary: format!(
                                "code chunk L{}-L{} (score: {:.3})",
                                start, end, hit.score
                            ),
                        },
                        hit.score,
                    ));
                }
            }
        }

        // Fallback: brute-force cosine on .embeddings.bin (when no HNSW exists yet)
        if memory_hits.is_empty() && !mem_hnsw_path.exists() {
            let store_path = dir.join(".embeddings.bin");
            if store_path.exists()
                && let Ok(store) = EmbeddingStore::load(&store_path)
            {
                let mut scores: Vec<(&str, f32)> = store
                    .entries
                    .iter()
                    .map(|e| {
                        let sim = cosine_similarity(&query_embedding, &e.embedding);
                        (e.file.as_str(), sim)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                if let Ok(mem_index) = MemoryIndex::load(dir) {
                    for (file, score) in scores.iter().take(10) {
                        if let Some(entry) = mem_index.entries.iter().find(|e| e.file == *file) {
                            memory_hits.push((dir.clone(), entry.clone(), *score));
                        }
                    }
                }
            }
        }
    }

    // Hierarchical: seed code search with top memory hit's embedding
    if let Some((top_dir, top_entry, _)) = memory_hits.first() {
        let store_path = top_dir.join(".embeddings.bin");
        if let Ok(store) = EmbeddingStore::load(&store_path)
            && let Some(emb) = store.get(&top_entry.file)
        {
            let seen: std::collections::HashSet<String> =
                code_hits.iter().map(|(_, e, _)| e.file.clone()).collect();
            for dir in dirs {
                let code_hnsw_path = dir.join(".code-index.hnsw");
                if code_hnsw_path.exists()
                    && let Ok(hnsw) = mnemonist_core::ann::hnsw::HnswIndex::load_from(&code_hnsw_path)
                    && let Ok(hits) = mnemonist_core::ann::AnnIndex::search(&hnsw, &emb.embedding, 5)
                {
                    for hit in hits {
                        let parts: Vec<&str> = hit.id.rsplitn(3, ':').collect();
                        if parts.len() == 3 {
                            let file = format!("{}:{}:{}", parts[2], parts[1], parts[0]);
                            if !seen.contains(&file) {
                                code_hits.push((
                                    dir.clone(),
                                    IndexEntry {
                                        title: parts[2].to_string(),
                                        file,
                                        summary: format!(
                                            "via {} (score: {:.3})",
                                            top_entry.file, hit.score
                                        ),
                                    },
                                    hit.score,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // Merge all hits — caller sorts by blended score
    let mut matched = memory_hits;
    matched.extend(code_hits);
    matched
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Extract a compressed cue from code content.
/// Prioritizes signatures and public items over raw truncation.
fn extract_code_cue(content: &str, max_chars: usize) -> String {
    const SIG_PREFIXES: &[&str] = &[
        "pub fn ",
        "pub async fn ",
        "pub struct ",
        "pub enum ",
        "pub trait ",
        "pub type ",
        "pub const ",
        "pub static ",
        "fn ",
        "async fn ",
        "struct ",
        "enum ",
        "trait ",
        "impl ",
        "type ",
        "const ",
        "class ",
        "def ",
        "async def ",
        "func ",
        "function ",
        "export ",
    ];

    let mut lines = Vec::new();
    let mut used = 0;
    for line in content.lines() {
        let trimmed = line.trim();
        if SIG_PREFIXES.iter().any(|p| trimmed.starts_with(p)) {
            if used + line.len() + 1 > max_chars {
                break;
            }
            lines.push(line);
            used += line.len() + 1;
        }
    }
    if !lines.is_empty() {
        return format!("{}...", lines.join("\n"));
    }
    // Fallback: first max_chars characters
    if content.len() > max_chars {
        format!("{}...", content.chars().take(max_chars).collect::<String>())
    } else {
        content.to_string()
    }
}

/// Try to embed a single file into the embedding store.
fn try_embed_single(dir: &std::path::Path, filename: &str) -> Result<()> {
    let embedder = mnemonist_core::CandleEmbedder::default_model()?;
    let store_path = dir.join(".embeddings.bin");

    let mut store = if store_path.exists() {
        EmbeddingStore::load(&store_path)?
    } else {
        EmbeddingStore::new(embedder.dimension()?)
    };

    let file_path = dir.join(filename);
    let content = std::fs::read_to_string(&file_path)?;
    let hash = mnemonist_core::embed::content_hash(&content);
    let embedding = embedder.embed(&content)?;

    store.upsert(mnemonist_core::EmbeddingEntry {
        file: filename.to_string(),
        hash,
        embedding,
    });
    store.save(&store_path)?;

    Ok(())
}

/// Build (or rebuild) the memory-layer HNSW index from all .md files in a directory.
/// Uses the existing .embeddings.bin as the source of vectors.
fn build_memory_hnsw(dir: &std::path::Path) -> Result<usize> {
    let store_path = dir.join(".embeddings.bin");
    if !store_path.exists() {
        return Ok(0);
    }
    let store = EmbeddingStore::load(&store_path)?;
    if store.entries.is_empty() {
        return Ok(0);
    }

    let mut hnsw = mnemonist_core::ann::hnsw::HnswIndex::with_defaults(store.dimension);
    for entry in &store.entries {
        mnemonist_core::ann::AnnIndex::insert(&mut hnsw, &entry.file, &entry.embedding)?;
    }

    let hnsw_path = dir.join(".memory-index.hnsw");
    mnemonist_core::ann::AnnIndex::save(&hnsw, &hnsw_path)?;
    Ok(store.entries.len())
}

/// Try to re-embed all memories in a directory. Returns (synced, total).
fn try_embed_all(dir: &std::path::Path) -> Result<(usize, usize)> {
    let embedder = mnemonist_core::CandleEmbedder::default_model()?;
    let store_path = dir.join(".embeddings.bin");

    let mut store = if store_path.exists() {
        EmbeddingStore::load(&store_path)?
    } else {
        EmbeddingStore::new(embedder.dimension()?)
    };

    let synced = store.sync(dir, &embedder)?;
    let total = store.entries.len();
    store.save(&store_path)?;

    Ok((synced, total))
}
