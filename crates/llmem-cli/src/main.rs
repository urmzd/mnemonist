use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llmem_core::{
    Config, Embedder, EmbeddingStore, Frontmatter, Inbox, InboxItem, IndexEntry, MemoryFile,
    MemoryIndex, MemoryType, global_dir, llmem_root, project_dir,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Read as _};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "llmem",
    about = "Memory for AI agents — learn, remember, consolidate"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Force JSON output (default when stdout is not a TTY)
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Initialize a memory directory
    Init {
        /// Initialize global memory (~/.llmem/global/) instead of project
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

    /// Manage llmem configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
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

fn resolve_dir(global: bool, root: &std::path::Path) -> Result<PathBuf> {
    if global {
        global_dir().context("could not determine config directory")
    } else {
        Ok(project_dir(root))
    }
}

/// Collect memory directories for a given level string.
fn level_dirs(level: &str, root: &std::path::Path) -> Result<Vec<PathBuf>> {
    match level {
        "project" => Ok(vec![project_dir(root)]),
        "global" => Ok(vec![
            global_dir().context("could not determine config directory")?,
        ]),
        _ => {
            let mut v = vec![project_dir(root)];
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

    if let Err(e) = run(cli) {
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

            // Notify daemon to reload indices
            daemon_notify_reload();

            info!("{action} {filename}");
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
                attention_score: 0.5, // default for manual notes
                created_at: now_iso(),
                file_source: None,
            };

            inbox.push(item);
            inbox.last_updated = Some(now_iso());
            inbox.save(&dir)?;

            // Notify daemon to reload
            daemon_notify_reload();

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

            // Try daemon first (indices warm in memory)
            if let Some(resp) = daemon_remember(&q, &level, budget, &root) {
                println!("{}", serde_json::to_string(&resp).unwrap());
            } else {
                // Fall back to local file-based search
                let dirs = level_dirs(&level, &root)?;
                let mut memories: Vec<MemoryOut> = Vec::new();
                let mut total_chars = 0usize;

                let mut matched = semantic_search(&q, &dirs, &root);
                if matched.is_empty() {
                    matched = text_search(&q, &dirs);
                }

                matched.sort_by_key(|(_, e)| type_priority_from_filename(&e.file));

                for (dir, entry) in &matched {
                    if total_chars >= budget {
                        break;
                    }

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
                                if total_chars + body_chars > budget && !memories.is_empty() {
                                    break;
                                }
                                total_chars += body_chars;
                                memories.push(MemoryOut {
                                    file: entry.file.clone(),
                                    memory_type: "code".to_string(),
                                    name: entry.title.clone(),
                                    description: entry.summary.clone(),
                                    body,
                                });
                            }
                        }
                        continue;
                    }

                    // Memory file entries
                    let path = dir.join(&entry.file);
                    if let Ok(mut mem) = MemoryFile::read(&path) {
                        let body_chars = mem.body.len();
                        if total_chars + body_chars > budget && !memories.is_empty() {
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
                        });
                    }
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

            // Phase 1: tree-sitter code extraction
            let mut code_index = llmem_index::code::CodeIndex::new(&ingest_path);
            let chunk_count = code_index.index()?;

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
                        "kind": c.kind,
                        "name": c.name,
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
            match llmem_core::FastEmbedder::default_model() {
                Ok(embedder) => {
                    let dim = embedder.dimension()?;
                    let mut hnsw = llmem_index::hnsw::HnswIndex::with_defaults(dim);
                    match code_index.build_ann(&embedder, &mut hnsw) {
                        Ok(n) => {
                            embedded_count = n;
                            let hnsw_path = mem_dir.join(".code-index.hnsw");
                            if let Err(e) = llmem_index::AnnIndex::save(&hnsw, &hnsw_path) {
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
                                let aniso = llmem_index::eval::anisotropy(&sample_embeddings);
                                let range = llmem_index::eval::similarity_range(&sample_embeddings);
                                eval_json = json!({
                                    "sample_size": sample_embeddings.len(),
                                    "anisotropy": (aniso * 10000.0).round() / 10000.0,
                                    "similarity_range": (range * 10000.0).round() / 10000.0,
                                });
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

            // Score chunks by importance heuristic:
            // - public items (pub, export) score higher
            // - functions/structs score higher than misc
            // - recently modified files score higher (approximated by order)
            for chunk in code_index.chunks() {
                let kind_score = match chunk.kind.as_str() {
                    "function_item" | "function_definition" => 0.8,
                    "struct_item" | "class_definition" => 0.9,
                    "impl_item" | "trait_item" => 0.85,
                    "enum_item" => 0.75,
                    _ => 0.5,
                };

                let visibility_bonus =
                    if chunk.content.contains("pub ") || chunk.content.contains("export ") {
                        0.1
                    } else {
                        0.0
                    };

                let _name_str = chunk.name.as_deref().unwrap_or("unknown");

                let item = InboxItem {
                    id: chunk.id(),
                    content: chunk.content.clone(),
                    source: "learn".to_string(),
                    attention_score: kind_score + visibility_bonus,
                    created_at: now.clone(),
                    file_source: Some(llmem_core::FileSource {
                        file: chunk.file.clone(),
                        start_line: Some(chunk.start_line),
                        end_line: Some(chunk.end_line),
                        kind: chunk.kind.clone(),
                    }),
                };

                inbox.push(item);
            }

            inbox.last_updated = Some(now.clone());
            inbox.save(&mem_dir)?;

            info!(
                "learned {} chunks from {} files, {} embedded, {} in inbox",
                chunk_count,
                file_count,
                embedded_count,
                inbox.len()
            );
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
                output_err("memory directory does not exist — run `llmem init` first");
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
                    body: item.content.clone(),
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

            if !dry_run {
                daemon_notify_reload();
            }

            info!(
                "consolidated: {} promoted, {} decayed{}",
                promoted,
                decayed,
                if dry_run { " (dry run)" } else { "" }
            );
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

            if index.remove(&file) {
                let path = dir.join(&file);
                if path.exists() {
                    std::fs::remove_file(&path)?;
                }

                // Also remove from embedding store
                let store_path = dir.join(".embeddings.bin");
                if store_path.exists()
                    && let Ok(mut store) = EmbeddingStore::load(&store_path)
                {
                    store.remove(&file);
                    let _ = store.save(&store_path);
                }

                index.save()?;
                info!("forgot {file}");
                output_ok(json!({
                    "file": file,
                    "action": "forgotten",
                }));
            } else {
                output_err(&format!("not found: {file}"));
            }
        }

        // -- config: manage configuration --
        Command::Config { action } => match action {
            ConfigAction::Show => {
                let config = llmem_core::Config::load();
                let toml_str =
                    toml::to_string_pretty(&config).context("failed to serialize config")?;
                output_ok(json!({
                    "config": toml_str,
                    "path": config.path().display().to_string(),
                }));
            }
            ConfigAction::Get { key } => {
                let config = llmem_core::Config::load();
                match config.get(&key) {
                    Some(value) => output_ok(json!({ "key": key, "value": value })),
                    None => output_err(&format!("unknown config key: {key}")),
                }
            }
            ConfigAction::Set { key, value } => {
                let mut config = llmem_core::Config::load();
                config.set(&key, &value)?;
                config.save()?;
                info!("set {key} = {value}");
                output_ok(json!({ "key": key, "value": value }));
            }
            ConfigAction::Init => {
                let config = llmem_core::Config::default();
                config.save()?;
                info!("created {}", config.path().display());
                output_ok(json!({
                    "path": config.path().display().to_string(),
                    "action": "created",
                }));
            }
            ConfigAction::Path => {
                let config = llmem_core::Config::load();
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
fn text_search(query: &str, dirs: &[PathBuf]) -> Vec<(PathBuf, IndexEntry)> {
    let mut matched = Vec::new();
    for dir in dirs {
        if let Ok(index) = MemoryIndex::load(dir) {
            for entry in index.search(query) {
                matched.push((dir.clone(), entry.clone()));
            }
        }
    }
    matched
}

/// Semantic search using HNSW index + embeddings.
fn semantic_search(
    query: &str,
    dirs: &[PathBuf],
    _root: &std::path::Path,
) -> Vec<(PathBuf, IndexEntry)> {
    let embedder = match llmem_core::FastEmbedder::default_model() {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let query_embedding = match embedder.embed(query) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut matched = Vec::new();

    for dir in dirs {
        // Try memory HNSW index (layered graph: memory layer)
        let mem_hnsw_path = dir.join(".memory-index.hnsw");
        if mem_hnsw_path.exists()
            && let Ok(hnsw) = llmem_index::hnsw::HnswIndex::load_from(&mem_hnsw_path)
            && let Ok(hits) = llmem_index::AnnIndex::search(&hnsw, &query_embedding, 10)
            && let Ok(mem_index) = MemoryIndex::load(dir)
        {
            for hit in hits {
                if let Some(entry) = mem_index.entries.iter().find(|e| e.file == hit.id) {
                    matched.push((dir.clone(), entry.clone()));
                }
            }
        }

        // Try code HNSW index
        let code_hnsw_path = dir.join(".code-index.hnsw");
        if code_hnsw_path.exists()
            && let Ok(hnsw) = llmem_index::hnsw::HnswIndex::load_from(&code_hnsw_path)
            && let Ok(hits) = llmem_index::AnnIndex::search(&hnsw, &query_embedding, 10)
        {
            // Resolve hit IDs (format: "file:start:end") to IndexEntry
            for hit in hits {
                let parts: Vec<&str> = hit.id.rsplitn(3, ':').collect();
                if parts.len() == 3 {
                    let file = parts[2].to_string();
                    let start = parts[1];
                    let end = parts[0];
                    matched.push((
                        dir.clone(),
                        IndexEntry {
                            title: file.clone(),
                            file: format!("{}:{}:{}", file, start, end),
                            summary: format!(
                                "code chunk L{}-L{} (score: {:.3})",
                                start, end, hit.score
                            ),
                        },
                    ));
                }
            }
        }

        // Fallback: brute-force cosine on .embeddings.bin (when no HNSW exists yet)
        if matched.is_empty() && !mem_hnsw_path.exists() {
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
                    for (file, _score) in scores.iter().take(10) {
                        if let Some(entry) = mem_index.entries.iter().find(|e| e.file == *file) {
                            matched.push((dir.clone(), entry.clone()));
                        }
                    }
                }
            }
        }
    }

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

/// Try to embed a single file into the embedding store.
fn try_embed_single(dir: &std::path::Path, filename: &str) -> Result<()> {
    let embedder = llmem_core::FastEmbedder::default_model()?;
    let store_path = dir.join(".embeddings.bin");

    let mut store = if store_path.exists() {
        EmbeddingStore::load(&store_path)?
    } else {
        EmbeddingStore::new(embedder.dimension()?)
    };

    let file_path = dir.join(filename);
    let content = std::fs::read_to_string(&file_path)?;
    let hash = llmem_core::embed::content_hash(&content);
    let embedding = embedder.embed(&content)?;

    store.upsert(llmem_core::EmbeddingEntry {
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

    let mut hnsw = llmem_index::hnsw::HnswIndex::with_defaults(store.dimension);
    for entry in &store.entries {
        llmem_index::AnnIndex::insert(&mut hnsw, &entry.file, &entry.embedding)?;
    }

    let hnsw_path = dir.join(".memory-index.hnsw");
    llmem_index::AnnIndex::save(&hnsw, &hnsw_path)?;
    Ok(store.entries.len())
}

/// Try to re-embed all memories in a directory. Returns (synced, total).
fn try_embed_all(dir: &std::path::Path) -> Result<(usize, usize)> {
    let embedder = llmem_core::FastEmbedder::default_model()?;
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

// -- Daemon client (Unix socket) --

/// Socket path for the llmem daemon.
fn daemon_socket_path() -> Option<PathBuf> {
    llmem_root().map(|r| r.join("llmem.sock"))
}

/// Send a request to the daemon over Unix socket. Returns the parsed JSON
/// response, or None if the daemon is not running.
fn daemon_request(path: &str) -> Option<Value> {
    use std::io::{BufRead, Write as _};
    use std::os::unix::net::UnixStream;

    let sock = daemon_socket_path()?;
    let mut stream = UnixStream::connect(&sock).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok()?;

    // Send HTTP/1.1 request
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).ok()?;

    // Read response
    let mut reader = std::io::BufReader::new(stream);

    // Skip HTTP status line
    let mut status_line = String::new();
    reader.read_line(&mut status_line).ok()?;
    if !status_line.contains("200") {
        return None;
    }

    // Skip headers until empty line
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        if line.trim().is_empty() {
            break;
        }
    }

    // Read body (may be chunked or content-length based, but with Connection: close
    // we can just read to EOF for simplicity)
    let mut body = String::new();
    reader.read_to_string(&mut body).ok()?;

    // Handle chunked transfer encoding (axum default for HTTP/1.1)
    let body = decode_chunked(&body).unwrap_or(body);

    serde_json::from_str(&body).ok()
}

/// Decode chunked transfer encoding.
fn decode_chunked(input: &str) -> Option<String> {
    let mut result = String::new();
    let mut remaining = input;

    loop {
        let remaining_trimmed = remaining.trim_start();
        let newline_pos = remaining_trimmed.find("\r\n")?;
        let size_str = &remaining_trimmed[..newline_pos];
        let size = usize::from_str_radix(size_str.trim(), 16).ok()?;
        if size == 0 {
            break;
        }
        let data_start = remaining_trimmed.get((newline_pos + 2)..)?;
        if data_start.len() < size {
            return None;
        }
        result.push_str(&data_start[..size]);
        remaining = &data_start[(size)..];
        // Skip trailing \r\n after chunk data
        if remaining.starts_with("\r\n") {
            remaining = &remaining[2..];
        }
    }

    Some(result)
}

/// Try to remember via the daemon. Returns the full JSON response if successful.
fn daemon_remember(
    query: &str,
    level: &str,
    budget: usize,
    root: &std::path::Path,
) -> Option<Value> {
    let root_str = root.display();
    let encoded_q = urlencoded(query);
    let path = format!("/remember?q={encoded_q}&level={level}&budget={budget}&root={root_str}");
    daemon_request(&path)
}

/// Fire-and-forget reload signal to the daemon.
fn daemon_notify_reload() {
    let _ = daemon_request("/reload");
}

/// Minimal URL encoding for query parameters.
fn urlencoded(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push_str(&format!("{b:02X}"));
            }
        }
    }
    out
}
