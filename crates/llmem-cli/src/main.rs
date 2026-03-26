use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use llmem_core::{
    Frontmatter, IndexEntry, MemoryFile, MemoryIndex, MemoryType, global_dir, project_dir,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "llmem", about = "Manage AI agent memory")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Initialize a memory directory
    Init {
        /// Initialize global memory (~/.config/llmem/) instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Add a new memory
    Add {
        /// Memory type: user, feedback, project, reference
        #[arg(value_parser = parse_memory_type)]
        memory_type: MemoryType,

        /// Short kebab-case name
        name: String,

        /// One-line description
        #[arg(long, short)]
        description: String,

        /// Memory body content
        #[arg(long, short)]
        body: Option<String>,

        /// Store in global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// List memories from the index
    List {
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
    /// Search memories by query
    Search {
        /// Search query
        query: String,

        /// Search level: project, global, or both
        #[arg(long, short, default_value = "both")]
        level: String,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Remove a memory by filename
    Remove {
        /// Filename to remove (e.g., feedback_standards.md)
        file: String,

        /// Remove from global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Sync embeddings and rebuild the ANN index
    Embed {
        /// Sync global memory instead of project
        #[arg(long, short)]
        global: bool,

        /// Project root (defaults to current directory)
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Context management
    Ctx {
        #[command(subcommand)]
        action: CtxAction,
    },
}

#[derive(Subcommand)]
enum CtxAction {
    /// Switch active project context
    Switch {
        /// Project root to switch to (defaults to current directory)
        #[arg(default_value = ".")]
        root: PathBuf,
    },
    /// Show the currently active context
    Show,
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

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Init { global, root } => {
            let dir = resolve_dir(global, &root)?;
            MemoryIndex::init(&dir)?;
            let level = if global { "global" } else { "project" };
            println!("initialized {level} memory at {}", dir.display());
        }

        Command::Add {
            memory_type,
            name,
            description,
            body,
            global,
            root,
        } => {
            let dir = resolve_dir(global, &root)?;
            let mut index = MemoryIndex::load(&dir)?;

            let mem = MemoryFile {
                frontmatter: Frontmatter {
                    name: name.clone(),
                    description: description.clone(),
                    memory_type,
                },
                body: body.unwrap_or_default(),
            };

            let filename = mem.filename();
            let entry = IndexEntry {
                title: name.replace('-', " "),
                file: filename.clone(),
                summary: description,
            };

            index.add(entry)?;
            mem.write(&dir.join(&filename))?;
            index.save()?;

            println!("added {filename}");
        }

        Command::List { global, all, root } => {
            let dirs = if all {
                let mut v = vec![project_dir(&root)];
                if let Some(g) = global_dir() {
                    v.push(g);
                }
                v
            } else {
                vec![resolve_dir(global, &root)?]
            };

            for dir in dirs {
                let index = MemoryIndex::load(&dir)?;
                if !index.entries.is_empty() {
                    println!("{}:", dir.display());
                    for entry in &index.entries {
                        println!("  {}", entry.to_line());
                    }
                }
            }
        }

        Command::Search { query, level, root } => {
            let dirs = match level.as_str() {
                "project" => vec![project_dir(&root)],
                "global" => vec![global_dir().context("could not determine config directory")?],
                _ => {
                    let mut v = vec![project_dir(&root)];
                    if let Some(g) = global_dir() {
                        v.push(g);
                    }
                    v
                }
            };

            for dir in dirs {
                let index = MemoryIndex::load(&dir)?;
                let results = index.search(&query);
                if !results.is_empty() {
                    println!("{}:", dir.display());
                    for entry in results {
                        println!("  {}", entry.to_line());
                    }
                }
            }
        }

        Command::Remove { file, global, root } => {
            let dir = resolve_dir(global, &root)?;
            let mut index = MemoryIndex::load(&dir)?;

            if index.remove(&file) {
                let path = dir.join(&file);
                if path.exists() {
                    std::fs::remove_file(&path)?;
                }
                index.save()?;
                println!("removed {file}");
            } else {
                println!("not found: {file}");
            }
        }

        Command::Embed { global, root } => {
            let dir = resolve_dir(global, &root)?;
            let store_path = dir.join(".embeddings.bin");

            // Load existing or create new
            let store = if store_path.exists() {
                llmem_core::EmbeddingStore::load(&store_path)?
            } else {
                // Dimension will be set by first embedding — placeholder
                llmem_core::EmbeddingStore::new(0)
            };

            println!(
                "embedding store at {} ({} entries, dim={})",
                store_path.display(),
                store.entries.len(),
                store.dimension
            );
            println!("note: provide an Embedder implementation to sync embeddings.");
            println!("      currently only the store format is managed by the CLI.");
            println!("      use llmem-server with a configured embedder for full sync.");

            store.save(&store_path)?;
        }

        Command::Ctx { action } => match action {
            CtxAction::Switch { root } => {
                let root =
                    std::fs::canonicalize(&root).context("could not resolve project root")?;
                let ctx_dir = global_dir().context("could not determine config directory")?;
                std::fs::create_dir_all(&ctx_dir)?;
                let ctx_file = ctx_dir.join(".active-ctx");
                std::fs::write(&ctx_file, root.display().to_string())?;
                println!("switched context to {}", root.display());
            }
            CtxAction::Show => {
                let ctx_dir = global_dir().context("could not determine config directory")?;
                let ctx_file = ctx_dir.join(".active-ctx");
                if ctx_file.exists() {
                    let ctx = std::fs::read_to_string(&ctx_file)?;
                    println!("{}", ctx.trim());
                } else {
                    println!("no active context (use `llmem ctx switch` to set one)");
                }
            }
        },
    }

    Ok(())
}
