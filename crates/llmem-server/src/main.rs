use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Query, State},
    routing::get,
};
use clap::Parser;
use hyper_util::{rt::TokioIo, service::TowerToHyperService};
use llmem_core::{EmbeddingStore, MemoryFile, MemoryIndex, global_dir, llmem_root, project_dir};
use llmem_index::{AnnIndex, hnsw::HnswIndex};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[derive(Parser)]
#[command(name = "llmem-server", about = "Memory daemon for llmem")]
struct Cli {
    /// Project root directory (omit for global-only mode)
    #[arg(long)]
    root: Option<PathBuf>,

    /// Bind to TCP address instead of Unix socket (for debugging)
    #[arg(long)]
    addr: Option<String>,
}

struct AppState {
    project_index: RwLock<Option<HnswIndex>>,
    global_index: RwLock<Option<HnswIndex>>,
    project_embeddings: RwLock<Option<EmbeddingStore>>,
    global_embeddings: RwLock<Option<EmbeddingStore>>,
    /// The original project root path (for display in /health).
    project_root: RwLock<Option<PathBuf>>,
    /// Resolved project memory directory (config-derived from project_root).
    project_mem_dir: RwLock<Option<PathBuf>>,
}

// -- Parameter types --

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_level")]
    level: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

#[derive(Deserialize)]
struct RememberParams {
    q: String,
    #[serde(default = "default_level")]
    level: String,
    #[serde(default = "default_budget")]
    budget: usize,
    /// Override project root for this request
    root: Option<String>,
}

#[derive(Deserialize)]
struct ReloadParams {
    root: Option<String>,
}

fn default_level() -> String {
    "both".to_string()
}

fn default_top_k() -> usize {
    10
}

fn default_budget() -> usize {
    2000
}

// -- Response types --

#[derive(Serialize)]
struct SearchResult {
    title: String,
    file: String,
    summary: String,
    level: String,
    score: f32,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    project_memories: usize,
    global_memories: usize,
    active_context: Option<String>,
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

// -- Handlers --

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let project_count = state
        .project_index
        .read()
        .ok()
        .and_then(|i| i.as_ref().map(|idx| idx.len()))
        .unwrap_or(0);
    let global_count = state
        .global_index
        .read()
        .ok()
        .and_then(|i| i.as_ref().map(|idx| idx.len()))
        .unwrap_or(0);
    let active_ctx = state
        .project_root
        .read()
        .ok()
        .and_then(|r| r.as_ref().map(|p| p.display().to_string()));

    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        project_memories: project_count,
        global_memories: global_count,
        active_context: active_ctx,
    })
}

async fn search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Json<Vec<SearchResult>> {
    let mut results = Vec::new();

    let search_level = |dir: &PathBuf, level: &str| -> Vec<SearchResult> {
        let index = MemoryIndex::load(dir).ok();
        index
            .map(|idx| {
                idx.search(&params.q)
                    .into_iter()
                    .map(|e| SearchResult {
                        title: e.title.clone(),
                        file: e.file.clone(),
                        summary: e.summary.clone(),
                        level: level.to_string(),
                        score: 1.0,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };

    let include_project = params.level != "global";
    let include_global = params.level != "project";

    if include_project && let Some(dir) = state.project_mem_dir.read().ok().and_then(|d| d.clone())
    {
        results.extend(search_level(&dir, "project"));
    }

    if include_global && let Some(dir) = global_dir() {
        results.extend(search_level(&dir, "global"));
    }

    results.truncate(params.top_k);
    Json(results)
}

/// Full remember endpoint — semantic search with Hebbian reinforcement.
/// This is the daemon's primary value: indices stay warm in memory.
async fn remember(
    State(state): State<Arc<AppState>>,
    Query(params): Query<RememberParams>,
) -> Json<serde_json::Value> {
    let include_project = params.level != "global";
    let include_global = params.level != "project";

    // Resolve dirs for this request
    let project_mem = if include_project {
        if let Some(ref root_str) = params.root {
            Some(project_dir(&PathBuf::from(root_str)))
        } else {
            state.project_mem_dir.read().ok().and_then(|d| d.clone())
        }
    } else {
        None
    };
    let global_mem = if include_global { global_dir() } else { None };

    // Collect (dir, entry) matches via HNSW then text fallback
    let mut matched: Vec<(PathBuf, llmem_core::IndexEntry)> = Vec::new();

    // HNSW semantic search using cached indices
    if let Some(ref dir) = project_mem {
        hnsw_search_from_state(
            &state.project_index,
            &state.project_embeddings,
            dir,
            &params.q,
            &mut matched,
        );
    }
    if let Some(ref dir) = global_mem {
        hnsw_search_from_state(
            &state.global_index,
            &state.global_embeddings,
            dir,
            &params.q,
            &mut matched,
        );
    }

    // Text fallback if semantic returned nothing
    if matched.is_empty() {
        if let Some(ref dir) = project_mem {
            text_search(dir, &params.q, &mut matched);
        }
        if let Some(ref dir) = global_mem {
            text_search(dir, &params.q, &mut matched);
        }
    }

    // Sort by type priority: feedback > project > user > reference
    matched.sort_by_key(|(_, e)| type_priority_from_filename(&e.file));

    // Load memory files up to budget, apply Hebbian reinforcement
    let mut memories: Vec<MemoryOut> = Vec::new();
    let mut total_chars = 0usize;

    for (dir, entry) in &matched {
        if total_chars >= params.budget {
            break;
        }
        let path = dir.join(&entry.file);
        if let Ok(mut mem) = MemoryFile::read(&path) {
            let body_chars = mem.body.len();
            if total_chars + body_chars > params.budget && !memories.is_empty() {
                break;
            }
            total_chars += body_chars;

            // Hebbian reinforcement
            mem.frontmatter.access_count += 1;
            mem.frontmatter.last_accessed = Some(chrono::Utc::now().to_rfc3339());
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

    Json(serde_json::json!({
        "ok": true,
        "data": {
            "memories": memories,
            "token_estimate": total_chars / 4,
        }
    }))
}

/// Search HNSW index from cached state, with brute-force cosine fallback.
fn hnsw_search_from_state(
    index_lock: &RwLock<Option<HnswIndex>>,
    embeddings_lock: &RwLock<Option<EmbeddingStore>>,
    dir: &Path,
    _query: &str,
    _matched: &mut Vec<(PathBuf, llmem_core::IndexEntry)>,
) {
    // Try HNSW if we have a query embedding (text-based HNSW search by index title)
    if let Ok(guard) = index_lock.read()
        && let Some(ref _hnsw) = *guard
    {
        // HNSW requires a query vector — we don't embed on the server side (yet).
        // Fall through to text search for now.
    }

    // Brute-force cosine on cached embedding store (no query embedding needed for text match)
    // This is a placeholder — full semantic search requires the embedder on the server.
    // For now, the HNSW/embedding path is a no-op and we rely on text fallback below.
    let _ = embeddings_lock;
    let _ = dir;
}

fn text_search(dir: &Path, query: &str, matched: &mut Vec<(PathBuf, llmem_core::IndexEntry)>) {
    if let Ok(index) = MemoryIndex::load(dir) {
        for entry in index.search(query) {
            matched.push((dir.to_path_buf(), entry.clone()));
        }
    }
}

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

async fn reload(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ReloadParams>,
) -> Json<serde_json::Value> {
    if let Some(root_str) = params.root {
        let root = PathBuf::from(&root_str);
        if let Ok(root) = std::fs::canonicalize(&root) {
            let mem_dir = project_dir(&root);
            reload_project(&state, &root, &mem_dir);
        }
    } else {
        // Reload current project index if set
        if let Ok(pd) = state.project_mem_dir.read()
            && let Some(ref mem_dir) = *pd
        {
            let root = state
                .project_root
                .read()
                .ok()
                .and_then(|r| r.clone())
                .unwrap_or_default();
            reload_project(&state, &root, mem_dir);
        }
    }

    // Always reload global index + embeddings
    if let Some(dir) = global_dir() {
        let hnsw_path = dir.join(".index.hnsw");
        if hnsw_path.exists()
            && let Ok(idx) = HnswIndex::load_from(&hnsw_path)
            && let Ok(mut gi) = state.global_index.write()
        {
            *gi = Some(idx);
        }
        let store_path = dir.join(".embeddings.bin");
        if store_path.exists()
            && let Ok(store) = EmbeddingStore::load(&store_path)
            && let Ok(mut ge) = state.global_embeddings.write()
        {
            *ge = Some(store);
        }
    }

    Json(serde_json::json!({"status": "reloaded"}))
}

fn reload_project(state: &AppState, root: &std::path::Path, mem_dir: &std::path::Path) {
    if let Ok(mut pr) = state.project_root.write() {
        *pr = Some(root.to_path_buf());
    }
    if let Ok(mut pd) = state.project_mem_dir.write() {
        *pd = Some(mem_dir.to_path_buf());
    }
    let hnsw_path = mem_dir.join(".index.hnsw");
    if hnsw_path.exists()
        && let Ok(idx) = HnswIndex::load_from(&hnsw_path)
        && let Ok(mut pi) = state.project_index.write()
    {
        *pi = Some(idx);
    }
    let store_path = mem_dir.join(".embeddings.bin");
    if store_path.exists()
        && let Ok(store) = EmbeddingStore::load(&store_path)
        && let Ok(mut pe) = state.project_embeddings.write()
    {
        *pe = Some(store);
    }
}

// -- Router --

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/search", get(search))
        .route("/remember", get(remember))
        .route("/reload", get(reload))
        .with_state(state)
}

// -- Socket path --

fn socket_path() -> PathBuf {
    llmem_root()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("llmem.sock")
}

fn pid_path() -> PathBuf {
    llmem_root()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("llmem.pid")
}

// -- Main --

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut project_root = None;
    let mut project_mem_dir = None;
    let mut project_hnsw = None;
    let mut global_hnsw = None;
    let mut project_emb = None;
    let mut global_emb = None;

    // Load project context from --root
    if let Some(ref root) = cli.root {
        let root = std::fs::canonicalize(root)?;
        let mem_dir = project_dir(&root);
        let hnsw_path = mem_dir.join(".index.hnsw");
        if hnsw_path.exists() {
            project_hnsw = HnswIndex::load_from(&hnsw_path).ok();
        }
        let store_path = mem_dir.join(".embeddings.bin");
        if store_path.exists() {
            project_emb = EmbeddingStore::load(&store_path).ok();
        }
        project_root = Some(root);
        project_mem_dir = Some(mem_dir);
    }

    // Load global
    if let Some(dir) = global_dir() {
        let hnsw_path = dir.join(".index.hnsw");
        if hnsw_path.exists() {
            global_hnsw = HnswIndex::load_from(&hnsw_path).ok();
        }
        let store_path = dir.join(".embeddings.bin");
        if store_path.exists() {
            global_emb = EmbeddingStore::load(&store_path).ok();
        }
    }

    let state = Arc::new(AppState {
        project_index: RwLock::new(project_hnsw),
        global_index: RwLock::new(global_hnsw),
        project_embeddings: RwLock::new(project_emb),
        global_embeddings: RwLock::new(global_emb),
        project_root: RwLock::new(project_root),
        project_mem_dir: RwLock::new(project_mem_dir),
    });

    let app = build_router(state);

    if let Some(ref addr) = cli.addr {
        // TCP mode (for debugging / non-Unix)
        println!("llmem-server listening on tcp://{addr}");
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
    } else {
        // Unix socket mode (default)
        let sock = socket_path();
        if let Some(parent) = sock.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Clean up stale socket
        if sock.exists() {
            std::fs::remove_file(&sock)?;
        }
        // Write PID file
        std::fs::write(pid_path(), std::process::id().to_string())?;

        println!("llmem-server listening on unix://{}", sock.display());

        let listener = tokio::net::UnixListener::bind(&sock)?;

        // Serve with hyper over Unix socket
        loop {
            let (stream, _addr) = listener.accept().await?;
            let svc = app.clone();

            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let hyper_svc = TowerToHyperService::new(svc);
                let conn =
                    hyper::server::conn::http1::Builder::new().serve_connection(io, hyper_svc);
                if let Err(e) = conn.await {
                    eprintln!("connection error: {e}");
                }
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::extract::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_state() -> Arc<AppState> {
        Arc::new(AppState {
            project_index: RwLock::new(None),
            global_index: RwLock::new(None),
            project_embeddings: RwLock::new(None),
            global_embeddings: RwLock::new(None),
            project_root: RwLock::new(None),
            project_mem_dir: RwLock::new(None),
        })
    }

    fn test_state_with_root(root: PathBuf) -> Arc<AppState> {
        Arc::new(AppState {
            project_index: RwLock::new(None),
            global_index: RwLock::new(None),
            project_embeddings: RwLock::new(None),
            global_embeddings: RwLock::new(None),
            project_root: RwLock::new(Some(root)),
            project_mem_dir: RwLock::new(None),
        })
    }

    fn test_state_with_mem_dir(root: PathBuf, mem_dir: PathBuf) -> Arc<AppState> {
        Arc::new(AppState {
            project_index: RwLock::new(None),
            global_index: RwLock::new(None),
            project_embeddings: RwLock::new(None),
            global_embeddings: RwLock::new(None),
            project_root: RwLock::new(Some(root)),
            project_mem_dir: RwLock::new(Some(mem_dir)),
        })
    }

    async fn response_json(app: Router, uri: &str) -> serde_json::Value {
        let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), 200);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = build_router(test_state());
        let json = response_json(app, "/health").await;

        assert_eq!(json["status"], "ok");
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(json["project_memories"], 0);
        assert_eq!(json["global_memories"], 0);
        assert!(json["active_context"].is_null());
    }

    #[tokio::test]
    async fn health_shows_active_context() {
        let state = test_state_with_root(PathBuf::from("/tmp/test-project"));
        let app = build_router(state);
        let json = response_json(app, "/health").await;

        assert_eq!(json["active_context"], "/tmp/test-project");
    }

    #[tokio::test]
    async fn search_empty_returns_empty_array() {
        let app = build_router(test_state());
        let json = response_json(app, "/search?q=anything").await;

        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn search_finds_matching_memories() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path().join("myproject");
        let mem_dir = tmp.path().join("mem");
        std::fs::create_dir_all(&mem_dir).unwrap();

        std::fs::write(
            mem_dir.join("MEMORY.md"),
            "- [prefer rust](feedback_prefer-rust.md) — use rust for CLI tools\n\
             - [write tests](feedback_write-tests.md) — always write tests\n",
        )
        .unwrap();

        let state = test_state_with_mem_dir(project_root, mem_dir);
        let app = build_router(state);

        let json = response_json(app, "/search?q=rust").await;
        let results = json.as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["title"], "prefer rust");
        assert_eq!(results[0]["level"], "project");
        assert_eq!(results[0]["score"], 1.0);
    }

    #[tokio::test]
    async fn search_respects_top_k() {
        let tmp = tempfile::tempdir().unwrap();
        let project_root = tmp.path().join("proj");
        let mem_dir = tmp.path().join("mem");
        std::fs::create_dir_all(&mem_dir).unwrap();

        std::fs::write(
            mem_dir.join("MEMORY.md"),
            "- [test one](feedback_test-one.md) — first test memory\n\
             - [test two](feedback_test-two.md) — second test memory\n",
        )
        .unwrap();

        let state = test_state_with_mem_dir(project_root, mem_dir);
        let app = build_router(state);

        let json = response_json(app, "/search?q=test&top_k=1").await;
        assert_eq!(json.as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn reload_returns_reloaded() {
        let app = build_router(test_state());
        let json = response_json(app, "/reload").await;
        assert_eq!(json["status"], "reloaded");
    }

    #[tokio::test]
    async fn remember_returns_empty_for_no_match() {
        let app = build_router(test_state());
        let json = response_json(app, "/remember?q=anything").await;

        assert_eq!(json["ok"], true);
        assert!(json["data"]["memories"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn remember_finds_memorized_content() {
        let tmp = tempfile::tempdir().unwrap();
        let mem_dir = tmp.path().join("mem");
        std::fs::create_dir_all(&mem_dir).unwrap();

        // Create MEMORY.md index
        std::fs::write(
            mem_dir.join("MEMORY.md"),
            "- [prefer rust](feedback_prefer-rust.md) — use rust for CLI tools\n",
        )
        .unwrap();

        // Create actual memory file
        std::fs::write(
            mem_dir.join("feedback_prefer-rust.md"),
            "---\nname: prefer-rust\ndescription: use rust for CLI tools\ntype: feedback\ncreated_at: \"2026-01-01T00:00:00Z\"\naccess_count: 0\nstrength: 1.0\nsource: memorize\n---\nUse Rust for CLI tools.\n",
        )
        .unwrap();

        let state = test_state_with_mem_dir(tmp.path().to_path_buf(), mem_dir);
        let app = build_router(state);

        let json = response_json(app, "/remember?q=rust").await;
        assert_eq!(json["ok"], true);
        let memories = json["data"]["memories"].as_array().unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0]["name"], "prefer-rust");
    }

    #[tokio::test]
    async fn snapshot_health_no_context() {
        let app = build_router(test_state());
        let mut json = response_json(app, "/health").await;
        json["version"] = serde_json::Value::String("[version]".to_string());
        insta::assert_json_snapshot!(json);
    }

    #[tokio::test]
    async fn snapshot_health_with_context() {
        let state = test_state_with_root(PathBuf::from("/tmp/my-project"));
        let app = build_router(state);
        let mut json = response_json(app, "/health").await;
        json["version"] = serde_json::Value::String("[version]".to_string());
        insta::assert_json_snapshot!(json);
    }

    #[tokio::test]
    async fn snapshot_search_with_results() {
        let tmp = tempfile::tempdir().unwrap();
        let mem_dir = tmp.path().join("mem");
        std::fs::create_dir_all(&mem_dir).unwrap();
        std::fs::write(
            mem_dir.join("MEMORY.md"),
            "- [prefer rust](feedback_prefer-rust.md) — use rust for CLI tools\n\
             - [write tests](feedback_write-tests.md) — always write tests\n\
             - [api docs](reference_api-docs.md) — REST API documentation\n",
        )
        .unwrap();

        let state = test_state_with_mem_dir(tmp.path().to_path_buf(), mem_dir);
        let app = build_router(state);
        let json = response_json(app, "/search?q=test").await;
        insta::assert_json_snapshot!(json);
    }

    #[tokio::test]
    async fn snapshot_search_empty() {
        let app = build_router(test_state());
        let json = response_json(app, "/search?q=nothing").await;
        insta::assert_json_snapshot!(json);
    }

    #[tokio::test]
    async fn snapshot_reload() {
        let app = build_router(test_state());
        let json = response_json(app, "/reload").await;
        insta::assert_json_snapshot!(json);
    }
}
