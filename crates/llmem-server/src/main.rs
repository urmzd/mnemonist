use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Query, State},
    routing::get,
};
use llmem_core::{MemoryIndex, global_dir, llmem_root, project_dir};
use llmem_index::{AnnIndex, hnsw::HnswIndex};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

struct AppState {
    project_index: RwLock<Option<HnswIndex>>,
    global_index: RwLock<Option<HnswIndex>>,
    /// The original project root path (for display in /health).
    project_root: RwLock<Option<PathBuf>>,
    /// Resolved project memory directory (config-derived from project_root).
    project_mem_dir: RwLock<Option<PathBuf>>,
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_level")]
    level: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_level() -> String {
    "both".to_string()
}

fn default_top_k() -> usize {
    10
}

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

    // Text-based fallback search using the memory index
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
                        score: 1.0, // text match
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

async fn reload(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // Re-read .active-ctx and reload project context
    if let Some(root_dir) = llmem_root() {
        let ctx_file = root_dir.join(".active-ctx");
        if ctx_file.exists()
            && let Ok(ctx) = std::fs::read_to_string(&ctx_file)
        {
            let root = PathBuf::from(ctx.trim());
            let mem_dir = project_dir(&root);

            if let Ok(mut pr) = state.project_root.write() {
                *pr = Some(root);
            }
            if let Ok(mut pd) = state.project_mem_dir.write() {
                *pd = Some(mem_dir.clone());
            }

            let hnsw_path = mem_dir.join(".index.hnsw");
            if hnsw_path.exists()
                && let Ok(idx) = HnswIndex::load_from(&hnsw_path)
                && let Ok(mut pi) = state.project_index.write()
            {
                *pi = Some(idx);
            }
        }
    }

    Json(serde_json::json!({"status": "reloaded"}))
}

/// Build the router with the given state. Extracted for testability.
fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/search", get(search))
        .route("/reload", get(reload))
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load initial context
    let mut project_root = None;
    let mut project_mem_dir = None;
    let mut project_hnsw = None;
    let mut global_hnsw = None;

    // Check for active context
    if let Some(root_dir) = llmem_root() {
        let ctx_file = root_dir.join(".active-ctx");
        if ctx_file.exists()
            && let Ok(ctx) = std::fs::read_to_string(&ctx_file)
        {
            let root = PathBuf::from(ctx.trim());
            let mem_dir = project_dir(&root);
            let hnsw_path = mem_dir.join(".index.hnsw");
            if hnsw_path.exists() {
                project_hnsw = HnswIndex::load_from(&hnsw_path).ok();
            }
            project_root = Some(root);
            project_mem_dir = Some(mem_dir);
        }
    }

    // Load global HNSW
    if let Some(dir) = global_dir() {
        let global_hnsw_path = dir.join(".index.hnsw");
        if global_hnsw_path.exists() {
            global_hnsw = HnswIndex::load_from(&global_hnsw_path).ok();
        }
    }

    let state = Arc::new(AppState {
        project_index: RwLock::new(project_hnsw),
        global_index: RwLock::new(global_hnsw),
        project_root: RwLock::new(project_root),
        project_mem_dir: RwLock::new(project_mem_dir),
    });

    let app = build_router(state);

    let addr = "127.0.0.1:3179";
    println!("llmem-server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

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
            project_root: RwLock::new(None),
            project_mem_dir: RwLock::new(None),
        })
    }

    fn test_state_with_root(root: PathBuf) -> Arc<AppState> {
        Arc::new(AppState {
            project_index: RwLock::new(None),
            global_index: RwLock::new(None),
            project_root: RwLock::new(Some(root)),
            project_mem_dir: RwLock::new(None),
        })
    }

    fn test_state_with_mem_dir(root: PathBuf, mem_dir: PathBuf) -> Arc<AppState> {
        Arc::new(AppState {
            project_index: RwLock::new(None),
            global_index: RwLock::new(None),
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
}
