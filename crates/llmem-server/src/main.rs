use anyhow::Result;
use axum::{
    Json, Router,
    extract::{Query, State},
    routing::get,
};
use llmem_core::{MemoryIndex, global_dir, project_dir};
use llmem_index::{AnnIndex, hnsw::HnswIndex};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

struct AppState {
    project_index: RwLock<Option<HnswIndex>>,
    global_index: RwLock<Option<HnswIndex>>,
    project_root: RwLock<Option<PathBuf>>,
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

    if include_project && let Some(root) = state.project_root.read().ok().and_then(|r| r.clone()) {
        let dir = project_dir(&root);
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
    if let Some(ctx_dir) = global_dir() {
        let ctx_file = ctx_dir.join(".active-ctx");
        if ctx_file.exists()
            && let Ok(ctx) = std::fs::read_to_string(&ctx_file)
        {
            let root = PathBuf::from(ctx.trim());
            if let Ok(mut pr) = state.project_root.write() {
                *pr = Some(root.clone());
            }

            let hnsw_path = project_dir(&root).join(".index.hnsw");
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

#[tokio::main]
async fn main() -> Result<()> {
    // Load initial context
    let mut project_root = None;
    let mut project_hnsw = None;
    let mut global_hnsw = None;

    // Check for active context
    if let Some(ctx_dir) = global_dir() {
        let ctx_file = ctx_dir.join(".active-ctx");
        if ctx_file.exists()
            && let Ok(ctx) = std::fs::read_to_string(&ctx_file)
        {
            let root = PathBuf::from(ctx.trim());
            let hnsw_path = project_dir(&root).join(".index.hnsw");
            if hnsw_path.exists() {
                project_hnsw = HnswIndex::load_from(&hnsw_path).ok();
            }
            project_root = Some(root);
        }

        // Load global HNSW
        let global_hnsw_path = ctx_dir.join(".index.hnsw");
        if global_hnsw_path.exists() {
            global_hnsw = HnswIndex::load_from(&global_hnsw_path).ok();
        }
    }

    let state = Arc::new(AppState {
        project_index: RwLock::new(project_hnsw),
        global_index: RwLock::new(global_hnsw),
        project_root: RwLock::new(project_root),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/search", get(search))
        .route("/reload", get(reload))
        .with_state(state);

    let addr = "127.0.0.1:3179";
    println!("llmem-server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
