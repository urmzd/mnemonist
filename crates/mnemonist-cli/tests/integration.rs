use insta::assert_json_snapshot;
use serde_json::Value;
use std::path::PathBuf;
use std::process::Command;

/// Get the path to the mnemonist binary built by cargo.
fn mnemonist_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_BIN_EXE_mnemonist"));
    // Fallback: if the macro doesn't resolve, try target/debug
    if !path.exists() {
        path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/mnemonist");
    }
    path
}

/// Run mnemonist with the given args in an isolated HOME directory.
/// Returns (stdout as JSON, exit code).
fn run(home: &std::path::Path, args: &[&str]) -> (Value, i32) {
    let output = Command::new(mnemonist_bin())
        .args(args)
        .env("HOME", home)
        .output()
        .expect("failed to execute mnemonist");

    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: Value =
        serde_json::from_str(&stdout).unwrap_or_else(|_| panic!("invalid JSON output: {stdout}"));
    (json, code)
}

#[test]
fn init_creates_memory_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("myproject");
    std::fs::create_dir_all(&project).unwrap();

    let (json, code) = run(&home, &["init", "--root", project.to_str().unwrap()]);
    assert_eq!(code, 0);
    assert_eq!(json["ok"], true);
    assert_eq!(json["data"]["level"], "project");

    // Verify MEMORY.md was created
    let mem_dir = PathBuf::from(json["data"]["path"].as_str().unwrap());
    assert!(mem_dir.join("MEMORY.md").exists());
}

#[test]
fn memorize_creates_memory_file() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    // Init first
    run(&home, &["init", "--root", project.to_str().unwrap()]);

    // Memorize
    let (json, code) = run(
        &home,
        &[
            "memorize",
            "always use tests",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0);
    assert_eq!(json["ok"], true);
    assert_eq!(json["data"]["action"], "created");
    assert!(
        json["data"]["file"]
            .as_str()
            .unwrap()
            .starts_with("feedback_")
    );
}

#[test]
fn memorize_with_type_and_name() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);

    let (json, code) = run(
        &home,
        &[
            "memorize",
            "user prefers vim",
            "-t",
            "user",
            "--name",
            "vim-preference",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0);
    assert_eq!(json["data"]["file"], "user_vim-preference.md");
}

#[test]
fn note_adds_to_inbox() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);

    let (json, code) = run(
        &home,
        &["note", "check logging", "--root", project.to_str().unwrap()],
    );
    assert_eq!(code, 0);
    assert_eq!(json["data"]["inbox_size"], 1);
    assert_eq!(json["data"]["capacity"], 7);
}

#[test]
fn remember_finds_memorized_content() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    run(
        &home,
        &[
            "memorize",
            "prefer rust for cli tools",
            "--root",
            project.to_str().unwrap(),
        ],
    );

    let (json, code) = run(
        &home,
        &[
            "remember",
            "rust",
            "--level",
            "project",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0);
    let memories = json["data"]["memories"].as_array().unwrap();
    assert_eq!(memories.len(), 1);
    assert!(memories[0]["body"].as_str().unwrap().contains("rust"));
}

#[test]
fn remember_returns_empty_for_no_match() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);

    let (json, code) = run(
        &home,
        &[
            "remember",
            "nonexistent",
            "--level",
            "project",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0);
    assert_eq!(json["data"]["memories"].as_array().unwrap().len(), 0);
}

#[test]
fn reflect_shows_memories_and_inbox() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    run(
        &home,
        &[
            "memorize",
            "prefer rust",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    run(
        &home,
        &["note", "todo item", "--root", project.to_str().unwrap()],
    );

    let (json, code) = run(&home, &["reflect", "--root", project.to_str().unwrap()]);
    assert_eq!(code, 0);
    assert_eq!(json["data"]["memories"].as_array().unwrap().len(), 1);
    assert_eq!(json["data"]["inbox"]["size"], 1);
}

#[test]
fn consolidate_promotes_inbox_items() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    run(
        &home,
        &[
            "note",
            "important finding",
            "--root",
            project.to_str().unwrap(),
        ],
    );

    // Dry run first
    let (json, code) = run(
        &home,
        &[
            "consolidate",
            "--dry-run",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0);
    assert_eq!(json["data"]["promoted"], 1);
    assert_eq!(json["data"]["dry_run"], true);

    // Real consolidation
    let (json, code) = run(&home, &["consolidate", "--root", project.to_str().unwrap()]);
    assert_eq!(code, 0);
    assert_eq!(json["data"]["promoted"], 1);
    assert_eq!(json["data"]["dry_run"], false);

    // Verify inbox is now empty
    let (json, _) = run(&home, &["reflect", "--root", project.to_str().unwrap()]);
    assert_eq!(json["data"]["inbox"]["size"], 0);
}

#[test]
fn forget_removes_memory() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    let (memorize_json, _) = run(
        &home,
        &[
            "memorize",
            "temp memory",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    let filename = memorize_json["data"]["file"].as_str().unwrap();

    // Forget it
    let (json, code) = run(
        &home,
        &["forget", filename, "--root", project.to_str().unwrap()],
    );
    assert_eq!(code, 0);
    assert_eq!(json["data"]["action"], "forgotten");

    // Verify it's gone
    let (json, _) = run(&home, &["reflect", "--root", project.to_str().unwrap()]);
    assert_eq!(json["data"]["memories"].as_array().unwrap().len(), 0);
}

#[test]
fn forget_nonexistent_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);

    let (json, code) = run(
        &home,
        &[
            "forget",
            "nonexistent.md",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1);
    assert_eq!(json["ok"], false);
}

#[test]
fn config_init_and_get() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();

    // Init config
    let (json, code) = run(&home, &["config", "init"]);
    assert_eq!(code, 0);
    assert_eq!(json["data"]["action"], "created");

    // Get a known key
    let (json, code) = run(&home, &["config", "get", "embedding.model"]);
    assert_eq!(code, 0);
    assert_eq!(json["data"]["value"], "all-MiniLM-L6-v2");
}

#[test]
fn config_set_updates_value() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();

    run(&home, &["config", "init"]);

    // Set
    let (json, code) = run(&home, &["config", "set", "embedding.model", "test-model"]);
    assert_eq!(code, 0);
    assert_eq!(json["data"]["value"], "test-model");

    // Verify
    let (json, _) = run(&home, &["config", "get", "embedding.model"]);
    assert_eq!(json["data"]["value"], "test-model");
}

#[test]
fn memorize_stdin_json() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);

    let input = serde_json::json!({
        "type": "user",
        "name": "stdin-test",
        "description": "test from stdin",
        "body": "detailed body content",
        "level": "project"
    });

    let output = Command::new(mnemonist_bin())
        .args([
            "memorize",
            "ignored",
            "--stdin",
            "--root",
            project.to_str().unwrap(),
        ])
        .env("HOME", &home)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(input.to_string().as_bytes())
                .unwrap();
            child.wait_with_output()
        })
        .expect("failed to run with stdin");

    assert!(output.status.success());
    let json: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["data"]["file"], "user_stdin-test.md");
}

#[test]
fn multiple_notes_respect_capacity() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);

    // Add more notes than the default capacity (7)
    for i in 0..10 {
        run(
            &home,
            &[
                "note",
                &format!("note number {i}"),
                "--root",
                project.to_str().unwrap(),
            ],
        );
    }

    let (json, _) = run(&home, &["reflect", "--root", project.to_str().unwrap()]);
    // Inbox should be capped at capacity (7)
    assert!(json["data"]["inbox"]["size"].as_u64().unwrap() <= 7);
}

/// Redact dynamic fields from CLI JSON output for deterministic snapshots.
fn redact_cli_json(mut json: Value) -> Value {
    // Redact timestamps and paths that vary between runs
    if let Some(data) = json.get_mut("data") {
        // Redact path fields
        for key in ["path", "context"] {
            if data.get(key).is_some_and(|v| v.is_string()) {
                data[key] = Value::String("[path]".to_string());
            }
        }
        // Redact embedded field (depends on Ollama availability)
        if data.get("embedded").is_some() {
            data["embedded"] = Value::String("[env-dependent]".to_string());
        }
        // Redact config show output (contains paths)
        if data.get("config").is_some_and(|v| v.is_string()) {
            data["config"] = Value::String("[toml]".to_string());
        }
        // Redact memories array timestamps
        if let Some(memories) = data.get_mut("memories") {
            if let Some(arr) = memories.as_array_mut() {
                for mem in arr {
                    for ts_key in ["last_accessed", "created_at", "indexed_at"] {
                        if mem.get(ts_key).is_some_and(|v| v.is_string()) {
                            mem[ts_key] = Value::String("[timestamp]".to_string());
                        }
                    }
                }
            }
        }
        // Redact inbox item timestamps
        if let Some(inbox) = data.get_mut("inbox") {
            if let Some(items) = inbox.get_mut("items") {
                if let Some(arr) = items.as_array_mut() {
                    for item in arr {
                        if item.get("created_at").is_some_and(|v| v.is_string()) {
                            item["created_at"] = Value::String("[timestamp]".to_string());
                        }
                    }
                }
            }
        }
        // Redact consolidation timestamps
        if let Some(actions) = data.get_mut("actions") {
            if let Some(arr) = actions.as_array_mut() {
                for action in arr {
                    for ts_key in ["created_at", "last_accessed"] {
                        if action.get(ts_key).is_some_and(|v| v.is_string()) {
                            action[ts_key] = Value::String("[timestamp]".to_string());
                        }
                    }
                }
            }
        }
    }
    json
}

#[test]
fn snapshot_init_output() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    let (json, _) = run(&home, &["init", "--root", project.to_str().unwrap()]);
    assert_json_snapshot!(redact_cli_json(json));
}

#[test]
fn snapshot_memorize_output() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    let (json, _) = run(
        &home,
        &[
            "memorize",
            "always write tests",
            "-t",
            "feedback",
            "--name",
            "write-tests",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    let json = redact_cli_json(json);
    assert_json_snapshot!(json);
}

#[test]
fn snapshot_note_output() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    let (json, _) = run(
        &home,
        &[
            "note",
            "investigate logging",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_json_snapshot!(json);
}

#[test]
fn snapshot_reflect_output() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    run(
        &home,
        &[
            "memorize",
            "prefer rust",
            "-t",
            "feedback",
            "--name",
            "prefer-rust",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    run(
        &home,
        &["note", "check logging", "--root", project.to_str().unwrap()],
    );

    let (json, _) = run(&home, &["reflect", "--root", project.to_str().unwrap()]);
    assert_json_snapshot!(redact_cli_json(json));
}

#[test]
fn snapshot_consolidate_dry_run_output() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    run(
        &home,
        &[
            "note",
            "important observation",
            "--root",
            project.to_str().unwrap(),
        ],
    );

    let (json, _) = run(
        &home,
        &[
            "consolidate",
            "--dry-run",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_json_snapshot!(redact_cli_json(json));
}

#[test]
fn snapshot_config_get_output() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();

    run(&home, &["config", "init"]);
    let (json, _) = run(&home, &["config", "get", "embedding.model"]);
    assert_json_snapshot!(json);
}

#[test]
fn snapshot_forget_error_output() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let project = tmp.path().join("proj");
    std::fs::create_dir_all(&project).unwrap();

    run(&home, &["init", "--root", project.to_str().unwrap()]);
    let (json, _) = run(
        &home,
        &[
            "forget",
            "nonexistent.md",
            "--root",
            project.to_str().unwrap(),
        ],
    );
    assert_json_snapshot!(json);
}
