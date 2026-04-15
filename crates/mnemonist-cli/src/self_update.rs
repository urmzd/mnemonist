use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct ReleaseAsset {
    name: String,
    browser_download_url: String,
}

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<ReleaseAsset>,
}

#[derive(Debug)]
pub enum UpdateResult {
    AlreadyUpToDate,
    Updated { from: String, to: String },
}

fn detect_target() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Ok("x86_64-unknown-linux-musl"),
        ("linux", "aarch64") => Ok("aarch64-unknown-linux-musl"),
        ("macos", "x86_64") => Ok("x86_64-apple-darwin"),
        ("macos", "aarch64") => Ok("aarch64-apple-darwin"),
        ("windows", "x86_64") => Ok("x86_64-pc-windows-msvc"),
        (os, arch) => bail!("unsupported platform: {os}/{arch}"),
    }
}

fn gh_get(url: &str) -> Result<ureq::Body> {
    let mut req = ureq::get(url).header("Accept", "application/vnd.github+json");

    if let Ok(token) = std::env::var("GH_TOKEN").or_else(|_| std::env::var("GITHUB_TOKEN")) {
        req = req.header("Authorization", format!("token {token}"));
    }

    let resp = req.call().context("HTTP request failed")?;
    Ok(resp.into_body())
}

fn fetch_latest_release(repo: &str) -> Result<Release> {
    let url = format!("https://api.github.com/repos/{repo}/releases/latest");
    let mut body = gh_get(&url)?;
    let release: Release = body.read_json().context("failed to parse release JSON")?;
    Ok(release)
}

fn download_bytes(url: &str) -> Result<Vec<u8>> {
    let body = gh_get(url)?;
    let mut bytes = Vec::new();
    body.into_reader()
        .read_to_end(&mut bytes)
        .context("failed to read response body")?;
    Ok(bytes)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn atomic_replace(target: &PathBuf, new_bytes: &[u8]) -> Result<()> {
    let backup = target.with_extension("old");
    let tmp = target.with_extension("new");

    fs::write(&tmp, new_bytes).context("failed to write new binary")?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp, fs::Permissions::from_mode(0o755))
            .context("failed to set executable permissions")?;
    }

    if target.exists() {
        fs::rename(target, &backup).context("failed to backup current binary")?;
    }
    if let Err(e) = fs::rename(&tmp, target) {
        if backup.exists() {
            let _ = fs::rename(&backup, target);
        }
        return Err(e).context("failed to replace binary");
    }

    let _ = fs::remove_file(&backup);
    Ok(())
}

pub fn self_update(
    repo: &str,
    current_version: &str,
    binary_name: &str,
) -> Result<UpdateResult> {
    let release = fetch_latest_release(repo)?;
    let latest_version = release.tag_name.trim_start_matches('v');

    if latest_version == current_version {
        return Ok(UpdateResult::AlreadyUpToDate);
    }

    let target = detect_target()?;
    let asset_name = format!("{binary_name}-{target}");

    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| anyhow::anyhow!("no asset found for {asset_name}"))?;

    let sha256_name = format!("{asset_name}.sha256");
    let expected_sha256 = release
        .assets
        .iter()
        .find(|a| a.name == sha256_name)
        .and_then(|a| {
            download_bytes(&a.browser_download_url)
                .ok()
                .and_then(|b| String::from_utf8(b).ok())
                .map(|s| s.trim().split_whitespace().next().unwrap_or("").to_string())
        });

    eprintln!("downloading {binary_name} {latest_version} for {target}...");

    let bytes = download_bytes(&asset.browser_download_url)?;

    if let Some(expected) = &expected_sha256 {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let actual = hex_encode(&hasher.finalize());
        if actual != *expected {
            bail!("SHA256 mismatch: expected {expected}, got {actual}");
        }
    }

    let current_exe = std::env::current_exe().context("cannot determine current executable")?;
    atomic_replace(&current_exe, &bytes)?;

    Ok(UpdateResult::Updated {
        from: current_version.to_string(),
        to: latest_version.to_string(),
    })
}
