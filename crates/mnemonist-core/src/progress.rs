//! Progress reporting for long-running operations.
//!
//! Core emits events; the CLI decides how to render them. `Option<&dyn Progress>`
//! is accepted by expensive loops (embedding, HNSW insert, chunk walking) so
//! callers can pass `None` when they don't care.

/// A handle to report progress on a phase of work.
pub trait Progress: Send + Sync {
    /// Begin a named phase. `total` is the expected unit count if known.
    fn start(&self, phase: &str, total: Option<usize>);

    /// Advance the current phase to `current` units.
    fn tick(&self, current: usize);

    /// Conclude the current phase. `detail` is an optional trailing note.
    fn finish(&self, detail: Option<&str>);
}

/// A no-op reporter. Use when you need to satisfy a `&dyn Progress` bound
/// but don't want any output.
pub struct NoopProgress;

impl Progress for NoopProgress {
    fn start(&self, _phase: &str, _total: Option<usize>) {}
    fn tick(&self, _current: usize) {}
    fn finish(&self, _detail: Option<&str>) {}
}
