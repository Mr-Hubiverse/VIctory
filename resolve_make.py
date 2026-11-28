import resolve1

P = 'crates/victory-gpu/src/pipeline/make.rs'
src, hs = resolve1.hunks(P)

h1 = """    /// Why it is not made, if it is not ([`left_out`]).
    left_out: Option<&'static str>,
    /// A pass over the screen ([`GraphicsState::screen_pass`]) or a confirmed
    /// opening-scene / bake producer that must never sit behind cold RT shaders.
    screen: bool,
    /// Confirmed first-scene / bake producer prepared early even on a cold cache.
    scene_input: bool,
    /// When a worker began it ([`slow`], [`stuck`]).
    begun: OnceLock<Instant>,
    /// Whether a worker was started in place of the one making it ([`stuck`]).
    replaced: AtomicBool,
    /// Whether the made pipeline has gone to a draw or dispatch yet
    /// (`crate::host::first_use`: for a device removal's report).
    handed: AtomicBool,
    /// The formats its state declares for what a draw draws into (zeros
    /// without a graphics state).
    pub(crate) formats: TargetFormats,
    /// The same shaders and state for other formats, bound at a draw
    /// ([`for_formats`]): a chain from the title's pipeline, one link per
    /// set of formats met, never shortened.
    variant: OnceLock<Arc<Pending>>,
}

static SHARED: OnceLock<Mutex<std::collections::HashMap<[u8; 32], Weak<Pending>>>> =
    OnceLock::new();

/// Whether some compute shader of `shaders` holds `name`.
"""

h3 = """        key.update(graphics_description.as_deref().unwrap_or_default().as_bytes());
        let scene_input = scene_input(&shaders);
        let screen = (graphics.as_ref().is_some_and(|g| g.screen_pass())
            && shaders.iter().all(|s| s.stage != "mesh"))
            || scene_input;
        let left_out = left_out(&shaders);
        Pending {
            screen,
            scene_input,
            formats: graphics.as_ref().map_or([0; 9], |g| g.formats()),
            variant: OnceLock::new(),
"""

h4 = """    pub(super) fn key(&self) -> &str {
        &self.key
    }
""" + hs[4][1].replace(
    "        let _ = self.begun.set(Instant::now());",
    "        let began = Instant::now();\n        let _ = self.begun.set(began);").replace(
    """            ctx.out
""",
    """            super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
            ctx.out
""")

h11 = """    if p.left_out.is_none() && !p.shaders.is_empty() && (p.scene_input || super::library::wanted_early(&p.key)) {
"""

h16 = """        } else if p.graphics.as_ref().is_some_and(|state| state.writes_depth_only()) {
            // Shadow maps and depth prepasses supply several later draws. Rebuild
            // their previously used PSOs first while no draw is requesting one.
            self.early_depth.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
"""

h17 = """    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        let mut p = self.screen.pop_front();
        if p.is_none() && !screens_only {
            p = self.requested.pop_front().or_else(|| {
                // An early entry can have been promoted and claimed already.
                std::iter::from_fn(|| self.early_depth.pop_front().or_else(|| self.early.pop_front()))
                    .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
            });
            if p.is_none() && self.slow_now < slow_limit {
                p = std::iter::from_fn(|| self.slow.pop_front())
                    .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY);
                if p.is_some() {
                    p.as_ref().unwrap().was_slow.store(true, Ordering::Relaxed);
                    self.slow_now += 1;
                }
            }
"""

h21 = """        let workers = std::env::var("VICTORY_PIPELINE_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .map_or_else(
                || {
                    worker_count_for(
                        std::env::var("VICTORY_PIPELINE_WORKERS")
                            .ok()
                            .and_then(|v| v.trim().parse().ok()),
                        std::thread::available_parallelism().map_or(2, |n| n.get()),
                        crate::residency::limited_host_ram(),
                    )
                },
                |n| n,
            );
        tracing::info!(workers, "pipeline workers");
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            running |= spawn_worker(i >= workers);
"""

h24 = hs[24][1].replace("queue.pop(false)", "queue.pop(false, 1)") + hs[24][2]

choices = ['ours', h1, 'theirs', h3, h4, 'both', 'both', 'ours', 'ours', 'theirs',
           'ours', h11, 'ours', 'ours', 'both', 'both', h16, h17, 'theirs', 'theirs',
           'both', h21, 'both', 'both', h24, 'theirs']
resolve1.resolve(P, choices)
