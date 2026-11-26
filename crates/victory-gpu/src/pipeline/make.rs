//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is 12 s per title frame and kind, a window the
//! recording threads wait in together ([`DEFAULT_WAIT_MS`]); it was half
//! a second. A pass the title runs only once and whose pipeline takes
//! longer is lost: while a world loads the title dispatches the passes
//! that unpack its models once. With a budget set below the defaults, a
//! frame may wait longer for a pipeline while a run's loading allowance
//! lasts ([`frame_budget`]), and never for one the driver has held for
//! long ([`held_too_long`]).
//! Ported from the sibling fork `joshuu/victory-amd`'s 4436f8f1 (its
//! loading waits only).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline, TargetFormats};

/// `VICTORY_DENY_PIPELINES=<16-hex shader hash>,...` (off unless set): the
/// hash of a shader of `shaders` that is listed there. Such a pipeline is
/// never built (no driver compile) and counts as refused, so its draws and
/// dispatches are dropped: a test of whether building a pipeline, rather
/// than running it, is what loses the card (runs 281 and 318: the removal
/// 32 and 81 ms after the 3-second build of `CS_RaytracingResolve` of the
/// two-layer vehicle wheel material, 57a3fad03cac36af and d5b9c94a30d7cf28;
/// gfx41). Unlike `VICTORY_SKIP_PIPELINES`, in release builds too.
fn denied(shaders: &[Arc<Shader>]) -> Option<&str> {
    static LIST: OnceLock<Vec<String>> = OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("VICTORY_DENY_PIPELINES")
            .unwrap_or_default()
            .split(',')
            .map(|h| h.trim().to_ascii_lowercase())
            .filter(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
            .collect()
    });
    if list.is_empty() {
        return None;
    }
    shaders.iter().find(|s| list.contains(&s.hash)).map(|s| s.hash.as_str())
}

/// The shaders of `p` (for logs).
pub(crate) fn shaders(p: &Pending) -> &[Arc<Shader>] {
    &p.shaders
}

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    /// Exact identity for sharing one in-flight host compilation among
    /// title pipeline objects that describe the same shaders and state.
    identity: [u8; 32],
    used: AtomicBool,
    /// Why it is not made, if it is not ([`left_out`]).
    left_out: Option<&'static str>,
    /// When a worker began it ([`slow`], [`stuck`]).
    begun: OnceLock<Instant>,
    /// A pass over the screen ([`GraphicsState::screen_pass`]).
    screen: bool,
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

/// Why a pipeline with these shaders is not made, if it is not.
fn left_out(shaders: &[Arc<Shader>]) -> Option<&'static str> {
    if without_structures(shaders) {
        Some("shades ray-tracing hits, left out")
    } else if too_dear(shaders) {
        Some("too dear to compile, left out")
    } else {
        None
    }
}

/// Whether some compute shader of `shaders` holds `name`.
fn named(shaders: &[Arc<Shader>], name: &[u8]) -> bool {
    shaders
        .iter()
        .any(|s| s.stage == "compute" && s.container.windows(name.len()).any(|w| w == name))
}

/// Whether a shader is left out for what it costs the host's driver:
/// `CS_ScreenProbes_ActivateRays` of the title
/// (39f0e9aeb156a61c, 66,501 instructions, 9,948 branches). As translated
/// before the clean-up passes (`emit::shader_opt`), AMD's took a core and
/// 0.8 GB more every 10 s for it and did not finish (RX 7800 XT: 47 GB after
/// 8 minutes); cleaned up, it takes 85 s and 1.3 GB, once.
/// `VICTORY_ACTIVATE_RAYS=0` leaves it out.
fn too_dear(shaders: &[Arc<Shader>]) -> bool {
    static OUT: OnceLock<bool> = OnceLock::new();
    let out = *OUT.get_or_init(|| std::env::var_os("VICTORY_ACTIVATE_RAYS").is_some_and(|v| v == "0"));
    out && named(shaders, b"CS_ScreenProbes_ActivateRays")
}

/// Whether a shader only shades what the console's ray-tracing structures
/// were hit by: about 1,000 compute programs of the title
/// (`CS_RaytracingResolve`, the `raytracingresolve_dispatch` pass of each
/// material), all bound at the world's first frame, each 5 to 9 s of AMD's
/// compiler. They stood in the queue before the passes that make the
/// picture, and with the structure builder off there is no hit to shade.
/// Made by default, the picture before loading time (with the builder on
/// they shade what reflections and indirect light hit);
/// `VICTORY_RT_RESOLVE=0` leaves them out while the builder is off.
fn without_structures(shaders: &[Arc<Shader>]) -> bool {
    static OUT: OnceLock<bool> = OnceLock::new();
    let out = *OUT.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        std::env::var_os("VICTORY_RT_RESOLVE").is_some_and(|v| v == "0") && !builder
    });
    out && named(shaders, b"CS_RaytracingResolve")
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let graphics_description = graphics.as_ref().map(|g| g.describe());
        let identity = Self::identity(&shaders, graphics_description.as_deref());
        Self::with_identity(shaders, graphics, identity, graphics_description)
    }

    fn with_identity(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Box<GraphicsState>>,
        identity: [u8; 32],
        graphics_description: Option<String>,
    ) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics_description.as_deref().unwrap_or_default().as_bytes());
        let left_out = left_out(&shaders);
        let screen = graphics.as_ref().is_some_and(|g| g.screen_pass());
        Pending {
            formats: graphics.as_ref().map_or([0; 9], |g| g.formats()),
            variant: OnceLock::new(),
            shaders,
            graphics,
            identity,
            left_out,
            begun: OnceLock::new(),
            replaced: AtomicBool::new(false),
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            screen,
            handed: AtomicBool::new(false),
        }
    }


    /// The full content and translated host state, without the short shader
    /// hashes used in diagnostics and the on-disk pipeline-library names.
    fn identity(shaders: &[Arc<Shader>], graphics: Option<&str>) -> [u8; 32] {
        let mut hash = blake3::Hasher::new();
        hash.update(&(shaders.len() as u64).to_le_bytes());
        for shader in shaders {
            hash.update(&(shader.stage.len() as u64).to_le_bytes());
            hash.update(shader.stage.as_bytes());
            hash.update(&(shader.container.len() as u64).to_le_bytes());
            hash.update(&shader.container);
        }
        match graphics {
            Some(description) => {
                hash.update(&[1]);
                hash.update(&(description.len() as u64).to_le_bytes());
                hash.update(description.as_bytes());
            }
            None => {
                hash.update(&[0]);
            }
        }
        *hash.finalize().as_bytes()
    }

    /// Returns the live pending pipeline for this exact title descriptor, or
    /// installs one. Weak entries disappear when no title pipeline or worker
    /// still needs the host pipeline, so the intern table does not retain the
    /// title's shaders for the whole run.
    pub(crate) fn shared(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Box<GraphicsState>>,
    ) -> Arc<Pending> {
        let graphics_description = graphics.as_ref().map(|g| g.describe());
        let identity = Self::identity(&shaders, graphics_description.as_deref());
        let mut shared = SHARED
            .get_or_init(|| Mutex::new(std::collections::HashMap::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = shared.get(&identity).and_then(Weak::upgrade) {
            return existing;
        }
        let pending = Arc::new(Self::with_identity(
            shaders,
            graphics,
            identity,
            graphics_description,
        ));
        shared.insert(identity, Arc::downgrade(&pending));
        pending
    }
    /// The shaders' hashes, for logging.
    fn shader_hashes(&self) -> String {
        self.shaders.iter().map(|s| s.hash.as_str()).collect::<Vec<_>>().join(",")
    }

    /// Whether it is a compute pipeline.
    fn computes(&self) -> bool {
        matches!(self.shaders.as_slice(), [only] if only.stage == "compute")
    }

    /// Whether the host's driver has been at it for longer than a frame
    /// waits for ([`held_too_long`]).
    fn slow(&self) -> bool {
        self.begun.get().is_some_and(|at| held_too_long(at.elapsed(), self.computes(), wait_budget()))
    }

    fn make(&self) {
        let _ = self.begun.set(Instant::now());
        let _making = Making::begin(self);
        self.made.get_or_init(|| {
            if let Some(hash) = denied(&self.shaders) {
                tracing::warn!(hash, shaders = ?self.shaders, "pipeline not built: its shader is in VICTORY_DENY_PIPELINES");
                return None;
            }
            static BAD_SHADERS: OnceLock<std::collections::HashSet<String>> = OnceLock::new();
            let bad = BAD_SHADERS.get_or_init(|| {
                let mut set = std::collections::HashSet::new();
                if let Ok(env) = std::env::var("VICTORY_SKIP_PIPELINES") {
                    for h in env.split(',') {
                        let h = h.trim().to_ascii_lowercase();
                        if !h.is_empty() {
                            set.insert(h);
                        }
                    }
                }
                if let Some(path) = super::library::bad_shaders_path()
                    && let Ok(contents) = std::fs::read_to_string(path)
                {
                    for line in contents.lines() {
                        let h = line.trim().to_ascii_lowercase();
                        if !h.is_empty() {
                            set.insert(h);
                        }
                    }
                }
                set
            });
            if self
                .shaders
                .iter()
                .any(|s| bad.contains(&s.hash.to_ascii_lowercase()))
            {
                tracing::warn!(shaders = self.shader_hashes(), "skipping host pipeline build (bad_shaders.txt / VICTORY_SKIP_PIPELINES)");
                return None;
            }
            unsafe extern "C" {
                fn victory_seh_call(
                    func: unsafe extern "C" fn(*mut std::ffi::c_void),
                    ctx: *mut std::ffi::c_void,
                ) -> i32;
            }
            struct CallCtx<'a> {
                shaders: &'a [Arc<Shader>],
                graphics: Option<&'a GraphicsState>,
                out: Option<HostPipeline>,
            }
            unsafe extern "C" fn run_make(raw: *mut std::ffi::c_void) {
                let ctx = unsafe { &mut *raw.cast::<CallCtx<'_>>() };
                ctx.out = HostPipeline::new(ctx.shaders, ctx.graphics);
            }
            let mut ctx = CallCtx {
                shaders: &self.shaders,
                graphics: self.graphics.as_deref(),
                out: None,
            };
            let ok = unsafe { victory_seh_call(run_make, (&raw mut ctx).cast()) };
            if ok == 0 {
                tracing::warn!(
                    shaders = self.shader_hashes(),
                    "host driver raised 0xc0000005 compiling pipeline; skipping and recording in bad_shaders.txt"
                );
                if let Some(path) = super::library::bad_shaders_path()
                    && let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                {
                    use std::io::Write;
                    for s in &self.shaders {
                        let _ = writeln!(f, "{}", s.hash);
                    }
                }
                return None;
            }
            ctx.out
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}


impl Drop for Pending {
    fn drop(&mut self) {
        // The registry itself is initialized by `shared`; avoid creating it
        // from a pipeline that was constructed directly (for example warmup).
        let Some(shared) = SHARED.get() else {
            return;
        };
        let mut shared = shared.lock().unwrap_or_else(|e| e.into_inner());
        if shared.get(&self.identity).is_some_and(|pending| pending.upgrade().is_none()) {
            shared.remove(&self.identity);
        }
    }
}
/// The pipelines being made now, by thread, for crash reports: a host
/// driver that faults while compiling one leaves no other trace of which
/// it was (its pipeline is logged only once made).
static IN_FLIGHT: Mutex<Vec<(std::thread::ThreadId, String)>> = Mutex::new(Vec::new());

/// A pipeline in [`IN_FLIGHT`] while it is being made.
struct Making(std::thread::ThreadId);

impl Making {
    fn begin(p: &Pending) -> Making {
        let thread = std::thread::current();
        let state = p
            .graphics
            .as_ref()
            .map(|g| format!(" state {}", g.describe()))
            .unwrap_or_default();
        let what = format!(
            "{:?} on thread {:?}, key {}{state}",
            p.shaders,
            thread.name().unwrap_or("?"),
            p.key
        );
        IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner()).push((thread.id(), what));
        Making(thread.id())
    }
}

impl Drop for Making {
    fn drop(&mut self) {
        let mut making = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = making.iter().position(|(thread, _)| *thread == self.0) {
            making.swap_remove(i);
        }
    }
}

/// For crash reports: one line per pipeline being made, the faulting
/// (calling) thread's marked. Never waits: nothing if another thread holds
/// the list at that moment.
pub(crate) fn being_made() -> String {
    let Ok(making) = IN_FLIGHT.try_lock() else {
        return "  pipelines being made: not known (the list was in use)\n".to_owned();
    };
    let here = std::thread::current().id();
    making
        .iter()
        .map(|(thread, what)| {
            let mark = if *thread == here { " (this thread)" } else { "" };
            format!("  pipeline being made{mark}: {what}\n")
        })
        .collect()
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        // Its worker may have died making it (a stack overflow).
        None if stuck(p) => Some(format!("stuck, its compile never came back {:?}", p.shaders)),
        None => Some(p.left_out.map_or_else(
            || format!("being made {:?}", p.shaders),
            str::to_owned,
        )),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// How long a pipeline may be in the making before it counts as stuck
/// (`VICTORY_PIPELINE_STUCK_S`, 15 s by default).
fn stuck_after() -> Duration {
    #[cfg(test)]
    return Duration::from_millis(50);
    #[cfg(not(test))]
    {
        static AFTER: OnceLock<Duration> = OnceLock::new();
        *AFTER.get_or_init(|| {
            Duration::from_secs(
                std::env::var("VICTORY_PIPELINE_STUCK_S")
                    .ok()
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .unwrap_or(15),
            )
        })
    }
}

/// Whether `p`'s worker began making it longer ago than [`stuck_after`]
/// and has not come back, whatever the frame budget ([`held_too_long`]
/// judges only budgets shorter than it). Under Proton, vkd3d-proton's DXIL
/// to SPIR-V conversion (dxil-spirv) recursed without end on three of the
/// title's ray-traced shadow programs (dxil-spirv#376): the worker's stack
/// overflowed whatever its size, the thread was gone, and the pipeline
/// stayed "being made": every frame then spent its whole wait budget on it
/// (12 s), and each such pipeline took a worker with it.
/// The first time one is found stuck, it is said in the log and a worker
/// is started in place of its own. A compile that is only slow (one of the
/// title's takes 85 s) is still used once made.
fn stuck(p: &Pending) -> bool {
    let Some(begun) = p.begun.get() else { return false };
    if p.made.get().is_some() || begun.elapsed() < stuck_after() {
        return false;
    }
    if !p.replaced.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            shaders = p.shader_hashes(),
            seconds = stuck_after().as_secs(),
            "a host pipeline's compile has not come back: it is not waited for any more, and a pipeline worker is started in place of the one making it"
        );
        if WORKERS_STARTED.load(Ordering::Relaxed) > 0 {
            spawn_worker(false);
        }
    }
    true
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    #[cfg(test)]
    if let Some(budget) = TEST_BUDGET.get() {
        return budget;
    }
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_WAIT_MS);
        tracing::info!(millis, "host pipeline wait budget per frame, per kind");
        Duration::from_millis(millis)
    })
}

/// The default of [`wait_budget`]: the picture before frame rate. A frame
/// waits for pipelines still being made rather than drop their draws and
/// dispatches, up to 12 s for each kind (graphics, compute), the threads
/// together ([`Sharing::Window`]). At worst a frame then waits 24 s, both
/// kinds' windows used up one after the other, under the title's GPU
/// watchdog, which ends a run whose frame takes more than 30 s (decision
/// 0007). Only a pipeline still not made after its frame's window loses
/// that frame's draws. `VICTORY_PIPELINE_WAIT_MS=500` waits as before.
const DEFAULT_WAIT_MS: u64 = 12_000;

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10,000): such a pass is a screen of a kind the title runs once, to fill
/// its lighting tables; behind a world's worth of pipelines it was dropped
/// and the run had no lamps and no diffuse light to its end (run
/// textures-d2; Belobug 18e57db8 found the same). A budget of their own,
/// and workers, keep them.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// The title's GPU watchdog: one frame of more than this ends the run.
const TITLE_WATCHDOG: Duration = Duration::from_secs(30);

/// Maximum wall-clock time since the most recent completed `Present` during
/// which pipeline waits are allowed by default. Combined graphics + compute
/// waits and overlapping in-flight frames share this single ceiling, leaving
/// 20 s of headroom for frame preparation, copies, RT work and presentation
/// before [`TITLE_WATCHDOG`] fires.
pub(crate) const WATCHDOG_WAIT_CEILING: Duration = Duration::from_secs(10);

// Both kinds' windows, one after the other, stay under the watchdog.
const _: () = assert!(2 * DEFAULT_WAIT_MS < TITLE_WATCHDOG.as_millis() as u64);
const _: () = assert!(WATCHDOG_WAIT_CEILING.as_millis() < TITLE_WATCHDOG.as_millis());

/// Caps `budget` so pipeline waiting never extends past `ceiling` since the
/// most recent completed `Present`.
fn watchdog_cap(budget: Duration, since_present: Option<Duration>, ceiling: Duration) -> Duration {
    match since_present {
        Some(elapsed) => budget.min(ceiling.saturating_sub(elapsed)),
        None => budget,
    }
}

/// Whether `VICTORY_PIPELINE_WAIT_MS` was explicitly configured by the user.
fn explicit_wait_budget() -> bool {
    static SET: OnceLock<bool> = OnceLock::new();
    *SET.get_or_init(|| {
        std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .is_some()
    })
}

#[cfg(test)]
thread_local! {
    /// A test's own frame budget, on its thread alone.
    static TEST_BUDGET: std::cell::Cell<Option<Duration>> = const { std::cell::Cell::new(None) };
}

/// What a frame may wait for one kind of pipeline while a world loads: they
/// are asked for by the hundred then, and a pass the title runs once is
/// lost if its pipeline is not there. Compute pipelines get more, apart:
/// the passes that unpack a world's models are dispatched once (the
/// sibling fork counted 130 compute pipelines dropped while a world loaded,
/// and walls, floors, ceilings and faces missing for the session).
const LOADING_FRAME: Duration = Duration::from_secs(4);
const LOADING_FRAME_COMPUTE: Duration = Duration::from_secs(12);

/// `VICTORY_PIPELINE_GATE_S` (120): the most a run waits past
/// [`wait_budget`] for each kind, in seconds, all frames together; 0 waits
/// as before. The title's GPU watchdog bounds one frame (30 s).
fn loading_allowance() -> Duration {
    static TOTAL: OnceLock<Duration> = OnceLock::new();
    *TOTAL.get_or_init(|| {
        let seconds = std::env::var("VICTORY_PIPELINE_GATE_S")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(120);
        tracing::info!(seconds, "host pipeline loading allowance per kind");
        Duration::from_secs(seconds)
    })
}

/// A frame's budget for a pipeline of one kind: the frame budget, or while
/// the run's allowance for that kind lasts (`spent` of it gone) up to
/// [`LOADING_FRAME`] or [`LOADING_FRAME_COMPUTE`].
fn frame_budget(base: Duration, compute: bool, allowance: Duration, spent: Duration) -> Duration {
    let loading = if compute { LOADING_FRAME_COMPUTE } else { LOADING_FRAME };
    base.max(loading.min(allowance.saturating_sub(spent)))
}

/// Whether the driver has held a pipeline for longer than most take, so no
/// frame waits for it (one compute program of the title takes 85 s): 10 s
/// for a compute pipeline, 3 s for others. Only while the frame budget
/// `base` (`VICTORY_PIPELINE_WAIT_MS`) is shorter than that: a budget set
/// at or above it is the bound alone, so a pipeline still being made is
/// waited for as long as the frame's budget lasts.
fn held_too_long(held: Duration, compute: bool, base: Duration) -> bool {
    let most = Duration::from_secs(if compute { 10 } else { 3 });
    base < most && held > most
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.left_out.is_some() {
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    // The first time the made pipeline goes to a draw or dispatch: noted
    // for a device removal's report (one load a call after that).
    if let Some(host) = made
        && !p.handed.load(Ordering::Relaxed)
        && !p.handed.swap(true, Ordering::Relaxed)
    {
        crate::host::first_use(&host.name);
    }
    made
}

/// Pipelines made again for the formats bound at a draw, since the start.
static VARIANTS: AtomicU64 = AtomicU64::new(0);

/// Whether a draw into views of other formats than its pipeline declares
/// gets a host pipeline for those formats ([`for_formats`]). On by
/// default; `VICTORY_TARGET_VARIANTS=0` sends such a draw to the host with
/// the title's pipeline, as before.
fn variants_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TARGET_VARIANTS").is_none_or(|v| v != "0"))
}

/// The host pipeline for a draw with `p` into views of the formats
/// `bound`, which `p` does not declare (`host::formats_differ`): that of
/// the same shaders and state for those formats. It is a pipeline like the
/// title's own: queued at its first draw, made by a worker, waited for
/// within the frame's budget, and the draw is dropped while it is not
/// there ([`ready`]).
#[cold]
pub(crate) fn for_formats<'a>(p: &'a Arc<Pending>, bound: &TargetFormats) -> Option<&'a HostPipeline> {
    match p.graphics.as_deref() {
        Some(g) if variants_on() => ready(variant(p, g, &super::host::formats_for(&p.formats, bound))),
        _ => ready(p),
    }
}

/// The pipeline of `p`'s shaders and its state `g` for the formats
/// `bound`: found in the chain that starts at `p`, or added to it (logged
/// then: once per pipeline and set of formats).
fn variant<'a>(p: &'a Arc<Pending>, g: &GraphicsState, bound: &TargetFormats) -> &'a Arc<Pending> {
    let mut at = p;
    // (`bound` is what `host::formats_for` gives, and a state made for it
    // declares exactly it, so the walk ends at the link made for it; the
    // count is a bound all the same: past it the title's own pipeline.)
    for _ in 0..64 {
        let next = at.variant.get_or_init(|| {
            let so_far = VARIANTS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                pipeline = ?p.shaders,
                declared = ?p.formats,
                bound = ?bound,
                so_far,
                "a pipeline is made again for the render target formats bound at its draw (eight render targets, then depth; VICTORY_TARGET_VARIANTS=0: the draw goes out with the title's pipeline)"
            );
            Arc::new(Pending::new(p.shaders.clone(), Some(Box::new(g.with_formats(bound)))))
        });
        // (A state made for `bound` declares exactly `bound`.)
        if next.formats == *bound {
            return next;
        }
        at = next;
    }
    p
}

/// Non-blocking check of whether `p` has a host pipeline: queues `p` on the
/// background workers at `REQUESTED` priority if not yet made, and returns
/// `Some(true)` if compiled, `Some(false)` if refused/left out, or `None` if
/// still being made.
pub(crate) fn poll(p: &Arc<Pending>) -> Option<bool> {
    if p.left_out.is_some() {
        return Some(false);
    }
    if let Some(made) = p.made.get() {
        return Some(made.is_some());
    }
    start(p, REQUESTED);
    p.made.get().map(|made| made.is_some())
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
#[allow(dead_code)]
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    // Its worker never came back: making it here would wait for that
    // worker for good (the one-time initialiser is still its).
    if p.left_out.is_some() || stuck(p) {
        return false;
    }
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// Translates the containers of compute programs and makes their host
/// pipelines, for the caches on disk; whether it has one (`None`: left out).
pub(crate) fn warm(shaders: Vec<Arc<Shader>>) -> Option<bool> {
    let p = Pending::new(shaders, None);
    if p.left_out.is_some() {
        return None;
    }
    Some(HostPipeline::new(&p.shaders, None).is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_none() && !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// How the threads that wait in one frame share its budget
/// (`VICTORY_PIPELINE_WAIT_SHARED`): by default a window of waiting time
/// ([`Sharing::Window`]); `reserve` the way before it, `0` the way before
/// that. Counted only when a wait had ended, the budget was one per
/// thread: each of the title's five recording threads began its first
/// wait of a frame with nothing counted yet and waited the whole half
/// second, 2.5 s of waiting in every frame in which pipelines were being
/// made (runs 34-49: "2504 ms waiting for pipelines" in each such frame).
/// Reserved, a thread that waited took the rest of the budget, and every
/// other thread's new pipeline in that frame was dropped at once: a run
/// of about 835 drops as "being made" whose frame lines show at most 9 ms
/// waited (run fork-essentials).
fn shared_budget() -> Sharing {
    static ON: OnceLock<Sharing> = OnceLock::new();
    *ON.get_or_init(|| sharing_from(std::env::var("VICTORY_PIPELINE_WAIT_SHARED").ok().as_deref()))
}

/// How the threads that wait in one frame count its budget for one kind
/// of pipeline (`VICTORY_PIPELINE_WAIT_SHARED`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sharing {
    /// The budget is a window of waiting time: every thread may wait while
    /// the time in which some thread has waited in this frame is shorter,
    /// and nothing is reserved, so no thread drops a pipeline because
    /// another is waiting. The threads' waits together take at most the
    /// budget.
    Window,
    /// `reserve`: a thread that waits takes what is left of the budget and
    /// gives back what it did not use; another thread asking meanwhile
    /// gets nothing.
    Reserve,
    /// `0`: each thread counts its waits once they have ended.
    PerThread,
}

/// The way of counting the switch's value asks for.
fn sharing_from(value: Option<&str>) -> Sharing {
    match value.map(str::trim) {
        Some("0") => Sharing::PerThread,
        Some("reserve") => Sharing::Reserve,
        _ => Sharing::Window,
    }
}

/// One frame's waiting for one kind of pipeline.
#[derive(Debug)]
struct FrameWait {
    frame: u64,
    /// Completed waiting in this frame, excluding open wait windows and reservations.
    spent: Duration,
    reserved: Duration,
    /// Waiting beyond the ordinary frame budget across frames of this kind.
    loading_spent: Duration,
    /// The window: threads waiting now, and since when one has been.
    waiting: u32,
    since: Option<Instant>,
}

impl FrameWait {
    const fn new() -> FrameWait {
        FrameWait {
            frame: u64::MAX,
            spent: Duration::ZERO,
            reserved: Duration::ZERO,
            loading_spent: Duration::ZERO,
            waiting: 0,
            since: None,
        }
    }

    fn budget(&self, frame: u64, base: Duration, compute: bool, allowance: Duration) -> Duration {
        // This frame already loses its completed waits in begin. Only
        // earlier frames reduce the loading allowance used for its budget.
        let prior = if self.frame == frame {
            self.loading_spent.saturating_sub(self.spent.saturating_sub(base))
        } else {
            self.loading_spent
        };
        frame_budget(base, compute, allowance, prior)
    }

    /// A thread is to wait in `frame` at `now`: how long it may.
    fn begin(&mut self, sharing: Sharing, frame: u64, budget: Duration, now: Instant) -> Duration {
        if self.frame != frame {
            *self = FrameWait { frame, loading_spent: self.loading_spent, ..FrameWait::new() };
        }
        if sharing == Sharing::Window {
            let open = self.since.map_or(Duration::ZERO, |at| now.saturating_duration_since(at));
            let left = budget.saturating_sub(self.spent + open);
            if !left.is_zero() {
                if self.waiting == 0 {
                    self.since = Some(now);
                }
                self.waiting += 1;
            }
            return left;
        }
        let left = budget.saturating_sub(self.spent + self.reserved);
        if sharing == Sharing::Reserve {
            self.reserved += left;
        }
        left
    }

    /// The thread that [`begin`](Self::begin) gave `left` (more than
    /// nothing) in `frame` has waited `waited`, and stops at `now`. A frame
    /// that began meanwhile has its own budget.
    fn end(
        &mut self,
        sharing: Sharing,
        frame: u64,
        base: Duration,
        left: Duration,
        waited: Duration,
        now: Instant,
    ) {
        if self.frame != frame {
            return;
        }
        let previous_extra = self.spent.saturating_sub(base);
        match sharing {
            Sharing::Window => {
                self.waiting = self.waiting.saturating_sub(1);
                if self.waiting == 0
                    && let Some(at) = self.since.take()
                {
                    self.spent += now.saturating_duration_since(at);
                }
            }
            Sharing::Reserve => {
                self.reserved = self.reserved.saturating_sub(left);
                self.spent += waited;
            }
            Sharing::PerThread => self.spent += waited,
        }
        self.loading_spent += self.spent.saturating_sub(base) - previous_extra;
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let screen = p.screen && screens_first();
    let base = if screen { screen_budget() } else { wait_budget() };
    if base.is_zero() || p.slow() || stuck(p) {
        return;
    }
    let compute = p.computes();
    let allowance = loading_allowance();
    let lane = if screen { 2 } else { usize::from(compute) };
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now), by kind.
    static FRAMES: [Mutex<FrameWait>; 3] = [const { Mutex::new(FrameWait::new()) }; 3];
    let frame_wait = &FRAMES[lane];
    let sharing = shared_budget();
    let started = Instant::now();
    let ceiling = if explicit_wait_budget() {
        base.max(WATCHDOG_WAIT_CEILING)
    } else {
        WATCHDOG_WAIT_CEILING
    };
    let left = {
        let mut wait = frame_wait.lock().unwrap_or_else(|e| e.into_inner());
        let budget = if screen { base } else { wait.budget(frame, base, compute, allowance) };
        let budget = watchdog_cap(budget, crate::queue::since_last_present(), ceiling);
        if budget.is_zero() {
            return;
        }
        wait.begin(sharing, frame, budget, started)
    };
    if left.is_zero() {
        return;
    }
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() && !p.slow() && !stuck(p) {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        // Woken now and then to see whether it has become a slow one.
        let rest = rest.min(Duration::from_millis(250));
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = p.shader_hashes(),
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    frame_wait
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .end(sharing, frame, base, left, waited, Instant::now());
    crate::framelog::add(crate::framelog::T::PipelineWaitUs, started.elapsed().as_micros() as u64);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    early_depth: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            early: VecDeque::new(),
            early_depth: VecDeque::new(),
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if p.graphics.as_ref().is_some_and(|state| state.writes_depth_only()) {
            // Shadow maps and depth prepasses supply several later draws. Rebuild
            // their previously used PSOs first while no draw is requesting one.
            self.early_depth.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self, screens_only: bool) -> Option<Arc<Pending>> {
        let mut p = self.screen.pop_front();
        if p.is_none() && !screens_only {
            p = self.requested.pop_front().or_else(|| {
                // An early entry can have been promoted and claimed already.
                std::iter::from_fn(|| self.early_depth.pop_front().or_else(|| self.early.pop_front()))
                    .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
            });
        }
        let p = p?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        if p.screen {
            AVAILABLE.notify_all();
        } else {
            AVAILABLE.notify_one();
        }
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop(screens_only) {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// Stack of each pipeline worker: room for the driver's shader compiler,
/// which runs on it (reserved address space; committed only as used).
const PIPELINE_STACK: usize = 64 << 20;

/// Starts the workers (half the logical processors, 2 to 8) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = std::env::var("VICTORY_PIPELINE_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8)));
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            running |= spawn_worker(i >= workers);
        }
        running
    })
}

/// Workers started since the start (each named by its number).
static WORKERS_STARTED: AtomicUsize = AtomicUsize::new(0);

/// Starts one more pipeline worker; whether it started. `screens_only`
/// workers take nothing but passes over the screen.
fn spawn_worker(screens_only: bool) -> bool {
    let i = WORKERS_STARTED.fetch_add(1, Ordering::Relaxed);
    let spawned = std::thread::Builder::new()
        .name(format!("victory-pipelines-{i}"))
        // The driver compiles pipelines on this thread, recursively:
        // NVIDIA's compiler (nvgpucomp64) overflowed the default 2 MB
        // on a world compute program and refused it as out of memory.
        .stack_size(PIPELINE_STACK)
        .spawn(move || {
            // Below the title's own threads (loading, recording); the
            // screen workers keep it too (their queues are few).
            // SAFETY: a plain call on our own thread.
            unsafe {
                use windows::Win32::System::Threading::*;
                let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            }
            loop {
                next(screens_only).make();
                if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                    drained();
                } else {
                    super::library::save(false);
                }
            }
        });
    if let Err(e) = &spawned {
        tracing::warn!(error = %e, "no pipeline worker");
    }
    spawned.is_ok()
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod loading_tests {
    use super::{
        FrameWait, LOADING_FRAME, LOADING_FRAME_COMPUTE, Sharing, WATCHDOG_WAIT_CEILING,
        frame_budget, held_too_long, sharing_from, watchdog_cap,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn watchdog_cap_stops_pipeline_waits_past_present_ceiling() {
        let budget = Duration::from_secs(12);
        // Before the first Present, no cap applies.
        assert_eq!(watchdog_cap(budget, None, WATCHDOG_WAIT_CEILING), budget);
        // Early in a frame, remaining time is capped to the ceiling.
        assert_eq!(
            watchdog_cap(budget, Some(Duration::from_secs(2)), WATCHDOG_WAIT_CEILING),
            Duration::from_secs(8)
        );
        // Once the frame has already spent >= ceiling since the last Present,
        // stop waiting immediately so the frame finishes before the 30 s watchdog.
        assert_eq!(
            watchdog_cap(budget, Some(WATCHDOG_WAIT_CEILING), WATCHDOG_WAIT_CEILING),
            Duration::ZERO
        );
        assert_eq!(
            watchdog_cap(budget, Some(Duration::from_secs(15)), WATCHDOG_WAIT_CEILING),
            Duration::ZERO
        );
    }

    /// Two threads ask for pipelines still being made in one frame, the
    /// second 1 ms after the first: by default both may wait, inside one
    /// window of the frame's budget, and neither drops its pipeline
    /// because the other is waiting. The window bounds their waiting
    /// together: once some thread has waited its whole length, a thread
    /// that asks gets nothing. A new frame has a window of its own.
    #[test]
    fn two_threads_in_one_frame_both_wait_inside_the_window() {
        let sharing = sharing_from(None);
        let (budget, t0, ms) = (Duration::from_secs(12), Instant::now(), Duration::from_millis(1));
        let mut w = FrameWait::new();
        let a = w.begin(sharing, 7, budget, t0);
        let b = w.begin(sharing, 7, budget, t0 + ms);
        assert_eq!(a, budget);
        assert_eq!(b, budget - ms, "the second thread drops its pipeline at once: {sharing:?} gives it {b:?}");
        // Both made after 2 s: 2 s of the window gone.
        w.end(sharing, 7, budget, a, Duration::from_secs(2), t0 + Duration::from_secs(2));
        w.end(sharing, 7, budget, b, Duration::from_secs(2) - ms, t0 + Duration::from_secs(2));
        assert_eq!(w.begin(sharing, 7, budget, t0 + Duration::from_secs(5)), Duration::from_secs(10));
        // That thread waits out the rest: the window is used up.
        w.end(sharing, 7, budget, Duration::from_secs(10), Duration::from_secs(10), t0 + Duration::from_secs(15));
        assert_eq!(w.begin(sharing, 7, budget, t0 + Duration::from_secs(16)), Duration::ZERO);
        // The next frame.
        assert_eq!(w.begin(sharing, 8, budget, t0 + Duration::from_secs(17)), budget);
    }

    /// `VICTORY_PIPELINE_WAIT_SHARED=reserve`, the default before the
    /// window: the second thread gets nothing while the first waits. `0`:
    /// each thread counts only waits that have ended.
    #[test]
    fn the_old_ways_of_counting_stay_behind_the_switch() {
        let (budget, t0) = (Duration::from_secs(12), Instant::now());
        let mut w = FrameWait::new();
        assert_eq!(w.begin(Sharing::Reserve, 1, budget, t0), budget);
        assert_eq!(w.begin(Sharing::Reserve, 1, budget, t0), Duration::ZERO);
        let mut w = FrameWait::new();
        assert_eq!(w.begin(Sharing::PerThread, 1, budget, t0), budget);
        assert_eq!(w.begin(Sharing::PerThread, 1, budget, t0), budget);
        w.end(Sharing::PerThread, 1, budget, budget, Duration::from_secs(3), t0);
        assert_eq!(w.begin(Sharing::PerThread, 1, budget, t0), Duration::from_secs(9));
        assert_eq!(sharing_from(Some("0")), Sharing::PerThread);
        assert_eq!(sharing_from(Some("reserve")), Sharing::Reserve);
    }

    /// While a run's allowance lasts a frame may wait 4 s for a graphics
    /// pipeline and 12 s for a compute one, then only the frame budget; a
    /// budget set above them stays.
    #[test]
    fn a_frame_waits_longer_while_the_loading_allowance_lasts() {
        let (base, allowance) = (Duration::from_millis(500), Duration::from_secs(120));
        assert_eq!(frame_budget(base, false, allowance, Duration::ZERO), LOADING_FRAME);
        assert_eq!(frame_budget(base, true, allowance, Duration::ZERO), LOADING_FRAME_COMPUTE);
        // 1 s of the allowance left: 1 s.
        assert_eq!(frame_budget(base, true, allowance, Duration::from_secs(119)), Duration::from_secs(1));
        // Spent: the frame budget, as before.
        assert_eq!(frame_budget(base, true, allowance, Duration::from_secs(130)), base);
        assert_eq!(frame_budget(base, false, Duration::ZERO, Duration::ZERO), base, "VICTORY_PIPELINE_GATE_S=0");
        assert_eq!(frame_budget(Duration::from_secs(20), true, allowance, Duration::ZERO), Duration::from_secs(20));
    }

    #[test]
    fn short_waits_use_the_loading_allowance_when_the_frame_exceeds_the_base() {
        let base = Duration::from_millis(500);
        let allowance = Duration::from_secs(1);
        let short = Duration::from_millis(400);
        let start = Instant::now();
        for sharing in [Sharing::Window, Sharing::Reserve, Sharing::PerThread] {
            let mut frame = FrameWait::new();
            for (offset, duration) in [
                (Duration::ZERO, short),
                (short, short),
                (short * 2, Duration::from_millis(200)),
            ] {
                let budget = frame.budget(1, base, false, allowance);
                let left = frame.begin(sharing, 1, budget, start + offset);
                assert_eq!(left, allowance - offset);
                frame.end(sharing, 1, base, left, duration, start + offset + duration);
            }
            assert_eq!(frame.loading_spent, base);
            assert_eq!(frame.budget(2, base, false, allowance), base);
        }
    }

    #[test]
    fn overlapping_waits_charge_one_window() {
        let start = Instant::now();
        let base = Duration::from_millis(500);
        let allowance = Duration::from_secs(1);
        let waited = Duration::from_millis(800);
        let mut frame = FrameWait::new();
        let first = frame.begin(Sharing::Window, 1, allowance, start);
        let second = frame.begin(Sharing::Window, 1, allowance, start);
        frame.end(Sharing::Window, 1, base, first, waited, start + waited);
        frame.end(Sharing::Window, 1, base, second, waited, start + waited);
        let extra = waited - base;
        assert_eq!(frame.loading_spent, extra);
        assert_eq!(frame.budget(2, base, false, allowance), allowance - extra);
    }

    /// No frame waits for a pipeline the driver has held past 10 s
    /// (compute) or 3 s (others), while the frame budget is shorter; a
    /// budget at or above that is the bound alone.
    #[test]
    fn a_pipeline_held_too_long_is_not_waited_for() {
        let base = Duration::from_millis(500);
        assert!(!held_too_long(Duration::from_secs(10), true, base));
        assert!(held_too_long(Duration::from_millis(10_001), true, base));
        assert!(!held_too_long(Duration::from_secs(3), false, base));
        assert!(held_too_long(Duration::from_millis(3_001), false, base));
        // VICTORY_PIPELINE_WAIT_MS=12000: waited for however long it is held.
        let long = Duration::from_secs(12);
        assert!(!held_too_long(Duration::from_secs(60), true, long));
        assert!(!held_too_long(Duration::from_secs(60), false, long));
        // 5 s: past it for compute pipelines only.
        let between = Duration::from_secs(5);
        assert!(held_too_long(Duration::from_secs(60), true, between));
        assert!(!held_too_long(Duration::from_secs(60), false, between));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_compute_pipelines_share_one_pending_compile_across_threads() {
        let shader = super::super::shader("compute", &[0x44, 0x58, 0x49, 0x4c, 1, 2, 3]);
        let workers: Vec<_> = (0..16)
            .map(|_| {
                let shader = shader.clone();
                std::thread::spawn(move || Pending::shared(vec![shader], None))
            })
            .collect();
        let pending: Vec<_> = workers.into_iter().map(|worker| worker.join().unwrap()).collect();
        assert!(pending
            .iter()
            .skip(1)
            .all(|candidate| Arc::ptr_eq(&pending[0], candidate)));

        let different_shader = super::super::shader("compute", &[0x44, 0x58, 0x49, 0x4c, 9]);
        let different = Pending::shared(vec![different_shader], None);
        assert!(!Arc::ptr_eq(&pending[0], &different));
    }

    #[test]
    fn pending_pipeline_identity_includes_graphics_state() {
        use windows::Win32::Graphics::Direct3D12::D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;

        let shader = super::super::shader("vertex", &[0x44, 0x58, 0x49, 0x4c, 5]);
        let first = Pending::shared(
            vec![shader.clone()],
            Some(Box::new(GraphicsState::default())),
        );
        let mut changed = GraphicsState::default();
        changed.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        let second = Pending::shared(vec![shader], Some(Box::new(changed)));
        assert!(!Arc::ptr_eq(&first, &second));
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A pipeline whose worker never came back is not waited for, `now`
    /// does not block on it, and a worker is started in its worker's place.
    #[test]
    fn a_stuck_pipeline_is_not_waited_for_and_its_worker_is_replaced() {
        assert!(workers());
        let p = Pending::new(Vec::new(), None);
        // A worker began it and never returned (the one-time initialiser
        // is never run here).
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let _ = p.begun.set(Instant::now());
        let p = Arc::new(p);
        assert!(!stuck(&p));
        std::thread::sleep(stuck_after() + Duration::from_millis(20));
        let before = WORKERS_STARTED.load(Ordering::Relaxed);
        let started = Instant::now();
        assert!(ready(&p).is_none());
        assert!(!now(&p));
        assert!(started.elapsed() < Duration::from_millis(100), "waited for a stuck pipeline");
        assert_eq!(WORKERS_STARTED.load(Ordering::Relaxed), before + 1);
        // Once only.
        assert!(stuck(&p));
        assert_eq!(WORKERS_STARTED.load(Ordering::Relaxed), before + 1);
        assert!(p.made.get().is_none());
    }

    /// A crash report names the pipeline being made, on the thread making
    /// it, and drops it once made. Other tests' workers may be making their
    /// own meanwhile: only this key is looked for.
    #[test]
    fn crash_report_names_the_pipeline_being_made() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "crash-report-test-key".to_owned();
        let line = |report: &str| {
            report
                .lines()
                .find(|l| l.contains("crash-report-test-key"))
                .map(str::to_owned)
        };
        let making = Making::begin(&p);
        let named = line(&being_made()).expect("the pipeline being made is listed");
        assert!(named.contains("(this thread)"), "{named}");
        let elsewhere = std::thread::scope(|s| s.spawn(being_made).join().unwrap());
        let named = line(&elsewhere).expect("listed for other threads too");
        assert!(!named.contains("(this thread)"), "{named}");
        drop(making);
        assert_eq!(line(&being_made()), None);
    }

    #[test]
    fn poll_queues_without_blocking_and_reports_completion() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        let first = poll(&p);
        assert!(matches!(first, None | Some(false)));
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        // The shared workers may be queued behind the rest of the suite for
        // a while: make it here rather than waiting them out (a worker
        // claiming it first makes this a no-op: `made` is set once).
        p.make();
        assert_eq!(poll(&p), Some(false));
    }

    /// A draw into other formats than its pipeline declares finds the
    /// pipeline for them, which exists once per set of formats and is
    /// queued by its first draw like any other.
    #[test]
    fn a_pipeline_for_bound_formats_is_made_once() {
        // The title's pipeline declares one render target, format 24.
        let p = Arc::new(Pending::new(Vec::new(), Some(Box::new(GraphicsState::declaring(24)))));
        assert_eq!(p.formats, [24, 0, 0, 0, 0, 0, 0, 0, 0]);
        let g = p.graphics.as_deref().unwrap();
        let (a, b) = ([26, 0, 0, 0, 0, 0, 0, 0, 0], [28, 0, 0, 0, 0, 0, 0, 0, 0]);
        let first = variant(&p, g, &a);
        assert_eq!(first.formats, a);
        assert_ne!(first.key, p.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        let second = variant(&p, g, &b);
        assert_eq!(second.formats, b);
        assert!(!Arc::ptr_eq(first, second));
        assert_ne!(second.key, first.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        assert!(Arc::ptr_eq(second, variant(&p, g, &b)));
        assert_eq!(first.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        assert_eq!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);
        // No frame budget on this thread (`VICTORY_PIPELINE_WAIT_MS=0`):
        // the request is not waited for, where the default would wait out
        // a 12 s window for a pipeline that is never made.
        super::TEST_BUDGET.set(Some(std::time::Duration::ZERO));

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false).unwrap(), &earlier));
        assert!(queue.pop(false).is_none());
    }

    #[test]
    fn early_depth_precedes_other_prewarming_and_requests_precede_both() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;
        let depth_pipeline = || {
            let mut depth = Box::new(GraphicsState::default());
            let format = [16u32, DXGI_FORMAT_D32_FLOAT.0 as u32];
            // SAFETY: a complete depth-format subobject of the title pipeline stream.
            unsafe { depth.read(16, format.as_ptr().cast()) };
            Arc::new(Pending::new(Vec::new(), Some(depth)))
        };
        let normal = Arc::new(Pending::new(Vec::new(), None));
        let promoted = depth_pipeline();
        let shadow = depth_pipeline();
        let mut queue = Queue::new();
        for p in [&normal, &shadow, &promoted] {
            assert_eq!(queue.push(p, EARLY), Some(true));
        }
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        for expected in [&promoted, &shadow, &normal] {
            assert!(Arc::ptr_eq(&queue.pop(false).unwrap(), expected));
        }
        assert!(queue.pop(false).is_none());
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true).unwrap(), &screen));
        assert!(queue.pop(true).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false).unwrap(), &world));
    }
}
