import re

SRC = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-gpu/src'

def edit(path, pairs, must=1):
    src = open(path, encoding='utf-8').read()
    for old, new in pairs:
        n = src.count(old)
        assert n == must, f'{path}: {n}x\n{old[:120]}'
        src = src.replace(old, new)
    open(path, 'w', encoding='utf-8').write(src)
    print(path, 'ok', len(pairs))

host = SRC + '/host.rs'
sub = SRC + '/host/submission.rs'

src = open(host, encoding='utf-8').read()

# 1) stale_under_flights: ours' Batch `spans` into theirs' Vec `stale_of`.
old = """    stale_of(spans)
}

/// Of `spans` (a copy shaders may be writing, and the span of what a
/// submission refreshes of it), those [`stale_under_flights`] answers."""
new = """    !stale_of(spans.iter().map(|(shadow, from, to)| (shadow.clone(), from, to)).collect()).is_empty()
}

/// Of `spans` (a copy shaders may be writing, and the span of what a
/// submission refreshes of it), those [`stale_under_flights`] answers."""
assert src.count(old) == 1
src = src.replace(old, new)

# 2) drop theirs' simple `rehash_runs` (keep ours' `alt` version further down).
old = '''fn rehash_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_REHASH_RUNS").is_some_and(|v| v == "1"))
}
'''
assert src.count(old) == 1, src.count(old)
src = src.replace(old, '')

# 3) drop the second AddressHash definition (identical).
pat = re.compile(r'(#\[derive\(Default\)\]\npub\(crate\) struct AddressHash\(u64\);\n\nimpl std::hash::Hasher for AddressHash \{\n(?:.*?\n)*?\})', re.S)
ms = list(pat.finditer(src))
assert len(ms) == 2, len(ms)
# keep the first (adjacent to declared_writes), drop the second
src = src[:ms[1].start()] + src[ms[1].end():]

# 4) drop ours' `late_met_known` returning bool; callers patched to theirs'
# Option version (queue.rs below).
old = """/// [`late_met`] without its settle: whether a bottom-of-pipe write of a
/// flight in the air, or of one being landed, is known to meet `wait`.
/// False says nothing about what else the flights may write there.
pub(crate) fn late_met_known(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    let line = line();
    let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
        Late::Write(at, what, n) => Some((at, what, n)),
        _ => None,
    }));
    for (at, what, n) in writes {
        if let crate::queue::Written::Value(v) = what
            && *at == address
            && *n >= bytes
            && wait.holds_for(*v)
        {
            LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}
"""
assert src.count(old) == 1, src.count(old)
src = src.replace(old, '')

# 5) drop the duplicate `fill_scratch` (theirs' `known`-named copy).
old = """    /// The host address of `len` bytes of scratch memory that hold the title
    /// memory at `src` as of the list's submission: one fill for every ask
    /// until the list next writes title memory.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: usize) -> Option<u64> {"""
if src.count(old) == 0:
    old = """    /// The host address of `len` bytes of scratch memory that hold the title
    /// memory at `src` as of the list's submission: one fill for every ask
    /// until the list next writes title memory.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {"""
i = src.index(old)
j = src.index("    /// As [`HostList::defer_fill`], for upload memory that only `slot`", i)
src = src[:i] + src[j:]

# 6) defer_prelude_upload: Step::Texture is 2-field (staging, texture).
old = """        let mut deferred = self.deferred();
        deferred.copies.push(Step::Texture(upload.staging()));
        deferred.prelude_uploads.push(upload);"""
new = """        let mut deferred = self.deferred();
        let staging = upload.staging();
        let copy = upload.copy.clone();
        deferred.copies.push(Step::Texture(staging, copy));
        deferred.prelude_uploads.push(upload);"""
assert src.count(old) == 1
src = src.replace(old, new)

# 7) joinable: ours' declared_writes is 2-arg / 3-tuple Declared.
old = """    let mut written: Vec<(usize, usize)> = declared_writes(&ahead).into_iter().flat_map(|(_, ranges)| ranges).collect();"""
assert src.count(old) == 1
src = src.replace(old, """    let mut written: Vec<(usize, usize)> = declared_writes(&ahead, false).into_iter().flat_map(|(_, ranges, _)| ranges).collect();""")

# 8) Meter: theirs' caption + copy_stats_frames fields and inits.
old = """struct Meter {
    since: std::time::Instant,
    frames: u32,
    drawn: u64,
    /// The last frame rate and frame time shown ([`readout`]).
    shown: Option<(f64, f64)>,
    logged: std::time::Instant,
    /// Process CPU time at the last report (microseconds).
    cpu_us: u64,
    /// Percent of the machine's CPU over the last report.
    cpu_pct: f64,
    /// Percent of the frame time the GPU was busy.
    gpu_pct: f64,
    /// Working set in MB.
    ram_mb: f64,
}"""
new = """struct Meter {
    since: std::time::Instant,
    frames: u32,
    drawn: u64,
    /// The last frame rate and frame time shown ([`readout`]).
    shown: Option<(f64, f64)>,
    logged: std::time::Instant,
    /// Process CPU time at the last report (microseconds).
    cpu_us: u64,
    /// Percent of the machine's CPU over the last report.
    cpu_pct: f64,
    /// Percent of the frame time the GPU was busy.
    gpu_pct: f64,
    /// Working set in MB.
    ram_mb: f64,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
}"""
assert src.count(old) == 1
src = src.replace(old, new)
old = """        Meter {
            since: now,
            frames: 0,
            drawn: crate::draw::drawn(),
            shown: None,
            logged: now,
            cpu_us: process_cpu_us().unwrap_or(0),
            cpu_pct: 0.0,
            gpu_pct: 0.0,
            ram_mb: 0.0,
        }"""
new = """        Meter {
            since: now,
            frames: 0,
            drawn: crate::draw::drawn(),
            shown: None,
            logged: now,
            cpu_us: process_cpu_us().unwrap_or(0),
            cpu_pct: 0.0,
            gpu_pct: 0.0,
            ram_mb: 0.0,
            caption: None,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
        }"""
assert src.count(old) == 1
src = src.replace(old, new)

open(host, 'w', encoding='utf-8').write(src)
print('host.rs core edits done')

# ---- submission.rs ----
s = open(sub, encoding='utf-8').read()
old = "                pipelines: Mutex::default(),"
assert s.count(old) == 1
s = s.replace(old, """                pipelines: std::sync::Arc::new(Mutex::default()),
                #[cfg(feature = "diagnostics")]
                indirects: indirects::Shared::default(),
                batch_args: Mutex::default(),
                coming: Mutex::new(None),
                upscale: AtomicU64::new(0),
                hudless: AtomicU64::new(0),
                left_open: [AtomicU64::new(0), AtomicU64::new(0)],""")
old = """            let uploads = list.run_deferred(&mut batch);"""
assert s.count(old) == 1
s = s.replace(old, """            let uploads = list.run_deferred(&mut batch);
            let preludes = list.prelude_uploads(&mut batch);""")
old = """            if let Some(prelude) = list.prelude(&uploads) {"""
assert s.count(old) == 1
s = s.replace(old, """            if let Some(prelude) = list.prelude(&uploads, &preludes) {""")
open(sub, 'w', encoding='utf-8').write(s)
print('submission.rs done')

# ---- queue.rs: late_met_known is now Option ----
q = SRC + '/queue.rs'
s = open(q, encoding='utf-8').read()
old = "if !wait.met() && !crate::host::late_met_known(wait) {"
assert s.count(old) == 1
s = s.replace(old, "if !wait.met() && crate::host::late_met_known(wait) != Some(true) {")
open(q, 'w', encoding='utf-8').write(s)
print('queue.rs done')

# ---- Step pattern fixes in theirs' submodule files ----
for path, pairs in [
    (SRC + '/host/card_steps.rs', [
        ("Step::Texture(staging) => {", "Step::Texture(staging, _) => {"),
        ("Step::Run(_) => Does::Run,", "Step::Run(..) => Does::Run,"),
    ]),
    (SRC + '/host/place.rs', [
        ("Step::Texture(staging) => {", "Step::Texture(staging, _) => {"),
    ]),
    (SRC + '/host/upcheck.rs', [
        ("Step::GpuWrite(_, at, len) => written.push((*at, *at + *len)),",
         "Step::GpuWrite(_, at, len, _) => written.push((*at, *at + *len)),"),
    ]),
]:
    s = open(path, encoding='utf-8').read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    open(path, 'w', encoding='utf-8').write(s)
    print(path, 'done')

# ---- port theirs' unique batch test into host/batch.rs ----
t109 = open('/tmp/hunks/h109_theirs.txt', encoding='utf-8').read().split('\n', 1)[1]
i = t109.index('    fn textures_are_kept_while_the_card_has_room()')
# find the end of that test fn (next `    fn ` or end of text)
rest = t109[i + 10:]
j = rest.find('\n    fn ')
test = t109[i:i + 10 + j] if j != -1 else t109[i:]
b = open(SRC + '/host/batch.rs', encoding='utf-8').read()
assert 'textures_are_kept_while_the_card_has_room' not in b
b = b.rstrip()[:-1].rstrip() + '\n\n' + test.rstrip() + '\n}\n'
open(SRC + '/host/batch.rs', 'w', encoding='utf-8').write(b)
print('batch.rs test appended')
