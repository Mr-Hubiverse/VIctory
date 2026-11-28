import re, sys
sys.path.insert(0, '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main')
from resolve1 import resolve

def O(n):
    return open(f'/tmp/hunks/h{n:03d}_ours.txt', encoding='utf-8').read().split('\n', 1)[1]

def T(n):
    return open(f'/tmp/hunks/h{n:03d}_theirs.txt', encoding='utf-8').read().split('\n', 1)[1]

C = {}

# h2: sweep() — theirs' book/residency/pool calls + ours' CAS/`force` trim (merged
# let_go_idle/trim_library all take `force`).
C[2] = T(2).replace(
    "    crate::pipeline::trim_library();\n    crate::shadow::let_go_idle();\n    crate::texture::let_go_idle();\n    crate::resource::let_go_idle();\n",
    "") + O(2)

# h3: theirs' retire_resource + bury_lag (drop their `fn bury`), then ours'
# retire_lean/SLOT_REWRITES/slot_rewritten/bury.
t3 = T(3)
i = t3.index("fn bury(finished: u64)")
C[3] = t3[:i] + O(3)

# h9: ours' create + theirs' tdr::log in the non-WARP branch + replay::record::from_env.
C[9] = """            let mut out: Option<ID3D12Device> = None;
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut out)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
            out.ok_or_else(windows::core::Error::empty)?
        };
        // Before anything is made with it (`VICTORY_RECORD`).
        #[cfg(feature = "diagnostics")]
        crate::replay::record::from_env(&device);
"""

# h10: union — ours' validated wave_lanes + theirs' RDNA flag + SM 6.6 check.
C[10] = """            .then_some(o1.WaveLaneCountMin..=o1.WaveLaneCountMax)
            .filter(|lanes| *lanes.start() != 0 && !lanes.is_empty());
        if let Some(lanes) = &wave_lanes {
            tracing::info!(min = lanes.start(), max = lanes.end(), "host wave lanes");
            victory_shader::emit::set_host_max_wave(*lanes.end());
            // (RDNA has waves of 32 and 64, the vendor's older cards of 64 alone.)
            let amd = crate::indirect::amd(&device);
            victory_shader::emit::set_host_rdna(amd && o1.WaveLaneCountMin == 32 && o1.WaveLaneCountMax == 64);
        }
        // Every translated shader is SM 6.6: below it each pipeline is refused (E_INVALIDARG),
        // and the title falls over within seconds (Windows 10's own D3D12, 2026-11-27).
        match crate::streamline::device_shader_model(&device) {
            Some(sm) if sm >= crate::streamline::SHADER_MODEL_NEEDED => {
                let runtime = crate::streamline::d3d12_core_path().map(|p| p.display().to_string());
                tracing::info!(model = %format!("{}.{}", sm.0, sm.1), ?runtime, "host shader model");
            }
            sm => tracing::error!(
                model = %sm.map_or("unknown".to_owned(), |sm| format!("{}.{}", sm.0, sm.1)),
                "host shader model below 6.6: no translated shader can run here. On Windows 10, \
                 `victory play` fetches Microsoft's newer Direct3D 12 runtime (D3D12\\D3D12Core.dll \
                 beside victory.exe); otherwise update the graphics driver"
            ),
        }
"""

# h12: ours' from_device/card_engine/execute_and_wait_locked/new_warp + theirs'
# streamline queue, upload::reserve_early, fallback heap fields, driver().
o12 = O(12)
o12 = o12.replace(
    """            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;""",
    """            // Streamline is given the device (DLSS, `VICTORY_DLSS`); for frame
            // generation (`VICTORY_FRAMEGEN`) the direct queue is made through
            // its device proxy.
            let queue: ID3D12CommandQueue = match crate::streamline::device_made(&device) {
                Some(queue) => queue,
                None => device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?,
            };""")
o12 = o12.replace(
    "                card: OnceLock::new(),\n            })",
    """                card: OnceLock::new(),
                fallback_view: OnceLock::new(),
                fallback_sampler: OnceLock::new(),
            })""")
o12 = o12.replace(
    "            Ok(Host {\n",
    "            crate::upload::reserve_early();\n            Ok(Host {\n")
C[12] = o12 + "\n    /// The driver's version, as the system counts it (0: not known).\n    pub(crate) fn driver(&self) -> i64 {\n        let adapter = self.adapter.as_ref();\n        // SAFETY: a plain query.\n        adapter.map_or(0, |a| unsafe { a.CheckInterfaceSupport(&IDXGIDevice::IID) }.unwrap_or(0))\n    }\n"

# h13: ours' mod submission + mod tests; + theirs' inline tests renamed
# `unit_tests` (the file `host/tests.rs` is `mod tests`), + theirs' BATCHES.
t13 = T(13)
C[13] = O(13) + "\n\n" + t13.replace("mod tests {", "mod unit_tests {", 1)

# h14: theirs' Arc<Mutex> pipelines + indirects field, + ours' queries field.
C[14] = """    pipelines: std::sync::Arc<Mutex<std::collections::BTreeSet<String>>>,
    /// Queries begun in the list and not ended yet: host heap, type, index,
    /// and whether a draw since was not run on the host (`missed_draw`).
    queries: Mutex<Vec<(ID3D12QueryHeap, D3D12_QUERY_TYPE, u32, bool)>>,
    /// What its indirect executions ran, for the report of a removed device.
    #[cfg(feature = "diagnostics")]
    indirects: indirects::Shared,
"""

# h15: 'both' — ours' window_check/clamp_window/open_queries/clamp_chunks +
# theirs' batch_args/coming/upscale/hudless/left_open.
C[15] = 'both'

# h16: 'both' — ours' make_prelude_on + Gather/Segments + theirs' fresh_open.
C[16] = 'both'

# h17: ours' Dred machinery + NAMED, + theirs' LIST_NOTES + list_number, +
# dred_light/dred_faults as dred_mode wrappers.
t17 = T(17)
i = t17.index("/// The lists alive while device-removed data is on")
C[17] = O(17) + "\n" + t17[i:] + """
/// Breadcrumbs and the faulting address, but no markers and no texts
/// (`VICTORY_D3D12_DRED=light`).
pub(crate) fn dred_light() -> bool {
    dred_mode() == Dred::Light
}

/// Whether only the faulting address of a removed device is kept
/// (`VICTORY_D3D12_DRED=faults`).
pub(crate) fn dred_faults() -> bool {
    dred_mode() == Dred::Faults
}
"""

# h18: ours' gpu_waits/log_gpu_waits/GPU_TIMES/log_gpu_times + a merged
# name/name_it (theirs' book + record naming, ours' NAMED counting).
o18 = O(18)
C[18] = o18 + """
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    #[cfg(feature = "diagnostics")]
    if crate::book::on()
        && let Ok(resource) = object.cast::<ID3D12Resource>()
    {
        let text = text();
        crate::book::made(&resource, || text.clone());
        return name_it(object, || text);
    }
    name_it(object, text);
}

fn name_it(object: &impl Interface, text: impl FnOnce() -> String) {
    #[cfg(feature = "diagnostics")]
    if crate::replay::record::on() {
        let text = text();
        crate::replay::record::name(object, &text);
        if dred_any()
            && let Ok(object) = object.cast::<ID3D12Object>()
        {
            named();
            // SAFETY: a plain call on a live object.
            let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text)) };
        }
        return;
    }
    if dred_any()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        named();
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text)) };
    }
}

/// Counts an object named, so that a log says what naming costs: one text
/// and one call each. Said at 4,096 and at every doubling after it.
fn named() {
    let named = NAMED[0].fetch_add(1, Ordering::Relaxed) + 1;
    if named >= 4096 && named.is_power_of_two() {
        tracing::info!(
            "device-removed data: {named} host objects named so far, {} of them made while frames are drawn",
            NAMED[1].load(Ordering::Relaxed)
        );
    }
}
"""

# h19: 'ours' (pso_wide field + ours' doc wording).
# h20: 'both' — ours' CARD_COMPARE_WIDE + theirs' piece_hash + card_pass.
C[20] = 'both'
# h21: 'ours' (sent_as/inflight/inflight_landed; theirs' looks/sends only used
# by the dropped inline submit).
# h22: union Step enum.
C[22] = """    /// Work that may write title memory; with the place it writes
    /// (address, bytes) when that is known (the title's memory-to-memory
    /// copies, `crate::list` `copy_memory`): [`near_writes`].
    Run(Box<dyn Fn() + Send>, Option<(usize, usize)>),
    /// Work that writes host memory only, and the title memory it reads
    /// (from, to; empty for none).
    Host(Box<dyn Fn() + Send>, (usize, usize)),
    /// Shaders write a range (address, bytes) of a buffer's host copy,
    /// the last value where they mark the pages they write in the views'
    /// bitmap instead (`victory_shader::emit` `SLOT_VIEW_MARKS`): no
    /// snapshot for it, the marked pages alone are compared
    /// (`crate::shadow::copy_back_marks`).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    ),
    /// Reads a CPU-written linear movie plane into a per-list upload buffer.
    Plane((usize, usize), Box<dyn Fn() + Send>),
    /// A title buffer copy that the list does on the card ([`place`]): what
    /// is done on the CPU for it.
    Placed(place::Placed),
"""
# h23: 'theirs' (AddressHash-hashed bound).
C[23] = 'theirs'
# h24: 'theirs' (same type; theirs' doc).
C[24] = 'theirs'
# h25: 'ours' — the walls-trace 6-tuple `backs` (shared read_backs body is ours').
# h26: 'ours' (after field + RawCopyCoverage).
# h27: 'theirs' (Deferred::title_written — called by the unioned defer paths).
C[27] = 'theirs'
# h28: 'none' — theirs' `own` field (Sent) is only used by the dropped submit.
C[28] = 'none'
# h29: 'ours' (texture_back_ranges + writeback_owners; drop theirs' own).
# h30/31: 'ours' — fill_ahead machinery, adapted to 4-field Fill.
C[31] = O(31).replace("&Step::Fill(_, src, len) =>", "&Step::Fill(_, src, len, _) =>")
# h32: union — ours' Run/Refresh/GpuWrite arm + theirs' Placed arm.
C[32] = """                Step::Run(..) => return Some(WHY_RUN),
                Step::Placed(p) => match p.waits(written) {
                    place::Waits::No => {}
                    place::Waits::ForReaders => placed.push((p.dst, p.dst + p.len)),
                    place::Waits::Yes => {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a buffer copy at its place", written, p.src, p.len);
                        }
                        return Some(WHY_READ);
                    }
                },
                // A refresh or GPU write that overlaps what flights still
                // in the air may publish must wait for them: their landing
                // merge needs title memory as it was.
                Step::Refresh(_, at, len) | Step::GpuWrite(_, at, len, _) => {
                    if crate::writeback::needs_new_snapshot(*at, at + len, written)
                        || touches(written, *at, at + len)
                    {
                        return Some(WHY_READ);
                    }
                }
                Step::Host(..) => {}
"""
# h33: 'theirs' — stale_under_flights returns Vec (stale_of is theirs' shared
# helper; ours' submission.rs/batch.rs call sites get `.is_empty()` patches).
# h34: custom — theirs' stale_of tail + theirs' joins/joinable/will_meet/split
# (call sites patched for 3-tuple Declared) + ours' rehash_runs/
# merge_runs_alt/sync_batched.
C[34] = T(34) + "\n" + O(34)[O(34).index("/// The cap of changed runs"):]
# h35: ours' submission_us + theirs' blamed (no double SUBMISSION_US add).
C[35] = "    submission_us(1, us);\n    #[cfg(feature = \"diagnostics\")]\n    blamed(why, us);\n"
# h36/37: 'ours' — submission.rs calls ours' 3-arg relay_hand_back.
# h38: 'both' — both stats-log blocks.
# h39: 'ours' — AddressHash + ours' 2-arg declared_writes (3-tuple Declared).
# h40/41: 'ours' — ours' backs tuple + `after` work queue.
# h42: 'ours' — the huge inline impl (submission.rs owns the submit machinery).
# h43: union — theirs' left_open reset + flush_indirect + ours' queries drain.
C[43] = T(43) + O(43)
# h44: union — ours' gather/cardmeter/shot::close + theirs' batches_common/take_coming.
C[44] = """        self.batches_common();
        self.take_coming();
""" + O(44)
# h45: 'both' — ours' keep/release/worked/waits + theirs' refreshes_coming & co.
C[45] = 'both'
# h46: 'both' — reset: ours' clamp_window=None + theirs' upscale/hudless/coming.
C[46] = 'both'
# h47: transition() — ours' head needs `texture` bound; take theirs' binding +
# ours' legal_state + shot::known (shared tail is ours' `let before` lookup).
C[47] = """        // SAFETY: as in `crate::resource::host_resource`; the list's states
        // hold the copy from here on.
        let texture: &'static ID3D12Resource = unsafe { &*std::ptr::from_ref(&copy.texture) };
        let wanted = legal_state(texture, wanted);
        // A frame shot names its pictures after the title's textures.
        #[cfg(feature = "diagnostics")]
        if crate::shot::recording() {
            crate::shot::known(std::sync::Arc::as_ptr(&copy) as usize, resource as usize);
        }
"""
# h48/64/68: 'theirs' (`let mut deferred` — harmless, theirs' code wants it).
C[48] = 'theirs'
# h49: 'ours' (shot::known/note_op for the texback read).
# h50: prelude signature — ours' uploads + add `preludes` param (call site in
# submission.rs is patched to pass `list.prelude_uploads(&mut batch)`).
C[50] = """    /// from the other, if a list submitted earlier left it behind. Title
    /// textures whose upload is decided at submission (`preludes`,
    /// `crate::texture::prelude_uploads`) are copied into their host copies
    /// here, right before the list runs (Belobug/VIctory-Belobug-AMD
    /// 5d6bd84).
    fn prelude(
        &self,
        uploads: &[(
            std::sync::Arc<crate::texture::Staging>,
            std::sync::Arc<crate::resource::HostTexture>,
        )],
        preludes: &[(crate::texture::PreludeUpload, bool)],
    ) -> Option<ID3D12CommandList> {
"""
# h51: union — theirs' aliasing/copying + ours' shot_begins/leaving/upload_pre.
C[51] = T(51) + O(51)
# h52: union — ours' sort+gather+early-return with `preludes` added.
C[52] = O(52).replace(
    "if uploads.is_empty() && ba",
    "if uploads.is_empty() && preludes.is_empty() && ba").replace(
    "if uploads.is_empty() && barriers.is_empty() && syncs.is_empty() && !gathered {",
    "if uploads.is_empty() && preludes.is_empty() && barriers.is_empty() && syncs.is_empty() && !gathered {")
# h53: 'both' (shot_span + aliasing barrier).
# h54: ours' upload_pre/record_copies/leaving + theirs' upload.record over
# `preludes`.
C[54] = O(54) + T(54).replace("for (upload, _) in uploads {", "for (upload, _) in preludes {")
# h55: theirs' ran_indirect + ours' #[allow(dead_code)] doc line.
C[55] = T(55) + O(55)
# h56: ours' block with theirs' 3-arg bind_pipeline swapped in + ours' accessor.
o56 = O(56)
i = o56.index("    /// Notes a pipeline the list binds and keeps it and its root signature")
j = o56.index("    /// With [`list_reuse`]: whether this is the first time")
C[56] = (o56[:i]
         + """    /// The list binds a pipeline: it and its root signature are kept until
    /// the list is reset, and its name noted, the first time only (a kept
    /// pipeline's address names no other).
    pub(crate) fn bind_pipeline(&self, name: &str, pso: &ID3D12PipelineState, root: &ID3D12RootSignature) {
        {
            let deferred = self.deferred();
            if !deferred.bound.insert(pso.as_raw() as usize) {
                return;
            }
            deferred.kept.push(pso.clone().into());
            deferred.kept.push(root.clone().into());
        }
        self.used_pipeline(name);
    }

"""
         + o56[j:])
# h57: union — theirs' title_written + changed_something; keep ours'
# invalidate_refreshes too (both track different state).
C[57] = "        deferred.invalidate_refreshes();\n        deferred.title_written();\n        self.changed_something();\n"
# h58: 'ours' (marked doc).
# h59: 'ours' (4-field GpuWrite push + declare_once).
# h60: union — theirs' shared_buffer for card_steps + ours' name_frame +
# hold_upload + readback.
o60 = O(60)
C[60] = o60.replace(
    "        let (buffer, data) = host.upload_buffer(size)?;",
    "        // With [`card_steps`] the card may be the one that fills it.\n        let (buffer, data) = if card_steps::stage() > 1 { host.shared_buffer(size)? } else { host.upload_buffer(size)? };")
# h61: 'both_rev' — theirs' scratch/scratch_held/scratch_at + ours' scratch_in.
C[61] = T(61) + O(61)
# h62: 'both' — ours' whole-chunk scratch_in branch + theirs' take_chunk arm.
C[62] = 'both'
# h63: union — theirs' chunk tail + ours' aligned scratch_in tail.
C[63] = T(63) + O(63)
# h64: 'theirs'.
C[64] = 'theirs'
# h65: 'theirs' (2-field Host step construction).
C[65] = 'theirs'
# h66: 'theirs'.
C[66] = 'theirs'
# h67: 'theirs' (batch_place; callers use it).
C[67] = 'theirs'
# h68: 'theirs'.
C[68] = 'theirs'
# h69: 'ours' (window_check field helper).
# h70: 'both' — ours' raw_memory_ranges/texture_alias + theirs' used/mark_upscale/hudless.
C[70] = 'both'
# h71: union — ours' defer Run-step/bindless + theirs' title_written/changed_something.
o71 = O(71)
t71 = T(71)
C[71] = o71.replace(
    """        let deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work), None));
        deferred.refreshed.clear();
        deferred.gpu_written.clear();
        deferred.filled.clear();
    }""",
    """        let deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work), None));
        deferred.refreshed.clear();
        deferred.gpu_written.clear();
        deferred.filled.clear();
        deferred.title_written();
        drop(deferred);
        self.changed_something();
    }""")
# h72: 'ours' — run_deferred keeps ours' signature (returns texture uploads).
# h73: union — ours' title_wrote path + theirs' fill/heap_marks/port.
t73 = T(73)
C[73] = O(73) + t73[t73.index("                    crate::port::copied"):]
# Insert theirs' batch.fill/title_written/heap_marks before port::copied,
# after ours' block:
C[73] = O(73) + """                    batch.fill();
                    batch.title_written();
                    // Into a heap's records: the rounds asked for earlier
                    // did not see it (`crate::heap_marks`).
                    if crate::heap_marks::on() && crate::descriptor::in_shader_heap(dst, len) {
                        crate::heap_marks::prep_wrote();
                    }
                    crate::port::copied(dst, src, len);
"""
# h74: union Fill arm (ours' copy_fill + theirs' wide_fills/port::filled) +
# theirs' Placed arm (air = none: free=false, conservative).
C[74] = """                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                        fill_check::runs(src, len);
                    }
                    crate::port::filled(src, len);
                    if wide_fills() {
                        batch.fills.push((dst, src, len));
                        continue;
                    }
                    #[cfg(feature = "diagnostics")]
                    let _reads_as = crate::backreads::reading_as(crate::backreads::TITLE_COPY, 0);
                    copy_fill(dst as *mut u8, src as *const u8, len);
                    len
                }
                Step::Placed(p) => {
                    if crate::heap_marks::on() && crate::descriptor::in_shader_heap(p.dst, p.len) {
                        crate::heap_marks::prep_wrote();
                    }
                    // The submissions in the air are not looked at here: not
                    // free means title memory is written by the landing.
                    let free = false;
                    batch.fill();
                    if free {
                        batch.title_written();
                    }
                    if let Some(bytes) = p.prepare(free) {
                        batch.placed.push((p.shadow.clone(), p.dst, bytes));
                    }
                    p.len
"""
# h75: 'ours' (refresh_count/refresh_bytes names match ours' h78 usage).
# h76: union Run/Host arm.
C[76] = """                Step::Run(work, Some((at, len))) if near_writes() => {
                    // Its place is known: only what it touches is brought up
                    // to date first and checked again after it, and the
                    // submission's one asking stands, told of the write, as
                    // for a title buffer copy ([`near_writes`]).
                    let written = batch.title_wrote_near(*at, *len, true);
                    work();
                    for &(from, to) in &written {
                        crate::watch::wrote(from, to - from);
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Run(work, _) | Step::Host(work, _) => {
                    batch.fill();
                    if matches!(step, Step::Run(..)) {
                        batch.title_written();
                        // (It may write anywhere, a heap's records too.)
                        if crate::heap_marks::on() {
                            crate::heap_marks::prep_wrote();
                        }
                    } else if !batch.due_textures.is_empty() {
                        // Host callbacks can use a refresh result, including the
                        // mip mask that maps a reserved texture for this upload.
                        batch.check_textures();
                    }
"""
# h77: union Texture arm.
C[77] = """                        if near_writes() {
                            // Where it lies, for a write of ours at a known
                            // place to leave it checked ([`Batch::title_wrote_near`]).
                            batch
                                .texture_spans
                                .insert(std::sync::Arc::as_ptr(staging) as *const () as usize, staging.memory());
                        }
                        batch.due.push(staging.clone());
                        if parallel_refresh() {
                            // Checked with the others of the batch, several
                            // at a time, before the next step that writes
                            // title memory ([`Batch::check_textures`]).
                            batch.due_textures.push(staging.clone());
                        } else if !wide_textures() {
                            batch.refresh_textures();
                        } else if staging.refresh() {
                            uploads.push((staging.clone(), copy.clone()));
"""
# h78: 'ours' (preparing() calls match ours' copy_us/copy_bytes vars).
# h79: 'theirs' (wide_refresh_enabled with tests).
C[79] = 'theirs'
# h80: union Batch fields.
C[80] = """    index: ByCopy<usize>,
    /// With [`refresh_once`], by copy (its address; the copy is held, so
    /// that the address names no other): the range of title addresses
    /// brought up to date in this submission that no step of ours has
    /// written since.
    current: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write,
    /// or due to be.
    textures: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Those due ([`Batch::refresh_textures`]).
    due: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// With [`parallel_refresh`]: textures due a check, not checked yet
    /// ([`Batch::check_textures`], before any write of ours and at every
    /// flush).
    due_textures: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// With [`near_writes`]: the title memory (address, bytes) of the
    /// staging buffers in `textures`, by the same key. A texture without
    /// an entry is checked again after any write of ours.
    texture_spans: std::collections::HashMap<usize, (usize, usize)>,
    /// What copies at their place in the submission's lists
    /// ([`Step::Placed`]) left for the landing: the destination's host
    /// copy, where and the bytes.
    placed: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, Vec<u8>)>,
    /// Fills not done yet (destination, source, bytes): [`wide_fills`].
    fills: Vec<(usize, usize, usize)>,
    /// Copies (by address) brought up to date ahead of the steps, and the
    /// range (from, to) ([`Batch::refresh_ahead`]).
    ahead: ByCopy<(usize, usize)>,
    /// View heaps the submission's draws index directly, and whether their
    /// shaders may write memory.
    bindless: Vec<(std::sync::Arc<crate::descriptor::host::HostHeap>, bool)>,
    #[allow(dead_code)]
    /// Their plans, which hold the host resources the slots view until the
    /// submission has run.
    plans: Vec<std::sync::Arc<crate::descriptor::host::BindlessPlan>>,
    /// Bindless unordered access ranges, copied back after the submission.
    bindless_writes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
"""
# (their refresh-front machinery fns come after the fields in their hunk)
C[80] = C[80] + T(80)[T(80).index("/// Whether the fills"):]
# h81: 'both' — ours' `current` early-out + theirs' ahead early-out.
C[81] = 'both'
# h82: union flush_one tail.
C[82] = """        let started = std::time::Instant::now();
        for &(from, to) in spans.as_slice() {
            if !(crate::watch::light() && shadow.nothing_new(from, to - from)) {
                Self::update(&shadow, from, to);
            }
            Self::brought(&mut self.current, &shadow, from, to);
        }
        let ns = started.elapsed().as_nanos() as u64;
        SPLIT[6].fetch_add(ns, Ordering::Relaxed);
        SPLIT[9].fetch_add(ns, Ordering::Relaxed);
        SPLIT[10].fetch_add(1, Ordering::Relaxed);
    }

    /// With [`refresh_once`]: `[from, to)` of `shadow`'s copy was brought
    /// up to date. Kept: its union with what was up to date if they touch,
    /// else the larger of the two.
    fn brought(
        current: &mut std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        from: usize,
        to: usize,
    ) {
        if !refresh_once() || from >= to {
            return;
        }
        let entry = current
            .entry(std::sync::Arc::as_ptr(shadow) as usize)
            .or_insert_with(|| (shadow.clone(), from, to));
        if from <= entry.2 && entry.1 <= to {
            (entry.1, entry.2) = (entry.1.min(from), entry.2.max(to));
        } else if to - from > entry.2 - entry.1 {
            (entry.1, entry.2) = (from, to);
        }
"""
# h83: 'both' — check_textures + refresh_textures.
C[83] = 'both'
# h84: 'both' — ours' unwatched filter + theirs' asked_ahead filter.
C[84] = 'both'
# h85: 'both' — ours' jobs/wide_run + theirs' FLUSHES/light path.
C[85] = 'both'
# h86: union — ours' plaincopy + theirs' watch::changes/again_after_change.
C[86] = O(86) + T(86)[:T(86).index("    // (A title destination")]
#    (above keeps ours' whole + theirs' `since`; re-derive whole properly)
C[86] = """    let since = crate::watch::changes();
    // Into host memory: in place first, if asked for (`crate::plaincopy`);
    // through the kernel if that is off or met a page that was not there.
    let whole = (!title_dst && crate::plaincopy::copy(dst, src, len)) || read(src, dst, len) || title_dst && crate::watch::again_after_change(since, || read(src, dst, len));
"""
# h87: 'both' — ours' victory_safe_copy asm + theirs' zero().
C[87] = 'both'
# h88: 'both' — ours' try_user_copy fast path + theirs' soft-views read.
C[88] = 'both'
# h89: 'both' — backreads::touched + watch::materialize.
C[89] = 'both'
# h90: union — ours' try_user_copy + theirs' safecopy path.
C[90] = 'both'
# h91: union — ours' E_OUTOFMEMORY sweep + theirs' residency log.
C[91] = """                if e.code().0 == crate::object::E_OUTOFMEMORY {
                    request_memory_sweep();
                }
                crate::residency::allocation_error(&e);
"""
# h92: 'both_rev' — theirs' composer/panel/notice/menu fields then ours' rtv+impl.
C[92] = 'both_rev'
# h93: 'both' — ours' SPREAD/HEAP_PASS_PARTS + theirs' merge::line.
C[93] = 'both'
# h94: 'both' — ours' log lines + theirs' two_queues/spans lines.
C[94] = 'both'
# h95: 'theirs' (superset format string + rehash_runs/wc_left args).
C[95] = 'theirs'
# h96: 'both' — ours' RECHECKED + 5-slot destructure, then theirs' 6-slot
# (shadowing; the shared tracing uses `again`).
C[96] = 'both'
# h97: 'both' — ours' SWEPT census + theirs' room/heap-pass logs.
C[97] = 'both'
# h98: union format string.
C[98] = """                 cpu {cpu_pct:.1}% gpu {gpu_pct:.0}% busy ram {ram_mb:.0} MB; \
                 raw view clamp (past end, shortened, sub-dword, unaligned) since start: SRV {raw_srv:?}, UAV {raw_uav:?}",
"""
# h99: union — ours' rewrites/flips/census/copystat + theirs' PUT/traffic.
C[99] = O(99) + T(99)[:T(99).index("                traffic::line")]
C[99] = O(99) + """                let put = [0, 1, 2, 3].map(|i| crate::shadow::PUT[i].swap(0, Ordering::Relaxed) as f64 / frames);
                tracing::info!(
                    "what shaders wrote put into title memory per frame (VICTORY_EXACT_BACK): {:.0} pieces in {:.0} writes, {:.1} MB; {:.0} pieces joined over gaps",
                    put[0],
                    put[1],
                    put[2] / 1e6,
                    put[3]
                );
                traffic::line(self.copy_stats_frames as f64);
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
"""
# h100: union — theirs' caption reading + framegen rate + ours' figures text
# with VIctory branding.
C[100] = T(100).replace('VIctory-NVIDIA', 'VIctory')
# h101: 'both' — ours' present_now + theirs' present_planes.
C[101] = 'both'
# h102: theirs' pre-barrier + CopyResource + ours' fpsbox::draw.
C[102] = """                ));
                list.ResourceBarrier(&pre);
                list.CopyResource(&back, texture);
                // The frame rate, into the corner of what is shown (not of the
                // title's own frame, which the lines below read back).
                crate::fpsbox::draw(self, list, &back, (width, height, format));
            }
"""
# h103: 'theirs' (composed/post barriers + fsr::before_present).
C[103] = 'theirs'
# h104: union — theirs' streamline/panel Present + ours' error reporting.
C[104] = """            #[cfg(feature = "diagnostics")]
            crate::replay::record::present(texture, state);
            let mut showing = crate::timeline::span("the host's Present");
            // Frame generation: the frame's markers, constants and tags.
            let presenting = crate::streamline::present_begins(
                frame,
                crate::streamline::Planes {
                    base: (texture, state_u32),
                    overlay: overlay.filter(|_| composed),
                    size: [width, height],
                },
            );
            let called = crate::streamline::present_called(&presenting);
            // What Present answers was thrown away: a removal it is the
            // first to report then looked as if the next call found it.
            // Said once (a status that is no error, as "occluded", is not).
            let shown = p.swap_chain.Present(0, DXGI_PRESENT(0));
            if shown.is_err() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::error!("Present answered {:#010x} (said once): the first call to report it, frame {frame}", shown.0 as u32);
                });
            }
            crate::streamline::present_returned(called);
            crate::streamline::present_ends(presenting);
            showing.next("the frame meter");
            p.panel.frame(p.meter.line_due());
            p.menu.frame();
"""
# h105: 'both' — ours' present_over + theirs' drop_presenter.
C[105] = 'both'
# h106: 'both' — ours' `rtv,` + theirs' composer/panel/notice/menu inits.
C[106] = 'both'
# h107: union — ours' E_OUTOFMEMORY + theirs' warn text + residency error.
C[107] = """                if e.code().0 == crate::object::E_OUTOFMEMORY {
                    request_memory_sweep();
                }
                crate::residency::allocation_error(&e);
                tracing::warn!(error = %e, size, "no buffer in card memory the CPU maps");
"""
# h108: ours' `mod batch;` + theirs' `mod refresh_tests` (their inline
# `mod batch` duplicates host/batch.rs; its unique test is ported there).
t108 = T(108)
i = t108.index("mod batch {")
C[108] = O(108) + "\n\n" + t108[:i]
# h109: 'ours' (wait_if_removed; theirs' tests are in host/batch.rs already).

choices = []
for i in range(110):
    choices.append(C.get(i, 'ours'))

path = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-gpu/src/host.rs'
resolve(path, choices)
