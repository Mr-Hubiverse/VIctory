//! Submission of command lists: the card work each list becomes, the
//! relay of its copies back, and the waits between them. A child
//! module so it shares the host's state.

use super::*;

impl Host {
    pub(crate) fn shader_model_6_6(&self) -> bool {
        static SUPPORTED: OnceLock<bool> = OnceLock::new();
        *SUPPORTED.get_or_init(|| {
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: writes the feature's structure of the given size.
            let query = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut model).cast(),
                    size_of_val(&model) as u32,
                )
            };
            query.is_ok() && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0
        })
    }

    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            // SAFETY: writes the feature's structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && self.shader_model_6_6();
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    /// The wave sizes the host runs (D3D12 options 1), which translated
    /// compute shaders that declare the console's wave size must fit;
    /// logged once. Unknown (the query fails): any.
    pub(crate) fn target(&self) -> victory_shader::emit::Target {
        static TARGET: OnceLock<victory_shader::emit::Target> = OnceLock::new();
        *TARGET.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes the structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let target = if ok && options.WaveOps.as_bool() && options.WaveLaneCountMin > 0 {
                victory_shader::emit::Target {
                    lanes_min: options.WaveLaneCountMin,
                    lanes_max: options.WaveLaneCountMax,
                }
            } else {
                victory_shader::emit::Target::ANY
            };
            tracing::info!(
                lanes_min = target.lanes_min,
                lanes_max = target.lanes_max,
                "host wave sizes"
            );
            if !target.runs(64) {
                tracing::warn!(
                    "the host has no 64-lane waves: console wave64 compute programs that read \
                     other lanes do not translate, and their dispatches are dropped"
                );
            }
            target
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    pub(super) fn make_prelude(&self) -> Option<Prelude> {
        make_prelude_on(&self.device)
    }

    pub(super) fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = OnceLock::new();
            if !lazy_prelude() {
                let _ = prelude.set(self.make_prelude());
            }
            Ok(HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: UnsafeCell::new(Deferred::default()),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                prelude,
                pipelines: Mutex::default(),
                queries: Mutex::default(),
                budget: OnceLock::new(),
                card_budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                window_check: OnceLock::new(),
                clamp_window: Mutex::new(None),
                open_queries: Mutex::new(Vec::new()),
                work: Mutex::default(),
                kept: Mutex::default(),
                last_fence: Default::default(),
                writes: Default::default(),
                clamp_chunks: Mutex::default(),
                _alive: Counted::new(),
                meter: AtomicU32::new(u32::MAX),
                pipeline_meter: AtomicU32::new(u32::MAX),
                gather: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
                #[cfg(feature = "diagnostics")]
                shot_block: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                shot_mark: Mutex::default(),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    pub(super) fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    pub(super) fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then)) = state.1.take() {
            card_clock_us(0, since.elapsed().as_micros() as u64);
            card_clock_us(1, worked().saturating_sub(worked_then));
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    pub(super) fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            let source = card_compare_source();
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            // With `alt`: the wide form's pipeline beside it (`pso` is then the thread-a-piece form, as
            // `card_compare_source` gives it without `=1`). None if it does not build: every submission keeps `pso`.
            let pso_wide = (card_hash_mode() == 2).then(|| self.make_card_compare_wide(&root)).flatten();
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                pso_wide,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                inflight: if inflight_fetch() { self.make_card_inflight(RESULTS) } else { None },
            })
        }
    }

    /// [`CARD_COMPARE_WIDE`]'s pipeline with the compare's root signature `root` (the two shaders declare the same
    /// one), for `VICTORY_CARD_HASH_WIDE=alt`; `None`, said in the log, if it cannot be made.
    pub(super) fn make_card_compare_wide(&self, root: &ID3D12RootSignature) -> Option<ID3D12PipelineState> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let made = (|| -> windows::core::Result<ID3D12PipelineState> {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; the out pointer is ours.
            unsafe {
                D3DCompile(
                    CARD_COMPARE_WIDE.as_ptr().cast(),
                    CARD_COMPARE_WIDE.len(),
                    windows::core::s!("card compare wide"),
                    None,
                    None,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )?;
            }
            let code = code.ok_or_else(windows::core::Error::empty)?;
            // SAFETY: the blob's own bytes; a root signature the shader declares as well.
            unsafe {
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
            }
        })();
        match made {
            Ok(pso) => Some(pso),
            Err(e) => {
                tracing::warn!(error = %e, "VICTORY_CARD_HASH_WIDE=alt: the wide compare could not be made; a thread a piece throughout");
                None
            }
        }
    }

    /// [`CardInflight`] for `results` bytes of the compare's results; `None`
    /// (and the old fetch) if the shader or the memory cannot be had.
    pub(super) fn make_card_inflight(&self, results: u64) -> Option<CardInflight> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let made = (|| -> windows::core::Result<CardInflight> {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; the out
            // pointer is ours.
            unsafe {
                D3DCompile(
                    CARD_FETCH.as_ptr().cast(),
                    CARD_FETCH.len(),
                    windows::core::s!("card fetch in flight"),
                    None,
                    None,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )?;
            }
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let slots = inflight_slots(inflight_fetch_bytes());
            let (table, table_data) = self.shared_buffer(4 + results).ok_or_else(windows::core::Error::empty)?;
            name(&table, || "card fetch in flight: tickets".into());
            let fetched_bytes = u64::from(slots) * INFLIGHT_SLOT;
            let (fetched, fetched_data) = self.shared_buffer(fetched_bytes).ok_or_else(windows::core::Error::empty)?;
            name(&fetched, || "card fetch in flight: pieces".into());
            // SAFETY: the mapping holds the counter; nothing of ours ran
            // on the card with it yet.
            unsafe { table_data.cast::<u32>().write_volatile(0) };
            // SAFETY: the blob's own bytes; the root signature is inside
            // the shader.
            unsafe {
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
                let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })?;
                Ok(CardInflight {
                    root,
                    pso,
                    table: (table, table_data as usize, 4 + results),
                    fetched: (fetched, fetched_data as usize, slots),
                    landed: 0,
                    brought: Vec::new(),
                })
            }
        })();
        match made {
            Ok(inflight) => {
                tracing::info!(
                    slots = inflight.fetched.2,
                    "card copies' flagged pieces come down inside their flights (VICTORY_CARD_FETCH_INFLIGHT=1)"
                );
                Some(inflight)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no fetch inside flights (VICTORY_CARD_FETCH_INFLIGHT=1): pieces are fetched the old way");
                None
            }
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`).
    ///
    /// `sent`: the last value signalled on the host's fence; the caller
    /// signals the next one behind this submission's lists.
    ///
    /// With `VICTORY_QUEUE2` bit 2 (`crate::queue2`; off by default) the
    /// changed bytes cross from the PC's memory to the card's on a copy
    /// queue, on the card's copy engine, while the direct queue still runs
    /// what was sent before: they go into a landing buffer in the card's
    /// memory that nothing else uses. The direct queue is told to wait for
    /// that copy and then moves the bytes from the landing buffer into the
    /// copies, inside the card's memory, in the same place of its order as
    /// before. So the copies change exactly when they did on one queue; what
    /// leaves the direct queue is the crossing from the PC's memory.
    pub(super) fn card_before(
        &self,
        runs: &[&HostList],
        flown: bool,
        sent: u64,
        bindless: &[(Arc<crate::shadow::Shadow>, usize, usize)],
    ) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        // The ranges this submission's shaders may write, per copy.
        let mut declared: Vec<Declared> = Vec::new();
        let indexed = step_index();
        if indexed {
            let given = given_to_shaders(runs, true);
            declared = declared_by_index(given.iter().map(|(shadow, range)| (shadow, *range)));
        }
        for list in if indexed { &[][..] } else { runs } {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len, marked) = step
                    && shadow.carded()
                {
                    // As []: a marked declaration holds the
                    // whole copy.
                    let range = if *marked { shadow.title_span() } else { (*address, *address + *len) };
                    match declared.iter_mut().find(|(s, _, _)| Arc::ptr_eq(s, shadow)) {
                        Some((_, ranges, m)) => {
                            *m |= *marked;
                            if !ranges.contains(&range) {
                                ranges.push(range);
                            }
                        }
                        None => declared.push((shadow.clone(), vec![range], *marked)),
                    }
                }
            }
        }
        for (shadow, address, len) in bindless {
            if !shadow.carded() {
                continue;
            }
            let range = (*address, *address + *len);
            match declared.iter_mut().find(|(s, _, _)| Arc::ptr_eq(s, shadow)) {
                Some((_, ranges, _)) => {
                    if !ranges.contains(&range) {
                        ranges.push(range);
                    }
                }
                None => declared.push((shadow.clone(), vec![range], false)),
            }
        }
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        let mut dirty = crate::shadow::card_dirty(sent_as);
        // With fetches on a copy queue (`crate::queue2`, bit 1 of
        // `VICTORY_QUEUE2`): the send's number is signalled on the direct
        // queue behind the send, so that a fetch knows which sends the card
        // has finished (`Second::order_done`).
        let numbered = || {
            if sent_as != 0
                && let Some(second) = crate::queue2::second().filter(|s| s.fetches())
            {
                second.sent_up(&self.queue, sent_as);
            }
        };
        if declared.is_empty() && dirty.is_empty() {
            numbered();
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // The hash form of this submission's passes, one for all of them (`card_hash_wide_now`).
        let wide_now = card_hash_wide_now();
        let (compare_pso, wide) = {
            let (pso, wide) = work.compare_for(wide_now);
            (pso.clone(), wide)
        };
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, ..) in &declared {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        // The changed bytes through the copy queue (`VICTORY_QUEUE2` bit
        // 2): the landing buffer they are in, in the order of `dirty`'s
        // ranges, once the direct queue has waited. `None`: straight from
        // the PC's memory on the direct queue, as before.
        let mut landed: Option<ID3D12Resource> = None;
        if let Some(second) = crate::queue2::second().filter(|s| s.uploads()) {
            let bytes: u64 = dirty
                .iter()
                .filter(|(shadow, _)| shadow.carded())
                .flat_map(|(_, ranges)| ranges.iter().map(|&(a, b)| (b - a) as u64))
                .sum();
            // SAFETY: a plain query.
            let completed = unsafe { self.fence.GetCompletedValue() };
            // Held until the card has passed this submission's signal: the
            // direct queue's copy out of it runs ahead of that.
            if bytes > 0
                && let Some(landing) = second.landing(bytes, completed, sent + 1)
            {
                // SAFETY: an open copy list of the lane's; live buffers,
                // and the ranges lie in them; the host's queue.
                let sent_up = unsafe {
                    (|| -> windows::core::Result<()> {
                        let copy = second.up.list()?;
                        let mut at = 0u64;
                        for (shadow, ranges) in &dirty {
                            let Some((staging, _, _)) = shadow.card_sides() else {
                                continue;
                            };
                            for &(a, b) in ranges {
                                copy.1.CopyBufferRegion(&landing, at, staging, a as u64, (b - a) as u64);
                                at += (b - a) as u64;
                            }
                        }
                        let job = second.up.send(copy, &[], bytes)?;
                        // Nothing sent to the direct queue from here on
                        // runs before the bytes have landed.
                        second.hold(&self.queue, job)
                    })()
                };
                if let Err(e) = sent_up {
                    fail(format_args!(
                        "host copies in the card's memory could not be brought up to date through the copy queue ({e}, VICTORY_QUEUE2)"
                    ));
                }
                landed = Some(landing);
            }
        }
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Sync);
                    #[cfg(feature = "diagnostics")]
                    let shot_span = crate::shot::span_begin(list);
                    // Where the next range lies in the landing buffer.
                    let mut at = 0u64;
                    // In three steps over all the copies ([`sync_batched`]):
                    // every changed range copied up, then every state
                    // change in one call, then every hash pass.
                    struct Synced<'a> {
                        shadow: &'a Arc<crate::shadow::Shadow>,
                        ranges: &'a [(usize, usize)],
                        before: Option<ID3D12Resource>,
                        size: u64,
                    }
                    let batched = sync_batched();
                    let mut synced: Vec<Synced> = Vec::with_capacity(dirty.len());
                    // No twin (it cost the card the copy's size again, 399 MB
                    // in run 75): `before` holds a 128-bit hash of each 64 KiB
                    // piece. The pieces just sent up are hashed again, all of
                    // them the first time; the compare after the lists hashes
                    // the declared pieces and flags those whose hash is another.
                    let hashes = |list: &ID3D12GraphicsCommandList, group: &[Synced]| {
                        // (a copy promoted the buffer to a copy destination)
                        let moves: Vec<_> = group
                            .iter()
                            .filter(|w| w.before.is_some() && !w.ranges.is_empty())
                            .map(|w| transition(&w.shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE))
                            .collect();
                        if !moves.is_empty() {
                            list.ResourceBarrier(&moves);
                        }
                        let mut bound = false;
                        for w in group {
                            let Some(before) = &w.before else {
                                continue;
                            };
                            if !bound {
                                bound = true;
                                list.SetComputeRootSignature(&work.root);
                                list.SetPipelineState(&compare_pso);
                                list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                            }
                            list.SetComputeRootShaderResourceView(1, w.shadow.buffer.GetGPUVirtualAddress());
                            list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                            let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, w.shadow)) {
                                vec![w.shadow.card_run(0, usize::MAX)]
                            } else {
                                w.ranges.iter().map(|&(a, b)| w.shadow.card_run(a, b)).collect()
                            };
                            // The same pieces in fewer passes (`rehash_runs`).
                            let runs = if rehash_runs() { crate::shadow::joined_piece_runs(runs) } else { runs };
                            CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                            CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                            for run in runs {
                                let constants = [run.first, run.skew, w.shadow.card_hashed().unwrap_or(w.size) as u32, run.pieces, 0, 0];
                                list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                                list.Dispatch(card_compare_groups_as(wide, run.pieces), 1, 1);
                            }
                        }
                    };
                    for (shadow, ranges) in &dirty {
                        let Some((staging, before, size)) = shadow.card_sides() else {
                            continue;
                        };
                        for &(a, b) in ranges {
                            match &landed {
                                Some(landing) => list.CopyBufferRegion(&shadow.buffer, a as u64, landing, at, (b - a) as u64),
                                None => list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64),
                            }
                            at += (b - a) as u64;
                            crate::shadow::CARD_STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                        }
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        synced.push(Synced { shadow, ranges, before, size });
                        if !batched {
                            // As before: this copy's hash passes behind its copies.
                            hashes(list, &synced[synced.len() - 1..]);
                        }
                    }
                    if batched {
                        hashes(list, &synced);
                    }
                    crate::cardmeter::end(list, timed);
                    #[cfg(feature = "diagnostics")]
                    crate::shot::span_end(list, shot_span, 0, || "SYNC LIST (changed bytes go up to the card's copies and are hashed)".to_string());
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!("host copies in the card's memory could not be brought up to date ({e})"));
            }
            used.push((false, sync));
        }
        numbered();
        drop(order);
        let mut plan = CardPlan {
            ranges: Vec::new(),
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            sent_as,
            inflight: false,
            inflight_landed: 0,
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        for (shadow, ranges, _) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Check);
                    #[cfg(feature = "diagnostics")]
                    let shot_span = crate::shot::span_begin(list);
                    // What the submission's shaders wrote, first.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&compare_pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let Some((_, Some(before), size)) = r.shadow.card_sides() else {
                            continue;
                        };
                        let hashed = r.shadow.card_hashed().unwrap_or(size) as u32;
                        let constants = [r.span.first, r.span.skew, hashed, r.span.pieces, r.at * 4, if flown { 2 } else { 1 }];
                        list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                        list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.Dispatch(card_compare_groups_as(wide, r.span.pieces), 1, 1);
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    // The fetch inside the flight (`inflight_fetch`): behind
                    // every compare of the list, the pieces they flagged go
                    // to slots in the PC's memory.
                    if flown && let Some(inflight) = &work.inflight {
                        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                        }]);
                        list.SetComputeRootSignature(&inflight.root);
                        list.SetPipelineState(&inflight.pso);
                        list.SetComputeRootUnorderedAccessView(2, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, inflight.table.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(4, inflight.fetched.0.GetGPUVirtualAddress());
                        let mut every = true;
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let Some((_, Some(_), size)) = r.shadow.card_sides() else {
                                every = false;
                                continue;
                            };
                            let groups = r.span.pieces.clamp(1, 65535);
                            let constants =
                                [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, inflight.landed, inflight.fetched.2, groups];
                            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            list.Dispatch(groups, 1, 1);
                        }
                        plan.inflight = every;
                        plan.inflight_landed = inflight.landed;
                    }
                    crate::cardmeter::end(list, timed);
                    #[cfg(feature = "diagnostics")]
                    crate::shot::span_end(list, shot_span, 0, || "CHECK LIST (the card compares the pieces shaders may have written)".to_string());
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    pub(super) fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    pub(super) fn card_done(&self, plan: CardPlan) {
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    pub(super) fn make_relay_fetch(&self) -> Option<RelayFetch> {
        let list = self.make_prelude()?;
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                fence,
                event,
                sent: 0,
                buffer: None,
            })
        }
    }

    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_pieces`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up.
    ///
    /// With [`inflight_fetch`] the pieces are first looked for where the
    /// flights' own compare lists put them ([`Host::relay_card_inflight`]);
    /// only a landing that cannot take them from there fetches, as here.
    /// `behind`: the value signalled on the host's fence behind the last of
    /// the flights. With `VICTORY_QUEUE2` bit 1 (`crate::queue2`) the fetch
    /// goes to a copy queue, behind that value and nothing else: it does
    /// not stand behind what the submitting thread sent meanwhile (run
    /// 118: 114-157 ms a frame of this thread's time). It then reads the
    /// card's side while later flights may be writing it: a byte is the
    /// old one or the new one, and the flight that writes it has a compare
    /// and a landing of its own behind it.
    ///
    /// `plans` come with their flights' numbers and `landed` is the number
    /// of the last flight being landed: a piece an earlier fetch on the
    /// copy queue brought down beside one of these flights is fetched
    /// whatever the compare says ([`BESIDE`]).
    pub(super) fn relay_card_back(&self, plans: &[(u64, &CardPlan)], fetch: &mut Option<RelayFetch>, behind: u64, landed: u64, owners: &crate::writeback::Owners) {
        if inflight_fetch() {
            let only: Vec<&CardPlan> = plans.iter().map(|p| p.1).collect();
            let taken = self.relay_card_inflight(&only, owners);
            inflight_tickets_landed(&only);
            if taken {
                return;
            }
        }
        let lane = crate::queue2::second().filter(|s| s.fetches());
        // The pieces fetched beside flights that had started, as noted by
        // earlier landings: looked at for these plans, renewed below.
        let beside = lane.is_some() && refetch();
        let noted: std::collections::HashMap<(usize, u32), u64> = if beside {
            BESIDE.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default()
        } else {
            Default::default()
        };
        let mut again = 0u64;
        // Every piece fetched now: (copy, piece number).
        let mut numbers: Vec<(usize, u32)> = Vec::new();
        let differs = {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
        };
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        if let Some((differs, dwords)) = differs {
            // SAFETY: the mapping of the results. The compares of these
            // plans have run, and their dwords are given to no other plan
            // before these are done with (`card_done`).
            let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
            let mut seen = std::collections::HashSet::new();
            for &(number, plan) in plans {
                let mut flags: Vec<(usize, u32, bool)> = plan
                    .ranges
                    .iter()
                    .flat_map(|r| {
                        (0..r.span.pieces).map(move |p| {
                            let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                            (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, flagged)
                        })
                    })
                    .collect();
                // A piece an earlier fetch brought down while this flight
                // may have been writing it: fetched, flagged or not.
                for flag in flags.iter_mut().filter(|f| !f.2) {
                    if fetched_beside(noted.get(&(flag.0, flag.1)).copied(), number) {
                        flag.2 = true;
                        again += 1;
                    }
                }
                let fetched = flagged_pieces(&flags);
                numbers.extend(fetched.iter().copied());
                for r in &plan.ranges {
                    let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        if !fetched.contains(&(copy, r.span.first + p)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo < hi && seen.insert((copy, lo, hi)) {
                            pieces.push((r.shadow.clone(), lo, hi, 0));
                        }
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                }
            }
        }
        BESIDE_STATS[0].fetch_add(again, Ordering::Relaxed);
        if pieces.is_empty() {
            if inflight_fetch() {
                // The next merge may be of a flight in the air now, with
                // bytes from when it ran: only sends up to these plans' are
                // older than it.
                if let Some(last) = plans.iter().map(|p| p.1.sent_as).max() {
                    crate::shadow::relay_fetched(last);
                }
                return;
            }
            if beside {
                // Notes whose flights have all landed are done with.
                let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
                note_beside(all.get_or_insert_with(Default::default), &[], landed, landed);
            }
            // Nothing to fetch: every send so far is older than the next
            // fetch (on a copy queue: every send the card has finished).
            crate::shadow::relay_fetched(match lane {
                Some(second) => second.order_done(),
                None => *QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()),
            });
            return;
        }
        // Fetch overlapping bytes once, but keep each original merge span and
        // its order: a larger CPU merge could cross an unreadable guest page.
        let mut order: Vec<_> = (0..pieces.len()).collect();
        order.sort_unstable_by_key(|&i| (std::sync::Arc::as_ptr(&pieces[i].0) as usize, pieces[i].1, pieces[i].2));
        let mut fetches: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for i in order {
            let (shadow, lo, hi, at) = &mut pieces[i];
            if let Some((last, from, to, out)) = fetches.last_mut()
                && std::sync::Arc::ptr_eq(last, shadow) && *lo <= *to
            {
                *at = *out + (*lo - *from) as u64;
                total += hi.saturating_sub(*to) as u64;
                *to = (*to).max(*hi);
            } else {
                *at = total;
                total += (*hi - *lo) as u64;
                fetches.push((shadow.clone(), *lo, *hi, *at));
            }
        }
        if inflight_fetch() {
            INFLIGHT_STATS[3].fetch_add(1, Ordering::Relaxed);
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY=1)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let waiting = std::time::Instant::now();
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (the copies are held by the plans); the fence and the
        // event are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                if let Some(second) = lane {
                    // The sends of title bytes the card has finished are in
                    // what comes down. A later one may be too, if the card
                    // runs it before this fetch: the merge leaves its bytes
                    // out as a send that came after, and they are the
                    // title's either way.
                    let after = second.order_done();
                    let copy = second.down.list()?;
                    for (shadow, lo, hi, at) in &fetches {
                        copy.1.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                    }
                    let job = second.down.send(copy, &[(&self.fence, behind)], total)?;
                    second.down.wait(job, event)?;
                    if beside {
                        // Whatever had started by now may have been
                        // writing these pieces while they came down
                        // (counted before a submission's lists go to the
                        // queue, `Host::submit`: nothing later ran yet).
                        let started = STARTED.load(Ordering::Acquire);
                        let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
                        let noted = note_beside(all.get_or_insert_with(Default::default), &numbers, landed, started);
                        BESIDE_STATS[1].fetch_add(noted, Ordering::Relaxed);
                    }
                    second.down.stats.waited_us.fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                    return Ok(after);
                }
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                copy_fetch_runs(&list, &buffer, fetches.iter().map(|(shadow, lo, hi, at)| (&shadow.buffer, *lo, *hi, *at)));
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                fence.SetEventOnCompletion(sent, event)?;
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY=1)")),
        };
        // This fetch may bring bytes of flights in the air now: their own
        // slots are older than what it puts in the CPU's side there
        // (`relay_card_inflight`). Sends up to `after` are numbered, and the
        // submission numbered `after` may not have its lists on the queue
        // yet; every one before it has (the submission lock).
        if inflight_fetch()
            && after > 1
            && let Some(inflight) = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|w| w.inflight.as_mut())
        {
            inflight.brought.extend(pieces.iter().map(|(shadow, lo, hi, _)| (shadow.clone(), *lo, *hi, after - 1)));
        }
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(fetches.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        }
        let mut title = Vec::new();
        let mut written = 0;
        for (shadow, lo, hi, at) in &pieces {
            // SAFETY: the mapping of the fetched bytes, `total` of them;
            // the copy into it has run.
            let gpu =
                unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
            let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title, owners);
            shadow.walls_merged(gpu.len(), merged);
            written += merged;
        }
        copy_back(1, written as u64);
        // Sends up to `after` are of no use to any later merge; with the
        // fetch inside flights only those up to these plans' are
        // (`fallback_fetched_mark`).
        let landed = plans.iter().map(|p| p.1.sent_as).max();
        crate::shadow::relay_fetched(fallback_fetched_mark(inflight_fetch(), after, landed));
    }
    /// [`Host::relay_card_back`] with [`inflight_fetch`]: the pieces these
    /// plans' compares flagged are taken from the slots their own compare
    /// lists copied them to ([`CARD_FETCH`]) and merged, plan by plan in the
    /// line's order, with no list or wait of ours. `false` (nothing merged)
    /// if any plan cannot land so; the caller then lands them all the old
    /// way, as without the switch.
    ///
    /// The merge's "fetched after" (`Shadow::card_merge_relay`) is the
    /// plan's own `sent_as`. `QUEUE_ORDER` is raised only in
    /// `Host::card_before`, under the submission lock, and that lock is held
    /// until the submission's lists and its compare's list are on the queue
    /// behind its sync list: so when the compare's list ran, the card had
    /// received exactly the sends 1..=`sent_as` of title bytes and no later
    /// one, which is what the merge's skip of later sends asks. (The old
    /// fetch, by contrast, ran whenever it reached the queue, and took the
    /// count at its own ExecuteCommandLists.)
    ///
    /// When a landing goes the old way instead, all of it does: a fetch the
    /// old way brings the card's bytes as they are now, newer than the slots
    /// of plans in the air then, and merging an older slot after it would
    /// put a flight's bytes back over a later flight's. For the same reason
    /// the bytes such a fetch brought (`CardInflight::brought`) are left
    /// out of the slots of flights whose lists were on the queue ahead of
    /// it: the CPU's side holds what the card held after them there. The
    /// rest of their slots is merged, newer than anything merged there
    /// before. (Left out whole, a flight sent before such a fetch would have
    /// had to land the old way, its fetch taking a later count still: one
    /// overflow kept every landing on the old way while flights were in
    /// the air.) Plans land in the line's order, each plan's slot as it was
    /// right behind its own lists, so every merge here is of bytes at least
    /// as new as the CPU's side holds.
    ///
    /// shortcut: all or nothing per landing, and a plan whose pieces overflow
    /// the slots lands the old way for all its pieces, not only the ones
    /// left out.
    pub(super) fn relay_card_inflight(&self, plans: &[&CardPlan], owners: &crate::writeback::Owners) -> bool {
        use std::sync::Arc;
        type Piece = (Arc<crate::shadow::Shadow>, usize, usize, usize, u64);
        let mut pieces: Vec<Piece> = Vec::new();
        {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            let Some(work) = work.as_ref() else {
                return false;
            };
            let Some(inflight) = &work.inflight else {
                return false;
            };
            let dwords = (work.differs.2 / 4) as usize;
            // SAFETY: the mappings of the results and the tickets. The
            // compares and fetches of these plans have run, and their dwords
            // are given to no other plan before these are done with
            // (`card_done`).
            let (differs, table) = unsafe {
                (
                    std::slice::from_raw_parts(work.differs.1 as *const u32, dwords),
                    std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords),
                )
            };
            let slots = inflight.fetched.2;
            for plan in plans {
                if plan.ranges.is_empty() {
                    continue;
                }
                if !plan.inflight || plan.ranges.iter().any(|r| !r.compared) {
                    return false;
                }
                // Pieces (copy, number) the compare flagged, with their ticket.
                let mut tickets: std::collections::HashMap<(usize, u32), u32> = std::collections::HashMap::new();
                let mut flags: Vec<(usize, u32, bool)> = Vec::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let at = (r.at + p) as usize;
                        let (Some(&flag), Some(&entry)) = (differs.get(at), table.get(at)) else {
                            return false;
                        };
                        if flag != 0 {
                            if entry == INFLIGHT_NONE {
                                return false;
                            }
                            if !inflight_gets_slot(entry, plan.inflight_landed, slots) {
                                INFLIGHT_STATS[2].fetch_add(1, Ordering::Relaxed);
                                return false;
                            }
                            tickets.insert((copy, r.span.first + p), entry);
                        }
                        flags.push((copy, r.span.first + p, flag != 0));
                    }
                }
                let fetched = flagged_pieces(&flags);
                let mut seen = std::collections::HashSet::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let piece = r.span.first + p;
                        if !fetched.contains(&(copy, piece)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo >= hi || !seen.insert((copy, lo, hi)) {
                            continue;
                        }
                        let from = inflight_piece_from(&r.span, p);
                        let Some(&k) = tickets.get(&(copy, piece)) else {
                            return false;
                        };
                        if lo < from || (hi - from) as u64 > INFLIGHT_SLOT {
                            return false;
                        }
                        let slot = inflight.fetched.1 + (k % slots) as usize * INFLIGHT_SLOT as usize;
                        // Bytes a fetch the old way brought after this flight ran.
                        let newer: Vec<(usize, usize)> = inflight
                            .brought
                            .iter()
                            .filter(|b| b.3 >= plan.sent_as && Arc::ptr_eq(&b.0, &r.shadow) && b.1 < hi && b.2 > lo)
                            .map(|b| (b.1, b.2))
                            .collect();
                        for (a, b) in crate::shadow::outside(lo, hi, newer) {
                            pieces.push((r.shadow.clone(), a, b, slot + (a - from), plan.sent_as));
                        }
                    }
                }
            }
        }
        // Outside the lock: no slot of these plans is written over before
        // their tickets are landed (`inflight_tickets_landed`, after this).
        let mut title = Vec::new();
        let (mut written, mut bytes) = (0, 0u64);
        for (shadow, lo, hi, at, after) in &pieces {
            // SAFETY: inside the mapping of the slots (checked above); the
            // copy into it ran in the plan's flight, which the card has run.
            let gpu = unsafe { std::slice::from_raw_parts(*at as *const u8, hi - lo) };
            written += shadow.card_merge_relay(*lo, gpu, *after, &mut title, owners);
            bytes += (hi - lo) as u64;
        }
        for plan in plans {
            for r in &plan.ranges {
                crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
            }
        }
        copy_back(1, written as u64);
        INFLIGHT_STATS[0].fetch_add(pieces.len() as u64, Ordering::Relaxed);
        INFLIGHT_STATS[1].fetch_add(bytes, Ordering::Relaxed);
        // The next merge is of a later plan, or a fetch the old way sent now.
        if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
            crate::shadow::relay_fetched(last);
        }
        true
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    pub(super) fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                fill_ahead::landing(flights.iter().map(|f| f.number).min().unwrap_or(0));
                let plans: Vec<(u64, &CardPlan)> = flights.iter().filter_map(|f| f.card.as_ref().map(|card| (f.number, card))).collect();
                let behind = flights.iter().map(|f| f.fence).max().unwrap_or(0);
                let landed = flights.iter().map(|f| f.number).max().unwrap_or(0);
                let owners = writeback_owners(flights.iter().copied());
                self.relay_card_back(&plans, fetch, behind, landed, &owners);
                let mut back = WriteGroups::new();
                let mut landed: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
                for flight in &flights {
                    for (shadow, ranges, marked) in &flight.back {
                        if *marked {
                            *landed.entry(std::sync::Arc::as_ptr(shadow) as usize).or_default() += 1;
                        }
                        back.add(shadow, ranges, *marked);
                    }
                }
                relay_hand_back(&back.groups, &owners, &landed);
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                for flight in &flights {
                    for (shadow, ranges, _) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                copy_back(0, landing.elapsed().as_micros() as u64);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        let mut last = 0;
        let mut landed = Vec::new();
        for item in items {
            match item {
                Late::Run(flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    landed.push((flight.lists, flight.order));
                }
                Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
            }
        }
        if last > 0 {
            crate::writeback::published(last);
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        self.log_messages();
        fill_ahead::landing(0);
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r, _)| r.iter().copied()));
                    ranges.extend(flight.texture_back_ranges.iter().copied());
                }
            }
            join_ranges(written, ranges);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    pub(super) fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
        owners: &crate::writeback::Owners,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (i, r) in plan.ranges.iter().enumerate() {
            for p in 0..r.span.pieces {
                if r.compared && differs[(r.at + p) as usize] == 0 {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi)) {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    // With `VICTORY_QUEUE2` bit 1 the fetch goes to the copy
                    // queue here too, behind everything sent to the direct
                    // queue. Nothing is gained (the card has run it all
                    // when this is reached); the path is the same one the
                    // relay uses, in the plainest run there is.
                    if let Some(second) = crate::queue2::second().filter(|s| s.fetches()) {
                        let copy = second.down.list()?;
                        for &(i, lo, hi, at) in &pieces {
                            copy.1.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                        }
                        let job = second.down.send(copy, &[(&self.fence, *submitted)], total)?;
                        let was_working = working(false);
                        second.down.wait(job, self.event)?;
                        stood(waiting);
                        working(was_working);
                        second.down.stats.waited_us.fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                        return Ok(());
                    }
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        let owners = owners.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe {
                                    std::slice::from_raw_parts(
                                        (fetched + at as usize) as *const u8,
                                        hi - lo,
                                    )
                                };
                                let merged = shadow.card_merge(lo, gpu, &mut title, &owners);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe {
                        std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo)
                    };
                    let merged = plan.ranges[i]
                        .shadow
                        .card_merge(lo, gpu, &mut title, owners);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            copy_back(1, written as u64);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // Behind what the queues were given before it and have not run
        // yet (`crate::queue::behind`).
        crate::queue::before_sync();
        let _ = self.submit(runs, false);
    }

    /// Runs title handles in bounded chunks without first materializing a
    /// second vector proportional to the whole queue submission.
    pub(crate) fn execute_handles(&self, handles: &[crate::object::This]) -> bool {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        if handles.is_empty() {
            if let Err((_, error)) = self.submit_locked(&[], false, submitted) {
                tracing::error!(error = %error, "could not reserve memory for an empty GPU submission");
                return false;
            }
            return true;
        }
        for range in submission_ranges(handles.len()) {
            let chunk = &handles[range];
            let mut runs = Vec::new();
            if let Err(error) = runs.try_reserve_exact(chunk.len()) {
                request_memory_sweep();
                tracing::error!(error = %error, lists = chunk.len(), "could not reserve bounded queue-list references");
                return false;
            }
            runs.extend(chunk.iter().filter_map(|&handle| crate::list::host_list(handle)));
            match self.submit_locked(&runs, false, submitted) {
                Ok((next, sent, _)) => {
                    debug_assert!(sent.is_none());
                    submitted = next;
                }
                Err((next, error)) => {
                    tracing::error!(error = %error, "could not reserve memory for a bounded GPU submission");
                    drop(next);
                    return false;
                }
            }
        }
        true
    }

    /// Runs detached lists in order, with only one bounded reference vector
    /// alive at a time.
    pub(crate) fn execute_detached(
        &self,
        lists: &[crate::list::Detached],
        order: &[usize],
    ) -> bool {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        if order.is_empty() {
            if let Err((_, error)) = self.submit_locked(&[], false, submitted) {
                tracing::error!(error = %error, "could not reserve memory for an empty GPU submission");
                return false;
            }
            return true;
        }
        for range in submission_ranges(order.len()) {
            let chunk = &order[range];
            let mut runs = Vec::new();
            if let Err(error) = runs.try_reserve_exact(chunk.len()) {
                request_memory_sweep();
                tracing::error!(error = %error, lists = chunk.len(), "could not reserve bounded detached-list references");
                return false;
            }
            runs.extend(chunk.iter().map(|&i| lists[i].host()));
            match self.submit_locked(&runs, false, submitted) {
                Ok((next, sent, _)) => {
                    debug_assert!(sent.is_none());
                    submitted = next;
                }
                Err((next, error)) => {
                    tracing::error!(error = %error, "could not reserve memory for a bounded GPU submission");
                    drop(next);
                    return false;
                }
            }
        }
        true
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, mut lists: Vec<crate::list::Detached>, mut order: Vec<usize>) {
        if order.is_empty() {
            crate::list::recycle(lists, &[]);
            return;
        }
        if order.len() > MAX_SUBMISSION_RUNS {
            // A single very large flight would need full-size dependency,
            // write-back and diagnostic vectors. Preserve queue order and
            // reclaim the detached lists, but wait between bounded chunks.
            let ran = self.execute_detached(&lists, &order);
            crate::list::recycle(lists, if ran { &order } else { &[] });
            return;
        }
        // Textures uploaded ahead (`crate::heap_worker`): their lists run
        // before the title's.
        let ahead = crate::heap_worker::take_ready();
        if !ahead.is_empty() {
            let first = lists.len();
            order.splice(0..0, first..first + ahead.len());
            lists.extend(ahead.into_iter().map(crate::list::Detached::ours));
        }
        late_threads();
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, true)
        };
        let Some((mut submitted, sent, already_run)) = sent else {
            // Allocation can still fail for a single-run submission after
            // every smaller workspace has been tried. Do not treat it as a
            // successful flight.
            crate::list::recycle(lists, &[]);
            return;
        };
        // Only the final segment still has read-backs to land in the flight.
        order.drain(..already_run);
        let told = !sent.writes.is_empty();
        let most = {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges, _)| ranges.iter().copied())
                    .chain(sent.texture_back_ranges.iter().copied()),
            );
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                texture_back_ranges: sent.texture_back_ranges,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            *flying >= late_most()
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            self.settle(&mut submitted, WHY_MOST);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        land();
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all.
    pub(super) fn execute_own(&self, list: &HostList) -> u64 {
        late_threads();
        let Some((submitted, sent, _)) = self.submit(&[list], true) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges, _)| ranges.iter().copied())
                    .chain(sent.texture_back_ranges.iter().copied()),
            );
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                texture_back_ranges: sent.texture_back_ranges,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    pub(super) fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let was_working = working(true);
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                // SAFETY: the fence and event are ours, under the
                // submission lock; `submitted` is the last value signalled.
                unsafe {
                    match self.fence.SetEventOnCompletion(*submitted, self.event) {
                        Ok(()) => {
                            working(false);
                            WaitForSingleObject(self.event, INFINITE);
                            stood(waiting);
                            working(true);
                        }
                        Err(e) => tracing::warn!(error = %e, "submissions sent ahead not waited for"),
                    }
                }
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                submission_us(1, gpu.as_micros() as u64);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                let owners = writeback_owners(items.iter().filter_map(|item| match item {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                }));
                let mut seen = std::collections::HashSet::new();
                let mut back = WriteGroups::new();
                let mut landed: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen, &owners);
                    }
                    for (shadow, ranges, marked) in &flight.back {
                        if *marked {
                            *landed.entry(std::sync::Arc::as_ptr(shadow) as usize).or_default() += 1;
                        }
                        back.add(shadow, ranges, *marked);
                    }
                }
                hand_back(&back.groups, &owners, &landed);
                copy_back(0, copying.elapsed().as_micros() as u64);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
                }
            }
            if last > 0 {
                crate::writeback::published(last);
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    pub(super) fn submit<'a>(
        &'a self,
        runs: &[&HostList],
        late: bool,
    ) -> Option<(std::sync::MutexGuard<'a, u64>, Sent, usize)> {
        let submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        match self.submit_locked(runs, late, submitted) {
            Ok((submitted, Some(sent), already_run)) => Some((submitted, sent, already_run)),
            Ok((_, None, _)) => None,
            Err((_, error)) => {
                tracing::error!(error = %error, lists = runs.len(), "could not reserve memory for a bounded GPU submission");
                None
            }
        }
    }

    /// Submits while keeping the caller's submission lock across adjacent
    /// bounded chunks, so another queue cannot be inserted between them.
    pub(super) fn submit_locked<'a>(
        &'a self,
        mut runs: &[&HostList],
        late: bool,
        mut submitted: std::sync::MutexGuard<'a, u64>,
    ) -> Result<
        (std::sync::MutexGuard<'a, u64>, Option<Sent>, usize),
        (std::sync::MutexGuard<'a, u64>, std::collections::TryReserveError),
    > {
        #[cfg(feature = "diagnostics")]
        let mut called = Instant::now();
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut already_run = 0;
        while let Some(cut) = texture_dependency_cut(runs) {
            let (next, sent, ran) = self.submit_segment(
                &runs[..cut],
                false,
                submitted,
                #[cfg(feature = "diagnostics")]
                called,
            )?;
            debug_assert!(sent.is_none());
            submitted = next;
            already_run += ran;
            runs = &runs[cut..];
            #[cfg(feature = "diagnostics")]
            { called = Instant::now(); }
        }
        let (submitted, sent, ran) = self.submit_segment(
            runs,
            late,
            submitted,
            #[cfg(feature = "diagnostics")]
            called,
        )?;
        already_run += ran;
        Ok((submitted, sent, already_run))
    }

    /// Submits a dependency-safe segment in bounded chunks. Only its final
    /// chunk may be sent ahead; earlier chunks are waited for in queue order.
    pub(super) fn submit_segment<'a>(
        &'a self,
        runs: &[&HostList],
        late: bool,
        mut submitted: std::sync::MutexGuard<'a, u64>,
        #[cfg(feature = "diagnostics")] mut called: Instant,
    ) -> Result<
        (std::sync::MutexGuard<'a, u64>, Option<Sent>, usize),
        (std::sync::MutexGuard<'a, u64>, std::collections::TryReserveError),
    > {
        if runs.is_empty() {
            return self.submit_adaptive(
                runs,
                late,
                submitted,
                #[cfg(feature = "diagnostics")]
                called,
            );
        }
        let mut already_run = 0;
        let mut ranges = submission_ranges(runs.len()).peekable();
        while let Some(range) = ranges.next() {
            let send_ahead = late && ranges.peek().is_none();
            let result = self.submit_adaptive(
                &runs[range],
                send_ahead,
                submitted,
                #[cfg(feature = "diagnostics")]
                called,
            );
            let (next, sent, ran) = result?;
            submitted = next;
            already_run += ran;
            if send_ahead {
                return Ok((submitted, sent, already_run));
            }
            #[cfg(feature = "diagnostics")]
            { called = Instant::now(); }
        }
        Ok((submitted, None, already_run))
    }

    /// Retries a failed bounded allocation as two smaller synchronous
    /// submissions. The right half keeps the caller's `late` choice, so the
    /// original order and the one-flight-at-the-end rule are preserved.
    pub(super) fn submit_adaptive<'a>(
        &'a self,
        runs: &[&HostList],
        late: bool,
        submitted: std::sync::MutexGuard<'a, u64>,
        #[cfg(feature = "diagnostics")] called: Instant,
    ) -> Result<
        (std::sync::MutexGuard<'a, u64>, Option<Sent>, usize),
        (std::sync::MutexGuard<'a, u64>, std::collections::TryReserveError),
    > {
        match self.submit_part(
            runs,
            late,
            submitted,
            #[cfg(feature = "diagnostics")]
            called,
        ) {
            Ok((submitted, sent)) => {
                let already_run = if sent.is_some() { 0 } else { runs.len() };
                Ok((submitted, sent, already_run))
            }
            Err((submitted, _error)) if runs.len() > 1 => {
                let middle = runs.len() / 2;
                let (submitted, sent, left_run) = self.submit_adaptive(
                    &runs[..middle],
                    false,
                    submitted,
                    #[cfg(feature = "diagnostics")]
                    Instant::now(),
                )?;
                debug_assert!(sent.is_none());
                let (submitted, sent, right_run) = self.submit_adaptive(
                    &runs[middle..],
                    late,
                    submitted,
                    #[cfg(feature = "diagnostics")]
                    Instant::now(),
                )?;
                Ok((submitted, sent, left_run + right_run))
            }
            Err(error) => Err(error),
        }
    }

    /// Keeps the submission lock across dependent segments so another
    /// submitter cannot insert commands inside the title's single submission.
    pub(super) fn submit_part<'a>(
        &'a self,
        runs: &[&HostList],
        late: bool,
        mut submitted: std::sync::MutexGuard<'a, u64>,
        #[cfg(feature = "diagnostics")] called: std::time::Instant,
    ) -> Result<
        (std::sync::MutexGuard<'a, u64>, Option<Sent>),
        (std::sync::MutexGuard<'a, u64>, std::collections::TryReserveError),
    > {
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        let was_working = working(true);
        submitting(1, runs.len());
        // Under the submission lock, before inspecting dependencies. A remap
        // racing preparation also invalidates this submission's old writes.
        crate::writeback::starting(STARTED.load(Ordering::Relaxed) + 1);
        self.video_memory_look();
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if let Some(why) = late_blocked(runs) {
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                relay_wait(why, |line| blocked_by(&line.written, runs, false).is_none());
            } else {
                self.settle(&mut submitted, why);
            }
        }
        // Descriptors the title copied into placed heaps directly, before
        // command lists that index those slots are submitted. With
        // `VICTORY_SYNC_BESIDE` on a thread of its own while this one
        // prepares, from after the sweep to before the lists go to the card
        // (`crate::descriptor::sync_beside`).
        let beside = crate::descriptor::sync_beside();
        if !beside {
            crate::descriptor::sync_heaps();
        }
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        // Reserve the command-list workspace before this submission gets a
        // number or changes descriptor/shadow state. A failed reservation
        // asks the adaptive caller to retry with fewer runs.
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::new();
        if let Err(error) = lists.try_reserve_exact(2 * runs.len() + 1) {
            request_memory_sweep();
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(Instant::now().saturating_duration_since(called).as_micros() as u64);
            working(was_working);
            return Err((submitted, error));
        }
        #[cfg(feature = "diagnostics")]
        let mut firsts = if timing() {
            let mut firsts = Vec::new();
            if let Err(error) = firsts.try_reserve_exact(runs.len()) {
                request_memory_sweep();
                submitting(0, 0);
                crate::copystat::submission_ends(Instant::now().saturating_duration_since(called).as_micros() as u64);
                working(was_working);
                return Err((submitted, error));
            }
            Some(firsts)
        } else {
            None
        };
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run (with the pass, beside, after it).
        if beside {
            crate::descriptor::sync_begin();
        } else {
            crate::descriptor::make_waiting();
        }
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut batch = Batch::default();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        // The parts of the preparing, for the line that splits it
        // ([`PREP_PARTS`]).
        let mut part = std::time::Instant::now();
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        if crate::watch::once() {
            let mut spans = Batch::default();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step
                        && !shadow.unwatched()
                    {
                        spans.add(shadow, *address, *len);
                    }
                }
            }
            let ranges: Vec<(usize, usize)> = spans.iter().map(|(_, from, to)| (from, to)).collect();
            crate::watch::ask_once(&ranges);
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if behind && stale_under_flights(runs) {
                let settling = std::time::Instant::now();
                self.settle(&mut submitted, WHY_STALE);
                settled += settling.elapsed();
                crate::watch::ask_once(&ranges);
            }
        } else if behind && stale_under_flights(runs) {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
        }
        part = prep_part(0, part);
        for list in runs {
            let uploads = list.run_deferred(&mut batch);
            #[cfg(feature = "diagnostics")]
            if let Some(firsts) = &mut firsts {
                firsts.push(lists.len());
            }
            if let Some(prelude) = list.prelude(&uploads) {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        part = prep_part(1, part);
        batch.prepare_bindless();
        let bindless_writes = std::mem::take(&mut batch.bindless_writes);
        crate::watch::ask_once_end();
        part = prep_part(2, part);
        // The pass over the view heap, if it ran beside: done before any
        // list of this submission goes to the card.
        if beside {
            crate::descriptor::sync_end();
        }
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            #[cfg(feature = "diagnostics")]
            let snapshots = std::time::Instant::now();
            crate::shadow::take_snapshots();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        if flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
        }
        part = prep_part(3, part);
        let card = self.card_before(runs, flown, *submitted, &bindless_writes);
        if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
            lists.push(Some(check));
        }
        prep_part(4, part);
        let prepared = std::time::Instant::now();
        if late {
            *submitted += 1;
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                self.queue.ExecuteCommandLists(&lists);
                self.queue.Signal(&self.fence, *submitted)
            };
            self.card_sent();
            prep_part(5, prepared);
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            let mut back = declared_writes(runs, false);
            for (shadow, address, len) in &bindless_writes {
                let range = (*address, *address + *len);
                match back.iter_mut().find(|(s, _, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                    Some((_, ranges, _)) => {
                        if !ranges.contains(&range) {
                            ranges.push(range);
                        }
                    }
                    None => back.push((shadow.clone(), vec![range], false)),
                }
            }
            let mut texture_back_ranges = Vec::new();
            for list in runs {
                texture_back_ranges.extend(list.deferred().texture_backs.iter().map(crate::texback::Back::range));
            }
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            submission_us(0, (prepared - started).saturating_sub(settled).as_micros() as u64);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                texture_back_ranges,
                writes,
            };
            working(was_working);
            return Ok((submitted, Some(sent)));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            let firsts = firsts.as_ref().expect("timing reserves per-run offsets");
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        submitting(2, runs.len());
        for (i, group) in groups.iter().enumerate() {
            // Diagnostics: everything but the GPU's work, to look at the
            // order of the title's submissions.
            #[cfg(feature = "diagnostics")]
            if no_work() {
                continue;
            }
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                self.card_sent();
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        working(false);
                        WaitForSingleObject(self.event, INFINITE);
                        stood(started);
                        working(true);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            // With timing on each group is one of `runs` (and the compare
            // of copies in the card's memory, with the last): every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    crate::copystat::command_took(&op, ms);
                }
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        submission_us(0, (prepared - started).as_micros() as u64);
        submission_us(1, gpu.as_micros() as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        submitting(3, runs.len());
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        let owners = crate::writeback::Owners::single(this_submission);
        if let Some(card) = &card {
            self.card_back(
                card,
                &mut submitted,
                &mut std::collections::HashSet::new(),
                &owners,
            );
        }
        let mut back = declared_writes(runs, false);
        for (shadow, address, len) in &bindless_writes {
            let range = (*address, *address + *len);
            match back.iter_mut().find(|(s, _, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                Some((_, ranges, _)) => {
                    if !ranges.contains(&range) {
                        ranges.push(range);
                    }
                }
                None => back.push((shadow.clone(), vec![range], false)),
            }
        }
        // One submission, waited for: nothing in the air withholds marks.
        hand_back(&back, &owners, &std::collections::HashMap::new());
        crate::writeback::published(this_submission);
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        copy_back(0, copying.elapsed().as_micros() as u64);
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        working(was_working);
        Ok((submitted, None))
    }

    #[allow(dead_code)]
    /// After a submission ran: what its shaders wrote goes back into title
    /// memory, then its read-backs, then the bottom-of-pipe writes that
    /// tell the title the GPU is done.
    pub(super) fn finish_submission(
        &self,
        runs: &[&HostList],
        back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    ) {
        let copying = std::time::Instant::now();
        if back.len() <= 1 {
            for (shadow, address, len) in &back {
                let written = shadow.copy_back(*address, *len);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
        } else {
            // Group by Shadow pointer so distinct shadows run in parallel
            // across CPU cores without lock contention on `before`/`pending`.
            let mut groups: Vec<Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>> =
                Vec::new();
            for item in back {
                if let Some(g) = groups
                    .iter_mut()
                    .find(|g| std::sync::Arc::ptr_eq(&g[0].0, &item.0))
                {
                    g.push(item);
                } else {
                    groups.push(vec![item]);
                }
            }
            if groups.len() <= 1 {
                for (shadow, address, len) in &groups[0] {
                    let written = shadow.copy_back(*address, *len);
                    COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                }
            } else {
                in_parallel(&groups, 8, |chunk| {
                    for g in chunk {
                        for (shadow, address, len) in g {
                            let written = shadow.copy_back(*address, *len);
                            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                        }
                    }
                });
            }
        }
        for list in runs {
            for rb in &list.deferred().readbacks {
                let written = if rb.dst_pitch == rb.row_bytes && rb.src_pitch == rb.row_bytes {
                    let bytes = rb.row_bytes * rb.lines;
                    crate::watch::writing(rb.dst, bytes);
                    write_mapped(rb.dst as *mut u8, rb.src as *const u8, bytes);
                    bytes
                } else {
                    let mut n = 0;
                    for line in 0..rb.lines {
                        let to = rb.dst + line * rb.dst_pitch;
                        let from = rb.src + line * rb.src_pitch;
                        crate::watch::writing(to, rb.row_bytes);
                        write_mapped(to as *mut u8, from as *const u8, rb.row_bytes);
                        n += rb.row_bytes;
                    }
                    n
                };
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        // Host timestamps the lists resolved, on the title's GPU clock.
        let mut clock = None;
        for list in runs {
            let deferred = list.deferred();
            for &(at, what, bytes) in &deferred.writes[deferred.early..] {
                let what = match what {
                    crate::queue::Written::HostTimestamp(raw) => match clock.get_or_insert_with(|| self.clock()) {
                        // SAFETY: the list's mapped read-back memory, which
                        // the submission that ran resolved into.
                        Some(c) => crate::queue::Written::Value(c.convert(unsafe { (raw as *const u64).read_volatile() })),
                        None => crate::queue::Written::GpuTimestamp,
                    },
                    what => what,
                };
                what.write(at as *mut u8, bytes);
            }
            deferred.early = 0;
        }
    }

    /// How host GPU timestamps map onto the title's GPU clock
    /// ([`crate::queue::Written::GpuTimestamp`]): the queue's calibration
    /// of its timestamps against the CPU's performance counter, which
    /// `Instant` reads too.
    pub(super) fn clock(&self) -> Option<Clock> {
        use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
        let (mut gpu, mut cpu, mut now, mut cpu_frequency) = (0, 0, 0, 0);
        // SAFETY: plain queries into our locals.
        let frequency = unsafe {
            let frequency = self.queue.GetTimestampFrequency().ok()?;
            self.queue.GetClockCalibration(&mut gpu, &mut cpu).ok()?;
            QueryPerformanceCounter(&mut now);
            QueryPerformanceFrequency(&mut cpu_frequency);
            frequency
        };
        let title_now = crate::queue::gpu_clock();
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        // The title's clock when the host GPU's read `gpu`.
        let since = (i128::from(now) - i128::from(cpu)) * ticks / i128::from(cpu_frequency.max(1));
        (frequency > 0).then_some(Clock { gpu, at: i128::from(title_now) - since, frequency, now: title_now })
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    pub(super) fn removal_data(&self, runs: &[&HostList]) -> String {
        self.removal_data_of(runs, true)
    }

    /// [`Host::removal_data`]; `tables`: whether a faulting address is also
    /// looked up in our own tables (title memory copies, view heaps), which
    /// takes their locks.
    ///
    /// With the data on in any form ([`Dred`]) the report says each of its
    /// three parts whatever the runtime answered: why the device went, the
    /// page fault (an address, none reported, or the call failed) and the
    /// breadcrumbs (how many lists the runtime returned, none, or the call
    /// failed). Run 277's report had neither a fault nor a breadcrumb, and
    /// nothing in it said whether the runtime had been asked in vain.
    pub(super) fn removal_data_of(&self, runs: &[&HostList], tables: bool) -> String {
        DATA_SAID.store(true, Ordering::Relaxed);
        let said = dred_any();
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return if said {
                "; device-removed data: this device has no interface for it".to_owned()
            } else {
                String::new()
            };
        };
        let mut text = String::new();
        // Whatever the runtime says: which lists in the air bind a pipeline
        // this run had not drawn with before.
        if !runs.is_empty() {
            text.push_str(&first_uses_text(runs, true));
        }
        let mut fault_va = None;
        if said {
            // SAFETY: a plain query.
            let reason = match unsafe { self.device.GetDeviceRemovedReason() } {
                Ok(_) => "none, the device is not removed".to_owned(),
                Err(e) => format!("{:#010X}", e.code().0),
            };
            text.push_str(&format!(
                "; device-removed data ({}): removal reason {reason}; {} host objects named, {} of them made while frames are drawn",
                dred_mode().text(),
                NAMED[0].load(Ordering::Relaxed),
                NAMED[1].load(Ordering::Relaxed)
            ));
            // The runtime can classify a removal even when it supplies no
            // fault address. Older runtimes still have the DRED1 report below.
            match dred.cast::<ID3D12DeviceRemovedExtendedData2>() {
                Ok(dred2) => {
                    // SAFETY: a query on the live DRED interface.
                    let state = unsafe { dred2.GetDeviceState() };
                    let name = match state {
                        D3D12_DRED_DEVICE_STATE_HUNG => "hung",
                        D3D12_DRED_DEVICE_STATE_FAULT => "fault",
                        D3D12_DRED_DEVICE_STATE_PAGEFAULT => "page fault",
                        D3D12_DRED_DEVICE_STATE_UNKNOWN => "unknown",
                        _ => "unrecognized",
                    };
                    text.push_str(&format!("; DRED device state {name} ({})", state.0));
                }
                Err(e) => text.push_str(&format!("; DRED device state unavailable ({:#010X})", e.code().0)),
            }
        }
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            let fault = dred.GetPageFaultAllocationOutput1();
            // Said either way: no address, or no answer at all.
            match &fault {
                Ok(f) if said && f.PageFaultVA == 0 => text.push_str("; page fault: no fault address reported"),
                Err(e) if said => text.push_str(&format!("; page fault: the call failed ({e})")),
                _ => {}
            }
            if let Ok(fault) = fault
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                let mut found = 0;
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        found += names.len();
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                if found == 0 {
                    text.push_str("; inside no allocation the runtime lists, alive or freed lately");
                }
                if tables {
                    text.push_str(&crate::shadow::near(va));
                    text.push_str(&crate::descriptor::host::near_heaps(va));
                }
            }
            // `faults` keeps no breadcrumbs: none are asked for.
            if dred_mode() == Dred::Faults {
                text.push_str("; breadcrumbs: not kept in this form");
                return text;
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let mut crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    if said {
                        text.push_str(&format!("; breadcrumbs: the call failed ({e})"));
                    } else {
                        text.push_str(&format!("; no breadcrumbs ({e})"));
                    }
                    return text;
                }
            };
            // An empty list within a millisecond of the removal (run 277):
            // asked again for half a second before it is believed.
            let mut asked = 0u32;
            while said && crumbs.pHeadAutoBreadcrumbNode.is_null() && asked < 10 {
                std::thread::sleep(std::time::Duration::from_millis(50));
                asked += 1;
                if let Ok(again) = dred.GetAutoBreadcrumbsOutput() {
                    crumbs = again;
                }
            }
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            if said {
                let again = if asked > 0 {
                    format!(", asked {asked} more times over {} ms", asked * 50)
                } else {
                    String::new()
                };
                if progress.is_empty() {
                    text.push_str(&format!("; breadcrumbs: the runtime returned none (nodes 0{again})"));
                } else {
                    text.push_str(&format!("; breadcrumbs: data returned (nodes {}{again})", progress.len()));
                }
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    }
                    // A list that never started; and in `light`, where no
                    // marker says what a command runs, every stopped list.
                    if (done == 0 || dred_mode() == Dred::Light)
                        && let Some(r) = run
                    {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// With the frame line (`VICTORY_D3D12_DEBUG=1`, diagnostics builds):
    /// the debug layer's messages since the last call go to our log, the
    /// first five of each kind in full, the rest counted, with one line of
    /// the counts by kind.
    ///
    /// They were only read when the device was lost ([`fail`]): a run that
    /// survived (run 205) said nothing of what the layer had found, and
    /// what it finds is what a driver's fault would have to be traced back
    /// to (an object let go while the card uses it, a resource in the
    /// wrong state, a view of something gone).
    pub(crate) fn drain_messages(&self) {
        let Some(queue) = &self.messages else {
            return;
        };
        static SEEN: Mutex<std::collections::BTreeMap<i32, (u64, i32)>> = Mutex::new(std::collections::BTreeMap::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        let stored = unsafe {
            let stored = queue.GetNumStoredMessages();
            for i in 0..stored {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let kind = seen.entry(m.ID.0).or_insert((0, m.Severity.0));
                    kind.0 += 1;
                    if kind.0 <= 5 {
                        let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(text);
                        let text = text.trim_end_matches('\0');
                        tracing::warn!(id = m.ID.0, severity = m.Severity.0, nth = kind.0, "D3D12 debug layer: {text}");
                    }
                }
            }
            queue.ClearStoredMessages();
            stored
        };
        if stored > 0 {
            let counts: Vec<String> = seen.iter().map(|(id, (n, severity))| format!("id {id} (severity {severity}) x{n}")).collect();
            tracing::info!(
                "D3D12 debug layer messages so far by kind (severity 0 corruption, 1 error, 2 warning, 3 info, 4 message): {}",
                counts.join(", ")
            );
        }
    }

    /// The host device (`crate::cardmeter`).
    pub(crate) fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// Ticks a second of the queue's timestamps; 0 if it has none.
    pub(crate) fn timestamp_frequency(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(0)
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}
