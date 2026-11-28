import re, sys
sys.path.insert(0, '.')
from resolve1 import resolve, hunks

P = 'crates/victory-gpu/src/descriptor/host.rs'
src, hs = hunks(P)
assert len(hs) == 68, len(hs)

# ---- custom literal replacements ----

# h1: ours' consts/helpers + theirs' Slots/Made (ours' text ends mid-fn `on`)
h1 = hs[1][1] + """        }
    }
}

""" + hs[1][2]

# h3: unified field set
h3 = """    made: Made,
    /// The same for a slot whose descriptor is made and stays (else 0): what
    /// [`HostHeap::mirror`] asks first, without the lock. Changed under it.
    quick: Vec<std::sync::atomic::AtomicU64>,
    /// The records of the title's heap as the pass over records copied in
    /// as bytes last saw them ([`Copied::changes`]). Made on first use.
    copied: Mutex<Vec<u8>>,
    /// A fingerprint of the record each slot held when the pass over
    /// records copied in as bytes last looked at it
    /// ([`HostHeap::mirror_copied`]; 0: never looked at). Made on first use.
    copied_prints: Mutex<Vec<u64>>,
    /// New ones of them that wait for a draw ([`HostHeap::copied_wait`]),
    /// by the frame they were found in.
    waiting: Mutex<std::collections::VecDeque<(u64, Vec<u32>)>>,
    /// The slots made from a texture whose memory was found away, with what
    /// they were made from and when that was
    /// ([`HostHeap::texture_slot_gone`]).
    away: Mutex<std::collections::HashMap<u32, (u64, u64)>>,
"""

# h8: Spans union fields (ours' spans/by_slot/displaced/sizes + theirs' first tuple)
h8 = """    spans: std::collections::BTreeMap<(u64, u64), ([u32; 14], std::collections::BTreeSet<u32>)>,
    by_slot: std::collections::HashMap<u32, (u64, u64)>,
    /// The level each slot's view starts at, and where what it is read from
    /// ends ([`Spans::shows`]).
    first: std::collections::HashMap<u32, (u32, u64)>,
    /// Materialized slots made null by a newer placed texture. Only the guest
    /// record key survives invalidation; no host resource is kept alive.
    displaced: std::collections::BTreeMap<u32, u64>,
    displaced_next: u32,
    /// The bytes a placed texture takes, by resource description and first
    /// mip: worked out once for each (a title writes the views of a few
    /// thousand descriptions, thousands of times a frame).
    sizes: std::collections::HashMap<([u32; 14], u32), Option<u64>>,
"""

# h9: first_viewed over ours' spans + theirs' (first, end) tuple
h9 = """        self.spans
            .range((start, 0)..(start.saturating_add(1), 0))
            .flat_map(|(_, slots)| slots.1.iter())
            .filter_map(|slot| self.first.get(slot))
            .map(|&(first, _)| first)
            .min()
"""

# h10: slots_holding over ours' spans + theirs' reaches filter
h10 = """            .chain(self.spans.range((start, 0)..(end, 0)))
            .flat_map(|(_, slots)| slots.1.iter().copied())
            .filter(|&slot| self.reaches(slot, start))
"""

# h12: ours' complete `due` + theirs' side minus their `fn due`
theirs12 = hs[12][2]
# drop theirs' `fn due` (doc comment + cfg(test) + fn body, up to its closing brace)
theirs12 = re.sub(r"    /// Up to `max` slots due at `frame`.*?\n    \}\n\n", "", theirs12, count=1, flags=re.S)
assert 'fn due_and_census' in theirs12 and 'fn due(' not in theirs12, theirs12[:400]
h12 = hs[12][1] + """            }
            retry = !retry;
        }
        order.truncate(max);
        order
    }

""" + theirs12

# h14: ours' slot_word + theirs' view_stride_and_kind/view_stride_word
h14 = """/// The strides-table dword of a heap slot holding `view`
/// (`victory_shader::emit::SLOT_STRIDES`): a buffer view's stride,
/// `SLOT_CONSTANT_BUFFER` for a constant buffer view, 0 for anything else.
fn slot_word(view: Option<&View>) -> u32 {
    match view {
        Some(View::Constant(..)) => victory_shader::emit::SLOT_CONSTANT_BUFFER,
        Some(v) => view_stride(v).unwrap_or(0),
        None => 0,
    }
}

/// Bytes per element and typed integer format bits of a buffer view, decoded
/// in a single pass over the view descriptor; `(None, 0)` if not a buffer view.
fn view_stride_and_kind(view: &View) -> (Option<u32>, u32) {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    let Some((stride, raw, format)) = view_elements(view) else {
        return (None, 0);
    };
    let bytes = if stride > 0 {
        stride
    } else if raw {
        1
    } else {
        crate::resource::element(format).map_or(0, |(bytes, _)| bytes as u32)
    };
    let kind = if stride == 0 && !raw {
        match format {
            3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => VIEW_INTEGER,
            4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => VIEW_INTEGER | VIEW_SIGNED,
            _ => 0,
        }
    } else {
        0
    };
    (Some(bytes), kind)
}

/// Packed [`victory_shader::emit::SLOT_STRIDES`] word for `view`.
fn view_stride_word(view: &View) -> u32 {
    let (stride, kind) = view_stride_and_kind(view);
    stride.unwrap_or(0) | kind
}

"""

# h15: ours' block + close `merged` + theirs' helpers
h15 = hs[15][1] + "}\n" + hs[15][2]

# h16: union probe_view_limit (tier/max_views query + raw twins sizing)
h16 = """        let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: the feature query fills this structure.
        let tier = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut caps).cast(),
                size_of_val(&caps) as u32,
            )
        }
        .is_ok()
        .then_some(caps.ResourceBindingTier);
        let mut limits = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
        // SAFETY: the feature query fills this structure on runtimes that support it.
        let max_views = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS19,
                (&raw mut limits).cast(),
                size_of_val(&limits) as u32,
            )
        }
        .is_ok()
        .then_some(limits.MaxViewDescriptorHeapSize)
        .filter(|&count| count >= PORTABLE_VIEWS);
        let twins = raw_twins() && fits(2 * RAW_TWINS);
        if twins {
            victory_shader::emit::set_raw_views(RAW_TWINS);
        }
        // (The twins lie past the null slot.)
        let views = if twins {
            RAW_TWINS
        } else {
            view_heap_size(tier, max_views, fits)
        };
        tracing::info!(
            views,
            binding_tier = ?tier,
            ?max_views,
            raw_twins = twins,
"""

# h19: `new()` head: wanted + raw + extra + 4-tuple binding
h19 = """        let wanted = count;
        let raw = if sampler { 0 } else { victory_shader::emit::raw_views() };
        let extra = if sampler { 0 } else { dynamic_slots(count, null) };
        let (heap, count, extra, size) = if null == 0 {
"""

# h20: `with()` head: view_marks + stride_table + queue marks + host heap literal
h20 = """        let view_marks = (!sampler)
            .then(|| {
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                // SAFETY: the buffer's whole mapping; a live buffer. Every
                // slot starts as marking nothing.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0xFF, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::name(&buffer, || "view write marks by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        let stride_table = (!sampler)
            .then(|| StrideTable::new(&host.device, size))
            .flatten();
        let marks = std::sync::Arc::new(QueueMarks::new(if sampler { 0 } else { count as usize }));
        let mut h = HostHeap {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            buffer_sweep: Mutex::default(),
            buffer_frame: std::sync::atomic::AtomicU64::new(u64::MAX),
"""

# h21: union field initializers
h21 = """            made: Made::new(count as usize),
            quick: (0..count).map(|_| std::sync::atomic::AtomicU64::new(0)).collect(),
            copied: Mutex::new(Vec::new()),
            copied_prints: Mutex::new(Vec::new()),
            waiting: Mutex::default(),
            away: Mutex::default(),
            kept: (0..KEPT_PIECES)
                .map(|_| Mutex::new(vec![None; (count + extra).div_ceil(KEPT_PIECES) as usize]))
                .collect(),
            kept_one: Mutex::new(()),
            kept_flat: Mutex::new(vec![None; size as usize]),
            kept_textures: Mutex::default(),
            bindless: Mutex::default(),
            recording: std::sync::atomic::AtomicBool::new(false),
            record_hint: std::sync::atomic::AtomicBool::new(false),
            record_looked: [AtomicU64::new(0), AtomicU64::new(0)],
            recheck_at: AtomicU64::new(u64::MAX),
            rewrites_seen: AtomicU64::new(0),
            memory_seen: Mutex::new(0),
            next: Mutex::new((0, taken, usize::MAX)),
            free: std::sync::atomic::AtomicU32::new(dynamic),
            too_long: std::sync::atomic::AtomicU32::new(u32::MAX),
            strides,
            stride_table,
            view_marks,
            marks,
            addresses: std::sync::OnceLock::new(),
            raw_pages: Mutex::new(None),
            buffers: Mutex::default(),
            changed: Slots::new(count as usize),
            textures: Mutex::new(TextureQueue::with_marks(marks.clone())),
            texture_sweep: Mutex::default(),
            material_demand: Default::default(),
            material_epochs: Default::default(),
            material_clock: Default::default(),
            raw,
            twins: (0..if raw == 0 { 0 } else { count + extra }).map(|_| std::sync::atomic::AtomicBool::new(false)).collect(),
            #[cfg(feature = "diagnostics")]
            written: (0..if sampler || !crate::book::on() { 0 } else { size })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
"""

# h22: theirs' twin-aware fill + keep ours' null_views retention
h22 = hs[22][2] + """            if !self.sampler {
                self.null_views = Some(block);
"""

# h26: ours' texture_slots_due signature ([u64;6] census) + theirs' fns minus their cfg(test) texture_slots_due
theirs26 = hs[26][2]
theirs26 = re.sub(r"    #\[cfg\(test\)\]\n    pub\(crate\) fn texture_slots_due[\s\S]*?\n    \}\n\n?", "", theirs26, count=1)
assert 'fn texture_slots_due_known' in theirs26 and 'fn texture_slots_due(' not in theirs26, theirs26[:300]
h26 = """    /// Up to `max` title slots whose texture view is due at `frame`, each
    /// with its write number and whether a GPU material re-enrolled it
    /// (only those need an ownership check during upload), in the order to
    /// serve them ([`TextureQueue::due`]), and the queue's counts
    /// ([`TextureQueue::census`]).
    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> (Vec<(u32, u64, bool)>, [u64; 6]) {
        let mut textures = self.queue(super::queue_site::DUE);
        (if max == 0 { Vec::new() } else { textures.due(frame, max) }, textures.census(frame))
    }

""" + theirs26

# h33: theirs' RESHOWN_AFTER promote + ours' quiet early-return
h33 = """        {
            let mut after = RESHOWN_AFTER.lock().unwrap_or_else(|e| e.into_inner());
            if !after.is_empty() {
                let finished = crate::host::submissions().1;
                let mut now = RESHOWN.lock().unwrap_or_else(|e| e.into_inner());
                after.retain(|&(at, start)| {
                    if at <= finished {
                        now.push(start);
                    }
                    at > finished
                });
            }
        }
        // The slots made again here are live slots of the heap the card
        // reads, and the making is the layer's own, at a moment the title
        // knows nothing of: a list already sent may be reading a slot while
        // it is written (Direct3D 12 asks that a descriptor a submitted
        // list reads is left alone until the list has run). Counted always;
        // with `VICTORY_VIEWS_REMADE_QUIET=1` the pass is put off while a
        // submission is in flight, and what is due stays due.
        let flying = crate::host::submissions_in_flight();
        if flying > 0 && views_remade_quiet() {
            if !RESHOWN.lock().unwrap_or_else(|e| e.into_inner()).is_empty() {
                VIEWS_REMADE[2].fetch_add(1, Relaxed);
            }
            return 0;
        }
"""

# h35: unmake writes both the lock-free mirror and the record
h35 = """        self.quick_is(index, 0);
        self.made.set(index, 0);
"""

# h36: buffer_slots over Mutex<BufferSlots> + theirs' buffer_slots_of adapted
h36 = """        let buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        buffers.indices.iter().copied().collect()
    }

    /// Of [`HostHeap::buffer_slots`], those of one of `parts` parts: every
    /// slot is in the same part each time.
    pub(crate) fn buffer_slots_of(&self, part: u32, parts: u32) -> Vec<u32> {
        let buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        buffers.indices.iter().copied().filter(|slot| slot % parts == part).collect()
    }
"""

# h38: note_buffer's span/stale block, theirs' shape over settled APIs
h38 = """            let span = placed_view.and_then(|(resource, start, first, held)| {
                Some((resource, start, start.saturating_add(placed.bytes(resource, first)?), first, held))
            });
            match span {
                Some((resource, start, end, first, held)) => {
                    SPANNED[0].fetch_min(start, Relaxed);
                    SPANNED[1].fetch_max(end, Relaxed);
                    self.spanned.put(index);
                    let stale = placed.note(index, start, end, resource, !seen && stale_out());
                    placed.shows(index, first, held);
                    // (Under the lock of `placed`: a later write of one of
                    // them comes by here before it makes the slot.)
                    stale.into_iter().map(|slot| (slot, self.made.get(slot).unwrap_or(0))).collect()
"""

# h39: note_buffer tail: strides/addresses/marks writes + acceleration + buffers sets
h39 = """        let (stride, kind) = view.map_or((None, 0), view_stride_and_kind);
        if let Some((_, mapping)) = self.addresses.get() {
            let address = view.and_then(guest_srv_address).unwrap_or(0);
            // SAFETY: note_buffer runs under the slot lock for a valid title slot.
            unsafe { (*mapping as *mut u64).add(index as usize + 2).write_volatile(address) };
        }
        self.write_stride(index, slot_word(view) | kind);
        if let Some((_, _, data)) = &self.view_marks
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            // The slot's mark entry: the page bit and page offset of the
            // view's first byte, where the copy it is of marks its pages
            // (`Shadow::mark_entry`); none where it does not (not a copy,
            // or in the card's memory). Base `VIEW_MARK_NONE`: the shader
            // marks nothing and the range is declared.
            let (base, phase) = view
                .and_then(buffer_copy)
                .and_then(|(shadow, address, _)| shadow.mark_entry(address))
                .unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
            // SAFETY: inside the mapping; shaders read it while we write (a
            // dword pair, either value is one the slot had).
            unsafe {
                let at = (*data as *mut u32).add(index as usize * 2);
                at.write_volatile(base);
                at.add(1).write_volatile(phase);
            }
        }
        let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let location = view.and_then(acceleration_location);
        let previous = if let Some(location) = location {
            buffers.acceleration.insert(index, location)
        } else {
            buffers.acceleration.remove(&index)
        };
        if previous != location {
            buffers.acceleration_locations = None;
        }
        let was = if stride.is_some() {
            buffers.indices.insert(index);
            true
        } else {
            buffers.indices.remove(&index)
"""

# h41: make_at union (twinned make + null fallbacks + stride tables + keep)
h41 = """        let kept = make_twinned(view, self.at(index), self.twin(index));
        let ok = kept.is_some();
        if !ok {
            if self.sampler {
                self.null_sampler(index);
            } else {
                null_for(view, self.at(index));
            }
        }
        self.put(index);
        self.twin_made(index, ok && twinned(view));
        self.keep_with_view(index, kept.flatten(), view);
        if ok {
            self.set_stride_count(index, view_stride(view).unwrap_or(0), view_count(view));
        } else {
            self.set_stride(index, 0);
        }
        self.write_stride(index, if ok { view_stride_word(view) } else { 0 });
"""

# h42: ours' make_null tail + theirs' twin fns (their `keep` dropped: ours' is kept);
# replacement ends leaving `fn clear_twin` open for the shared `    }` to close
h42 = """        self.set_stride(index, 0);
    }

    /// Where the raw twin of slot `index` is made.
    fn twin(&self, index: u32) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        (index < self.twins.len() as u32).then(|| staged(self.sampler, 1).unwrap_or_else(|| self.cpu(self.raw + index)))
    }

    /// The raw twin of slot `index` was made by this thread ([`make_twinned`]),
    /// or is a null view if it was not.
    fn twin_made(&self, index: u32, made: bool) {
        if !made {
            return self.clear_twin(index);
        }
        if let Some(shown) = self.twins.get(index as usize) {
            self.put_twin(index);
            shown.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// The raw twin of slot `index` is a null view again.
    fn clear_twin(&self, index: u32) {
        if let Some(made) = self.twins.get(index as usize)
            && made.swap(false, std::sync::atomic::Ordering::Relaxed)
        {
            null_buffer_view(self.at(self.raw + index));
            self.put(self.raw + index);
        }
"""

# h45: mirror_if head: union fast-path + theirs' slot lock + ours' diag guard + shown
h45 = """        // Made from this record already: what the lock would find, without
        // it (the lock-free read is `quick`, kept in step under it; the view
        // is looked up first, as under the lock: a record whose view is gone
        // is not served here).
        if tag == VIEW_TAG
            && (heap_locks() || quick_mirror())
            && self.quick.get(index as usize).is_some_and(|k| k.load(std::sync::atomic::Ordering::Acquire) == key)
            && let Some(view) = view_at(i, generation)
        {
            #[cfg(feature = "diagnostics")]
            mirror_how(MIRROR_FAST);
            if !self.sampler {
                if view.is_buffer() {
                    let placed = matches!(&view, View::Placed(false, r, _) if r[0] != 1);
                    self.track_bindless(index, (i, generation), placed);
                } else {
                    self.untrack_bindless(index);
                }
            }
            return Ok(Some(view));
        }
        let held = self.made.lock(index);
        #[cfg(feature = "diagnostics")]
        let held_diag = Held::start(for_draw);
        let shown = self.made.get(index).unwrap_or(0);
"""

# h46: not-a-view branch union
h46 = """            #[cfg(feature = "diagnostics")]
            mirror_how(MIRROR_NOT_VIEW);
            if !self.sampler {
                self.untrack_bindless(index);
            }
            if shown != 0 {
                #[cfg(feature = "diagnostics")]
                held_diag.slot_changed();
                self.quick_is(index, 0);
                self.made.set(index, 0);
"""

# h47: displaced-view check + owner-change check + stale noting (inside `if !current`)
h47 = """        if queued_write.is_some() && let View::Placed(false, resource, desc) = &view {
            let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            if placed.displaced.get(&index) == Some(&key) {
                // The ordinary sweep can enqueue from an older heap snapshot.
                // Only install a displaced view once its exact span was
                // reclaimed. A later alias revokes that claim under 'made'.
                let start = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let span = placed.bytes(resource, crate::resource::first_mip(desc))
                    .map(|bytes| (start, start.saturating_add(bytes)));
                if span.is_none() || placed.by_slot.get(&index) != span.as_ref() {
                    return Ok(None);
                }
            }
        }
        // A title can replace the resource at a placed UAV's address without
        // changing a single byte of its descriptors (including its SRV).
        let changed_owner = shown == key && for_draw && match &view {
            View::Placed(_, r, _) if r[0] != 1 && r[12] & 4 != 0 => !self.shows_placed_owner(index, &view),
            _ => !self.buffer_owner_current(index, &view),
        };
        let current = shown == key && !changed_owner;
        if !allow(&view, current) {
            return Err(());
        }
        if !current {
            #[cfg(feature = "diagnostics")]
            held_diag.slot_changed();
            // Being made again: not "made" to a reader without the lock
            // until it is (the store after `keep` below).
            self.quick_is(index, 0);
            // Views of other textures in the memory this one describes
            // are stale: their host slots read nothing from here on, and
            // let go of the host copies they held.
            #[cfg(feature = "diagnostics")]
            let noting = std::time::Instant::now();
            let stale = self.note_buffer(index, Some(&view), for_draw);
            if !stale.is_empty() {
                let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
                for &(slot, key) in &stale {
                    if slot != index && key != 0 {
                        placed.displaced.insert(slot, key);
                    }
                }
                drop(placed);
            }
            #[cfg(feature = "diagnostics")]
            if mirror_timing() {
                MIRROR_US[0].fetch_add(noting.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            }
"""

# h48: kept-view / kin / show_new_later union (inside `if later && keep_views() && let Some(old)`)
h48 = """                let kin = kin(&old, &view);
                if kin == Kin::Same {
                    KEPT_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    #[cfg(feature = "diagnostics")]
                    mirror_how(MIRROR_KEPT);
                    // The slot goes on showing what it was made from.
                    if !self.sampler {
                        if view.is_buffer() {
                            let placed = matches!(&view, View::Placed(false, r, _) if r[0] != 1);
                            self.track_bindless(index, (i, generation), placed);
                        } else {
                            self.untrack_bindless(index);
                        }
                    }
                    drop(held);
                    self.unmake_stale(index, stale);
                    return Ok(Some(view));
                }
                // And of another texture that can stand for it
                // ([`keep_shown`]): black, then grey for a frame or two is
                // what popped.
                if is_texture_view(&old) {
                    let kept = kin.kept(keep_shown());
                    rewritten(kin, kept);
                    if kept {
                        #[cfg(feature = "diagnostics")]
                        mirror_how(MIRROR_KEPT);
                        if !self.sampler {
                            if view.is_buffer() {
                                let placed = matches!(&view, View::Placed(false, r, _) if r[0] != 1);
                                self.track_bindless(index, (i, generation), placed);
                            } else {
                                self.untrack_bindless(index);
                            }
                        }
                        drop(held);
                        self.unmake_stale(index, stale);
                        return Ok(Some(view));
                    }
                }
            }
            // A texture made while lists sent before it may still run: those
            // read this slot by index, and must not find it ([`show_new_later`]).
            // The slot shows what it showed (or the neutral texture), and the
            // new view when they have all run.
            if !later
                && !self.sampler
                && show_new_later()
                && is_texture_view(&view)
                && let Some(after) = begun_before(&view)
            {
                // Asked again (each draw that reads the slot, each pass)
                // while it waits: nothing more to do.
                let own = (std::ptr::from_ref(self) as usize, index, key);
                if !NEW_PENDING.lock().unwrap_or_else(|e| e.into_inner()).insert(own) {
                    drop(held);
                    self.unmake_stale(index, stale);
                    return Ok(Some(view));
                }
                let old = view_at(shown as u32, (shown >> 32) as u32);
                let kept = old.is_some_and(|old| match kin(&old, &view) {
                    Kin::Same => true,
                    kin => kin.kept(keep_shown()),
                });
                if shown == 0 || !kept {
                    #[cfg(feature = "diagnostics")]
                    note_before(index, shown, "neutral, new texture");
                    self.quick_is(index, 0);
                    self.made.set(index, 0);
                    let kept = neutral_for(&view, self.at(index));
                    self.put(index);
                    self.clear_twin(index);
                    self.keep(index, kept);
                }
                let late = Some(crate::frame::current());
                let flip = Flip { after, heap: std::ptr::from_ref(self) as usize, index, handle, key, late };
                FLIPS.lock().unwrap_or_else(|e| e.into_inner()).push(flip);
                NEW_LATER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                drop(held);
                self.unmake_stale(index, stale);
                return Ok(Some(view));
            }
            #[cfg(feature = "diagnostics")]
            if !self.sampler && is_texture_view(&view) {
                note_before(index, shown, if later { "null, waits" } else { "at once" });
            }
            self.quick_is(index, 0);
            self.made.set(index, if later { 0 } else { key });
            #[cfg(feature = "diagnostics")]
            mirror_how(if later { MIRROR_LATER } else { MIRROR_MADE });
            #[cfg(feature = "diagnostics")]
            let creating = std::time::Instant::now();
"""

# h49: make/keep tail union
h49 = """            #[cfg(feature = "diagnostics")]
            let keeping = std::time::Instant::now();
            if kept.is_some() || !self.sampler {
                self.put(index);
            }
            self.twin_made(index, kept.is_some() && twinned(&view));
            self.keep(index, kept.flatten());
            #[cfg(feature = "diagnostics")]
            if mirror_timing() {
                use std::sync::atomic::Ordering::Relaxed;
                MIRROR_US[1].fetch_add((keeping - creating).as_micros() as u64, Relaxed);
                MIRROR_US[2].fetch_add(keeping.elapsed().as_micros() as u64, Relaxed);
            }
            // After the slot is made and kept: a reader without the lock
            // takes the record as made only once it is.
            self.quick_is(index, if later { 0 } else { key });
            drop(held);
            self.unmake_stale(index, stale);
"""

# h62: ours' placed_texture + theirs' want_level
h62 = hs[62][1].replace(
    "    Some(PlacedTexture {",
    "    crate::resource::want_level(texture, crate::resource::first_mip(desc));\n    Some(PlacedTexture {",
)
assert 'want_level' in h62 and 'PlacedTexture' in h62

# h64: ours' complete magenta test (with its shared-tail body inline) + theirs' side
h64 = """    /// The texture that stands for missing ones can be made and viewed.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_magenta_texture_is_made() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let texture = magenta().expect("the texture");
        // SAFETY: a slot of our heap and a live texture.
        unsafe { host.device.CreateShaderResourceView(texture, None, heap.cpu(0)) };
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

""" + hs[64][2]

C = [
    'ours',            # 0 imports
    h1,                # 1
    'ours',            # 2 NEXT_ID dup -> ours' allow(dead_code)
    h3,                # 3
    'ours',            # 4 kept sharded
    'ours',            # 5 buffers: Mutex<BufferSlots>
    'both',            # 6
    'both',            # 7
    h8,                # 8
    h9,                # 9
    h10,               # 10
    'ours',            # 11
    h12,               # 12
    'ours',            # 13
    h14,               # 14
    h15,               # 15
    h16,               # 16
    'both',            # 17
    'ours',            # 18
    h19,               # 19
    h20,               # 20
    h21,               # 21
    h22,               # 22
    'both',            # 23
    'ours',            # 24
    'ours',            # 25
    h26,               # 26
    'theirs',          # 27
    'ours',            # 28
    'both',            # 29
    'theirs',          # 30
    'theirs',          # 31
    'ours',            # 32
    h33,               # 33
    'theirs',          # 34
    h35,               # 35
    h36,               # 36
    'theirs',          # 37
    h38,               # 38
    h39,               # 39
    'ours',            # 40
    h41,               # 41
    h42,               # 42
    'theirs',          # 43
    'theirs',          # 44
    h45,               # 45
    h46,               # 46
    h47,               # 47
    h48,               # 48
    h49,               # 49
    'theirs',          # 50
    'ours',            # 51
    'theirs',          # 52
    'both',            # 53
    'ours',            # 54
    'both',            # 55
    'ours',            # 56
    'both',            # 57
    'theirs',          # 58
    'both',            # 59
    'both',            # 60
    'theirs',          # 61
    h62,               # 62
    'theirs',          # 63
    h64,               # 64
    'theirs',          # 65
    'ours',            # 66
    'both',            # 67
]
assert len(C) == 68
resolve(P, C)
