import resolve1

P = 'crates/victory-hle/src/modules/xmem.rs'

h0 = """    virtual_alloc(
        base,
        size,
        kind,
        flags,
        protect,
        victory_gpu::title_memory_committed,
    )
}

fn virtual_alloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
    on_commit: impl FnOnce(usize, usize),
) -> *mut c_void {
    start_memory_log();
    // Asked of addresses a lazy view shows: as if the view were mapped.
    if !base.is_null() {
        materialize_for(base as usize, size, Touch::TitleCall);
    }
"""

h1 = """    if kind & MEM_RESERVE_PLACEHOLDER != 0 && !at.is_null() {
        physical().reservations.push((at as usize, size));
    }
    if !constrained && kind & MEM_COMMIT != 0 && !at.is_null() {
        victory_gpu::title_memory_committed(at as usize, size);
    }
    if at.is_null() {
        let error = std::io::Error::last_os_error();
        if kind & MEM_COMMIT != 0 {
            commit_refused(size, "virtual memory", error);
        } else {
            tracing::warn!(
                size,
                kind = format_args!("{kind:#x}"),
                flags = format_args!("{flags:#x}"),
                %error,
                "XMemVirtualAlloc refused"
            );
        }
"""

# h3: theirs' Section + ours' SECTION_BYTES static
src, hs = resolve1.hunks(P)
h3 = hs[3][2] + """
/// Bytes of the physical-page sections alive (reserved; committed as
/// mapped or written).
static SECTION_BYTES: AtomicUsize = AtomicUsize::new(0);
"""

h4 = hs[4][2] + """        SECTION_BYTES.fetch_sub(self.pages * PAGE, Ordering::Relaxed);
"""

h6 = """    SECTION_BYTES.fetch_add(size, Ordering::Relaxed);
""" + hs[6][2]

h7 = """        let committed = commit(at.Value as usize, len);
        victory_gpu::framelog::add(victory_gpu::framelog::T::DiskViews, 1);
        victory_gpu::framelog::add(victory_gpu::framelog::T::DiskViewUs, started.elapsed().as_micros() as u64);
        if !committed {
            tracing::warn!(error = %std::io::Error::last_os_error(), n, "physical pages not committed for a write");
            return None;
        }
        Some(view)
"""

h8 = """    use victory_gpu::framelog::{T, add};
    let entered = std::time::Instant::now();
    let mapped = map_pages(address, n, pages, lazy_views() || soft_views() & SOFT_POOL != 0);
    add(T::RemapDirectCalls, 1);
    add(T::RemapDirectPages, n as u64);
    add(T::RemapDirectUs, entered.elapsed().as_micros() as u64);
    mapped
}

/// [`XMemMapPhysicalPages`]; `lazy`: views are mapped on the host when first
/// touched ([`lazy_views`], [`soft_views`]); the caller counts the call for
/// the frame log.
fn map_pages(address: *mut c_void, n: usize, pages: *const usize, lazy: bool) -> *mut c_void {
    let base = address as usize;
    // Read before any lock is taken: a fault in title memory may wait for
    // the mapping call that holds them.
    // SAFETY: the title passes `n` page numbers, or null.
    let pages = (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) }.to_vec());
    if port_on()
        && let Some(pages) = &pages
    {
        arrange(base, pages);
    }
    let mut costed = Costed::begin(Some(n));
    let _called = victory_gpu::timeline::span("the title's XMemMapPhysicalPages");
    let _busy = Busy::begin();
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`). Taken before
    // the views, as everywhere: a reader that holds it may fault into a
    // lazy view and need the views to make it real ([`materialize`]).
    let _changing = victory_gpu::title_memory_remapping_range(physical().unmap_extent(base, base + n * PAGE));
    let mut physical = physical();
    if let Some(costed) = &mut costed {
        costed.waited = costed.started.elapsed().as_nanos() as u64;
"""

h10 = """    use victory_gpu::framelog::{T, add};
    // From entry to return, the wait for the page table's lock included.
    let entered = std::time::Instant::now();
    let _costed = Costed::begin(None);
    let mapped = alias(to, from, len);
    add(T::RemapAliasCalls, 1);
    add(T::RemapAliasUs, entered.elapsed().as_micros() as u64);
    mapped
}

thread_local! {
    /// The page this thread mapped last through [`map_alias`]: the address
    /// after it, its section and its index there. For the frame log's
    /// count of pages that one view could have shown together.
    static ALIASED: std::cell::Cell<(usize, usize, usize)> = const { std::cell::Cell::new((0, 0, 0)) };
}

/// [`map_alias`], not counted as a call of its own.
fn alias(to: u64, from: u64, len: u64) -> bool {
    alias_as(to, from, len, alias_runs(), soft_views() & SOFT_COPIES != 0)
}

/// [`alias`]; `runs`: a view per run of consecutive section pages
/// ([`alias_runs`]), else a view per page; `lazy`: the views are kept in the
/// books alone ([`soft_views`]), a page each.
fn alias_as(to: u64, from: u64, len: u64, runs_in_one: bool, lazy: bool) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let lazy = lazy.then_some(Soft::NotedCopies);
    let _busy = Busy::begin();
    // As in `XMemMapPhysicalPages`: no read in place meets the gap (taken
    // before the views, as there).
    let _changing = victory_gpu::title_memory_remapping_range(physical().unmap_extent(to, to + len));
    let mut physical = physical();
"""

h11 = hs[11][2].replace(
    """    for (k, slice) in slices.into_iter().enumerate() {
        if !place(&mut physical, to + k * PAGE, PAGE, slice, lazy) {""",
    """    for (k, slice) in slices.into_iter().enumerate() {
        use victory_gpu::framelog::{T, add};
        // Counted only: whether this page follows the one mapped just
        // before, in address and in its section (one view could show
        // both), or is that very page again (no view can).
        let (section, index) = (Arc::as_ptr(&slice.section) as usize, slice.index);
        let (after, last_section, last_index) = ALIASED.get();
        add(T::RemapAliasPages, 1);
        if after == to + k * PAGE && last_section == section {
            if index == last_index + 1 {
                add(T::RemapAliasRunPages, 1);
            } else if index == last_index {
                add(T::RemapAliasSamePages, 1);
            }
        }
        ALIASED.set((to + k * PAGE + PAGE, section, index));
        if !place(&mut physical, to + k * PAGE, PAGE, slice, lazy) {""")

h14 = """        shown_mut().forget(at, size, &slice);
        for k in 0..size / PAGE {
            if let std::collections::hash_map::Entry::Occupied(mut o) =
                physical.shown.entry(slice.key(k))
            {
                o.get_mut().retain(|&shown| shown != at + k * PAGE);
                if o.get().is_empty() {
                    o.remove();
                }
            }
        }
        if at < base || at + size > end {
            // Parts of it stay, and are mapped again below.
            victory_gpu::title_memory_splitting(at, size);
        }
        victory_gpu::title_memory_unmapping(at, size);
        use victory_gpu::framelog::{T, add};
        let started = std::time::Instant::now();
        victory_gpu::title_memory_remapped(at, size);
        let noted = std::time::Instant::now();
        add(T::RemapNotesUs, (noted - started).as_micros() as u64);
        let meter = Meter::begin();
        victory_gpu::title_memory_view_gone(at, size);
        if unlazy(at) {
            // A view only in our books ([`map_lazy`]): no host view to
            // unmap; its own placeholder is there, and what stays of it
            // stays lazy.
            count(Soft::Gone, 1);
            if at >= base && at + size <= end {
                physical.holes.insert(at, size);
            }
            if at < base {
                map_lazy(physical, at, base - at, slice.clone(), None);
            }
            if at + size > end {
                let index = slice.index + (end - at) / PAGE;
                map_lazy(physical, end, at + size - end, Slice { index, ..slice }, None);
            }
            continue;
        }
"""

h15 = """        });
        meter.end(false);
        add(T::RemapUnmapUs, noted.elapsed().as_micros() as u64);
        if gone != 0 && at >= base && at + size <= end {
            physical.holes.insert(at, size);
        }
        // What of the view lies outside the range asked for is mapped
        // again below: the whole view went for it.
        let outside = base.saturating_sub(at) + (at + size).saturating_sub(end);
        if outside != 0 {
            add(T::RemapCuts, 1);
            add(T::RemapCutBytes, outside as u64);
"""

choices = [h0, h1, 'both', h3, h4, 'none', h6, h7, h8, 'theirs', h10, h11, 'theirs', 'none', h14, h15]
resolve1.resolve(P, choices)
