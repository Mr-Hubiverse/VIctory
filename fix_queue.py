import resolve1 as r

h2 = '''    let _called = crate::timeline::span("the title's queue Signal");
    #[cfg(windows)]
    if behind::on() && !this.is_null() {
        behind::signal(this, fence, value);
        return S_OK;
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::fence::pace::signal_called(fence as usize, value);
'''

h3 = '''            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                let now = (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64;
                #[cfg(all(windows, feature = "diagnostics"))]
                crate::host::spans::stamped(at as usize, now);
                as_loaded(now, at as usize, gpu_load())
            }
            Written::HostTimestamp(_) => gpu_clock(),
'''

h4 = '''    #[cfg(windows)]
    let presenting = std::time::Instant::now();
    let hr = present_now(this, planes, plane_parameters);
    #[cfg(windows)]
    {
        crate::framelog::add(crate::framelog::T::PresentUs, presenting.elapsed().as_micros() as u64);
        crate::framelog::frame();
        #[cfg(feature = "diagnostics")]
        crate::fence::pace::frame();
    }
    hr
}

fn present_now(this: This, planes: u32, plane_parameters: *const Plane) -> HRESULT {
    #[cfg(windows)]
    let presenting = std::time::Instant::now();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::copystat::present_begins();
    let called = crate::timeline::span("the title's Present");
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::title_thread();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::profile::frame();
    // With `present_behind`: the frame is shown by our thread, behind what
    // the queue holds, and nothing is run or waited for here.
    #[cfg(windows)]
    let sent_behind = present_behind() && send_behind(this, planes, plane_parameters);
    #[cfg(not(windows))]
    let _ = (this, planes, plane_parameters);
    // The frame shown and, of several planes, a second overlaid on it.
    #[cfg(windows)]
    let (shown, over) = plane_refs(planes, plane_parameters);
    // Submissions sent to the card and not waited for are settled first,
    // and what that frees runs in its own order, before anything is forced.
    #[cfg(windows)]
    if sent_behind || behind::on() {
        // The present goes into the line of what the queues were given.
    } else if crate::host::late() && !crate::host::late_free_present() {
        // (With `behind`: what the queues hold runs here first, and what
        // it sent is settled with the rest.)
        if behind() {
            release_held_now(false);
        }
'''

h5 = '''    if !sent_behind && !behind::on() {
        release_held(forces());
    }
    #[cfg(windows)]
    crate::framelog::add(crate::framelog::T::PresentLineUs, presenting.elapsed().as_micros() as u64);
'''

h6 = '''    if behind::on() && !this.is_null() {
        const COPY_SOURCE: u32 = 0x800;
        let base_res = if planes > 0
            // SAFETY: the title passes `planes` plane descriptions.
            && let Some(plane) = unsafe { plane_parameters.as_ref() }
            && plane.count > 0
            && !plane.resources.is_null()
        {
            // SAFETY: the plane lists `count` resources.
            Some(unsafe { *plane.resources })
        } else {
            None
        };
        let take_copy = |resource: This| {
            crate::resource::host_copy(resource).map(|copy| {
                let state = crate::resource::host_transition(resource, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before);
                crate::resource::host_transition(resource, state);
                // (Lists not translated yet leave it in no state yet: the
                // one it is in when its turn in the line comes.)
                let state = if crate::list::thin::wanted() { u32::MAX } else { state };
                (resource as usize, copy, state)
            })
        };
        let texture = base_res.and_then(take_copy);
        let overlay = (planes > 1 && !plane_parameters.is_null())
            .then(|| overlay_resource(plane_parameters))
            .flatten()
            .filter(|&over_res| base_res != Some(over_res))
            .and_then(take_copy);
        let resource = texture.as_ref().map_or(0, |t| t.0);
        behind::present(
            this,
            texture.map(|(_, copy, state)| (copy, state)),
            overlay.map(|(_, copy, state)| (copy, state)),
            Shown { frame, resource },
        );
        called.end();
        #[cfg(feature = "diagnostics")]
        frame_ends();
        return S_OK;
    }
    #[cfg(windows)]
    if !sent_behind
        && let Some(over) = plane(over.as_ref())
        && let Some(host) = crate::host::host()
    {
        host.present_over(plane(shown.as_ref()), over, frame);
    } else if planes > 0
'''

h7 = '''    #[cfg(windows)]
    LAST_PRESENT_MS.store(crate::host::now_ms().max(1), Ordering::Relaxed);
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    called.end();
    #[cfg(all(windows, feature = "diagnostics"))]
    frame_ends();
    S_OK
}
/// The first plane's resource (the frame shown) and, of several planes,
/// the second's (overlaid on it), as [`object::Ref`]s.
#[cfg(windows)]
fn plane_refs(planes: u32, plane_parameters: *const Plane) -> (Option<object::Ref>, Option<object::Ref>) {
    let mut shown = None;
    let mut over = None;
    if planes > 1 && !plane_parameters.is_null() {
        over = second_plane(plane_parameters).map(object::Ref::new);
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::info!(planes, found = over.is_some(), "several planes presented"));
    }
    if planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
    {
        // SAFETY: the plane lists `count` resources.
        let resource = unsafe { *plane.resources };
        shown = (!resource.is_null()).then(|| object::Ref::new(resource));
    }
    (shown, over)
}

/// A presented resource's host texture and the host state it is in.
#[cfg(windows)]
fn plane(
    resource: Option<&object::Ref>,
) -> Option<(&'static windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)> {
    let resource = resource?.get();
    let texture = crate::resource::host_resource(resource)?;
    const COPY_SOURCE: u32 = 0x800;
    let state = crate::resource::host_transition(resource, COPY_SOURCE)
        .map_or(COPY_SOURCE, |(_, before)| before);
    crate::resource::host_transition(resource, state);
    Some((texture, state))
}

/// A present that found its queue holding work, run when its turn comes
/// (the queue's line runs it through [`release_held_now`]).
#[cfg(windows)]
fn show_held(shown: Option<object::Ref>, over: Option<object::Ref>) {
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if let Some(over) = plane(over.as_ref())
        && let Some(host) = crate::host::host()
    {
        host.present_over(plane(shown.as_ref()), over, frame);
    } else if let Some((texture, state)) = plane(shown.as_ref())
        && let Some(host) = crate::host::host()
    {
        let _ = host.present(texture, state, frame);
    }
    LAST_PRESENT_MS.store(crate::host::now_ms().max(1), Ordering::Relaxed);
}

/// Diagnostics: the title's present is over.
#[cfg(all(windows, feature = "diagnostics"))]
fn frame_ends() {
    crate::timeline::presented();
    crate::faultbook::frame();
    crate::ab::presented(|switch, was, will| {
        if switch == "VICTORY_ASYNC_SUBMIT" && was && !will {
            behind::leave();
        }
    });
}

/// The frame a present shows, and the title's resource it shows.
#[cfg(windows)]
#[derive(Clone, Copy)]
struct Shown {
    frame: u64,
    resource: usize,
}

/// Shows `texture` (with optional `overlay` plane blended over it) in our window.
#[cfg(windows)]
fn show_planes(
    texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    state: u32,
    overlay: Option<(&windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32)>,
    what: Shown,
) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let shown = host.present_planes(texture, state, overlay, what.frame);
    #[cfg(feature = "diagnostics")]
    blinks(what.frame, what.resource as This, state, shown);
    #[cfg(not(feature = "diagnostics"))]
    let _ = (shown, what.resource);
'''

h8 = '''    // Read now: the batch and range arrays belong to the title (often on its
    // stack) and may be reused before a held-back queue runs them.
    let _called = crate::timeline::span("the title's CopyPageMappings");
'''

h9 = '''    let mut ops = Vec::new();
    // The remapping meter: what the title asks for, by kind of range.
    crate::framelog::add(crate::framelog::T::PageMapCalls, 1);
    let mut ok = true;
    let many = crate::resource::many_page_mapper();
    let mut gathered: Vec<(u64, u64, u64)> = Vec::new();
    let mut map = |to: u64, from: u64, len: u64| match many {
        Some(_) => {
            gathered.push((to, from, len));
            true
        }
        None => map(to, from, len),
    };
'''

h10 = '''    if let Some(many) = many {
        ok &= many(&gathered);
        let _ = ops;
        return if ok { S_OK } else { E_FAIL };
    }
    // Behind what the queue holds, as it is on the console; at once if it
    // holds nothing.
    #[cfg(windows)]
    let held = hold(this, || {
        let ops = ops.clone();
        Held::Work(Box::new(move || apply_page_mappings(map, pool, &ops)))
    });
    #[cfg(not(windows))]
    let held = false;
    if !held {
        apply_page_mappings(map, pool, &ops);
    }
    S_OK
}

fn apply_page_mappings(map: crate::resource::PageMapper, pool: u64, ops: &[PageOp]) {
    let mut ok = true;
    for &op in ops {
        ok &= match op {
            PageOp::Copy { at, source, len } => map(at, source, len),
            PageOp::Fill { at, source, count } => {
                (0..count).all(|k| map(at + k * PAGE, source, PAGE))
            }
            PageOp::Unmap { at, len } => map(at, 0, len),
            PageOp::Invalid => false,
        };
'''

r.resolve('crates/victory-gpu/src/queue.rs', [
    'ours', 'ours', h2, h3, h4, h5, h6, h7, h8, 'ours', 'ours',
])
print('done')
