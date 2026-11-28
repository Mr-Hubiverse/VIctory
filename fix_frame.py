src = open('crates/victory-gpu/src/frame.rs', encoding='utf-8').read()
old1 = """<<<<<<< HEAD
/// Default runtime frame-rate cap (240 Hz OLED target); `VICTORY_FPS=0`
/// uncaps completely, `VICTORY_FPS=<n>` caps at `<n>`. Tests keep the title's
/// own interval.
const DEFAULT_FPS: u32 = 240;

fn fps_override() -> Option<u32> {
    if cfg!(test) {
        return None;
    }
    static CAP: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| match std::env::var("VICTORY_FPS") {
        Ok(v) => v.trim().parse::<u32>().ok().filter(|&n| n > 0),
        Err(_) => Some(DEFAULT_FPS),
    })
}

fn uncapped() -> bool {
    !cfg!(test) && fps_override().is_none()
}

=======
"""
new1 = old1.replace("<<<<<<< HEAD\n", "").replace("=======\n", "")
assert old1 in src
src = src.replace(old1, new1, 1)
old2 = """<<<<<<< HEAD
    let interval = if let Some(fps) = fps_override() {
        Duration::from_nanos(1_000_000_000 / u64::from(fps))
    } else {
        match length {
            0 => SIXTIETH,
            us => Duration::from_micros(u64::from(us)),
        }
    };
    tracing::info!(?interval, uncapped = uncapped(), "frame interval");
=======
    let asked_by_title = match length {
        0 => SIXTIETH,
        us => Duration::from_micros(u64::from(us)),
    };
    // VICTORY_FRAME_INTERVAL_MS (with the title's own rate patched to match)
    // first, then VICTORY_FRAME_INTERVAL_US (our clock alone).
    let interval = asked().or_else(interval_override).unwrap_or(asked_by_title);
    tracing::info!(?interval, title = ?asked_by_title, period, flags, "frame interval");
>>>>>>> nvidia/main
"""
new2 = """    let asked_by_title = match length {
        0 => SIXTIETH,
        us => Duration::from_micros(u64::from(us)),
    };
    // VICTORY_FRAME_INTERVAL_MS (with the title's own rate patched to match)
    // first, then VICTORY_FRAME_INTERVAL_US (our clock alone), then the
    // VICTORY_FPS cap (240 by default, so the title's own rate applies only
    // with VICTORY_FPS=0).
    let fps = fps_override().map(|fps| Duration::from_nanos(1_000_000_000 / u64::from(fps)));
    let interval = asked().or_else(interval_override).or(fps).unwrap_or(asked_by_title);
    tracing::info!(?interval, uncapped = uncapped(), title = ?asked_by_title, period, flags, "frame interval");
"""
assert old2 in src
src = src.replace(old2, new2, 1)
open('crates/victory-gpu/src/frame.rs','w',encoding='utf-8').write(src)
print('markers left:', '<<<<<<<' in src)
