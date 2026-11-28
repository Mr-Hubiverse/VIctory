import re
src = open('crates/victory-hle/src/modules/dstorage.rs', encoding='utf-8').read()
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
hunks = list(pat.finditer(src))
assert len(hunks) == 6, len(hunks)

# hunk0: ours' submit + submit_as (single signature set)
m0 = hunks[0].group(1)

# hunk1: ours' held logic + theirs' trace_batch (with `early` derived)
m1 = '''        let now = std::time::Instant::now();
        let early = !explicit;
        // The title's Submit releases what a self-submit held: how long it waited.
        let released = if explicit && !lane.held.is_empty() {
            lane.held_since
                .take()
                .map(|since| (lane.held.len() as u64, now.saturating_duration_since(since).as_micros() as u64))
        } else {
            None
        };
        trace_batch(id, &lane.unsubmitted, early);
        let mut commands = if explicit { std::mem::take(&mut lane.held) } else { Vec::new() };
        commands.append(&mut lane.unsubmitted);
        for command in commands {
            let hold = !explicit
                && match &command {
                    Command::Read(r) => r.options & PAGE_DESTINATION != 0,
                    Command::Status(_) => !lane.held.is_empty(),
                };
            if hold {
                lane.held.push(command);
                continue;
            }
'''

# hunk2: both blocks; theirs' call adapted to submit_as
m2 = '''        // Full with reads held for a Submit the title cannot make while it
        // waits here: they go now, page lists as they are.
        if !lane.held.is_empty() {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!("dstorage: a full queue's reads into physical pages start before the title's Submit");
            }
            lanes.submit(id);
            scheduler.work.notify_all();
''' + '''
        if !lane.unsubmitted.is_empty() && lane.reads.is_empty() && lane.running.is_empty() {
            tracing::warn!(
                queue = id,
                commands = lane.unsubmitted.len(),
                "dstorage: a queue full of commands that are not submitted submits itself"
            );
            lanes.submit_as(id, false);
            scheduler.work.notify_all();
            continue;
'''

# hunk3: theirs' gate + ours' submit_as
m3 = '''    if half_submits() && lane.unsubmitted.len() >= (lane.capacity / 2).max(1) {
        let capacity = lane.capacity;
        lanes.submit_as(id, false);
'''

# hunk4,5: theirs' / both
m4 = hunks[4].group(2)
m5 = hunks[5].group(1) + hunks[5].group(2)

out = (src[:hunks[0].start()] + m0 + src[hunks[0].end():hunks[1].start()] + m1
       + src[hunks[1].end():hunks[2].start()] + m2 + src[hunks[2].end():hunks[3].start()]
       + m3 + src[hunks[3].end():hunks[4].start()] + m4 + src[hunks[4].end():hunks[5].start()]
       + m5 + src[hunks[5].end():])
open('crates/victory-hle/src/modules/dstorage.rs', 'w', encoding='utf-8').write(out)
print('done', '<<<<<<<' in out)
