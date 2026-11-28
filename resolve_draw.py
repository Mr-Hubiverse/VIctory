#!/usr/bin/env python3
"""Resolve the merge conflicts in crates/victory-gpu/src/draw.rs (deep union)."""
import re, sys

PATH = 'crates/victory-gpu/src/draw.rs'
src = open(PATH).read()
lines = src.split('\n')
marks = [i for i, l in enumerate(lines) if l.startswith(('<<<<<<<', '=======', '>>>>>>>'))]
assert len(marks) % 3 == 0, len(marks)
N = len(marks) // 3

hunks = []  # (start_line_idx, mid_idx, end_idx)
for k in range(0, len(marks), 3):
    hunks.append((marks[k], marks[k + 1], marks[k + 2]))

def ours(n):
    s, m, e = hunks[n]
    return '\n'.join(lines[s + 1:m])

def theirs(n):
    s, m, e = hunks[n]
    return '\n'.join(lines[m + 1:e])

I8 = '        '
I12 = '            '
I16 = '                '

choices = {}

# ---------------- hunk 0: PASS_BUDGET (theirs' 3-tuple) + ours' statics
o0 = ours(0)
# ours' first line is the 2-tuple PASS_BUDGET decl; drop it, keep the rest
o0_rest = '\n'.join(o0.split('\n')[1:])
choices[0] = theirs(0) + '\n' + o0_rest

choices[1] = 'ours'
choices[2] = 'both'

# ---------------- hunk 3: fresh_tables must be FreshTable-typed (theirs'), keep ours' other fields
o3 = ours(3).split('\n')
# ours: fresh_tables decl (drop), then fresh_records..made + doc tail
o3_rest = '\n'.join(o3[1:])
choices[3] = theirs(3).split('\n')[0] + '\n' + o3_rest + '\n' + '\n'.join(theirs(3).split('\n')[1:])

choices[4] = 'theirs'
choices[5] = 'ours'
choices[6] = 'ours'
choices[7] = 'both'
choices[8] = 'ours'

# ---------------- hunk 9: ours' fns + theirs' `fn set -> bool` signature
o9 = ours(9)
assert o9.rstrip().endswith('fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {'), o9[-200:]
o9 = o9.rstrip()
o9 = o9[: o9.rfind('\n')] + '\n' + '    /// Sets root argument `slot`: whether that changed it.\n    fn set(roots: &mut Vec<Root>, slot: u32, root: Root) -> bool {'
choices[9] = o9

choices[10] = 'ours'
choices[11] = 'ours'
choices[12] = 'ours'
choices[13] = 'ours'
choices[14] = 'ours'
choices[15] = 'ours'

# ---------------- hunk 16: ours' import + USER_ROOT_ADDRESS
o16 = ours(16)
o16 = o16.replace('USER_MEMORY_BASE,\n            USER_VIEW_ADDRESS',
                  'USER_MEMORY_BASE,\n            USER_ROOT_ADDRESS, USER_VIEW_ADDRESS')
assert 'USER_ROOT_ADDRESS' in o16
choices[16] = o16

choices[17] = 'theirs'
choices[18] = 'theirs'

# ---------------- hunk 19: theirs' memory_ready body + ours' `fn memory` + `fn ambient` (renamed `lit`)
t19 = theirs(19)
o19 = ours(19)
# drop ours' memory_ready body: keep from "    /// For shaders that reach memory"
cut = o19.find('    /// For shaders that reach memory')
assert cut > 0
o19_tail = o19[cut:]
# rename ours' ambient (2-arg) to `lit` to avoid clashing with theirs' 3-arg ambient
o19_tail = o19_tail.replace('fn ambient(', 'fn lit(', 1)
assert 'fn lit(' in o19_tail
choices[19] = t19 + '\n    }\n\n' + o19_tail

# ---------------- hunk 20: ours' by_address/memory + theirs' lent_heap + heap bind block
o20 = ours(20)
# keep ours' by_address + memory lines, drop ours' `if self.host_heaps != ...` + heaps vec
cut = o20.find('        if self.host_heaps != self.heaps')
assert cut > 0
o20_head = o20[:cut]
choices[20] = o20_head + theirs(20)

# ---------------- hunk 21: ours' list_reuse keep + theirs' bind_pipeline
choices[21] = ours(21) + '\n' + theirs(21)

# ---------------- hunk 22: theirs' picked prelude + union-sig `tables` call
t22 = theirs(22)
t22 = t22.replace('let made = self.tables(list, &t.tables, compute);',
                  'let made = self.tables(list, &t.tables, &t.host.half_views, compute, &spans, &mut staged);')
assert '&t.host.half_views' in t22
choices[22] = t22

choices[23] = 'theirs'
choices[24] = 'theirs'

# ---------------- hunk 25: ours' material_demand block + theirs' user-diagnostics block
o25 = ours(25)
t25 = theirs(25)
choices[25] = o25 + '\n' + I16 + '}\n' + t25

choices[26] = 'theirs'

# ---------------- hunks 27/28: theirs' settle_for_fence retry + ours' slot.index/hold_dynamic
def dyn_slot(n):
    o = ours(n)
    t = theirs(n)
    # theirs': shadow line, dynamic().or_else(settle), else{return}, push/insert tail
    # ours': dynamic() else { note_why; return } ; slot.index = Some(index); hold_dynamic
    return (I16 + 'let Some(index) = host.dynamic(1).or_else(|| {\n'
            + I16 + '    crate::host::settle_for_fence();\n'
            + I16 + '    host.dynamic(1)\n'
            + I16 + '}) else {\n'
            + I16 + '    #[cfg(feature = "diagnostics")]\n'
            + I16 + '    crate::pipeline::note_why("no free dynamic slot for the one-view table of a root view".to_owned());\n'
            + I16 + '    return false;\n'
            + I16 + '};\n'
            + I16 + 'slot.index = Some(index);\n'
            + I16 + 'self.hold_dynamic(key.0, host.id(), index, 1);')
choices[27] = dyn_slot(27)
choices[28] = dyn_slot(28)

choices[29] = 'theirs'

# ---------------- hunk 30: ours' heap_worker kick + theirs' stage/HEAP_SPLIT wrap
t30 = theirs(30)
t30 = t30.replace('let _ = self.heap_textures(list, heap, host, None);',
                  'let textures_began = std::time::Instant::now();\n'
                  + I8 + 'if !crate::heap_worker::kick(heap) {\n'
                  + I8 + '    Self::heap_textures(Into::List(list), heap, host, None);\n'
                  + I8 + '}')
t30 += '\n' + I8 + 'HEAP_PASS_PARTS[0].fetch_add(textures_began.elapsed().as_micros() as u64, Ordering::Relaxed);'
choices[30] = t30

choices[31] = 'both'
choices[32] = 'both'

# ---------------- hunk 33: theirs' bake block; adapt heap_textures calls to Into::List
t33 = theirs(33)
t33 = t33.replace('served = self.heap_textures(bake_list, heap, host, Some(&mut sync));',
                  'served = Self::heap_textures(Into::List(bake_list), heap, host, Some(&mut sync));')
t33 = t33.replace('let _ = self.heap_textures(list, heap, host, None);',
                  'let _ = Self::heap_textures(Into::List(list), heap, host, None);')
assert 'self.heap_textures' not in t33, t33
choices[33] = t33

# ---------------- hunk 34: ours' sources/only/forced prelude + theirs' room(memory)
choices[34] = (I8 + '// A skin-bake draw\'s own textures (`State::bake_sources_pass`): a\n'
    + I8 + '// run for them alone, on the thread that records the draw. Nothing\n'
    + I8 + '// of the queue\'s upkeep is done in it, which a pass of this frame\n'
    + I8 + '// did or will do.\n'
    + I8 + 'let sources = PASS_SOURCES.take();\n'
    + I8 + 'let only = sources.is_some();\n'
    + I8 + '// A bake draw\'s own run with `VICTORY_BAKE_SOURCES=full` goes on\n'
    + I8 + '// as if the card had room, within its own bytes.\n'
    + I8 + 'let forced = only && bake_sources_mode().0 == BakeMode::Full;\n'
    + I8 + 'let mut room = forced || crate::resource::room(memory);\n'
    + I8 + 'host.set_texture_pressure(!room);')

# ---------------- hunk 35: theirs' baking cell + ours' `if !only` + theirs' before-wrap
choices[35] = (I8 + 'let baking = crate::texture::BAKING.get();\n'
    + I8 + 'if !only {\n'
    + I12 + 'before(0, || {')

# ---------------- hunk 36: union of upkeep + due/known/census acquisition
choices[36] = (I12 + '// Levels on demand: the slots of textures whose first level\n'
    + I12 + '// with memory (levels on demand), or whose host copy for one\n'
    + I12 + '// with more levels, changed since the last pass are made again\n'
    + I12 + '// from it; those whose new view\'s upload has run are made from\n'
    + I12 + '// that.\n'
    + I12 + 'host.texture_views_remade();\n'
    + I12 + 'host.texture_slots_flip();\n'
    + I12 + 'host.texture_sweep(heap, frame);\n'
    + I12 + 'if baking {\n'
    + I12 + '    host.texture_slots_bake_written(frame);\n'
    + I12 + '}\n'
    + I12 + 'host.texture_slots_short(frame);\n'
    + I12 + '// Slots that read zeros for good and nothing asks for again\n'
    + I12 + '// (`VICTORY_REASK_BLANK=1`; off by default; diagnostics builds).\n'
    + I12 + '#[cfg(feature = "diagnostics")]\n'
    + I12 + 'host.texture_slots_blank(frame);\n'
    + I12 + '});\n'
    + I8 + '}\n'
    + I8 + '// And with room, the textures whose upload left a held level\n'
    + I8 + '// without the card\'s memory are uploaded again.\n'
    + I8 + 'if room && !only {\n'
    + I8 + '    before(0, || host.texture_slots_waiting(frame));\n'
    + I8 + '}\n'
    + I8 + 'let wanted = crate::material_demand::enabled().then(|| host.material_demand.take(heap, host, frame));\n'
    + I8 + 'let required = PASS_BUDGET.get().is_some_and(|(_, _, bake)| bake);\n'
    + I8 + '// (And what the queue knew of each, `known`: see `Tells`.)\n'
    + I8 + 'let (mut due, mut known, census) = match &sources {\n'
    + I8 + '    // The bake draw\'s own, due or not, held or not.\n'
    + I8 + '    Some(slots) => {\n'
    + I8 + '        let due: Vec<(u32, u64, bool)> = host.texture_slots_queued(slots);\n'
    + I8 + '        let n = due.len();\n'
    + I8 + '        (due, vec![None; n], [0; 6])\n'
    + I8 + '    }\n'
    + I8 + '    None => {\n'
    + I8 + '        let (due, known, census) =\n'
    + I8 + '            before(2, || host.texture_slots_due_known(frame, if baking { 1 << 16 } else { 4096 }));\n'
    + I8 + '        let due: Vec<(u32, u64, bool)> = due.into_iter().map(|(i, w)| (i, w, false)).collect();\n'
    + I8 + '        // What the slots shaders pick by index show, every five seconds.\n'
    + I8 + '        #[cfg(feature = "diagnostics")]\n'
    + I8 + '        host.placed_slot_census();\n'
    + I8 + '        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);\n'
    + I8 + '        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {\n'
    + I8 + '            count.store(now, Ordering::Relaxed);\n'
    + I8 + '        }\n'
    + I8 + '        // Where the card\'s memory is, and what the due slots would\n'
    + I8 + '        // need (`VICTORY_CARD_ACCOUNTING=1`; a line when the card\n'
    + I8 + '        // first has no room, and every five seconds after it).\n'
    + I8 + '        Self::card_accounting(heap, host, frame, room);\n'
    + I8 + '        (due, known, census)\n'
    + I8 + '    }\n'
    + I8 + '};\n'
    + I8 + '// (Those that are due later after all that are due.)\n'
    + I8 + 'let on_time = due.len();\n'
    + I8 + 'if baking && !only {\n'
    + I8 + '    before(2, || {\n'
    + I8 + '        let taken: std::collections::HashSet<u32> = due.iter().map(|&(index, _, _)| index).collect();\n'
    + I8 + '        static AHEAD: AtomicU64 = AtomicU64::new(u64::MAX);\n'
    + I8 + '        let again = AHEAD.swap(frame, Ordering::Relaxed) == frame;\n'
    + I8 + '        let (ahead, ahead_known, _) = host.texture_slots_due_known(frame + BAKE_AHEAD[usize::from(again)], 1 << 16);\n'
    + I8 + '        for (slot, knew) in ahead.into_iter().zip(ahead_known) {\n'
    + I8 + '            if !taken.contains(&slot.0) {\n'
    + I8 + '                due.push((slot.0, slot.1, false));\n'
    + I8 + '                known.push(knew);\n'
    + I8 + '            }\n'
    + I8 + '        }\n'
    + I8 + '    });\n'
    + I8 + '}\n'
    + I8 + 'if let Some(wanted) = wanted {\n'
    + I8 + '    crate::material_demand::promote(&mut due, &wanted, frame);\n'
    + I8 + '}\n'
    + I8 + '// What a run for a bake draw\'s own may upload.\n'
    + I8 + 'let sourced_at_most = bake_sources_mode().1;\n'
    + I8 + 'let mut sourced_bytes = 0u64;')

# ---------------- hunk 37: ours' `plain` + theirs' on_time rule + ours' catching_up bump
choices[37] = (I8 + 'let plain = span;\n'
    + I8 + 'let (per_frame, span) = if catchup && on_time >= 1024 { (per_frame * 3, span * 3) } else { (per_frame, span) };\n'
    + I8 + 'let (per_frame, span) = if catchup && census[0] > LONG && catching_up(std::time::Duration::ZERO) {\n'
    + I8 + '    (per_frame.max(PER_FRAME_CATCHING_UP), span.max(SPAN_CATCHING_UP))\n'
    + I8 + '} else {\n'
    + I8 + '    (per_frame, span)\n'
    + I8 + '};')

# ---------------- hunk 38: theirs' prelude + loop over (due, known) + ours' only/ahead/record
t38 = theirs(38)
choices[38] = t38 + '\n' + (
    I12 + '// A run for a bake draw\'s own ends at its own bytes, not at the\n'
    + I12 + '// frame\'s budget or time.\n'
    + I12 + 'if only && sourced_bytes >= sourced_at_most {\n'
    + I12 + '    once!("a skin-bake draw samples more than VICTORY_BAKE_SOURCES_MB of textures not on the card yet: the rest take their turn");\n'
    + I12 + '    break;\n'
    + I12 + '}\n'
    + I12 + 'if matches!(into, Into::Ahead(_)) && crate::heap_worker::make_way() {\n'
    + I12 + '    break;\n'
    + I12 + '}\n'
    + I12 + '// The record the slot has now, and the texture it names held\n'
    + I12 + '// while a thread of ours works on it. (Released meanwhile, or\n'
    + I12 + '// not a texture: nothing to upload.)\n'
    + I12 + 'let record = heap.handle(index).and_then(crate::descriptor::view_and_key);\n'
    + I12 + 'let _held = match &record {\n'
    + I12 + '    Some((View::Shader(object, _), _)) if matches!(into, Into::Ahead(_)) => {\n'
    + I12 + '        let held = crate::resource::hold(*object as This);\n'
    + I12 + '        if held.is_none() {\n'
    + I12 + '            tells.tell(index, write, Told::Done(None));\n'
    + I12 + '            continue;\n'
    + I12 + '        }\n'
    + I12 + '        held\n'
    + I12 + '    }\n'
    + I12 + '    _ => None,\n'
    + I12 + '};')
# adjust loop head: theirs' `for ((index, write), knew)` -> include recovered
choices[38] = choices[38].replace('for ((index, write), knew) in due.into_iter().zip(known) {',
                                  'for ((index, write, recovered), knew) in due.into_iter().zip(known) {')

# ---------------- hunk 39: theirs' checked/knew + ours' recheck_every fallback arm
t39 = theirs(39)
t39 = t39.replace(
    "None => Told::Done(None),",
    "None => match recheck_every() {\n"
    + I16 + "    0 => Told::Done(None),\n"
    + I16 + "    every => Told::Recheck(frame + every, (at, len), print, looks, uploaded),\n"
    + I16 + "},")
choices[39] = t39

# ---------------- hunk 40: theirs' held/there + ours' material_upload_allowed gate (recovered)
t40 = theirs(40)
choices[40] = t40 + '\n' + (
    I12 + 'if recovered && !host.material_upload_allowed(index, write, address, len, mapped) {\n'
    + I12 + '    tells.tell(index, write, Told::Done(None));\n'
    + I12 + '    continue;\n'
    + I12 + '}')

choices[41] = 'theirs'

# ---------------- hunk 42: theirs' mapped test + ours' only-gate + PASS_SOURCED
o42 = ours(42)
t42 = theirs(42)
choices[42] = (
    I16 + "// (One of a bake draw's own is left so only if its host\n"
    + I16 + "// slot shows what was uploaded then.)\n"
    + I16 + "if mapped <= rest(0, || host.texture_slot_mapped(index))\n"
    + I16 + "    && !(only\n"
    + I16 + "        && !heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle)))\n"
    + I16 + "{\n"
    + I16 + "    if only {\n"
    + I16 + "        let [uploaded, bytes, unmapped, as_before] = PASS_SOURCED.get();\n"
    + I16 + "        PASS_SOURCED.set([uploaded, bytes, unmapped, as_before + 1]);\n"
    + I16 + "    }")

choices[43] = 'theirs'

# ---------------- hunk 44: theirs' !room gate + ours' mirror_queued attempt (with tells)
choices[44] = (
    I12 + 'if !room {\n'
    + I12 + '    let has_its_texture = reupload_when_full()\n'
    + I12 + '        && rest(1, || heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle)));\n'
    + I12 + '    if !has_its_texture {\n'
    + I12 + '        refused += 1;\n'
    + I12 + '        outcome(slot_outcome::HELD);\n'
    + I12 + '        if full_card_pass() {\n'
    + I12 + '            // Held back for the frames of about four seconds\n'
    + I12 + '            // (`most`), or until the card has room; the pass\n'
    + I12 + '            // goes on, within a count that bounds one pass.\n'
    + I12 + '            tells.tell(index, write, Told::Held(frame + most));\n'
    + I12 + '            if refused > 1024 {\n'
    + I12 + '                break;\n'
    + I12 + '            }\n'
    + I12 + '        } else if refused > 256 {\n'
    + I12 + '            break;\n'
    + I12 + '        }\n'
    + I12 + '        continue;\n'
    + I12 + '    }\n'
    + I12 + '}\n'
    + I12 + 'let Some(handle) = heap.handle(index) else {\n'
    + I12 + '    tells.tell(index, write, Told::Done(None));\n'
    + I12 + '    continue;\n'
    + I12 + '};\n'
    + I12 + 'let upload = (room || reupload_when_full())\n'
    + I12 + '    .then(|| host.mirror_queued(index, handle, write, !room))\n'
    + I12 + '    .and_then(Result::ok);\n'
    + I12 + 'let Some((view, _residence)) = upload else {\n'
    + I12 + '    // Give ready uploads and unmapped slots behind this one\n'
    + I12 + '    // their turn before retrying the blocked prefix. A slot\n'
    + I12 + '    // refused for want of room on the card comes back the\n'
    + I12 + '    // frame room returns (`TextureQueue::room`), not only after\n'
    + I12 + '    // `most`.\n'
    + I12 + '    tells.tell(index, write, Told::Later(frame, most));\n'
    + I12 + '    if !room {\n'
    + I12 + '        tells.tell(index, write, Told::Held(frame + most));')

# ---------------- hunk 45: theirs' residency/sync_bake gate + Into::Ahead view arm
t45 = theirs(45)
t45 = t45.replace(
    '} else {\n' + I16 + '    (heap.handle(index).and_then(|handle| host.mirror(index, handle, true)), None)\n' + I16 + '}',
    '} else if let Into::Ahead(_) = &into {\n'
    + I16 + '    // On the worker the slot shows the view when the upload is\n'
    + I16 + '    // in the queue (`crate::heap_worker::Batch::publish`).\n'
    + I16 + '    (record.as_ref().map(|(view, _)| view.clone()), None)\n'
    + I16 + '} else {\n'
    + I16 + '    (heap.handle(index).and_then(|handle| host.mirror(index, handle, true)), None)\n'
    + I16 + '}')
assert 'Into::Ahead' in t45, 'h45 patch failed'
choices[45] = t45

# ---------------- hunk 46: theirs' same/ahead path + ours' upload_again + Into::Ahead publish
t46 = theirs(46)
t46 = t46.replace(
    I12 + 'let tabled = rest(6, || crate::texture::is_staged(resource));',
    I12 + 'let tabled = rest(6, || crate::texture::is_staged(resource));\n'
    + I12 + 'if tabled {\n'
    + I12 + '    crate::texture::upload_again(resource);\n'
    + I12 + '}')
# append the worker's publish at the end of theirs' side
t46 += '\n' + (
    I12 + '// On the worker: the slot shows the view when the upload is in\n'
    + I12 + '// the queue (`crate::heap_worker::Batch::publish`).\n'
    + I12 + 'if let Into::Ahead(batch) = &mut into\n'
    + I12 + '    && let (Some(handle), Some((_, key))) = (heap.handle(index), &record)\n'
    + I12 + '{\n'
    + I12 + '    batch.publish(index, handle, *key);\n'
    + I12 + '}')
choices[46] = t46

# ---------------- hunk 47: theirs' timed room, with ours' forced
choices[47] = I16 + 'room = forced || rest(6, || crate::resource::room(crate::host::host().and_then(|h| h.video_memory())));'

# ---------------- hunk 48: theirs' tells.flush tail + ours' catching_up
choices[48] = theirs(48) + '\n' + I8 + 'catching_up(began.elapsed().saturating_sub(plain));'

# ---------------- hunk 49: dispatch signatures union
choices[49] = (
    '    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32, flags: u32) {\n'
    + '        self.dispatch_x(list, x, y, z, flags);\n'
    + '    }\n'
    + '\n'
    + '    /// Runs a console `DispatchX(x, y, z, flags)` on the host: with\n'
    + '    /// `USE_THREAD_DIMENSIONS` (flags bit 5) the counts are threads, made\n'
    + '    /// groups here, and the program leaves out the threads past them\n'
    + '    /// ([`dispatch_dimensions`]; fork pull request #646).\n'
    + '    pub(crate) fn dispatch_x(&mut self, list: &HostList, x: u32, y: u32, z: u32, flags: u32) {')

# ---------------- hunk 50: ours' Dispatch tail + theirs' ambient call + theirs' fn ambient (3-arg)
o50 = ours(50)
t50 = theirs(50)
o50 = o50.replace('let lit = self.ambient(list, p);', 'let lit = self.lit(list, p);')
assert 'self.lit(list, p)' in o50
# insert theirs' buffers_written + ambient-role block after ours' DISPATCHED line
o50 = o50.replace(
    '        DISPATCHED.fetch_add(1, Ordering::Relaxed);',
    '        DISPATCHED.fetch_add(1, Ordering::Relaxed);\n'
    '        self.buffers_written();\n'
    '        if let Some(role) = p.stages.first().and_then(|s| s.shader.ambient)\n'
    '            && crate::ambient::on()\n'
    '            && self.ambient(list, p, role)\n'
    '        {\n'
    '            self.host_pipeline = 0;\n'
    '            self.host_roots[1] = 0;\n'
    '        }',
    1)
# theirs' side: keep only the fn ambient definition (from its doc comment)
cut = t50.find('    /// The stand-in for the indirect light')
assert cut > 0
t50_ambient = t50[cut:]
choices[50] = o50 + '\n    }\n\n' + t50_ambient

# ---------------- hunk 51: union tables signature + theirs' plan-driven body
t51 = theirs(51)
t51 = t51.replace('        plan: &TablePlan,\n        compute: bool,',
                  '        plan: &TablePlan,\n        half: &[(u8, u32)],\n        compute: bool,\n        spans: &[(u8, u32, u32)],\n        staged: &mut Vec<(u64, u32)>,')
assert 'half: &[(u8, u32)]' in t51
choices[51] = t51

# ---------------- hunk 52: theirs' h.add + ours' half-in-key
choices[52] = (
    I16 + 'h.add(unsafe { (at as *const [u32; 3]).read_unaligned() });\n'
    + I16 + '// (Another table than the same slots made for a shader\n'
    + I16 + '// that reads them in the title\'s formats.)\n'
    + I16 + 'h.add([u32::from(half.contains(&(slot, k * 8))), 0, 0]);')

# ---------------- hunk 53: theirs' n + ours' spans widening
choices[53] = (
    I12 + 'let n = (group.max_dword / 8).max(selected.last().copied().unwrap_or(0)) + 1;\n'
    + I12 + '// With `table_stages`: as many slots as the stage of this draw\n'
    + I12 + '// that reads furthest into the table needs among those that\n'
    + I12 + '// make it afresh too: the ones reading a constant view in it,\n'
    + I12 + '// or every stage where the table reaches past the host heap\'s\n'
    + I12 + '// slots (`spans` is empty with the switch off).\n'
    + I12 + 'let n = match spans.iter().find(|s| s.0 == slot) {\n'
    + I12 + '    Some(&(_, with_constants, all)) => {\n'
    + I12 + '        let past = heap\n'
    + I12 + '            .index(table)\n'
    + I12 + '            .is_some_and(|at| u64::from(at) + u64::from(all) > u64::from(host.count()));\n'
    + I12 + '        n.max(if past { all } else { with_constants })\n'
    + I12 + '    }\n'
    + I12 + '    None => n,\n'
    + I12 + '};')

# ---------------- hunk 54: theirs' FreshTable lookup + ours' tally + staged_title
t54 = theirs(54)
choices[54] = (
    I12 + '#[cfg(feature = "diagnostics")]\n'
    + I12 + 'tally(0, 1);\n'
    + t54.replace(
        '            if let Some(known) = self.fresh_tables.get_mut(&key) {\n',
        '            if let Some(known) = self.fresh_tables.get_mut(&key) {\n'
        + I12 + '    #[cfg(feature = "diagnostics")]\n'
        + I12 + '    tally(1, 1);\n')
    + '\n'
    + I16 + '// Found in the table an earlier stage of this draw made\n'
    + I16 + '// ([`table_stages`]): the title\'s own slots of the texture\n'
    + I16 + '// views this stage reads are stamped and made as the\n'
    + I16 + '// making of a table of its own did it (below).\n'
    + I16 + 'let staged_title = staged\n'
    + I16 + '    .contains(&(table, n))\n'
    + I16 + '    .then(|| heap.index(table).filter(|at| at + n <= host.count()))\n'
    + I16 + '    .flatten();')

# ---------------- hunk 55: ours' clock + theirs' dynamic-retry
choices[55] = (
    I12 + '#[cfg(feature = "diagnostics")]\n'
    + I12 + 'let mut clock = table_asked(&mut self.recording, table, n);\n'
    + theirs(55))

# ---------------- hunk 56: ours' lap/staged + theirs' FreshTable insert
choices[56] = (
    I12 + '#[cfg(feature = "diagnostics")]\n'
    + I12 + 'clock.lap(21);\n'
    + theirs(56) + '\n'
    + I12 + 'if !spans.is_empty() {\n'
    + I12 + '    staged.push((table, n));\n'
    + I12 + '}')

choices[57] = 'theirs'

# ---------------- hunk 58: ours' note_op/capture + theirs' native-1440p viewport
choices[58] = (
    I12 + 'list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false, false));\n'
    + I8 + '#[cfg(feature = "diagnostics")]\n'
    + I8 + 'crate::capture::pass(list, self.targets.first().copied().unwrap_or(0), || self.op_text(format!("draw {count}x{instances}"), false, true));\n'
    + I8 + 'let frontend = if count == 3 && instances == 1 { self.native_frontend_viewport() } else { None };\n'
    + I8 + 'if let Some(viewport) = frontend {\n'
    + I8 + '    viewports(list, &[viewport]);\n'
    + I8 + '}')

choices[59] = (
    I12 + 'if frontend.is_some() {\n'
    + I12 + '    viewports(list, &self.viewports);\n'
    + I12 + '}\n'
    + I12 + 'self.drew(crate::pipeline::host_pipeline(self.pipeline as This));\n'
    + I12 + '#[cfg(feature = "diagnostics")]\n'
    + I12 + 'self.dump_after(list);')

# ---------------- hunk 60: ours' draws_rectangles + theirs' native_frontend_viewport
choices[60] = ours(60) + '\n    }\n\n' + theirs(60)

choices[61] = (
    I12 + 'self.drew(crate::pipeline::host_pipeline(self.pipeline as This));\n'
    + I12 + '#[cfg(feature = "diagnostics")]\n'
    + I12 + 'self.dump_after(list);')

# ---------------- hunk 62: ours' indirect_rectangles + theirs' batch fns
choices[62] = ours(62) + '\n    }\n\n' + theirs(62)

# ---------------- hunk 63: ours' hit/gathered arms adapted to 4-tuple + theirs' batch arm
o63 = ours(63)
# adapt ours' tuple-producing arms to (buffer, at, step, clamped)
o63 = o63.replace('(scratch, at)\n            }\n            (None, Some((scratch, at, recorded)))',
                  '(scratch, at, step, false)\n            }\n            (None, Some((scratch, at, recorded)))')
o63 = o63.replace('(scratch, at)\n            }\n            (None, None) => {',
                  '(scratch, at, stride, recorded)\n            }\n            (None, None) => match batch {\n                Some(b) => b,\n                None => {')
o63 = o63.replace('let (bounded, at) = match (hit, gathered) {',
                  'let (bounded, at, step, clamped) = match (hit, gathered) {')
choices[63] = o63

# ---------------- hunk 64: ours' arms (4-tuple) + theirs' merge machinery w/ ours' split-execute
o64 = ours(64)
t64 = theirs(64)
# ours' arm endings -> 4-tuple
o64 = o64.replace('(bounded, at)\n                } else {', '(bounded, at, step, true)\n                } else {')
o64 = o64.replace('(bounded, 0)\n                }', '(bounded, 0, step, true)\n                }')
# drop ours' `let ran ... ExecuteIndirect` part (theirs' merge machinery replaces it),
# keep ours' `#[cfg] last_indirect` block which precedes `let ran`
cut = o64.find('        let ran = match kind {')
assert cut > 0
o64_head = o64[:cut]
# theirs' `let ran` inside `if !joined`: bind(list,p,true) -> (list,p,false) for merged bind(direct_builder)
t64 = t64.replace('DISPATCH => self.bind(list, p, true),', 'DISPATCH => self.bind(list, p, false),')
assert 'self.bind(list, p, false)' in t64
# theirs' 3-arg op_text -> 4-arg
t64 = t64.replace('self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)',
                  'self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH, false)')
assert ', kind == DISPATCH, false)' in t64
# splice ours' parts/SPLIT execute into theirs' else-arm
t64 = t64.replace(
    '''                unsafe {
                    list.list.ExecuteIndirect(
                        &signature,
                        max,
                        &bounded,
                        at,
                        counter.as_ref().map(|(s, _)| &s.buffer),
                        counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                    )
                };''',
    '''                unsafe {
                    if let Some(((param, dword), signature)) = &parts {
                        for part in 0..crate::indirect::SPLIT {
                            list.list.SetComputeRoot32BitConstant(*param, part * crate::indirect::PART, *dword);
                            let at = u64::from(part) * crate::indirect::PART_BYTES;
                            list.list.ExecuteIndirect(signature, 1, &bounded, at, None, 0);
                        }
                        list.list.SetComputeRoot32BitConstant(*param, 0, *dword);
                    } else {
                        list.list.ExecuteIndirect(
                            &signature,
                            max,
                            &bounded,
                            at,
                            counter.as_ref().map(|(s, _)| &s.buffer),
                            counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                        )
                    }
                };''')
assert 'SPLIT' in t64, 'h64 splice failed'
choices[64] = o64_head + t64

# ---------------- hunk 65: ours' compute_word call-site + theirs' const/fns, one signature
o65 = ours(65)
t65 = theirs(65)
# ours' side ends with `fn compute_word(p: &crate::pipeline::host::HostPipeline, kind: u32) -> Option<(u32, u32)> {`
# theirs' ends with `fn compute_word(p: &HostPipeline, kind: u32) -> Option<(u32, u32)> {`
# keep ours' side minus its trailing signature, add theirs' side (which ends in the signature)
o65 = o65[: o65.rfind('fn compute_word')]
choices[65] = o65 + t65

choices[66] = 'theirs'

# ---------------- hunk 67: theirs' types + readable (3-arg wrapper) + readable_marked (4-arg)
t67 = theirs(67)
assert t67.rstrip().endswith('fn readable(list: &HostList, kind: ResourceType, view: &View) {')
t67 = t67[: t67.rfind('fn readable(list: &HostList, kind: ResourceType, view: &View) {')]
choices[67] = t67 + (
    'fn readable(list: &HostList, kind: ResourceType, view: &View) {\n'
    + '    readable_marked(list, kind, view, None)\n'
    + '}\n'
    + '\n'
    + 'fn readable_marked(\n'
    + '    list: &HostList,\n'
    + '    kind: ResourceType,\n'
    + '    view: &View,\n'
    + '    mark_slot: Option<(&crate::descriptor::host::HostHeap, u32)>,\n'
    + ') {')

# ---------------- hunk 68: theirs' wide_write/gpu_writes + ours' mark_at
choices[68] = (
    I16 + '#[cfg(feature = "diagnostics")]\n'
    + I16 + 'crate::host::wide_write(address, len, 0);\n'
    + I16 + '// A view of the heap whose slot has a mark entry marks its\n'
    + I16 + '// pages (`HostHeap::mark_at`, which also gives the slot its\n'
    + I16 + '// entry where it still held none).\n'
    + I16 + 'let _ = mark_slot.is_some_and(|(heap, index)| heap.mark_at(index, view));\n'
    + I16 + 'shadow.gpu_writes(list, address, len);')

choices[69] = 'both'
choices[70] = 'theirs'
choices[71] = 'ours'

# ---------------- hunk 72: ours' mod tests + theirs' fn + theirs' tests (renamed mod)
t72 = theirs(72)
t72 = t72.replace('mod tests {', 'mod budget_tests {')
choices[72] = 'BOTH_CUSTOM:' + t72  # handled below: ours + t72

choices[73] = 'both'

# =====================================================================
# Apply, walking from the last hunk back to the first.
out = lines[:]
for n in range(N - 1, -1, -1):
    s, m, e = hunks[n]
    c = choices.get(n)
    assert c is not None, f'hunk {n} undecided'
    if c == 'ours':
        new = lines[s + 1:m]
    elif c == 'theirs':
        new = lines[m + 1:e]
    elif c == 'both':
        new = lines[s + 1:m] + lines[m + 1:e]
    elif c.startswith('BOTH_CUSTOM:'):
        new = lines[s + 1:m] + c[len('BOTH_CUSTOM:'):].split('\n')
    else:
        new = c.split('\n')
    out[s:e + 1] = new

open(PATH, 'w').write('\n'.join(out))
print('resolved', N, 'hunks')
