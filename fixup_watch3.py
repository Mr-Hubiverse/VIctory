import re

F = 'crates/victory-gpu/src/watch.rs'
s = open(F, encoding='utf-8').read()

# 1. `j.asked += 1; let now = j.asked;` -> `let now = next_asking();` (Journal has no `asked` field).
s = s.replace('            j.asked += 1;\n            let now = j.asked;', '            let now = next_asking();')
s = s.replace('    j.asked += 1;\n    let now = j.asked;', '    let now = next_asking();')

# 2. Delete ours' `fn guard` + `fn guard_at` (theirs' guard at ~3002 stays).
# ours' guard: `fn guard(start, end) -> Option<Vec<usize>>` that calls guard_at.
import re as _re
g = s.find('fn guard(start: usize, end: usize) -> Option<Vec<usize>> {\n    let keep')
assert g != -1
# find start of its doc
doc = s.rfind('///', 0, g)
while doc > 0 and s[:doc].split('\n')[-1].strip().startswith('///'):
    # walk back while preceding lines are doc comments
    prev = s.rfind('\n', 0, doc - 1)
    line_start = prev + 1
    if s[line_start:doc].strip().startswith('///'):
        doc = line_start
    else:
        break
# find end of guard_at: the next top-level item after fn guard_at
ga = s.index('fn guard_at(', g)
# find the closing of guard_at: next line starting with 'fn ' at col 0 or '///' + 'fn'
m = _re.search(r'\n}\n', s[ga:])
end = ga + m.end()
s = s[:doc] + s[end:]

# 3. Merge Guards: delete ours' struct (first `struct Guards {` .. `}`) and graft ours'-only
# fields into theirs'; delete ours' static GUARDS; graft inits into theirs'.
first_struct = s.index('struct Guards {')
second_struct = s.index('struct Guards {', first_struct + 1)
ours_fields = '''
    /// Blocks the kernel is about to write for the runtime ([`writing`]),
    /// from, to, and until when they are left writable: a read may be
    /// asynchronous and land after the call that asked for it returned, and
    /// the kernel does not fault into our handler, it fails. Until then
    /// they are not made read-only, and every check reports them written.
    pins: Vec<(usize, usize, std::time::Instant)>,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: BTreeSet<usize>,
    /// With [`absent_once`]: for blocks of `absent`, when ([`guard_ms`])
    /// Windows last said they are uncommitted.
    verified: BTreeMap<usize, u64>,
    /// With the light refresh and notes kept for good ([`told`]): the
    /// blocks of `verified` that [`publish`] has looked at. Those of them
    /// that lie in a placeholder reservation ([`placeholder`]) are known by
    /// their bits ([`PUBLISHED`]) until the runtime tells of them; the
    /// others are answered from their notes, which then grow old
    /// ([`note_life`]).
    holes: BTreeSet<usize>,
'''
# graft ours' fields into theirs' struct (insert before closing `}` of second struct)
sc = s.index('\n}\n', second_struct)
s = s[:sc] + ours_fields + s[sc:]
# now delete ours' struct (first), its static GUARDS, keep its impl Guards (pinned)
gs = s.index('static GUARDS: Mutex<Guards> = Mutex::new(Guards {', first_struct)
gi_end = s.index('});', gs) + 3
s = s[:first_struct] + s[gi_end:]
# add ours' inits to theirs' static GUARDS (the remaining one)
tg = s.index('static GUARDS: Mutex<Guards> = Mutex::new(Guards {')
tgi = s.index('});', tg)
s = s[:tgi] + '    pins: Vec::new(),\n    viewed: BTreeSet::new(),\n    verified: BTreeMap::new(),\n    holes: BTreeSet::new(),\n' + s[tgi:]

# 4. Delete ours' `fn placeholder` (first of two remaining; theirs' stays).
ph1 = s.index('fn placeholder(block: usize) -> bool {')
ph2 = s.index('fn placeholder(block: usize) -> bool {', ph1 + 1)
# ours' is the FIRST (inside ours' machinery region before theirs' Bits block? check which is ours')
# ours' placeholder mentions `told`/absent — theirs' is self-contained; delete the one in the
# LIGHT_REFRESH/absent region (first, since ours' block precedes theirs' in file).
docp = s.rfind('/// Whether `block` lies in a placeholder', 0, ph1)
if docp == -1:
    docp = ph1
pend = s.index('\n}\n', ph1) + 3
s = s[:docp] + s[pend:]

# 5. Delete my inserted duplicate `known_view` (the one before `fn told`'s doc).
# Find second `fn known_view`
k1 = s.index('fn known_view(start: usize) -> bool {')
k2 = s.index('fn known_view(start: usize) -> bool {', k1 + 1)
kd = s.rfind('/// Whether the block holding `start` is view memory', 0, k2)
kend = s.index('\n}\n', k2) + 3
s = s[:kd] + s[kend:]

# 6. Rename ours' look_at_holes(most).
s = s.replace('fn look_at_holes(most: usize)', 'fn look_at_holes_most(most: usize)')
s = s.replace('look_at_holes(HOLES_A_LOOK);', 'look_at_holes_most(HOLES_A_LOOK);')

# 7. Delete ours' FIRST title_memory_arrived + try_open_guarded pair (j.asked version).
ta1 = s.index('pub fn title_memory_arrived(at: usize, len: usize) {')
ta_doc = s.rfind('///', 0, ta1)
while True:
    prev = s.rfind('\n', 0, ta_doc - 1)
    ls = prev + 1
    if s[ls:ta_doc].strip().startswith('///'):
        ta_doc = ls
    else:
        break
tg2 = s.index('pub(crate) fn try_open_guarded(at: usize) -> bool {', ta1)
tg2end = s.index('\n    false\n}\n', tg2) + len('\n    false\n}\n')
s = s[:ta_doc] + s[tg2end:]

# 8. Add missing REMAPPING static + Remapping struct + Drop impl before `fn title_memory_remapping`.
add = '''
/// The runtime's word that it changes what an address of title memory
/// shows: while the write lock is held no refresh guards a view that is
/// being unmapped, and `changes`/`again_after_change` notice it.
static REMAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Held while the runtime changes what title memory shows ([`title_memory_remapping`]):
/// blocks the change left different are opened when it is done.
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

impl Drop for Remapping {
    fn drop(&mut self) {
        // The change is done: what is there now may be guarded.
        if CHANGING.replace((false, false)).1 {
            guards().changing.clear();
        }
        CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

'''
tmr = s.index('pub fn title_memory_remapping() -> Remapping {')
s = s[:tmr] + add + s[tmr:]

# 9. Restore ours' title_memory_remapping_range (returns RemapGuard) — before `pub struct Reading`.
rr = '''
/// Protects every address that can be unmapped during the operation,
/// including whole old views that Windows temporarily removes.
pub fn title_memory_remapping_range(range: Range<usize>) -> RemapGuard {
    let asked = std::time::Instant::now();
    let writer = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    let mut state = MAPPING.lock().unwrap_or_else(|e| e.into_inner());
    state.changing = Some(range.clone());
    drop(
        MAPPING_CHANGED
            .wait_while(state, |s| s.readers.iter().any(|r| overlaps(r, &range)))
            .unwrap_or_else(|e| e.into_inner()),
    );
    let previous = crate::inplace::physical_spans(&range);
    crate::framelog::add(crate::framelog::T::Remaps, 1);
    crate::framelog::add(crate::framelog::T::RemapWaitUs, asked.elapsed().as_micros() as u64);
    RemapGuard {
        _writer: writer,
        range,
        previous,
        asked,
    }
}

'''
rd = s.index('pub struct Reading(Range<usize>);')
s = s[:rd] + rr + s[rd:]
open(F, 'w', encoding='utf-8').write(s)
print('done')
