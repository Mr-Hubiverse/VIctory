import re

F = 'crates/victory-gpu/src/watch.rs'
s = open(F, encoding='utf-8').read()

def cut(start_marker, end_marker, keep_end=True):
    """Delete text from start_marker (incl.) up to end_marker (excl if keep_end)."""
    global s
    a = s.index(start_marker)
    b = s.index(end_marker, a)
    s = s[:a] + (s[b:] if keep_end else '')

# 1. Delete ours' `pub(crate) fn light` (its doc stays? delete doc+fn from "/// Whether an asking" to static LIGHT_ASKS).
a = s.index('/// Whether an asking about view memory is answered from the bits')
b = s.index('static LIGHT_ASKS')
s = s[:a] + s[b:]

# 2. Delete ours' Bits decl/impl + PUBLISHED + publish + published (ours' versions).
a = s.index('const BITS_WORDS')
b = s.index('/// Whether a block noted uncommitted stays noted')
s = s[:a] + s[b:]

# 3. Delete ours' 4-arg nothing_new + its doc+#[allow].
a = s.index('#[allow(dead_code)]\n/// Whether nothing of `[start, end)` was written')
b = s.index('/// Whether a refresh of `[start, end)`')
s = s[:a] + s[b:]

# 4. nothing_noted: fix JOURNAL.lock -> journal(), published(&j -> published.
s = s.replace("    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());\n    // The bits first, then the journal, both under its lock ([`published`]).",
              "    let j = journal();\n    // The bits first, then the journal, both under its lock ([`published`]).")
s = s.replace('published(&j, first * PAGE, last * PAGE)', 'published(first * PAGE, last * PAGE)')

# 5. JOURNAL.lock() -> journal()/journal_mut() remaining.
s = s.replace('let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());', 'let mut j = journal_mut();')
s = s.replace('let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());', 'let j = journal();')
s = s.replace('JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).written.get(&page).copied().unwrap_or(0)',
              'journal().written.get(&page).copied().unwrap_or(0)')

# 6. Blocks fields take usize args: strip `&` on contains/insert/remove for protected/clean/absent/aliased/held.
s = re.sub(r'\.(protected|clean|absent|aliased|held)\.(contains|insert|remove)\(&', r'.\1.\2(', s)
# Blocks::range yields usize: `.copied()` after those ranges breaks; closures |&x| break.
s = re.sub(r'\.(protected|clean|absent|aliased|held)\.range\(([^)]*)\)\.copied\(\)', r'.\1.range(\2)', s)

# 7. Rename theirs' MAPPING RwLock -> REMAPPING (def + one use).
s = s.replace('static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());',
              'static REMAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());')
s = s.replace('let changing = MAPPING.write().unwrap_or_else(|e| e.into_inner());',
              'let changing = REMAPPING.write().unwrap_or_else(|e| e.into_inner());')

# 8. Rename ours' struct Remapping -> RemapGuard + drop impl + remapping_range return type.
s = s.replace('/// The only active change of physical mappings.\npub struct Remapping {',
              '/// The only active change of physical mappings.\npub struct RemapGuard {')
s = s.replace('pub fn title_memory_remapping_range(range: Range<usize>) -> Remapping {',
              'pub fn title_memory_remapping_range(range: Range<usize>) -> RemapGuard {')
s = s.replace('impl Drop for Remapping {', 'impl Drop for RemapGuard {')

# 9. Delete ours' title_memory_remapping (the one calling remapping_range); keep theirs'.
s = s.replace('''/// Protects a mapping change whose extent is unknown.
pub fn title_memory_remapping() -> Remapping {
    title_memory_remapping_range(ALL_MEMORY)
}

''', '')

# 10. Restore ours' known_view + open_here + try_open_guarded + title_memory_arrived.
KNOWN_VIEW = '''
/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    });
    if ask {
        return false;
    }
    let first = start / BLOCK;
    if light() && PUBLISHED.all(first, first + 1) {
        return true;
    }
    let g = guards();
    g.protected.contains(first) || g.absent.contains(first) || (view_known() && g.viewed.contains(&first))
}

'''
# Insert before `fn early` region? place near `published` def end — put before `fn told`.
s = s.replace('/// Whether a block noted uncommitted stays noted', KNOWN_VIEW + '/// Whether a block noted uncommitted stays noted')

OPEN_HERE = '''
/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    unguard(g, &(first..last).collect::<Vec<_>>());
    for block in first..last {
        PUBLISHED.clear(block);
    }
}

'''
s = s.replace('/// notes them written; how many there were.', OPEN_HERE + '/// notes them written; how many there were.')

TRIED = '''
/// What the runtime told of title memory that came to be: it is noted
/// written (a host copy made before kept what it held then for good), and
/// its region's epoch bumped.
pub fn title_memory_arrived(at: usize, len: usize) {
    if len != 0 {
        bump_region_epochs(at, at.saturating_add(len));
        {
            let mut j = journal_mut();
            let now = next_asking();
            for page in at / PAGE..(at + len).div_ceil(PAGE) {
                j.note(page, WHOLE, now);
            }
        }
        crate::resource::textures_rewritten(at as u64, (at + len) as u64);
    }
}

/// Opens a guarded block at `at` (or returns `true` if another thread already
/// made it read-write). Called by `host::safe_copy`'s VEH handler on write
/// faults.
pub(crate) fn try_open_guarded(at: usize) -> bool {
    let mut g = guards();
    if open(&mut g, at, at + 1) != 0 {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardFaults, 1);
        return true;
    }
    if writable(at) {
        return true;
    }
    if view_guard() && !aliases(at).is_empty() {
        let mut old = 0;
        let block = at - at % BLOCK;
        // SAFETY: only the protection of title view memory changes.
        let opened = unsafe {
            windows_sys::Win32::System::Memory::VirtualProtect(
                block as *const _,
                BLOCK,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
                &mut old,
            )
        } != 0;
        if opened {
            bump_region_epochs(block, block + BLOCK);
            tracing::warn!(at = format_args!("{at:#x}"), "a read-only view block that no guard knew was opened");
        }
        return opened;
    }
    false
}

'''
s = s.replace('/// Registers the handler that takes writes to guarded blocks', TRIED + '/// Registers the handler that takes writes to guarded blocks')

# 11. Graft ours' `sub` (dirty4k) bookkeeping into Journal::note.
s = s.replace('''    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
''', '''    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        if dirty4k_kept() {
            let before = self.sub.get(&page).map_or(0, |&(asked, _, _)| asked);
            self.sub.insert(page, (now, before, parts));
        }
''')
open(F, 'w', encoding='utf-8').write(s)
print('done')
