import sys
sys.path.insert(0, '.')
from resolve1 import hunks, resolve

PATH = 'crates/victory-gpu/src/shadow.rs'
src, hs = hunks(PATH)
assert len(hs) == 42, len(hs)
ours = [h[1] for h in hs]
thrs = [h[2] for h in hs]
c = [None] * 42

# h0: union of imports (AtomicUsize + RwLock + Weak all needed)
c[0] = ('use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};\n'
        'use std::sync::{Arc, Mutex, RwLock, Weak};\n')
# h1: both field sets (uploaded + changes_only/wc/mapped)
c[1] = 'both'
# h2: theirs (VRAM-budget-aware card_limits)
c[2] = 'theirs'
# h3-h5: theirs (TOUCHED list, card stats with WINDOW_MAPPED_BYTES, window_mapped machinery)
c[3] = 'theirs'; c[4] = 'theirs'; c[5] = 'theirs'
# h6: ours' window_card_room (called by unmarked window_upload_for) + theirs' SentRuns
c[6] = 'both'
# h7-h9: theirs' TOUCHED walk; send_copied needs pending 3-tuple fix
c[7] = 'theirs'; c[8] = 'theirs'
t9 = thrs[9].replace(
    '.iter().copied().filter(|&(a, b)| a < to && b > from).collect()',
    '.iter().copied().filter(|&(a, b, _)| a < to && b > from).map(|(a, b, _)| (a, b)).collect()')
c[9] = t9
# h10: theirs' snapshot-pool give_back
c[10] = 'theirs'
# h11: both (GUARDED and WATCHED_GUARDED consts are both used unmarked) + refresh_count
c[11] = 'both'
# h12: both counters; note_stale also feeds MERGE_GAVE_UP
t12 = thrs[12].replace(
    'fn note_stale(why: usize) {\n    STALE_WHY.store(why, Ordering::Relaxed);\n}',
    'fn note_stale(why: usize) {\n    STALE_WHY.store(why, Ordering::Relaxed);\n    MERGE_GAVE_UP[why.min(3)].fetch_add(1, Ordering::Relaxed);\n}')
c[12] = ours[12] + t12
# h13/h14: both (changed_runs_wide both cfgs + theirs' new fns)
c[13] = 'both'; c[14] = 'both'
# h15: merged use line + ours' test (theirs' side is use-line only)
o15 = ours[15].split('\n', 1)[1]
c[15] = ('    use super::{PIECE, back_piece, changed_chunks, changed_runs, dword_runs, put_changed, word_runs};\n' + o15)
# h16: theirs' tests
c[16] = 'theirs'
# h17: union of Drop counters
c[17] = ours[17] + thrs[17]
# h18: theirs minus duplicate stale_pages_in (already unmarked); rename mod stale_tests
t18 = thrs[18]
start = t18.index('/// The pages of `[a, e)`')
end = t18.index('/// `runs` (from, to, whether')
t18 = t18[:start] + t18[end:]
t18 = t18.replace('mod stale_tests', 'mod stale_page_tests')
c[18] = t18
# h19: ours' refresh_params + theirs' width() + ours' 6-arg new
o19 = ours[19]
w = ('    /// The bytes of the host buffer.\n'
     '    pub(crate) fn width(&self) -> u64 {\n'
     '        // SAFETY: a live buffer.\n'
     '        *self.width.get_or_init(|| unsafe { self.buffer.GetDesc() }.Width)\n'
     '    }\n\n')
c[19] = o19.replace('    fn new(', w + '    fn new(')
# h20/h21: both init sets
c[20] = 'both'; c[21] = 'both'
# h22/h23: theirs' put_changed tails (merge_race/owners.publish machinery absent)
c[22] = 'theirs'; c[23] = 'theirs'
# h24: ours (pending is 3-tuple, free is 2-tuple ranges)
c[24] = 'ours'
# h25/h27: theirs' note_stale; h26: destructure 3-tuple busy, marked unused
c[25] = 'theirs'
c[26] = '        for (a, e, _) in busy {\n'
c[27] = 'theirs'
# h28: theirs (matches shared merge_page_under_writes/under_wide body)
c[28] = 'theirs'
# h29: ours' 3-arg update_free (unmarked callers pass lazy)
c[29] = 'ours'
# h30: ours' subs peekable + theirs' ranged/runs/again_at
c[30] = ('        let mut subs = subs.iter().peekable();\n'
         '        // Pages the range holds in part are never noted in `copied`, and\n'
         '        // were copied again at every refresh of the range, written or not:\n'
         '        // of the heap\'s buffer views alone two ends for each of 4,400\n'
         '        // copies, every frame. A range that was taken whole, and not written\n'
         '        // since, is as it was.\n'
         '        let ranged = if crate::watch::light() { self.ranged(from, to) } else { 0 };\n'
         '        let (mut runs, mut run) = (Vec::new(), None::<(usize, bool)>);\n'
         '        // Pages taken before and written since: in the order of their runs.\n'
         '        let mut again_at: Vec<(usize, u64)> = Vec::new();\n')
# h31: theirs (taken = max(at, ranged))
c[31] = 'theirs'
# h32: ours' dirty4k narrow path (adapted to tuple run) + theirs' again/run tracking
c[32] = ('                // Only the 4 KiB pages written since the copy took the page: the asking before the last one that\n'
         '                // found it written came no later than the copy\'s taking.\n'
         '                let found = sub.filter(|&(before, _)| taken != 0 && before <= taken).map(|(_, bits)| bits);\n'
         '                if crate::watch::dirty4k_kept() {\n'
         '                    crate::watch::dirty4k_count(found, narrow && found.is_some());\n'
         '                }\n'
         '                if narrow && let Some(bits) = found {\n'
         '                    if let Some((first, known)) = run.take() {\n'
         '                        runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));\n'
         '                    }\n'
         '                    let mut k = 0;\n'
         '                    while k < PAGE / KERNEL_PAGE {\n'
         '                        if bits & (1 << k) == 0 {\n'
         '                            k += 1;\n'
         '                            continue;\n'
         '                        }\n'
         '                        let a = k;\n'
         '                        while k < PAGE / KERNEL_PAGE && bits & (1 << k) != 0 {\n'
         '                            k += 1;\n'
         '                        }\n'
         '                        let (lo, hi) = ((page * PAGE + a * KERNEL_PAGE).max(from), (page * PAGE + k * KERNEL_PAGE).min(to));\n'
         '                        if lo < hi {\n'
         '                            self.copy(lo, hi);\n'
         '                        }\n'
         '                    }\n'
         '                    continue;\n'
         '                }\n'
         '                // Cut where pages taken before meet pages never taken.\n'
         '                let again = taken != 0;\n'
         '                if again {\n'
         '                    again_at.push((page, taken));\n'
         '                }\n'
         '                if let Some((first, known)) = run\n'
         '                    && known != again\n'
         '                {\n'
         '                    runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));\n'
         '                    run = None;\n'
         '                }\n'
         '                run.get_or_insert((page, again));\n'
         '            } else if let Some((first, known)) = run.take() {\n'
         '                runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));\n')
# h33: theirs (touched/TOUCHED scheme + helper methods)
c[33] = 'theirs'
# h34/h35: theirs (put_changed tails, test impl; fix 5-arg Shadow::new calls)
c[34] = 'theirs'
t35 = thrs[35].replace('Shadow::new(none(), data, 0, start, len)', 'Shadow::new(none(), data, 0, start, len, None)')
t35 = t35.replace('Shadow::new(buffer.clone(), data, 0, title as u64, len as u64)',
                  'Shadow::new(buffer.clone(), data, 0, title as u64, len as u64, None)')
c[35] = t35
# h36: theirs' RwLock BUFFERS + ours' buffer_generation (on BUFFERS_CHANGED) + FindCache
o36 = ours[36]
findcache = o36[o36.index("/// A recording thread's last registered-buffer hit."):]
c[36] = (thrs[36] +
         '\n/// Changes when the same guest buffer descriptor can resolve to a new owner.\n'
         'pub(crate) fn buffer_generation() -> u64 {\n'
         '    BUFFERS_CHANGED.load(Ordering::Acquire)\n'
         '}\n\n' + findcache)
# h37: ours' lookup machinery + theirs' peek signature (its body follows unmarked)
c[37] = 'both'
# h38: union: own_marks + carded()
c[38] = ('        own_marks(&shadow);\n'
         '        if shadow.card.is_some() {\n'
         '            carded(&shadow);\n')
# h39: union: ours' 6-arg new + theirs' window flag/carded/note
c[39] = ('    let mut shadow = Shadow::new(buffer, data, gpu, start, len, card);\n'
         '    shadow.window = true;\n'
         '    let shadow = Arc::new(shadow);\n'
         '    if shadow.card.is_some() {\n'
         '        carded(&shadow);\n'
         '    }\n'
         '    note(&shadow);\n')
# h40: both (exact_peek + OWN_MADE)
c[40] = 'both'
# h41: merged use + both test sets
o41 = ours[41].split('\n', 1)[1]
t41 = thrs[41].split('\n', 4)[4]
c[41] = ('    use super::{\n'
         '        WINDOW, cover, default_card_limits_mb, grid, holder, holder_below, longest_tail,\n'
         '        resolve_card_limits, tail_of, window_of,\n'
         '    };\n' + o41 + '\n' + t41)

resolve(PATH, c)
