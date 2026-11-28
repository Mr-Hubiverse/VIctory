import re

SRC = 'crates/victory-gpu/src/watch.rs'
src = open(SRC, encoding='utf-8').read()
pat = re.compile(r'<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n?', re.S)
ms = list(pat.finditer(src))
assert len(ms) == 28, len(ms)

rep = {}

def side(i, which):
    return ms[i].group(which)

# h0: theirs' doc + union imports + ours' REGION_EPOCHS block.
o0, t0 = side(0, 1), side(0, 2)
idx = o0.index('/// Per-64-MiB-region epoch table')
region_block = o0[idx:]
rep[0] = (
    t0.rstrip()
    + "\nuse std::ops::Range;\nuse std::sync::atomic::{AtomicU64, Ordering};\n\n"
    + region_block
)

# h1/h2: both sides' fields/init.
for i in (1, 2):
    rep[i] = side(i, 1).rstrip() + '\n' + side(i, 2)

# h3-h8: theirs.
for i in (3, 4, 5, 6, 7, 8):
    rep[i] = side(i, 2)

# h9: both (ours' whole machinery block + theirs' light/Bits).
rep[9] = side(9, 1).rstrip() + '\n' + side(9, 2)

# h10-h22: theirs.
for i in range(10, 23):
    rep[i] = side(i, 2)

# h23: theirs' open-continuation + close of title_memory_made_real, then ours'
# viewed/holes bookkeeping folded in, then title_memory_arrived + try_open_guarded.
o23, t23 = side(23, 1), side(23, 2)
# ours' side: [view_known tail of fn + '}' ] + title_memory_arrived + try_open_guarded(unclosed)
# take ours' side from '/// The runtime wrote new contents' onward = the two fns
extra = o23[o23.index('/// The runtime wrote new contents'):]
# fix theirs' side additions: theirs ends with the fn body continuation
inner = (
    t23.rstrip()
    + '\n'
    + '''
    // What the range shows is new: no longer known to be a view.
    if view_known() {
        g.viewed.extend(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK));
    } else {
        let viewed: Vec<usize> = g.viewed.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).copied().collect();
        for block in viewed {
            g.viewed.remove(&block);
        }
    }
    let end = at.saturating_add(len);
    let holes: Vec<usize> = g
        .holes
        .range(at / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    let changed = !holes.is_empty();
    for block in holes {
        g.holes.remove(&block);
    }
    if changed {
        bump_region_epochs(at, end);
    }
    drop(g);
    crate::shadow::remapped(at as u64, end as u64);
    crate::resource::evict_placed_textures(at as u64, end as u64);
    crate::descriptor::host::bindless_memory_changed(at as u64, end as u64);
}
'''
)
rep[23] = inner + extra

# h24-h27: theirs.
for i in range(24, 28):
    rep[i] = side(i, 2)

out = []
last = 0
for i, m in enumerate(ms):
    out.append(src[last:m.start()])
    out.append(rep[i])
    last = m.end()
out.append(src[last:])
open(SRC, 'w', encoding='utf-8').write(''.join(out))
print('resolved 28 hunks')
