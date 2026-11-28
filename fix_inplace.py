import re
src = open('crates/victory-gpu/src/inplace.rs', encoding='utf-8').read()
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
hunks = list(pat.finditer(src))
assert len(hunks) == 4, len(hunks)

# hunk0: union — ours' OWN-map purge AND theirs' whole-view purge
m0 = '''    {
        let mut v = views_mut();
        v.sections.retain(|s| s.0 != section);
        // A handle's number given out again: the view was of the section that
        // had it before (it stays mapped: someone may be writing through it).
        v.whole.retain(|w| w.0 != section);
        v.sections.push((section, bytes, committed));
    }
''' + hunks[0].group(1).split('{', 1)[-1].join if False else '''    {
        let mut v = views_mut();
        v.sections.retain(|s| s.0 != section);
        // A handle's number given out again: the view was of the section that
        // had it before (it stays mapped: someone may be writing through it).
        v.whole.retain(|w| w.0 != section);
        v.sections.push((section, bytes, committed));
    }
    // A view of our own made on an earlier section that had this handle's
    // number shows that section's pages, not this one's: it is forgotten
    // (left mapped: nothing reads it any more) and the next store through
    // a view of ours maps this section.
    let gone = OWN.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|own| own.remove(&section)).flatten();
    if gone.is_some() {
        tracing::info!(section, "a section handle's number came round again: our own view of the earlier section is forgotten");
    }
'''

# hunk1: both function sets
m1 = hunks[1].group(1) + hunks[1].group(2)

# hunks 2,3: both guards in each test
m2 = ('        let _memory_guard = crate::watch::TEST_TITLE_MEMORY_LOCK.lock().unwrap_or_else(|e| e.into_inner());\n'
      '        let _turn = crate::indirect::tests::warp_turn();\n')
m3 = m2

out = (src[:hunks[0].start()] + m0 + src[hunks[0].end():hunks[1].start()] + m1
       + src[hunks[1].end():hunks[2].start()] + m2 + src[hunks[2].end():hunks[3].start()]
       + m3 + src[hunks[3].end():])
open('crates/victory-gpu/src/inplace.rs', 'w', encoding='utf-8').write(out)
print('done', '<<<<<<<' in out)
