import re
src = open('crates/victory-gpu/src/pipeline/library.rs', encoding='utf-8').read()
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
hunks = list(pat.finditer(src))
assert len(hunks) == 6, len(hunks)

m0 = hunks[0].group(1) + hunks[0].group(2)   # both fns
# hunk1: ours' crashed/Making prelude + theirs' filter
m1 = '''    if crashed(name) {
        // E_FAIL.
        return Err(windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)));
    }
    let create = || {
        let _making = Making::new(name);
        create()
    };
    let Some(library) = library().filter(|_| !STATIC_SAMPLERS.get()) else {
'''
# hunk2: ours' used snapshot + USED_CHANGED reset, theirs' library_written
m2 = '''    // With what earlier runs listed: a run that asks for no title pipeline
    // (`victory warm`) wrote an empty list over theirs, and the next world
    // began with every pipeline still to be asked for.
    let used: Vec<String> = {
        let used = used();
        USED_CHANGED.store(false, Ordering::Relaxed);
        used.iter().flatten().cloned().collect()
    };
    let library_written = if bytes > 0 { replace_library(dir, &tmp) } else { Ok(()) };
    if bytes > 0 && library_written.is_ok() {
'''
m3 = hunks[3].group(2)   # theirs' slow+early chain
m4 = hunks[4].group(2)   # theirs' serialized write_atomically + slow_ones
m5 = hunks[5].group(1) + hunks[5].group(2)  # both tests

out = (src[:hunks[0].start()] + m0 + src[hunks[0].end():hunks[1].start()] + m1
       + src[hunks[1].end():hunks[2].start()] + m2 + src[hunks[2].end():hunks[3].start()]
       + m3 + src[hunks[3].end():hunks[4].start()] + m4 + src[hunks[4].end():hunks[5].start()]
       + m5 + src[hunks[5].end():])
open('crates/victory-gpu/src/pipeline/library.rs', 'w', encoding='utf-8').write(out)
print('done', '<<<<<<<' in out)
