p = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-gpu/src/host.rs'
s = open(p, encoding='utf-8').read()

# 1) split(): missing closing brace.
old = """    let [a, b] = halves;
    let [x, y] = orders;
    Ok([(a, x), (b, y)])

/// The cap of changed runs"""
new = """    let [a, b] = halves;
    let [x, y] = orders;
    Ok([(a, x), (b, y)])
}

/// The cap of changed runs"""
assert s.count(old) == 1
s = s.replace(old, new)

# 2) theirs' prelude record: the `for (copy, real, first) in &leaving` block
# never closed (concat ate the brace).
old = """                for (copy, real, first) in &leaving {
                    crate::shot::leaves(list, std::ptr::from_ref(self) as usize, None, shot_begins, copy, *real, *first);
                for (upload, _) in preludes {"""
new = """                for (copy, real, first) in &leaving {
                    crate::shot::leaves(list, std::ptr::from_ref(self) as usize, None, shot_begins, copy, *real, *first);
                }
                for (upload, _) in preludes {"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('ok')
