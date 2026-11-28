p = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-gpu/src/host.rs'
s = open(p, encoding='utf-8').read()

# close scratch_at() before scratch_in's signature begins
old = """        let (data, buffer) = (s.chunks[s.current].0.data, s.chunks[s.current].0.buffer.clone());
        let at = s.used;
        s.used += len;
        Some((buffer, at, (data + at as usize) as *mut u8))
        let chunk = &s.chunks[s.current].0;
        let at = s.used.next_multiple_of(align);
        let made = f(&chunk._buffer, at, chunk.gpu + at, (chunk.data + at as usize) as *mut u8);
        s.used = at + len;
        Some(made)
    }"""
new = """        let (data, buffer) = (s.chunks[s.current].0.data, s.chunks[s.current].0.buffer.clone());
        let at = s.used;
        s.used += len;
        Some((buffer, at, (data + at as usize) as *mut u8))
    }

    /// `f` is called with the next `len` aligned scratch bytes: their host
    /// buffer and the offset of the bytes in it, the GPU address of the
    /// bytes and their host address. The result of `f` comes back.
    /// [`HostList::scratch_in`]
    fn scratch_with<R>(
        &self,
        align: u64,
        len: u64,
        f: impl FnOnce(&ID3D12Resource, u64, u64, *mut u8) -> R,
    ) -> Option<R> {
        let s = &mut *self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let chunk = &s.chunks[s.current].0;
        let at = s.used.next_multiple_of(align);
        let made = f(&chunk._buffer, at, chunk.gpu + at, (chunk.data + at as usize) as *mut u8);
        s.used = at + len;
        Some(made)
    }"""
i = s.count(old)
assert i == 1, i
s = s.replace(old, new)

# close alias_source_changed
old = """        self.deferred().texture_aliases.insert(view, source) != Some(source)
    /// Whether `key` was asked"""
assert s.count(old) == 1
s = s.replace(old, """        self.deferred().texture_aliases.insert(view, source) != Some(source)
    }

    /// Whether `key` was asked""")

open(p, 'w', encoding='utf-8').write(s)
print('ok')
