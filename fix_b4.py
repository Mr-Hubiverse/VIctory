p = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-gpu/src/host.rs'
s = open(p, encoding='utf-8').read()

# The fused scratch_at/scratch_in: replace from `fn scratch_at` signature
# through `Some(made)` + `}` with the clean two-fn union.
start = s.index('    fn scratch_at(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {')
end = s.index('    /// The bytes of scratch chunks the list holds now', start)

new = '''    fn scratch_at(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name_frame(&buffer, || format!("list scratch {len:#x}"));
            self.deferred().hold_upload(buffer.clone(), len);
            return Some((buffer, 0, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let (data, buffer) = (s.chunks[s.current].0.data, s.chunks[s.current].0.buffer.clone());
        let at = s.used;
        s.used += len;
        Some((buffer, at, (data + at as usize) as *mut u8))
    }

    /// `f` is given the next `len` scratch bytes (aligned to `align`): their
    /// buffer, the offset of the bytes in it, the GPU address of the bytes
    /// and their host address. What `f` gives back is given back.
    pub(crate) fn scratch_in<R>(
        &self,
        len: u64,
        align: u64,
        f: impl FnOnce(&ID3D12Resource, u64, u64, *mut u8) -> R,
    ) -> Option<R> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name_frame(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let made = f(&buffer, 0, gpu, data);
            self.deferred().hold_upload(buffer, len);
            return Some(made);
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let chunk = &s.chunks[s.current].0;
        let at = s.used.next_multiple_of(align);
        let made = f(&chunk._buffer, at, chunk.gpu + at, (chunk.data + at as usize) as *mut u8);
        s.used = at + len;
        Some(made)
    }

'''
s = s[:start] + new + s[end:]

# close alias_source_changed (missing })
old = """        self.deferred().texture_aliases.insert(view, source) != Some(source)
    /// Whether `key` was asked"""
assert s.count(old) == 1
s = s.replace(old, """        self.deferred().texture_aliases.insert(view, source) != Some(source)
    }

    /// Whether `key` was asked""")
open(p, 'w', encoding='utf-8').write(s)
print('scratch fns + alias fixed')
