p = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-gpu/src/host.rs'
s = open(p, encoding='utf-8').read()

# Replace the whole interleaved Batch::add with the clean union: ours'
# checked_add/Spans/current form + theirs' `ahead` early-out.
i = s.index('    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {')
j = s.index('    fn iter(&self) -> impl Iterator<Item = (&std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {', i)
new = """    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let Some(end) = address.checked_add(len).filter(|&e| e > address) else {
            return;
        };
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        // Brought up to date in this submission already ([`refresh_once`];
        // nothing is in `current` without the switch).
        if let Some(&(_, from, to)) = self.current.get(&key)
            && from <= address
            && end <= to
        {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, len as u64);
            }
            return;
        }
        // In front of all the steps already (the writes under flights').
        if self.ahead.get(&key).is_some_and(|&(from, to)| from <= address && end <= to) {
            return;
        }
        match self.index.get(&key) {
            Some(&at) => self.refreshes[at].1.add(address, end),
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes
                    .push((shadow.clone(), Spans::One((address, end))));
            }
        }
    }

"""
s = s[:i] + new + s[j:]
open(p, 'w', encoding='utf-8').write(s)
print('add fixed')
