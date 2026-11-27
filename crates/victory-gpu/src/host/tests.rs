    /// Run once with VICTORY_CARD_SYNC_BATCH=0 and once with it set to 1.
    #[test]
    #[ignore = "requires a D3D12 adapter and the card hash pipeline; run alone"]
    fn two_dirty_card_buffers_reach_readback_in_the_same_submission() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const MIB: usize = 1 << 20;
        const PIECE: usize = 64 << 10;
        let host = super::host().expect("a D3D12 device");
        assert!(host.card_ready(), "the card hash pipeline must be available");
        // SAFETY: a private allocation held through the fenced submissions.
        let allocation = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * MIB, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        };
        assert!(!allocation.is_null());
        let bases = [allocation as usize, allocation as usize + MIB];
        let shadows = bases.map(|base| {
            crate::shadow::register(base, base as u64, MIB as u64);
            let copy = crate::shadow::find(base as u64, MIB as u64).unwrap();
            assert!(copy.carded(), "both title buffers must have a card copy");
            copy
        });
        let readback = host.read_back_buffer(16).unwrap();
        let hash_readback = host.read_back_buffer(64).unwrap();
        let mut previous = [0u128; 4];
        for round in [0x11u32, 0x55] {
            let list = host.new_list().unwrap();
            let mut expected = [0u32; 4];
            for (i, (shadow, &base)) in shadows.iter().zip(&bases).enumerate() {
                for (j, offset) in [32, PIECE + 20].into_iter().enumerate() {
                    let k = 2 * i + j;
                    expected[k] = round * 0x0101_0101 + k as u32;
                    // SAFETY: aligned dword in the private allocation.
                    unsafe { ((base + offset) as *mut u32).write_volatile(expected[k]); }
                    shadow.refresh(&list, (base + offset) as u64, 4);
                    // Request the before-hash for each of the two distinct pieces.
                    shadow.gpu_writes(&list, (base + offset) as u64, 4, false);
                    // SAFETY: the readback and the copy outlive this fenced list.
                    unsafe { list.list.CopyBufferRegion(&readback, (k * 4) as u64, &shadow.buffer, offset as u64, 4); }
                }
            }
            list.close();
            host.execute(&[&list]);
            let mut mapped = std::ptr::null_mut();
            // SAFETY: execute waited for all four writes, and the buffer holds 16 bytes.
            let got = unsafe {
                readback.Map(0, None, Some(&mut mapped)).unwrap();
                let got = mapped.cast::<[u32; 4]>().read_unaligned();
                readback.Unmap(0, None);
                got
            };
            assert_eq!(got, expected, "both copied pieces of each buffer");
            let hash_list = host.new_list().unwrap();
            for (i, shadow) in shadows.iter().enumerate() {
                let hashes = shadow.card_sides().unwrap().1.expect("before-hashes were made");
                // SAFETY: two consecutive 16-byte piece hashes from each live buffer.
                unsafe { hash_list.list.CopyBufferRegion(&hash_readback, (32 * i) as u64, &hashes, 0, 32); }
            }
            hash_list.close();
            host.execute(&[&hash_list]);
            // SAFETY: all 64 bytes of the readback were written and fenced.
            let current = unsafe {
                hash_readback.Map(0, None, Some(&mut mapped)).unwrap();
                let words = mapped.cast::<[u128; 4]>().read_unaligned();
                hash_readback.Unmap(0, None);
                words
            };
            for (i, hash) in current.iter().enumerate() {
                assert_ne!(*hash, previous[i], "piece {i} must be rehashed");
                assert!(!current[..i].contains(hash), "each piece has different input");
            }
            previous = current;
        }
        for base in bases {
            crate::shadow::unregister(base, base as u64);
        }
        drop(shadows);
        // SAFETY: the lists completed and the copies were unregistered.
        unsafe { VirtualFree(allocation, 0, MEM_RELEASE); }
    }

    #[test]
    fn submission_ranges_bound_each_batch_without_reordering() {
        for len in [0, 1, super::MAX_SUBMISSION_RUNS, super::MAX_SUBMISSION_RUNS + 1, super::MAX_SUBMISSION_RUNS * 3 + 7] {
            let mut next = 0;
            for range in super::submission_ranges(len) {
                assert_eq!(range.start, next);
                assert!(range.len() <= super::MAX_SUBMISSION_RUNS);
                next = range.end;
            }
            assert_eq!(next, len);
        }
    }

    #[test]
    fn raw_memory_reads_follow_each_list_and_guest_copies() {
        let mut first = super::Deferred::default();
        let mut second = super::Deferred::default();
        assert_eq!(first.raw_copies.advance(2, false), (0..2, 2..2));
        assert_eq!(second.raw_copies.advance(5, true), (0..5, 0..5));
        assert_eq!(first.raw_copies.advance(5, true), (2..5, 0..5));
        assert_eq!(first.raw_copies.advance(5, true), (5..5, 5..5));
        first.invalidate_refreshes();
        assert_eq!(first.raw_copies.advance(5, false), (0..5, 5..5));
        assert_eq!(first.raw_copies.advance(6, true), (5..6, 5..6));
    }

    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    /// A list with no work before its waits, as the title's queues meet one
    /// another halfway: each write goes as soon as the waits before it are
    /// met, and work holds the later ones back.
    #[test]
    fn writes_before_an_unmet_wait_go_early() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut memory = [0u64; 4];
        let at = memory.as_mut_ptr();
        // SAFETY: inside the array.
        let slot = |i: usize| unsafe { at.add(i) };
        let wait = |i: usize| crate::memwait::Wait {
            address: slot(i) as usize,
            comparison: 7,
            reference: 5,
            bytes: 8,
        };
        let write = |i: usize| list.defer_write(slot(i).cast(), crate::queue::Written::Value(5), 8);
        // SAFETY: as above.
        let value = |i: usize| unsafe { slot(i).read_volatile() };
        list.note_wait(wait(0));
        write(1);
        list.note_wait(wait(2));
        list.count_work();
        write(3);
        assert!(!list.write_early() && value(1) == 0);
        // SAFETY: as above.
        unsafe { slot(0).write_volatile(5) };
        assert_eq!(list.unmet(), [wait(2)]);
        assert!(list.write_early() && !list.write_early());
        assert_eq!((value(1), value(3)), (5, 0));
        // SAFETY: as above.
        unsafe { slot(2).write_volatile(5) };
        assert!(list.unmet().is_empty() && !list.write_early());
        // Run: what is left, then all of it anew for another submission.
        list.close();
        host.execute(&[&list]);
        assert_eq!(value(3), 5);
        // SAFETY: as above.
        unsafe { slot(1).write_volatile(0) };
        host.execute(&[&list]);
        assert_eq!(value(1), 5);
    }

    #[test]
    fn the_caption_ends_with_how_long_the_process_has_run() {
        use super::{Duration, caption};

        let figures = "VIctory | 1 fps 953.5 ms/frame | 1526 draws/frame";
        let up = |secs| caption(figures, Duration::from_secs(secs));
        assert_eq!(up(0), format!("{figures} | up 0:00"));
        assert_eq!(up(59), format!("{figures} | up 0:59"));
        assert_eq!(up(754), format!("{figures} | up 12:34"));
        assert_eq!(up(3723), format!("{figures} | up 1:02:03"));
    }

    #[test]
    fn the_running_time_counts_from_the_process_start_and_moves_on() {
        use super::{Duration, process_age, process_uptime};

        // The test process was created before this test ran, and not long ago.
        let age = process_age().expect("the kernel's creation time");
        assert!(age > Duration::ZERO && age < Duration::from_secs(3600), "{age:?}");
        let first = process_uptime();
        std::thread::sleep(Duration::from_millis(20));
        let later = process_uptime();
        assert!(first >= age, "{first:?} is before the process's age {age:?}");
        assert!(later >= first + Duration::from_millis(20), "{first:?} then {later:?}");
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// CPU-only A/B of taking the deferred-work lock on every access versus
    /// the list-lifetime invariant used here. Run with
    /// `cargo test --release -p victory-gpu deferred_access_without_lock -- --ignored --nocapture --test-threads=1`.
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn deferred_access_without_lock() {
        let Some(host) = super::host() else { return };
        let Ok(list) = host.new_list() else { return };
        let locked = std::sync::Mutex::new(super::Deferred::default());
        const CALLS: usize = 500_000;
        let mut samples = Vec::new();
        for sample in 0..5 {
            let started = std::time::Instant::now();
            for _ in 0..CALLS {
                let deferred = locked.lock().unwrap_or_else(|e| e.into_inner());
                std::hint::black_box(deferred.copies.len());
            }
            let old = started.elapsed().as_secs_f64() * 1e9 / CALLS as f64;
            let started = std::time::Instant::now();
            for _ in 0..CALLS {
                let deferred = list.deferred();
                std::hint::black_box(deferred.copies.len());
            }
            let new = started.elapsed().as_secs_f64() * 1e9 / CALLS as f64;
            println!("sample {sample}: lock {old:.1} ns, direct {new:.1} ns");
            samples.push((old, new));
        }
        let mut old: Vec<f64> = samples.iter().map(|s| s.0).collect();
        let mut new: Vec<f64> = samples.iter().map(|s| s.1).collect();
        old.sort_by(f64::total_cmp);
        new.sort_by(f64::total_cmp);
        println!(
            "median: lock {:.1} ns, direct {:.1} ns",
            old[old.len() / 2],
            new[new.len() / 2]
        );
    }

    /// Batching must leave the same steps, in the same order, as one call per
    /// range (including deduplication of repeated ranges).
    #[test]
    fn defer_refreshes_matches_one_call_per_range() {
        let Some(host) = host() else { return };
        let Ok(one) = host.new_list() else { return };
        let Ok(batched) = host.new_list() else { return };
        static DATA: [u8; 0x300] = [0; 0x300];
        let address = DATA.as_ptr() as u64;
        const ID: usize = 0x52454231;
        crate::shadow::register(ID, address, DATA.len() as u64);
        let Some(shadow) = crate::shadow::find(address, DATA.len() as u64) else {
            crate::shadow::unregister(ID, address);
            return;
        };
        let ranges = [
            (shadow.clone(), address as usize, 0x100usize),
            (shadow.clone(), address as usize + 0x100, 0x100),
            (shadow.clone(), address as usize, 0x100),
        ];
        for (shadow, address, len) in &ranges {
            one.defer_refresh(shadow, *address, *len);
        }
        batched.defer_refreshes(&ranges);
        let one = one.deferred();
        let batched = batched.deferred();
        assert_eq!(one.copies.len(), 2);
        assert_eq!(batched.copies.len(), 2);
        for (a, b) in one.copies.iter().zip(&batched.copies) {
            let (Step::Refresh(a, aa, al), Step::Refresh(b, ba, bl)) = (a, b) else {
                panic!("only refreshes were deferred");
            };
            assert_eq!(std::sync::Arc::as_ptr(a), std::sync::Arc::as_ptr(b));
            assert_eq!((aa, al), (ba, bl));
        }
        crate::shadow::unregister(ID, address);
    }

    /// CPU-only A/B of deferring the buffers of one draw: one lock for all
    /// ranges against one lock per range. Run with
    /// `cargo test --release -p victory-gpu defer_refresh_batching -- --ignored --nocapture --test-threads=1`.
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn defer_refresh_batching() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        static DATA: [u8; 0x800] = [0; 0x800];
        let address = DATA.as_ptr() as u64;
        const ID: usize = 0x52454232;
        crate::shadow::register(ID, address, DATA.len() as u64);
        let Some(shadow) = crate::shadow::find(address, DATA.len() as u64) else {
            crate::shadow::unregister(ID, address);
            return;
        };
        const BUFFERS: usize = 8;
        const ROUNDS: usize = 50_000;
        let ranges: Vec<_> = (0..BUFFERS)
            .map(|i| (shadow.clone(), address as usize + i * 0x100, 0x100))
            .collect();

        let mut samples = Vec::new();
        for sample in 0..5 {
            let started = std::time::Instant::now();
            for _ in 0..ROUNDS {
                for (s, address, len) in &ranges {
                    list.defer_refresh(s, *address, *len);
                }
            }
            let one_each = started.elapsed().as_secs_f64() * 1e9 / (ROUNDS * BUFFERS) as f64;
            let started = std::time::Instant::now();
            for _ in 0..ROUNDS {
                list.defer_refreshes(&ranges);
            }
            let batched = started.elapsed().as_secs_f64() * 1e9 / (ROUNDS * BUFFERS) as f64;
            println!("sample {sample}: one each {one_each:.1} ns, batched {batched:.1} ns");
            samples.push((one_each, batched));
        }
        let mut old: Vec<f64> = samples.iter().map(|s| s.0).collect();
        let mut new: Vec<f64> = samples.iter().map(|s| s.1).collect();
        old.sort_by(f64::total_cmp);
        new.sort_by(f64::total_cmp);
        println!(
            "median per range: one each {:.1} ns, batched {:.1} ns",
            old[old.len() / 2],
            new[new.len() / 2]
        );
        crate::shadow::unregister(ID, address);
    }
