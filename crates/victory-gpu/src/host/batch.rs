    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        use super::vram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        use super::ram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }

    #[test]
    fn coalesced_spans_keep_disjoint_ranges_separate_and_merge_overlaps() {
        use super::{MAX_BATCH_SPANS, Spans};

        let mut spans = Spans::One((100, 160));
        // Subset and touching/overlapping merges stay a single span.
        spans.add(110, 140);
        assert_eq!(spans.as_slice(), &[(100, 160)]);
        spans.add(160, 200);
        assert_eq!(spans.as_slice(), &[(100, 200)]);
        spans.add(80, 120);
        assert_eq!(spans.as_slice(), &[(80, 200)]);
        spans.add(300, 300); // empty span ignored
        assert_eq!(spans.as_slice(), &[(80, 200)]);

        // Disjoint out-of-order inserts stay sorted and separate.
        spans.add(10, 30);
        spans.add(500, 560);
        spans.add(300, 360);
        assert_eq!(
            spans.as_slice(),
            &[(10, 30), (80, 200), (300, 360), (500, 560)]
        );

        // Bridging adjacent/overlapping spans coalesces across them.
        spans.add(200, 300);
        assert_eq!(spans.as_slice(), &[(10, 30), (80, 360), (500, 560)]);

        // Exceeding MAX_BATCH_SPANS merges only the smallest gap.
        let mut bounded = Spans::One((0, 10));
        for i in 1..MAX_BATCH_SPANS {
            let start = i * 100;
            bounded.add(start, start + 10);
        }
        assert_eq!(bounded.as_slice().len(), MAX_BATCH_SPANS);
        // Insert one more very close to slot 1 ([100, 110)) so gap (15..25 vs 100 = 75) is smallest.
        bounded.add(120, 125);
        assert_eq!(bounded.as_slice().len(), MAX_BATCH_SPANS);
        assert!(bounded.as_slice().contains(&(100, 125)));
    }

    #[test]
    fn stale_under_flights_and_batch_flush_ignore_unrequested_gaps() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };

        let Some(host) = super::host() else {
            return;
        };
        const PAGE: usize = 64 << 10;
        let bytes = 4 * PAGE;
        // Allocate non-write-watched memory so Shadow tracks refreshes across the full span.
        // SAFETY: plain private anonymous RW pages freed before return.
        let start = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(start, 0);
        // SAFETY: zero-initialized allocation of `bytes`.
        unsafe { std::ptr::write_bytes(start as *mut u8, 0, bytes) };
        crate::shadow::register(start, start as u64, bytes as u64);
        let shadow = crate::shadow::find(start as u64, 1).unwrap();

        // Populate guest pages 0, 2, and 3 AFTER creating the shadow so only refreshed ranges update.
        // SAFETY: distinct pages inside the 4-page allocation.
        unsafe {
            std::ptr::write_bytes(start as *mut u8, 0x11, 64);
            std::ptr::write_bytes((start + 2 * PAGE) as *mut u8, 0x22, 64);
            std::ptr::write_bytes((start + 3 * PAGE) as *mut u8, 0x33, 64);
        }
        // Mark page 1 as an in-flight shader write.
        shadow.writing(start + PAGE, 64, true);

        let list1 = host.new_list().unwrap();
        let list2 = host.new_list().unwrap();
        let list3 = host.new_list().unwrap();
        list1.defer_refresh(&shadow, start, 64);
        list2.defer_refresh(&shadow, start + 3 * PAGE, 64);
        list3.defer_refresh(&shadow, start + PAGE, 64);

        // Disjoint refreshes on pages 0 and 3 do not intersect page 1's pending write.
        assert!(!super::stale_under_flights(&[&list1, &list2]));
        // A refresh covering page 1 itself does trigger stale detection.
        assert!(super::stale_under_flights(&[&list1, &list3]));

        // Flushing disjoint spans on pages 0 and 3 copies those endpoints without copying page 2.
        let mut batch = Batch::default();
        batch.add(&shadow, start, 64);
        batch.add(&shadow, start + 3 * PAGE, 64);
        batch.flush();
        let (p0, p2, p3) = (
            shadow.byte_at(0),
            shadow.byte_at(2 * PAGE),
            shadow.byte_at(3 * PAGE),
        );
        assert_eq!(p0, 0x11);
        assert_eq!(p2, 0x00, "unrequested middle page must not be copied");
        assert_eq!(p3, 0x33);

        drop(batch);
        drop(list1);
        drop(list2);
        drop(list3);
        shadow.copy_back_done(&[(start + PAGE, start + PAGE + 64)]);
        drop(shadow);
        crate::shadow::unregister(start, start as u64);
        assert_ne!(unsafe { VirtualFree(start as _, 0, MEM_RELEASE) }, 0);
    }
