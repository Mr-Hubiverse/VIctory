//! The pair of host waves a console wave is cut into (`exchange`, `pair_flags`): the
//! relabelling of a thread's console index, the exchange of wave-wide values over
//! LDS, and the handshake that meets the two halves where a group barrier would
//! wait for console waves that never come.
use super::super::bitcode::{Attr, Bin, Rmw, Term, V, pred};
use super::super::shader::flags;
use super::{Lower, meeting_mode};

impl<'a> Lower<'a> {
    /// A console wave64 program in host waves of 32: the first console thread of this host wave
    /// (0, 32, 64, ...). A GPU need not give a host wave consecutive thread indices (NVIDIA's
    /// 8 x 8 groups), so each wave takes the next 32 from an LDS counter instead: every thread
    /// zeroes it, the group meets, each wave's first lane adds 32 and the wave takes what it
    /// found. Branch-free, in the entry block, which every thread runs.
    pub(super) fn wave_base(&mut self) -> Option<V> {
        let slot = self.halves?;
        if let Some(v) = self.wave_base {
            return Some(v);
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let lds = self.s.lds();
        let k = self.s.m.i32(slot);
        let ptr = self.s.m.element(e, lds, k);
        let zero = self.s.m.i32(0);
        self.s.m.store(e, ptr, zero);
        self.group_sync(e);
        let lane = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let first = self.s.m.cmp(e, pred::EQ, lane, zero);
        let k32 = self.s.m.i32(32);
        let add = self.s.m.select(e, first, k32, zero);
        let ptr = self.s.m.element(e, lds, k);
        let old = self.s.m.atomic(e, Rmw::Add, ptr, add);
        let base = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[old]);
        self.s.flags |= flags::WAVE_OPS;
        self.wave_base = Some(base);
        Some(base)
    }

    /// The console thread index of this thread in the group: `wave_base` plus the host lane
    /// (host waves of 32), else the wave's first thread index rounded down plus the lane.
    pub(super) fn console_thread(&mut self) -> V {
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let lane = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        if let Some(base) = self.wave_base() {
            return self.s.m.bin(e, Bin::Add, base, lane);
        }
        let flat = self.s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        let first = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[flat]);
        let wave = self.s.wave_size.unwrap_or(32);
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        self.s.m.bin(e, Bin::Add, base, lane)
    }

    /// This thread's console lane (0..63): its relabelled thread index (`thread_id`), whose host
    /// wave holds a 32-aligned run of console lanes. Without `exchange`: the host lane.
    pub(super) fn console_lane(&mut self) -> V {
        if self.exchange.is_none() {
            return self.lane_index();
        }
        if let Some(v) = self.clane {
            return v;
        }
        let (e, i32t) = (self.s.entry, self.s.t.i32);
        let f = self.console_thread();
        let _ = i32t;
        let k63 = self.s.m.i32(63);
        let cl = self.s.m.bin(e, Bin::And, f, k63);
        self.s.flags |= flags::WAVE_OPS;
        self.clane = Some(cl);
        cl
    }

    /// A group barrier with an LDS fence in `block`.
    pub(super) fn group_sync(&mut self, block: usize) {
        let mode = self.s.m.i32(meeting_mode());
        let void = self.s.t.void;
        self.s.op(block, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// `exchange`: each host wave gives `mine` (the same in all its lanes) for its half; both get
    /// (lower half's, upper half's). A group with no upper half gets 0 for it.
    /// Returns the block the exchange ends at (a pair handshake ends `at`
    /// with control flow: what follows goes on in the block it returns).
    pub(super) fn exchange_halves(&mut self, at: usize, mine: V) -> (usize, (V, V)) {
        let Some((slot, small, _)) = self.exchange.as_ref().map(|(a, b, _)| (*a, *b, ())) else {
            let z = self.s.m.i32(0);
            return (at, (mine, z));
        };
        if small {
            let z = self.s.m.i32(0);
            return (at, (mine, z));
        }
        if self.local {
            // each host wave on its own: its half holds `mine`, the other nothing
            let cl = self.console_lane();
            let (k32, z) = (self.s.m.i32(32), self.s.m.i32(0));
            let upper = self.s.m.cmp(at, pred::IUGE, cl, k32);
            let lo = self.s.m.select(at, upper, z, mine);
            let hi = self.s.m.select(at, upper, mine, z);
            return (at, (lo, hi));
        }
        // the pair of dwords of this thread's console wave: its half's, then both
        let pair = self.exchange_pair(slot);
        let cl = self.console_lane();
        let five = self.s.m.i32(5);
        let at = self.s.m.cont(at);
        let half = self.s.m.bin(at, Bin::LShr, cl, five);
        let index = self.s.m.bin(at, Bin::Add, pair, half);
        let lds = self.s.lds();
        if self.pair_flags.is_some() {
            // A group of several console waves: only this wave's two host
            // waves may meet (a group barrier inside a branch the other
            // console wave does not take would wait for waves that never
            // come). A console wave of one host wave gives 0 for its other
            // half and meets no one.
            let paired = self.paired();
            let (hs, join) = (self.s.m.new_block(), self.s.m.new_block());
            // `at` can be a closed block's tail — a lane mask emitted lazily
            // long after its block ended: the exchange's branch replaces that
            // terminator, and the merge continues where `at` was going.
            let resume = self.s.m.term(at);
            self.s.m.terminate(at, Term::CondBr(paired, hs, join));
            self.s.m.close(at, join);
            let ptr = self.s.m.element(hs, lds, index);
            self.s.m.store(hs, ptr, mine);
            let e = self.pair_meet(hs, 0);
            let one = self.s.m.i32(1);
            let k1 = self.s.m.bin(e, Bin::Add, pair, one);
            let p0 = self.s.m.element(e, lds, pair);
            let p1 = self.s.m.element(e, lds, k1);
            let lo = self.s.m.load(e, p0);
            let hi = self.s.m.load(e, p1);
            // the slots are written again only after the other half read them
            let e = self.pair_meet(e, 1);
            self.s.m.terminate(e, Term::Br(join));
            self.s.m.close(e, join);
            let ty = self.s.m.type_of(mine);
            let (lp, hp) = (self.s.m.phi(join, ty), self.s.m.phi(join, ty));
            let z = self.s.m.i32(0);
            self.s.m.set_phi(lp, vec![(lo, e), (mine, at)]);
            self.s.m.set_phi(hp, vec![(hi, e), (z, at)]);
            if let Some(t) = resume {
                self.s.m.terminate(join, t);
            }
            return (join, (lp, hp));
        }
        let ptr = self.s.m.element(at, lds, index);
        self.s.m.store(at, ptr, mine);
        self.group_sync(at);
        let one = self.s.m.i32(1);
        let k1 = self.s.m.bin(at, Bin::Add, pair, one);
        let p0 = self.s.m.element(at, lds, pair);
        let p1 = self.s.m.element(at, lds, k1);
        let lo = self.s.m.load(at, p0);
        let hi = self.s.m.load(at, p1);
        // the slots are written again only after every thread read them
        self.group_sync(at);
        (at, (lo, hi))
    }

    /// Whether this thread's console wave is two host waves, which meet each
    /// other (made in the entry block): the group's last console wave has
    /// only its lower half when the thread count is not a multiple of 64,
    /// and it meets no one.
    pub(super) fn paired(&mut self) -> V {
        if let Some(v) = self.paired {
            return v;
        }
        let (_, other) = self.pair_lanes();
        let e = self.s.entry;
        let z = self.s.m.i32(0);
        let v = self.s.m.cmp(e, pred::NE, other, z);
        self.paired = Some(v);
        v
    }

    /// This host wave's active lanes and the other host wave of this console
    /// wave's (0 when the console wave is one host wave), made in the entry
    /// block. A group whose thread count is not a multiple of 64 gives the
    /// last console wave fewer than 32 lanes on a half, and the pair's ticket
    /// counts a round in what each wave actually adds.
    pub(super) fn pair_lanes(&mut self) -> (V, V) {
        if let Some(v) = self.pair_lanes {
            return v;
        }
        let e = self.s.entry;
        let thread = self.console_thread();
        let k63 = self.s.m.i32(!63u32);
        let base = self.s.m.bin(e, Bin::And, thread, k63);
        let group = self.s.m.i32(self.group_threads());
        // the console wave's lanes from its first thread (1..64)
        let rest = self.s.m.bin(e, Bin::Sub, group, base);
        let k32 = self.s.m.i32(32);
        let big = self.s.m.cmp(e, pred::SGT, rest, k32);
        let lower = self.s.m.select(e, big, k32, rest);
        let over = self.s.m.bin(e, Bin::Sub, rest, k32);
        let huge = self.s.m.cmp(e, pred::SGT, over, k32);
        let capped = self.s.m.select(e, huge, k32, over);
        let z = self.s.m.i32(0);
        let upper = self.s.m.select(e, big, capped, z);
        let cl = self.console_lane();
        let five = self.s.m.i32(5);
        let half = self.s.m.bin(e, Bin::LShr, cl, five);
        let is_upper = self.s.m.cmp(e, pred::NE, half, z);
        let my = self.s.m.select(e, is_upper, upper, lower);
        let other = self.s.m.select(e, is_upper, lower, upper);
        let v = (my, other);
        self.pair_lanes = Some(v);
        v
    }

    /// The threads of the group the program declares (`numthreads`).
    pub(super) fn group_threads(&self) -> u32 {
        self.s.threads.map(|t| t.iter().product()).unwrap_or(1)
    }

    /// The first flag dword of this thread's console wave (made in the entry
    /// block): the quads [`Self::pair_flags`] lays out per console wave.
    pub(super) fn flag_pair(&mut self) -> V {
        if let Some(v) = self.flag_pair {
            return v;
        }
        let f = self.pair_flags.expect("a pair handshake without its flags");
        let e = self.s.entry;
        let thread = self.console_thread();
        let six = self.s.m.i32(6);
        let wave = self.s.m.bin(e, Bin::LShr, thread, six);
        let two = self.s.m.i32(2);
        let quad = self.s.m.bin(e, Bin::Shl, wave, two);
        let base = self.s.m.i32(f);
        let v = self.s.m.bin(e, Bin::Add, base, quad);
        // The ticket counts of `pair_meet` start at 0 and groupshared is not
        // initialised: every thread zeroes its wave's four and the group
        // meets once, in the entry block every thread runs (`wave_base`).
        let lds = self.s.lds();
        let zero = self.s.m.i32(0);
        for k in 0..4 {
            let kk = self.s.m.i32(k);
            let d = self.s.m.bin(e, Bin::Add, v, kk);
            let ptr = self.s.m.element(e, lds, d);
            self.s.m.store(e, ptr, zero);
        }
        self.group_sync(e);
        self.flag_pair = Some(v);
        v
    }

    /// The two host waves of this thread's console wave meet on its flag
    /// pair `kind` (0 arrivals, 1 leavings), each adding to its own half's
    /// dword and spinning on the other's: the adds give a wave 32 tickets a
    /// round, so both halves of round `t/32` are in when the other's flag
    /// reaches `32*(t/32)+32` for this wave's ticket `t`. What a wave wrote
    /// before an arrivals meet the other reads after it; the leavings meet
    /// keeps a wave from overwriting a slot the other still reads.
    /// A console wave of one host wave meets no one and skips the meet.
    /// Ends `at` (which it leaves terminated); returns the block the meet
    /// continues in.
    pub(super) fn pair_meet(&mut self, at: usize, kind: u32) -> usize {
        let paired = self.paired();
        let (hs, join) = (self.s.m.new_block(), self.s.m.new_block());
        let at = self.s.m.cont(at);
        let resume = self.s.m.term(at);
        self.s.m.terminate(at, Term::CondBr(paired, hs, join));
        self.s.m.close(at, join);
        let e = self.s.entry;
        let cl = self.console_lane();
        let five = self.s.m.i32(5);
        let half = self.s.m.bin(e, Bin::LShr, cl, five);
        let one = self.s.m.i32(1);
        let other = self.s.m.bin(e, Bin::Xor, half, one);
        let k2k = self.s.m.i32(kind * 2);
        let base = self.flag_pair();
        let my = self.s.m.bin(e, Bin::Add, base, k2k);
        let my = self.s.m.bin(e, Bin::Add, my, half);
        let ot = self.s.m.bin(e, Bin::Add, base, k2k);
        let ot = self.s.m.bin(e, Bin::Add, ot, other);
        let lds = self.s.lds();
        let mine = self.s.m.element(e, lds, my);
        let other = self.s.m.element(e, lds, ot);
        // this wave's writes before are seen by the other after
        let mode = self.s.m.i32(8);
        let void = self.s.t.void;
        self.s.op(hs, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let t = self.s.m.atomic(hs, Rmw::Add, mine, one);
        let (lanes, partner) = self.pair_lanes();
        let round = self.s.m.bin(hs, Bin::UDiv, t, lanes);
        let round = self.s.m.bin(hs, Bin::Add, round, one);
        let want = self.s.m.bin(hs, Bin::Mul, round, partner);
        let lp = self.s.m.new_block();
        self.s.m.terminate(hs, Term::Br(lp));
        let z = self.s.m.i32(0);
        let cur = self.s.m.atomic(lp, Rmw::Add, other, z);
        let ok = self.s.m.cmp(lp, pred::IUGE, cur, want);
        self.s.m.terminate(lp, Term::CondBr(ok, join, lp));
        // ...and what the other wrote is seen by what this wave reads after
        self.s.op(join, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        if let Some(t) = resume {
            self.s.m.terminate(join, t);
        }
        join
    }

    /// The first of the two exchange dwords of this thread's console wave (made in the entry
    /// block): a group of several console waves has a pair for each.
    pub(super) fn exchange_pair(&mut self, slot: u32) -> V {
        if let Some(v) = self.pair {
            return v;
        }
        let e = self.s.entry;
        let thread = self.console_thread();
        let six = self.s.m.i32(6);
        let wave = self.s.m.bin(e, Bin::LShr, thread, six);
        let one = self.s.m.i32(1);
        let twice = self.s.m.bin(e, Bin::Shl, wave, one);
        let base = self.s.m.i32(slot);
        let v = self.s.m.bin(e, Bin::Add, base, twice);
        self.pair = Some(v);
        v
    }
    /// The two host waves of a console wave meet (a group barrier with a shared-memory
    /// fence; in a group of several console waves, the wave's own pair's handshake
    /// instead — a group barrier inside a branch only this console wave takes would
    /// wait for host waves that never come): LDS writes before are seen by reads after.
    pub(super) fn lds_meeting(&mut self) {
        if self.pair_flags.is_some() {
            let c = self.pair_meet(self.cur, 0);
            self.cur = c;
            return;
        }
        let mode = self.s.m.i32(meeting_mode());
        let void = self.s.t.void;
        self.s.op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }
}
