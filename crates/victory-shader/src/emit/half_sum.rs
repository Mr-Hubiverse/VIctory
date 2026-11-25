//! Wave sums of 16-bit floats, bit for bit as the console adds them.
use super::super::bitcode::{Attr, Bin, V, pred};
use super::super::shader::flags;
use super::{Lower, Why, all_lane_blocks, need};

impl<'a> Lower<'a> {
    /// `WaveActiveSum` of 16-bit floats (`x`: each lane's term in its low half, +0 in the
    /// lanes outside the mask), bit for bit as the console adds it: every addition rounds to
    /// 16 bits, so the order counts, and the compiler's is the balanced tree over the lanes
    /// in their order (lanes 2k and 2k+1, then pairs, ... then the wave's two halves;
    /// [`super::wave::buddies`]). Each lane adds its partner's partial sum, read from the
    /// lane whose number differs in one bit, step by step; two host waves of 32 then exchange
    /// their halves' sums for the last addition. The high half of the result is lane 0's,
    /// which no `v_add_f16` writes.
    /// That needs every lane of a full wave there to be read: where not every lane runs, one
    /// host wave of 64 goes by [`Self::half_sum_running`] and two of 32 are refused; refused
    /// too in a thread group that does not fill its waves, and for each half-wave on its own
    /// (half a sum is no sum).
    pub(super) fn half_sum(&mut self, x: V) -> Result<V, Why> {
        const WHERE: Why = "a wave sum of 16-bit floats where not every lane runs";
        let threads = self.s.threads.ok_or("a wave sum of 16-bit floats outside compute")?;
        if threads.iter().product::<u32>() % 64 != 0 {
            return Err("a wave sum of 16-bit floats in a wave that is not full");
        }
        if self.local {
            return Err("a wave sum of 16-bit floats in a half-wave");
        }
        // two host waves of 32 that exchange, or one host wave of 64 lanes
        let halves = match (&self.exchange, self.s.wave_size) {
            (Some((_, false, _)), _) => true,
            (None, Some(64)) if self.x.p.wave64 => false,
            _ => return Err("a wave sum of 16-bit floats on these waves"),
        };
        let every = if halves {
            self.all_lanes(self.cur_ir)
        } else {
            all_lane_blocks(self.x.st).is_some_and(|all| all.get(self.cur_ir).copied().unwrap_or(false))
        };
        if !every {
            if halves {
                need(self.cur_ir);
                return Err(WHERE);
            }
            return Ok(self.half_sum_running(x));
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let mut sum = x;
        for step in 0..if halves { 5 } else { 6 } {
            let bit = self.s.m.i32(1 << step);
            let partner = self.s.m.bin(c, Bin::Xor, lane, bit);
            let theirs = self.read_lane(sum, partner);
            sum = self.add_halves(sum, theirs);
        }
        let (high, low, zero) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF), self.s.m.i32(0));
        let first = self.read_lane(x, zero);
        let first = self.s.m.bin(c, Bin::And, first, high);
        let sum = self.s.m.bin(c, Bin::And, sum, low);
        let mine = self.s.m.bin(c, Bin::Or, first, sum);
        if !halves {
            return Ok(mine);
        }
        let (lo, hi) = self.exchange_halves(c, mine);
        let c = self.cur;
        let total = self.add_halves(lo, hi);
        let total = self.s.m.bin(c, Bin::And, total, low);
        let first = self.s.m.bin(c, Bin::And, lo, high);
        Ok(self.s.m.bin(c, Bin::Or, first, total))
    }

    /// [`Self::half_sum`] in one host wave of 64 lanes where not every lane
    /// runs, still bit for bit as the console adds it. The console's section
    /// turns every lane on (`s_or_saveexec_b64 -1`) and gives the lanes that
    /// were off a term of +0 (`v_cndmask` on the saved mask), so its tree
    /// runs over all 64 lanes; on the host those lanes do not run, and their
    /// partial sums cannot be read. Each step of the tree adds the partial
    /// sum of the partner's aligned run of lanes: every running lane of a run
    /// holds that run's sum, so it is read from the run's first running lane
    /// (the wave's ballot), and a run with none running sums to +0, as the
    /// console's +0 terms add up to. The high half is lane 0's term where
    /// lane 0 runs, and 0 where it does not (`v_cndmask` gave it a whole 0).
    fn half_sum_running(&mut self, x: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let yes = self.s.m.bool(true);
        let running = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[yes]);
        let (lo, hi) = (self.s.m.extract(c, running, 0), self.s.m.extract(c, running, 1));
        let (zero, k31, k32) = (self.s.m.i32(0), self.s.m.i32(31), self.s.m.i32(32));
        let mut sum = x;
        for step in 0..6u32 {
            let c = self.cur;
            let size = 1u32 << step;
            let bit = self.s.m.i32(size);
            let partner = self.s.m.bin(c, Bin::Xor, lane, bit);
            let align = self.s.m.i32(!(size - 1));
            let start = self.s.m.bin(c, Bin::And, partner, align);
            // the half of the ballot that holds the run, its lanes at bit 0
            let upper = self.s.m.bin(c, Bin::And, start, k32);
            let upper = self.s.m.cmp(c, pred::NE, upper, zero);
            let word = self.s.m.select(c, upper, hi, lo);
            let shift = self.s.m.bin(c, Bin::And, start, k31);
            let mut run = self.s.m.bin(c, Bin::LShr, word, shift);
            if size < 32 {
                let mask = self.s.m.i32((1u32 << size) - 1);
                run = self.s.m.bin(c, Bin::And, run, mask);
            }
            let any = self.s.m.cmp(c, pred::NE, run, zero);
            // FirstbitLo (dx.op 32): the run's first running lane
            let first = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[run]);
            let from = self.s.m.bin(c, Bin::Add, start, first);
            // (with none running this lane reads itself; the value is not used)
            let from = self.s.m.select(c, any, from, lane);
            let theirs = self.read_lane(sum, from);
            let theirs = self.s.m.select(c, any, theirs, zero);
            sum = self.add_halves(sum, theirs);
        }
        let c = self.cur;
        let (high, low, one) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF), self.s.m.i32(1));
        // where lane 0 runs it is the wave's first running lane
        let first = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
        let first = self.s.m.bin(c, Bin::And, first, high);
        let zero_runs = self.s.m.bin(c, Bin::And, lo, one);
        let zero_runs = self.s.m.cmp(c, pred::NE, zero_runs, zero);
        let first = self.s.m.select(c, zero_runs, first, zero);
        let sum = self.s.m.bin(c, Bin::And, sum, low);
        self.s.m.bin(c, Bin::Or, first, sum)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::S_ENDPGM;
    use super::super::*;
    use crate::native::ComputeIo;
    use crate::{io, ir, resource, structure};

    /// A compute program of `threads` in a wave64 build, lowered for a host
    /// whose waves reach 64 lanes (AMD), with 512 bytes of LDS.
    fn lower_wave64_host(words: &[u32], threads: [u32; 3]) -> Result<Lowered, crate::ShaderError> {
        TEST_HOST_WAVE.set(Some(64));
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 256,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15 | 2 << 1,
                threads,
            }),
            code,
        };
        let p = ir::build(&native.code, true, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        let r = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        });
        TEST_HOST_WAVE.set(None);
        r
    }

    /// The console's `WaveActiveSum` of 16-bit floats (the tree of
    /// `wave::tests::a_sum_of_halves_over_the_wave_is_followed_in_the_order_of_its_tree`)
    /// inside a branch only lanes 0-31 take, in one host wave of 64: the
    /// section turns every lane on (`s_or_saveexec_b64 -1`) and gives the
    /// lanes outside the branch +0 (`v_cndmask`), which the host, whose
    /// lanes outside the branch do not run, takes as +0 too. Refused before
    /// ("where not every lane runs"): compute `b22a16a178976e1b` of the
    /// title, dropped 28-42 times a frame in the world on AMD.
    #[test]
    fn a_wave_sum_of_halves_where_not_every_lane_runs_lowers_on_64_lane_hosts() {
        let words = [
            0x7D88_00A0, // v_cmp_gt_u32_e32 vcc, 32, v0
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
            0xBF88_0014, // s_cbranch_execz 20 (past the section)
            0xBE84_25C1, // s_or_saveexec_b64 s[4:5], -1
            0xD501_0001,
            0x0012_0E80, // v_cndmask_b32 v1, 0, v7, s[4:5]
            0x6402_02E9,
            0xDE54_C101, // v_add_f16 v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
            0x6402_02E9,
            0x9360_1201, // v_add_f16 v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
            0x6402_02E9,
            0x0009_2401, // v_add_f16 v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
            0x6402_02FA,
            0xFF09_4001, // v_add_f16 v1, v1, v1 row_mirror bound_ctrl:1
            0xD778_0002,
            0x0201_0101, // v_permlanex16_b32 v2, v1, 0, 0
            0x6402_0302, // v_add_f16 v1, v2, v1
            0xD760_0007,
            0x0001_4101, // v_readlane_b32 s7, v1, 32
            0x6402_0207, // v_add_f16 v1, s7, v1
            0xD760_0007,
            0x0001_0101, // v_readlane_b32 s7, v1, 0
            0xBEFE_0404, // s_mov_b64 exec, s[4:5]
            0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
            0x7E06_0207, // v_mov_b32 v3, s7
            0xD834_0000,
            0x0000_0300, // ds_write_b32 v0, v3
            S_ENDPGM,
        ];
        lower_wave64_host(&words, [64, 1, 1]).unwrap();
    }
}
