//! Image, sampler, and buffer accesses, a child module so it shares the emitter's state.

use super::*;

impl<'a> Lower<'a> {
    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let sampler = self.descriptors.get(&((b, i), true)).copied();
        if let Some(Source::Phi { block, args }) = sampler
            && let [first @ Source::Constant(_), second @ Source::Constant(_)] = args.as_slice()
            && let [left, right] = self.x.p.blocks[*block].preds.as_slice()
        {
            // Immediate samplers are separate static bindings. Sample both with
            // the original derivatives, then select the texel on the same edge
            // that supplied the console's sampler.
            if self.block[*left].is_none() {
                return self.image_with_sampler(inst, b, i, Some(second));
            }
            if self.block[*right].is_none() {
                return self.image_with_sampler(inst, b, i, Some(first));
            }
            let edge = |pred: usize| {
                self.edges.get(&(pred, *block)).copied().or(self.tail[pred])
                    .ok_or("a sampler choice with an unfinished predecessor")
            };
            let (left, right) = (edge(*left)?, edge(*right)?);
            let join = self.block[*block].ok_or("a sampler choice without a host block")?;
            let chosen = self.s.m.phi(join, self.s.t.i1);
            let (yes, no) = (self.s.m.bool(true), self.s.m.bool(false));
            self.s.m.set_phi(chosen, vec![(yes, left), (no, right)]);
            self.image_with_sampler(inst, b, i, Some(first))?;
            let left = inst.writes.iter()
                .map(|&(_, v)| self.i32_of(v))
                .collect::<Result<Vec<_>, _>>()?;
            self.image_with_sampler(inst, b, i, Some(second))?;
            for (&(_, v), a) in inst.writes.iter().zip(left) {
                let b = self.i32_of(v)?;
                let result = self.s.m.select(self.cur, chosen, a, b);
                self.vals.insert(v, Val::I32(result));
            }
            return Ok(());
        }
        self.image_with_sampler(inst, b, i, sampler)
    }

    pub(super) fn image_with_sampler(
        &mut self,
        inst: &Inst,
        b: usize,
        i: usize,
        sampler: Option<&Source>,
    ) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        if name.starts_with("image_bvh") {
            // shortcut: rays hit nothing (no node after this one): nothing is
            // in shadow, and indirect light is the sky's.
            let guard = self.guard(b, i)?;
            let Operand::Vector { reg, .. } = *data else {
                return Err("a ray intersection's result");
            };
            if name == "image_bvh64_intersect_ray" && !*a16 && self.x.res.addresses {
                let mut regs = Vec::new();
                for a in addr {
                    if let Operand::Vector { reg, dwords } = *a {
                        regs.extend((0..dwords).map(|k| reg + k));
                    }
                }
                let words = regs
                    .iter()
                    .map(|&r| {
                        let v = inst.read(Reg::V(r)).ok_or("a ray")?;
                        self.i32_of(v)
                    })
                    .collect::<Result<Vec<V>, Why>>()?;
                let words: [V; 12] = words.try_into().map_err(|_| "a ray")?;
                let r = self.intersect(&words, guard);
                for (k, x) in r.into_iter().enumerate() {
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let none = self.s.m.i32(u32::MAX);
            for k in 0..4 {
                self.write(inst, Reg::V(reg + k), none, guard);
            }
            return Ok(());
        }
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.effect_guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            6 if name == "image_load" || name == "image_get_resinfo" => (3, 2),
            7 if name == "image_load" || name == "image_get_resinfo" => (8, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // Full 32-bit address VGPR at `reg_idx` (offset, bias, z-compare).
        let addr_u32 = |l: &mut Self, reg_idx: usize| -> Result<V, Why> {
            let r = *regs.get(reg_idx).ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            l.i32_of(v)
        };
        // With a16, two 16-bit components per VGPR starting at `base_reg`, low half first.
        let addr_packed = |l: &mut Self, base_reg: usize, k: usize| -> Result<V, Why> {
            let x = addr_u32(l, base_reg + if *a16 { k / 2 } else { k })?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> { addr_packed(l, 0, k) };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let uint = !uav && !*d16 && *dmask == 1
            && matches!(name, "image_load" | "image_gather4_lz") && self.scalar_uint_srv(src, kind);
        let props = (kind | if uav { 0x1000 } else { 0 }, if uint { 0x105 } else { 0x409 });
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // GLC returns the old value. DXIL implements subtraction by adding
            // the two's complement of the operand at the same width.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 {
                return Err("a 16-bit image atomic");
            }
            // Compare-exchange consumes a replacement followed by a comparand.
            let compare = code == u32::MAX;
            let wide = match (compare, *dmask) {
                (false, 1) | (true, 3) => false,
                (false, 3) | (true, 15) => true,
                _ => return Err("an image atomic data mask"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            let width = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, data, if compare { 2 * width } else { width })?;
            let (ty, component) = if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64;
                (self.s.m.int(64), 0x107) // one uint64_t in an R32G32_UINT view
            } else {
                (self.s.t.i32, 0x105)
            };
            let original = if wide {
                self.join64(vals[0], vals[1])
            } else {
                vals[0]
            };
            let value = if op == "sub" {
                let zero = self.s.m.int_const(ty, 0);
                self.s.m.bin(c, Bin::Sub, zero, original)
            } else {
                original
            };
            let h = self.handle(site, src, false, (kind | 0x1000, component))?;
            let mut args = vec![h];
            let (intrinsic, opcode) = if compare {
                let comparand = if wide {
                    self.join64(vals[2], vals[3])
                } else {
                    vals[1]
                };
                args.extend(cs);
                args.extend([comparand, value]);
                let intrinsic = if wide {
                    "atomicCompareExchange.i64"
                } else {
                    "atomicCompareExchange.i32"
                };
                (intrinsic, 79)
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                (
                    if wide {
                        "atomicBinOp.i64"
                    } else {
                        "atomicBinOp.i32"
                    },
                    78,
                )
            };
            let at = self.begin_guarded(guard);
            let r = self
                .s
                .op(self.cur, intrinsic, Attr::NoUnwind, ty, opcode, &args);
            if cache.glc {
                let old = self.end_guarded(at, &[(r, original)])[0];
                if wide {
                    let (lo, hi) = self.split64(old);
                    self.write(inst, Reg::V(reg), lo, None);
                    self.write(inst, Reg::V(reg + 1), hi, None);
                } else {
                    self.write(inst, Reg::V(reg), old, None);
                }
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format). The
            // data VGPRs hold the dmask's channels in order; the others are
            // written as 0.
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for ch in 0..4u8 {
                if dmask >> ch & 1 == 0 {
                    vals.push(self.s.m.f32(0));
                    continue;
                }
                let k = (dmask & ((1 << ch) - 1)).count_ones() as u8;
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let form = sample_form(name);
        let f = form.unwrap_or_default();
        let gather = f.gather;
        let has_cl = name.ends_with("_cl");
        // Address word positions. Derivatives are 16-bit in the g16 opcodes,
        // coordinates and levels with a16 (GFX10.3 keeps the two apart, as
        // LLVM's lowerImage does where hasG16): two per dword, low half
        // first, each direction starting a dword.
        let spatial = coords - usize::from(kind == 6 || kind == 7);
        let whole = usize::from(f.o) + usize::from(f.b) + usize::from(f.c);
        let bias_at = (usize::from(f.o), a16.then_some(0));
        // A cube's gradients are those of its two face coordinates.
        let grads = match (f.d, kind) {
            (false, _) => 0,
            (true, 5) => 2,
            (true, _) => spatial,
        };
        let g16 = f.g16;
        let per_direction = if g16 { grads.div_ceil(2) } else { grads };
        let grad_at = |dir: usize, k: usize| {
            if g16 {
                (whole + dir * per_direction + k / 2, Some(k as u32 % 2))
            } else {
                (whole + dir * grads + k, None)
            }
        };
        let first_at = whole + 2 * per_direction;
        // Coordinate k; k = coords is the level or the level clamp.
        let coord_at = |k: usize| {
            if *a16 {
                (first_at + k / 2, Some(k as u32 % 2))
            } else {
                (first_at + k, None)
            }
        };
        let raw = |l: &mut Self, at: usize| -> Result<V, Why> { addr_u32(l, at) };
        // A float address: a dword, or an f16 half of one.
        let float = |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<V, Why> {
            let x = raw(l, at)?;
            let Some(h) = half else {
                return Ok(l.as_float(x));
            };
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * h), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            let x = l.s.m.bin(c, Bin::And, x, mask);
            let f32t = l.s.t.f32;
            Ok(l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]))
        };
        // A float address known at translation (an f16 only when 0).
        let known =
            |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<Option<f32>, Why> {
                let x = raw(l, at)?;
                Ok(l.s.m.known(x).and_then(|bits| match half {
                    None => Some(f32::from_bits(bits as u32)),
                    Some(h) => (bits >> (16 * h) & 0x7FFF == 0).then_some(0.0),
                }))
            };
        // The level an `_l` form names where it is known at translation, and
        // whether a gather at an explicit level needs emulating below.
        let mut level = None;
        let mut gather_at_level = false;
        let (op, sample) = match name {
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            "image_get_lod" if self.s.pixel => (81, true),
            _ => {
                if form.is_none() {
                    return Err(name);
                }
                level = if f.l {
                    known(self, coord_at(coords))?
                } else {
                    f.lz.then_some(0.0)
                };
                let level_zero = level == Some(0.0);
                gather_at_level = f.gather && f.l && !f.c && !level_zero;
                // The DXIL operation. SM 6.6 has no comparison at a level, with
                // a bias or with gradients (SampleCmpLevel is SM 6.7,
                // SampleCmpBias and SampleCmpGrad 6.8) and gathers level 0 only;
                // those forms are taken where a level or bias known to be 0 makes
                // them one SM 6.6 has.
                let op = if f.gather {
                    if f.b {
                        return Err("a gather with a level bias");
                    }
                    if !f.l && !f.lz {
                        return Err("a gather with implicit level");
                    }
                    if !level_zero && !gather_at_level {
                        return Err("a gather at a level not known to be 0");
                    }
                    if gather_at_level && kind != 2 {
                        return Err("a gather with explicit level on a non-2D texture");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    if !matches!(kind, 2 | 5 | 7) {
                        return Err("a gather from a 1D or 3D image");
                    }
                    if f.c { 74 } else { 73 }
                } else if f.c {
                    if f.d {
                        return Err("a comparison sample with explicit derivatives");
                    }
                    if f.l && !level_zero {
                        return Err("a comparison sample at a level not known to be 0");
                    }
                    if f.b && known(self, bias_at)?.is_none_or(|b| b != 0.0) {
                        return Err("a comparison sample with a level bias");
                    }
                    if f.l || f.lz {
                        65
                    } else if self.s.pixel
                        || self.s.threads.is_some() && self.s.mesh.is_none()
                    {
                        // Implicit derivatives: pixel and compute shaders.
                        64
                    } else {
                        return Err(
                            "a comparison sample with implicit derivatives outside pixel and compute shaders",
                        );
                    }
                } else if f.d {
                    63
                } else if f.l || f.lz {
                    62
                } else if f.b {
                    61
                } else {
                    60
                };
                (op, true)
            }
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address: DXIL's dimensions; a channel DXIL leaves
            // out reads 1 (UAVs: one level).
            let h = self.handle(site, src, false, props)?;
            let mip = if uav || matches!(kind, 3 | 8) {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            let present = match kind {
                1 => [true, false, false, true],
                2 | 3 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav || !present[ch as usize] {
                    one
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        // A gather at a named level, done as four texel reads (below).
        let mut gathered: Option<[V; 4]> = None;
        let r = if sample {
            let ss = sampler.ok_or("an untracked sampler")?;
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = f.c;
            if cmp && dmask.count_ones() != 1 {
                return Err("a comparison sample of several channels");
            }
            let sh = self.handle(site, ss, true, (14 | if cmp { 0x8000 } else { 0 }, 0))?;
            // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
            // signed (programmable offsets: gathers only).
            let first = whole;
            let coord_base = first_at;
            let mut cs = vec![fu; 4];
            // With a16, sampling coordinates, gradients, LOD and clamp are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            if kind == 5 && (*unorm || op == 81) {
                return Err("a cube sample with texel coordinates or a level query");
            }
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                let x = addr_packed(self, coord_base, k)?;
                *slot = to_float(self, x);
            }
            if kind == 5 {
                let dir = self.cube_direction(cs[0], cs[1], cs[2]);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                // The size's dimensions only: an array slice is undef.
                let mut lc = [fu; 3];
                let n = lod_dims(kind);
                lc[..n].copy_from_slice(&cs[..n]);
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let lod = self.s.op(
                        c,
                        "calculateLOD.f32",
                        Attr::ReadOnly,
                        f32t,
                        81,
                        &[h, sh, lc[0], lc[1], lc[2], clamped],
                    );
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if kind == 5 { 0 } else { spatial.min(3) };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            if f.o {
                // Texel offsets: x in bits 0-5, y in 8-13, z in 16-21, signed.
                let w = raw(self, 0)?;
                let field = |w: i64, k: usize| (w as i32) << (26 - 8 * k) >> 26;
                let immediate = self
                    .s
                    .m
                    .known(w)
                    .filter(|&w| (0..offsets).all(|k| (-8..=7).contains(&field(w, k))));
                if let (Some(w), false) = (immediate, f.gather) {
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        *o = self.s.m.i32(field(w, k) as u32);
                    }
                } else {
                    // Bits at..at+5 sign-extended: shift up, then down.
                    let k26 = self.s.m.i32(26);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let left = self.s.m.i32(26 - 8 * k as u32);
                        let t = self.s.m.bin(c, Bin::Shl, w, left);
                        *o = self.s.m.bin(c, Bin::AShr, t, k26);
                    }
                }
                if !f.gather && immediate.is_none() {
                    // A sample's offsets are immediates in -8..7 (gathers
                    // take any, per lane). At a level known here, the
                    // coordinates move by offset / that level's size instead:
                    // the same texels, the coordinate rounded once more (the
                    // console adds the offset to the texel coordinate). A
                    // sampler whose LOD clamp or bias moves the sampled
                    // level away from the one given, or a level past the
                    // view's last, differs.
                    let Some(n) = level.filter(|n| *n >= 0.0 && n.fract() == 0.0) else {
                        return Err(
                            "a texel offset past -8..7 or per lane at a level not known here",
                        );
                    };
                    let i32t = self.s.t.i32;
                    let f32t = self.s.t.f32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let n = self.s.m.i32(n as u32);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, n]);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let size = self.s.m.extract(c, d, k as u32);
                        let size = self.s.m.cast(c, Cast::UiToFp, size, f32t);
                        let texels = self.s.m.cast(c, Cast::SiToFp, *o, f32t);
                        let shift = self.s.m.bin(c, Bin::SDiv, texels, size);
                        cs[k] = self.s.m.bin(c, Bin::Add, cs[k], shift);
                        *o = zero;
                    }
                }
            }
            let mut args = vec![h, sh];
            args.extend(&cs);
            // The level clamp, after the coordinates.
            let clamp = if f.cl {
                float(self, coord_at(coords))?
            } else {
                fu
            };
            match op {
                // Outside a pixel shader there are no screen derivatives:
                // the validator refuses `sample` there ("features
                // incompatible with the shader stage": 730 compute programs,
                // among them the G-buffer resolves and the sky), and what a
                // driver does with it is its own affair.
                // shortcut: such a sample reads level 0 (a bias: that
                // level). The console takes derivatives from neighbouring
                // lanes there; compute them from the lanes if a resolve
                // shows the wrong mip.
                60 | 61 if !self.s.pixel => {
                    // Outside pixel shaders the console still takes the level
                    // from each quad of lanes (lane 1 minus lane 0 across,
                    // lane 2 minus lane 0 down); DXIL before validator 1.8
                    // has no implicit derivatives there, so the quad's
                    // differences become explicit gradients. A bias scales
                    // them by 2^bias, the same level for isotropic filtering.
                    self.cross_lane(true)?;
                    let scale = if op == 61 {
                        let bias = float(self, bias_at)?;
                        Some(self.unary(21, bias))
                    } else {
                        None
                    };
                    let g = self.quad_gradients(&cs[..spatial], scale);
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                60 => {
                    args.extend(&os);
                    args.push(clamp);
                    self.s
                        .op(c, "sample.f32", Attr::ReadOnly, self.s.t.res_ret, 60, &args)
                }
                61 => {
                    // With a16 the bias is an f16 in the low half of its
                    // word (the word is whole, the value is not).
                    let bias = addr_u32(self, usize::from(f.o))?;
                    let bias = if *a16 {
                        let low = self.s.m.i32(0xFFFF);
                        let bias = self.s.m.bin(c, Bin::And, bias, low);
                        to_float(self, bias)
                    } else {
                        self.as_float(bias)
                    };
                    args.extend(&os);
                    if self.s.pixel {
                        args.extend([bias, clamp]);
                        self.s.op(
                            c,
                            "sampleBias.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            61,
                            &args,
                        )
                    } else {
                        // As above: the bias is the level.
                        args.push(bias);
                        self.s.op(
                            c,
                            "sampleLevel.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            62,
                            &args,
                        )
                    }
                }
                63 if kind == 5 => {
                    // A cube with explicit gradients: the host wants them
                    // as changes of the direction, the console gives the
                    // changes of the face coordinates (the face is one
                    // unit wide). The level they select is taken from the
                    // longer of the two, in texels, and sampled directly.
                    // shortcut: no anisotropic filtering for these samples
                    // (the longer axis alone decides the level).
                    let mut len = [fu; 2];
                    for (axis, slot) in len.iter_mut().enumerate() {
                        let mut sum = self.s.m.f32(0);
                        for k in 0..2 {
                            let x = float(self, grad_at(axis, k))?;
                            let sq = self.s.m.bin(c, Bin::Mul, x, x);
                            sum = self.s.m.bin(c, Bin::Add, sum, sq);
                        }
                        *slot = self.unary(24, sum); // square root
                    }
                    let longer = self.binary(35, len[0], len[1]); // maximum
                    let i32t = self.s.t.i32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let z = self.s.m.i32(0);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                    let width = self.s.m.extract(c, d, 0);
                    let width = self.s.m.cast(c, Cast::UiToFp, width, self.s.t.f32);
                    let texels = self.s.m.bin(c, Bin::Mul, longer, width);
                    // DXIL `Log` is already base 2 (it was scaled by
                    // log2(e) once more: levels 1.44 times too coarse). A
                    // length of 0 gives minus infinity, which the sampler
                    // clamps to level 0. `sampleLevel` has no clamp: the
                    // console's minimum level (`_cl`) is applied here.
                    let lod = self.unary(23, texels);
                    let lod = if has_cl { self.binary(35, lod, clamp) } else { lod };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                63 => {
                    // shortcut: the coarse forms (`_cd`) compute one level
                    // per quad; the host's per pixel is the same when the
                    // gradients agree across the quad.
                    let mut g = vec![fu; 6];
                    for k in 0..spatial {
                        g[k] = float(self, grad_at(0, k))?;
                        g[3 + k] = float(self, grad_at(1, k))?;
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    self.s.op(
                        c,
                        "sampleGrad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        63,
                        &args,
                    )
                }
                64 | 65 => {
                    let reference = addr_u32(self, first - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(clamp);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    self.s
                        .op(c, name, Attr::ReadOnly, self.s.t.res_ret, op, &args)
                }
                62 => {
                    let lod = if f.l {
                        float(self, coord_at(coords))?
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    self.s.op(
                        c,
                        "sampleLevel.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        62,
                        &args,
                    )
                }
                _ => {
                    if gather_at_level {
                        // The host's gather has no level: it read level 0
                        // where the title names one (the depth pyramid its
                        // occlusion tests read: most of the world was
                        // culled against the wrong depths). The four texels
                        // are read at the level instead, in the gather's
                        // order: (u0,v1) (u1,v1) (u1,v0) (u0,v0).
                        // With an offset too (`image_gather4_l_o`): the
                        // offset is in texels of that level.
                        // The title's meshlet culling takes a 4x4 block of
                        // the pyramid as one gather and three offset ones
                        // ((2,0), (0,2), (2,2)); the three read level 0,
                        // two texels beside the first one's place, so a
                        // meshlet three or four texels wide was tested
                        // against a quarter of what it covers.
                        // shortcut: the level is rounded to a whole one and
                        // coordinates past the edge are clamped, whatever
                        // the sampler says.
                        let lod = addr_packed(self, coord_base, coords)?;
                        let lod = to_float(self, lod);
                        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
                        let zero = self.s.m.f32(0);
                        let lod = self.binary(35, lod, zero); // maximum
                        let lod = self.unary(26, lod); // round to nearest
                        let mip = self.s.m.cast(c, Cast::FpToUi, lod, i32t);
                        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                        let d = self
                            .s
                            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
                        let half = self.s.m.f32(0.5f32.to_bits());
                        let one = self.s.m.f32(1.0f32.to_bits());
                        let (mut lo, mut hi) = ([iu; 2], [iu; 2]);
                        for k in 0..2 {
                            let n = self.s.m.extract(c, d, k as u32);
                            let n = self.s.m.cast(c, Cast::UiToFp, n, f32t);
                            let last = self.s.m.bin(c, Bin::Sub, n, one);
                            let t = self.s.m.bin(c, Bin::Mul, cs[k], n);
                            let t = self.s.m.bin(c, Bin::Sub, t, half);
                            let first = self.unary(27, t); // round down
                            // The offset word's field (0 without one).
                            let by = self.s.m.cast(c, Cast::SiToFp, os[k], f32t);
                            let first = self.s.m.bin(c, Bin::Add, first, by);
                            let next = self.s.m.bin(c, Bin::Add, first, one);
                            for (slot, x) in [(&mut lo[k], first), (&mut hi[k], next)] {
                                let x = self.binary(35, x, zero); // maximum
                                let x = self.binary(36, x, last); // minimum
                                *slot = self.s.m.cast(c, Cast::FpToSi, x, i32t);
                            }
                        }
                        let ch = dmask.trailing_zeros();
                        let mut g = [fu; 4];
                        let mut read = None;
                        let texels = [(lo[0], hi[1]), (hi[0], hi[1]), (hi[0], lo[1]), (lo[0], lo[1])];
                        for (slot, (x, y)) in g.iter_mut().zip(texels) {
                            let r = self.s.op(
                                c,
                                "textureLoad.f32",
                                Attr::ReadOnly,
                                self.s.t.res_ret,
                                66,
                                &[h, mip, x, y, iu, iu, iu, iu],
                            );
                            *slot = self.s.m.extract(c, r, ch);
                            read = Some(r);
                        }
                        gathered = Some(g);
                        read.expect("four reads")
                    } else {
                        args.extend(&os[..2]);
                        args.push(self.s.m.i32(dmask.trailing_zeros()));
                        if op == 74 {
                            // The comparison reference, after the offset word.
                            let reference =
                                float(self, (usize::from(f.o), None))?;
                            args.push(reference);
                            self.s.op(
                                c,
                                "textureGatherCmp.f32",
                                Attr::ReadOnly,
                                self.s.t.res_ret,
                                74,
                                &args,
                            )
                        } else {
                            let ret = self.resource_return_type(uint);
                            self.s.op(
                                c,
                                if uint {
                                    "textureGather.i32"
                                } else {
                                    "textureGather.f32"
                                },
                                Attr::ReadOnly,
                                ret,
                                73,
                                &args,
                            )
                        }
                    }
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" || matches!(kind, 3 | 8) {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            let ret = self.resource_return_type(uint);
            self.s.op(
                c,
                if uint { "textureLoad.i32" } else { "textureLoad.f32" },
                Attr::ReadOnly,
                ret,
                66,
                &args,
            )
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if gather {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = match gathered {
                Some(g) => g[ch as usize],
                None => self.s.m.extract(c, r, ch),
            };
            xs.push(if uint {
                x
            } else if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel.
    // shortcut: the view is taken as a cube, not a cube array (whose face
    // operand adds 8 × the slice).
    pub(super) fn cube_direction(&mut self, s: V, t: V, face: V) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let f = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        // Per face +x, -x, +y, -y, +z, -z.
        self.cube_face(
            f,
            [
                [one, mone, u, u, u, nu],
                [nv, nv, one, mone, nv, nv],
                [nu, u, v, nv, one, mone],
            ],
        )
    }

    /// Each row's entry for the face (an integer, +x -x +y -y +z -z).
    pub(super) fn cube_face(&mut self, face: V, table: [[V; 6]; 3]) -> [V; 3] {
        let c = self.cur;
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, face, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }

    pub(super) fn raw_dword(&mut self, h: V, coords: [V; 2]) -> V {
        let c = self.cur;
        let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, self.s.t.res_ret, 139,
            &[h, coords[0], coords[1], mask, align]);
        let x = self.s.m.extract(c, r, 0);
        self.as_int(x)
    }

    pub(super) fn buffer_compare_exchange(&mut self, h: V, coords: [V; 3], old: V, new: V, wide: bool) -> V {
        let name = if wide { "atomicCompareExchange.i64" } else { "atomicCompareExchange.i32" };
        let ty = self.s.m.type_of(old);
        self.s.op(self.cur, name, Attr::NoUnwind, ty, 79,
            &[h, coords[0], coords[1], coords[2], old, new])
    }

    /// A CAS must run on a real lane and at an in-range address: outside
    /// those cases DXIL can return an unspecified old value.
    pub(super) fn buffer_bounds(&mut self, h: V, index: Option<V>, off: V, stride: u32, size: u32) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let unused = self.s.m.undef(i32t);
        let dimensions = self.s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, unused]);
        let length = self.s.m.extract(c, dimensions, 0);
        if let Some(index) = index {
            let record = self.s.m.cmp(c, pred::IULT, index, length);
            let limit = self.s.m.i32(stride - (size - 1));
            let member = self.s.m.cmp(c, pred::IULT, off, limit);
            return self.s.m.bin(c, Bin::And, record, member);
        }
        let tail = self.s.m.i32(size - 1);
        let last = self.s.m.bin(c, Bin::Add, off, tail);
        let fits = self.s.m.cmp(c, pred::IULT, last, length);
        let no_wrap = self.s.m.cmp(c, pred::IUGE, last, off);
        self.s.m.bin(c, Bin::And, fits, no_wrap)
    }

    pub(super) fn buffer_store_guard(&mut self, bounds: V, guard: Option<V>) -> V {
        let c = self.cur;
        let g = guard.map_or(bounds, |g| self.s.m.bin(c, Bin::And, g, bounds));
        if !self.s.pixel {
            return g;
        }
        let helper = self.s.op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        let no = self.s.m.bool(false);
        self.s.m.select(c, helper, no, g)
    }

    pub(super) fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err("an MTBUF store requiring a packed-format write");
        }
        if name.contains("_format_d16_hi_") {
            return Err("a typed buffer access to the high half with an untracked numeric format");
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        let atomic = name.strip_prefix("buffer_atomic_");
        let wide = atomic.is_some_and(|n| n.ends_with("_x2"));
        let atomic = atomic.map(|n| n.strip_suffix("_x2").unwrap_or(n));
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let native_atomic = match atomic {
            None => None,
            Some("add" | "sub") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap" | "inc" | "dec" | "csub") => None,
            Some("fcmpswap" | "fmin" | "fmax") => {
                // Mesa aco_insert_fp_mode.cpp, vmem_default_needs.
                return Err("a buffer floating-point atomic requiring the console's denormal mode");
            }
            Some(_) => return Err(name),
        };
        let sub_load = name.strip_prefix("buffer_load_").and_then(subword_load);
        let sub_store = match name {
            "buffer_store_byte" | "buffer_store_byte_d16_hi" => Some(8),
            "buffer_store_short" | "buffer_store_short_d16_hi" => Some(16),
            _ => None,
        };
        // d16: two 16-bit channels per VGPR, low half first (buffer and
        // tbuffer formats alike).
        let d16 = name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || sub_load.is_some() || sub_store.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let stride = used.map_or(0, |u| u.stride);
        // Sixteen bits a channel through a view of a table: the bits go as
        // they are, as unsigned integers, and the host is told, which shows
        // the shader a 16-bit float view there as its unsigned twin
        // ([`super::super::HostShader::half_views`]).
        // A format load of whole dwords whose every use reads one half of a
        // result as a half float moves 16-bit integers too (the head
        // program's reads of the face network's output), where every
        // access through its view does ([`Self::places_read_as_halves`]).
        // A d16 access too only where every access does: a whole float
        // read through the same place would see the unsigned twin.
        let place = typed
            .then(|| self.table_place(src))
            .flatten()
            .filter(|p| {
                self.halves_only.contains(p)
                    && (d16 || self.format_load_read_as_halves(inst))
            });
        let halves = place.is_some();
        if let Some(place) = place {
            self.half_views.insert(place);
        }
        if self.by_address(src)
            && let Fields::Buffer { rsrc, .. } = &inst.fields
        {
            if typed && format.is_none() {
                return Err("a typed buffer access by address");
            }
            let (h, start, stride) = self.made_buffer(inst, rsrc)?;
            self.made.insert(site, h);
            off = self.s.m.bin(c, Bin::Add, off, start);
            if let Some(i) = index {
                let rows = self.s.m.bin(c, Bin::Mul, i, stride);
                off = self.s.m.bin(c, Bin::Add, off, rows);
            }
        }
        let index = if self.by_address(src) { None } else { index };
        let (base, extra) = if self.by_address(src) {
            (0, None)
        } else {
            self.built_offset(src)?
        };
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            if store || atomic.is_some() {
                return Err("a write or atomic operation on a constant buffer");
            }
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            if let Some((width, signed, half)) = sub_load {
                let x = self.read_subword(byte, width, signed, |l, at| Ok(l.cbuffer_dword(h, at)))?;
                self.write_subword_result(inst, reg, x, half, guard)?;
                return Ok(());
            }
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        // A dword access by index through a descriptor built in code: the
        // stride is the code's own, so the element becomes a byte offset in
        // the raw root view.
        let (index, off) = match index {
            Some(i) if !typed && format.is_none() => match self.built_element(src, i, off)? {
                Some(byte) => (None, byte),
                // Structured buffers of 16-byte multiples are raw views of
                // their bytes on the host: D3D12 takes structured views only
                // on multiples of the stride, and AMD reads element 0 for all
                // of a raw view's. (Others find a start they fit, and raw
                // views are to start on 16 bytes.)
                None if stride >= 16 && stride.is_multiple_of(16) => {
                    let st = self.s.m.i32(stride);
                    let first = self.s.m.bin(c, Bin::Mul, i, st);
                    (None, self.s.m.bin(c, Bin::Add, first, off))
                }
                None => (Some(i), off),
            },
            x => (x, off),
        };
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        if sub_load.is_some() || sub_store.is_some() {
            let data = data.as_ref().ok_or("buffer sub-dword data")?;
            let Operand::Vector { reg, .. } = *data else {
                return Err("buffer sub-dword data");
            };
            let h = if index.is_some() {
                if stride < 4 || !stride.is_multiple_of(4) {
                    return Err("a structured sub-dword buffer access without a dword-aligned stride");
                }
                self.handle(site, src, false, (12 | u, stride))?
            } else {
                self.handle(site, src, false, (11 | u, 0))?
            };
            let coords = |byte| index.map_or([byte, iu], |i| [i, byte]);
            if let Some((width, signed, half)) = sub_load {
                // Root SRVs have no size in DXIL.
                let bounds = (!matches!(src, Source::Built { .. } | Source::Address))
                    .then(|| self.buffer_bounds(h, index, off, stride, width / 8));
                let at = self.begin_guarded(bounds);
                let x = self.read_subword(off, width, signed, |l, at| Ok(l.raw_dword(h, coords(at))))?;
                let zero = self.s.m.i32(0);
                let x = self.end_guarded(at, &[(x, zero)])[0];
                self.write_subword_result(inst, reg, x, half, guard)?;
            } else if let Some(width) = sub_store {
                let mut x = self.vgprs(inst, data, 1)?[0];
                if name.ends_with("_hi") {
                    let sixteen = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, sixteen);
                }
                let bounds = self.buffer_bounds(h, index, off, stride, width / 8);
                let active = self.buffer_store_guard(bounds, guard);
                let at = self.begin_guarded(Some(active));
                self.write_subword(off, width, x, |l, byte, mask, bits| {
                    let [a, b] = coords(byte);
                    let zero = l.s.m.i32(0);
                    l.atomic_update(zero, |l, old| {
                        let x = l.atomic_new_value(AtomicUpdate::MaskOr, old, mask, bits);
                        l.buffer_compare_exchange(h, [a, b, iu], old, x, false)
                    });
                });
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(op) = atomic {
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            if op == "csub" && !returns {
                // LLVM BUFInstructions.td only selects the returning form
                // on GFX10.3.
                return Err("buffer_atomic_csub without the required GLC bit");
            }
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let words = if wide { 2 } else { 1 };
            let vals = self.vgprs(inst, d, if op == "cmpswap" { 2 * words } else { words })?;
            let val = if wide { self.join64(vals[0], vals[1]) } else { vals[0] };
            let ty = self.s.m.type_of(val);
            let zero = self.s.m.int_const(ty, 0);
            let (h, coords) = match index {
                Some(index) => {
                    if stride >= 4 && stride.is_multiple_of(4) {
                        if wide && !stride.is_multiple_of(8) {
                            return Err("a 64-bit buffer atomic without an eight-byte-aligned stride");
                        }
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        (h, [index, off, iu])
                    } else {
                        if wide {
                            return Err("a 64-bit buffer atomic through a structured buffer of unknown stride");
                        }
                        let byte = self.runtime_stride_offset(src, index, off)?;
                        (self.handle(site, src, false, (11 | u, 0))?, [byte, iu, iu])
                    }
                }
                None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            if wide {
                self.s.flags |= flags::INT64 | flags::ATOMIC_HEAP64;
            }
            let active = if wide || matches!(op, "sub" | "inc" | "dec" | "csub") {
                let bounds = self.buffer_bounds(h, index, off, stride, 4 * u32::from(words));
                Some(self.buffer_store_guard(bounds, guard))
            } else {
                guard
            };
            let at = self.begin_guarded(active);
            let r = if op == "cmpswap" {
                // cmpswap: data is (new, compare).
                let compare = if wide { self.join64(vals[2], vals[3]) } else { vals[1] };
                self.buffer_compare_exchange(h, coords, compare, val, wide)
            } else if let Some(code) = native_atomic {
                let name = if wide { "atomicBinOp.i64" } else { "atomicBinOp.i32" };
                let code = self.s.m.i32(code);
                let data = if op == "sub" { self.s.m.bin(self.cur, Bin::Sub, zero, val) } else { val };
                self.s.op(self.cur, name, Attr::NoUnwind, ty, 78,
                    &[h, code, coords[0], coords[1], coords[2], data])
            } else {
                let update = match op {
                    "inc" => AtomicUpdate::Inc,
                    "dec" => AtomicUpdate::Dec,
                    "csub" => AtomicUpdate::ClampSub,
                    _ => unreachable!(),
                };
                self.atomic_update(zero, |l, old| {
                    let x = l.atomic_new_value(update, old, val, zero);
                    l.buffer_compare_exchange(h, coords, old, x, wide)
                })
            };
            if returns {
                let r = self.end_guarded(at, &[(r, zero)])[0];
                self.write_atomic_result(inst, reg, r, wide, guard);
            } else {
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // Through a view of a table each half is stored as it is
                // (`halves`). Reached another way:
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    if halves {
                        let low = self.s.m.i32(0xffff);
                        wide.push(self.s.m.bin(c, Bin::And, x, low));
                        continue;
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format (four unsigned integers for halves). The
                // channels the opcode leaves out are written as 0: the
                // validator has typed stores write all four.
                if *offen || *offset != 0 || matches!(off, V::Inst(_)) {
                    return Err("a typed buffer store at an offset");
                }
                let component = if halves { 0x405 } else { 0x409 };
                args.push(self.handle(site, src, false, (10 | u, component))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(if halves { x } else { self.as_float(x) });
                }
                args.resize(
                    3 + 4,
                    if halves {
                        self.s.m.i32(0)
                    } else {
                        self.s.m.f32(0)
                    },
                );
                (
                    69,
                    if halves {
                        "bufferStore.i32"
                    } else {
                        "bufferStore.f32"
                    },
                )
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride >= 4 && stride.is_multiple_of(4) {
                            args.push(self.handle(site, src, false, (12 | u, stride))?);
                            args.extend([index, off]);
                        } else {
                            let byte = self.runtime_stride_offset(src, index, off)?;
                            args.push(self.handle(site, src, false, (11 | u, 0))?);
                            args.extend([byte, iu]);
                        }
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("an indexed MTBUF load requiring a raw view of a structured descriptor");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let mut xs = self.format_load(h, off, f, count)?;
            if d16 {
                xs = self.d16_channels(f, xs)?;
            }
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uint = name == "buffer_load_format_x" && !uav && self.scalar_uint_srv(src, 10);
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let index = index.unwrap_or(self.s.m.i32(0));
            if halves {
                let h = self.handle(site, src, false, (10 | u, 0x405))?;
                let i32t = self.s.t.i32;
                let ret = self.s.m.named_struct("dx.types.ResRet.i32", vec![i32t; 5]);
                self.s.op(
                    c,
                    "bufferLoad.i32",
                    Attr::ReadOnly,
                    ret,
                    68,
                    &[h, index, iu],
                )
            } else {
                let h = self.handle(site, src, false, (10 | u, if uint { 0x105 } else { 0x409 }))?;
                let ret = self.resource_return_type(uint);
                self.s.op(
                    c,
                    if uint { "bufferLoad.i32" } else { "bufferLoad.f32" },
                    Attr::ReadOnly,
                    ret,
                    68,
                    &[h, index, iu],
                )
            }
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride >= 4 && stride.is_multiple_of(4) {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, index, off, mask, align],
                        )
                    } else {
                        let byte = self.runtime_stride_offset(src, index, off)?;
                        let h = self.handle(site, src, false, (11 | u, 0))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, byte, iu, mask, align],
                        )
                    }
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0); through
            // a view of a table, the 16 bits of each as they are.
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = if halves {
                        let low = self.s.m.i32(0xffff);
                        self.s.m.bin(c, Bin::And, f, low)
                    } else {
                        self.s
                            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
                    };
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}
