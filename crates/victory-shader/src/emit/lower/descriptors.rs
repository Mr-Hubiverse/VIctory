//! Loads and stores through descriptors, group-shared memory, and the descriptor heap tables, a child module so it shares the emitter's state.

use super::*;

impl<'a> Lower<'a> {
    pub(super) fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name.starts_with("s_scratch_") {
            return Err("scalar access to the console scratch aperture");
        }
        if name.contains("_store_") {
            return Err("a scalar store without root UAV address support");
        }
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        if self.scalar.get(&site).is_some_and(|a| a.param == MEMORY) {
            return self.memory_smem(inst);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            if !self.scalar.contains_key(&site)
                && self.x.res.addresses
                && let Some(count) = name.strip_prefix("s_load_dword").and_then(smem_dwords)
                && let Some(Operand::Scalar { code, .. }) = *data
            {
                let Fields::Smem {
                    base: Operand::Scalar { code: at, .. },
                    ..
                } = inst.fields
                else {
                    return Err("a scalar load's address");
                };
                let (lo, hi) = (
                    inst.read(Reg::S(at)).ok_or("a scalar load's address")?,
                    inst.read(Reg::S(at + 1)).ok_or("a scalar load's address")?,
                );
                let mut address = (self.i32_of(lo)?, self.i32_of(hi)?);
                if *offset != 0 {
                    let x = self.s.m.i32(*offset as u32);
                    address = self.add_address(address, x, true);
                }
                if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
                    let so = self.operand(inst, soffset)?;
                    address = self.add_address(address, so, false);
                }
                let xs = self.memory_dwords(address, count, false);
                for (k, x) in xs.into_iter().enumerate() {
                    self.write(inst, Reg::S(code + k as u8), x, None);
                }
                return Ok(());
            }
            if !self.scalar.contains_key(&site)
                && crate::resource::traces_rays(self.x.p)
                && let Some(count) = name.strip_prefix("s_load_dword").and_then(smem_dwords)
                && let Some(Operand::Scalar { code, .. }) = *data
            {
                // shortcut: out of the acceleration structure.
                let z = self.s.m.i32(0);
                for k in 0..count {
                    self.write(inst, Reg::S(code + k as u8), z, None);
                }
                return Ok(());
            }
            if let Some(a) = self.scalar.get(&site).copied()
                && self
                    .x
                    .native
                    .layout
                    .get(a.param)
                    .is_some_and(|p| p.root == RootType::Descriptor)
            {
                return self.root_smem(inst, site, a);
            }
            return self.descriptor_data(inst, site);
        };
        let count = smem_dwords(rest).ok_or("a scalar buffer load")?;
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        if self.by_address(src)
            && let Fields::Smem { base, .. } = &inst.fields
        {
            let (h, start, _) = self.made_buffer(inst, base)?;
            let mut addr = self.s.m.i32(*offset as u32);
            if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
                let so = self.operand(inst, soffset)?;
                addr = self.s.m.bin(c, Bin::Add, addr, so);
            }
            let addr = self.s.m.bin(c, Bin::Add, addr, start);
            let words = self.view_dwords(h, Some(0), addr, None, count);
            for (k, x) in words.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // A buffer the program picks from the view heap by an index it
        // computes (`ResourceHeap`) has no used-resource entry, and the
        // heap holds a buffer view there, not a constant buffer: read as
        // raw bytes, as a table's SRV is below. Read as a constant buffer
        // (the host's validation: "Descriptor Type in Heap: SRV, Register
        // Type: CBV", heap index 17) the loads gave nothing:
        // the title's meshlet culling (BatchCullMeshlets,
        // BatchCullPreSkinnedMeshlets, BatchCullBatchCSMeshlets) reads its
        // batch records, mesh headers and instance transforms this way, so
        // it passed no meshlet and every indirect mesh draw ran empty.
        // The host's table of strides by heap slot tells a buffer view (a
        // stride), a constant buffer (`SLOT_CONSTANT_BUFFER`) and anything
        // else (0) apart; only the load that fits the slot runs, and a slot
        // that is neither reads zeros, as the console's hardware gives for
        // a descriptor with no buffer behind it. (Read as a constant buffer,
        // a texture's or an empty slot faulted the card at address 0 on the
        // host's D3D12 on Linux: fork issue #698, item 4.)
        let picked = match kind {
            None => self.slot_stride(src)?,
            Some(_) => None,
        };
        if let Some(stride) = picked {
            let zero = self.s.m.i32(0);
            let constant = self.s.m.i32(super::super::shader::SLOT_CONSTANT_BUFFER);
            let c = self.cur;
            let some = self.s.m.cmp(c, pred::NE, stride, zero);
            let not_cb = self.s.m.cmp(c, pred::NE, stride, constant);
            let is_buffer = self.s.m.bin(c, Bin::And, some, not_cb);
            let at = self.begin_guarded(Some(is_buffer));
            let h = self.handle(site, src, false, (11, 0))?;
            let mut raw = Vec::new();
            let wide = Self::wide_loads();
            if wide {
                let dwords = self.raw_dwords(h, addr, count);
                raw.extend(dwords.into_iter().map(|x| (x, zero)));
            }
            for k in 0..if wide { 0 } else { count } {
                let b = self.cur;
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(b, Bin::Add, addr, off);
                let mask = self.s.m.i8(1);
                let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(
                    b,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, u, mask, al],
                );
                let x = self.s.m.extract(b, r, 0);
                let x = self.as_int(x);
                raw.push((x, zero));
            }
            let raw = self.end_guarded(at, &raw);
            let other = self.s.m.cmp(self.cur, pred::EQ, stride, constant);
            let at = self.begin_guarded(Some(other));
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let mut both = Vec::new();
            for (k, &r) in raw.iter().enumerate() {
                let b = self.cur;
                let off = self.s.m.i32(4 * k as u32);
                let a = self.s.m.bin(b, Bin::Add, addr, off);
                let x = self.cbuffer_dword(h, a);
                both.push((x, r));
            }
            let both = self.end_guarded(at, &both);
            for (k, x) in both.into_iter().enumerate() {
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        if matches!(kind, Some(u) if u.kind == ResourceType::Srv) {
            let h = self.handle(site, src, false, (11, 0))?;
            if Self::wide_loads() {
                for (k, x) in self.raw_dwords(h, addr, count).into_iter().enumerate() {
                    self.write(inst, Reg::S(code + k as u8), x, None);
                }
                return Ok(());
            }
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let mask = self.s.m.i8(1);
                let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, a, u, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let words = self.view_dwords(h, None, addr, constant, count);
        for (k, x) in words.into_iter().enumerate() {
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// `count` dwords from byte `addr` (`constant` when known) of a raw
    /// buffer (`raw` = `Some(0)`), a structured buffer of that stride
    /// (`raw` = `Some(stride)`) or a constant buffer view (`None`) `h`. The
    /// address's two low bits are ignored, as scalar memory does.
    pub(super) fn view_dwords(
        &mut self,
        h: V,
        raw: Option<u32>,
        addr: V,
        constant: Option<u32>,
        count: u32,
    ) -> Vec<V> {
        let c = self.cur;
        let mut out = Vec::new();
        if let Some(stride) = raw {
            let not3 = self.s.m.i32(!3);
            let addr = self.s.m.bin(c, Bin::And, addr, not3);
            let st = (stride != 0).then(|| self.s.m.i32(stride));
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let (idx, elem_off) = match st {
                    Some(st) => (
                        self.s.m.bin(c, Bin::UDiv, a, st),
                        self.s.m.bin(c, Bin::URem, a, st),
                    ),
                    None => (a, self.s.m.undef(self.s.t.i32)),
                };
                let mask = self.s.m.i8(1);
                let al = self.s.m.i32(4);
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, idx, elem_off, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                out.push(self.as_int(x));
            }
            return out;
        }
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base.wrapping_add(4 * k);
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            out.push(x);
        }
        out
    }

    /// `s_load_dword*` from the memory of a root CBV or SRV: constants the
    /// code reads through scalar memory at an address it computes from the
    /// root descriptor's, read through the root view (a raw buffer for an
    /// SRV, a constant buffer otherwise).
    // shortcut: like the console's scalar loads, the host's root views are
    // not bounds-checked, so a read past the view's data is undefined on
    // both.
    pub(super) fn root_smem(&mut self, inst: &Inst, site: Site, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = inst
            .name()
            .strip_prefix("s_load_dword")
            .and_then(smem_dwords)
            .ok_or("a scalar memory read as data")?;
        let src = Source::Built {
            address: a,
            bits: 0,
            words: [0, 0],
        };
        let srv = self
            .used_kind(&src)
            .is_some_and(|u| u.kind == ResourceType::Srv);
        let props = if srv { (11, 0) } else { (13, 0x1_0000) };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = self.dynamic_term(&a)? {
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        let constant = a.dynamic.is_none().then_some(a.offset);
        let words = self.view_dwords(h, srv.then_some(0), addr, constant, count);
        for (k, x) in words.into_iter().enumerate() {
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// Two scalar loads of the same dwords: same opcode and immediate
    /// offset, base read from the same values, and the same or an
    /// equivalent register offset.
    pub(super) fn twin_loads(&self, a: Site, b: Site) -> bool {
        let (ia, ib): (&Inst, &Inst) = (
            &self.x.p.blocks[a.0].insts[a.1],
            &self.x.p.blocks[b.0].insts[b.1],
        );
        let (
            Fields::Smem {
                base: Operand::Scalar { code: ba, .. },
                offset: oa,
                soffset: sa,
                ..
            },
            Fields::Smem {
                base: Operand::Scalar { code: bb, .. },
                offset: ob,
                soffset: sb,
                ..
            },
        ) = (&ia.fields, &ib.fields)
        else {
            return false;
        };
        let same_base = (0..2).all(|k| {
            let va = ia.read(Reg::S(ba + k));
            va.is_some() && va == ib.read(Reg::S(bb + k))
        });
        let same_soffset = match (sa, sb) {
            (Operand::Scalar { code: NULL, .. }, Operand::Scalar { code: NULL, .. }) => true,
            (Operand::Scalar { code: ca, .. }, Operand::Scalar { code: cb, .. }) => {
                match (ia.read(Reg::S(*ca)), ib.read(Reg::S(*cb))) {
                    (Some(x), Some(y)) => self.same_scalar(x, y),
                    _ => false,
                }
            }
            _ => false,
        };
        ia.name() == ib.name() && oa == ob && same_base && same_soffset
    }

    /// The same value, or two `v_readfirstlane_b32` of the same VGPR value
    /// (each waterfall loop reads the index anew).
    pub(super) fn same_scalar(&self, x: Value, y: Value) -> bool {
        if x == y {
            return true;
        }
        let first_lane_of = |v: Value| {
            self.x
                .p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| i.writes.iter().any(|w| w.1 == v))
                .filter(|i| i.name() == "v_readfirstlane_b32")
                .and_then(|i| {
                    i.reads
                        .iter()
                        .find(|r| matches!(r.0, Reg::V(_)))
                        .map(|r| r.1)
                })
        };
        matches!((first_lane_of(x), first_lane_of(y)), (Some(a), Some(b)) if a == b)
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program. A cube reads as its six faces (not a cube array).
    pub(super) fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // How the program uses the descriptor tells the view's kind: this
        // load's dwords, or those of another load of the same descriptor
        // (the code often loads the whole descriptor for the resource and a
        // dword or two of it again for its size). `skew` is the dword of
        // the descriptor this load's first dword is; another load's
        // descriptor counts only if this load starts inside it. Prefer an
        // image or vector buffer use when a descriptor has both scalar and
        // vector uses, since those carry format and indexing cues.
        let find = |own: bool| {
            let mut best = None;
            for d in self
                .x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
            {
                for l in d.source.leaves() {
                    let Source::Loaded { load, first } = *l else {
                        continue;
                    };
                    if (load == site) != own {
                        continue;
                    }
                    // A twin load of the same dwords (a non-uniform index
                    // reloads the descriptor in each waterfall loop, one for
                    // the size bits, another for the access) is at skew 0.
                    let Some(skew) = self
                        .load_skew(inst, site, load)
                        .or_else(|| (!own && self.twin_loads(site, load)).then_some(0))
                        .map(|s| s - i32::from(first))
                    else {
                        continue;
                    };
                    if !own && !(0..8).contains(&skew) {
                        continue;
                    }
                    let ui = &self.x.p.blocks[d.site.0].insts[d.site.1];
                    let found = (d.site, load, first, skew);
                    if matches!(ui.fields, Fields::Image { .. } | Fields::Buffer { .. }) {
                        return Some(found);
                    }
                    best.get_or_insert(found);
                }
            }
            best
        };
        let use_at = find(true).or_else(|| find(false));
        let (skew, src, props, image, uav, used) = if let Some((at, load, first, skew)) = use_at
        {
            let src = Source::Loaded { load, first };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store")
                || un.contains("atomic")
                || used.is_some_and(|u| u.kind == ResourceType::Uav);
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = match &ui.fields {
                // A ray-tracing structure in a buffer of the host: read raw.
                Fields::Image { .. } if un.starts_with("image_bvh") => ((11, 0), None),
                Fields::Image { dim, .. } => {
                    let kind = match dim {
                        0 => 1,
                        1 => 2,
                        2 => 4,
                        3 => 5,
                        4 => 6,
                        5 => 7,
                        6 => 3,
                        7 => 8,
                        _ => return Err("an image dimension (cube or MSAA)"),
                    };
                    let ty = if !uav && self.scalar_uint_srv(&src, kind) {
                        0x105
                    } else {
                        0x409
                    };
                    ((kind | u, ty), Some(kind))
                }
                Fields::Buffer { idxen, format, .. } => {
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if un.starts_with("buffer_load_format")
                        || un.starts_with("buffer_store_format")
                    {
                        // (Unsigned integers where the access moves halves, and
                        // the host is told the place shows its unsigned twin.)
                        let place = self.table_place(&src).filter(|p| {
                            self.halves_only.contains(p)
                                && (un.contains("_format_d16_")
                                    || self.format_load_read_as_halves(ui))
                        });
                        let halves = place.is_some();
                        if let Some(place) = place {
                            self.half_views.insert(place);
                        }
                        let ty = if halves {
                            0x405
                        } else if !uav && self.scalar_uint_srv(&src, 10) {
                            0x105
                        } else {
                            0x409
                        };
                        (10 | u, ty)
                    } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4)
                    {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        self.stale.extend(inst.writes.iter().map(|w| w.1));
                        return Err("a constant buffer's descriptor read as data");
                    }
                    (props, None)
                }
                // Read by scalar buffer loads alone: the raw view  reads it through.
                Fields::Smem { .. } if used.is_some_and(|u| u.kind == ResourceType::Srv) => {
                    ((11, 0), None)
                }
                // A constant buffer: nothing of its descriptor is a number.
                Fields::Smem { .. } => {
                    self.stale.extend(inst.writes.iter().map(|w| w.1));
                    return Err("a constant buffer's descriptor read as data");
                }
                _ => return Err("a scalar memory read as data"),
            };
            (skew, src, props, image, uav, used)
        } else {
            // A descriptor kept per lane in VGPRs (`Resources::lane_copied`): the code only
            // copies its dwords and compares them, lane against first lane, and every load
            // that reaches those lanes reads one address. Each dword stands in as its place
            // in the table, so the lanes compare equal as they do on the console.
            let address = self.scalar.get(&site).copied();
            if self.x.res.lane_copied.contains(&site)
                && let Some(a) = self.scalar.get(&site).copied()
            {
                for k in 0..count {
                    let x = self.s.m.i32(a.offset / 4 + k);
                    self.write(inst, Reg::S(code + k as u8), x, None);
                }
                return Ok(());
            }
            // A sampler's descriptor: no program reads its dwords as numbers, so where one
            // meets numbers at a join the register is dead on that path (`dead`), as a
            // descriptor's address or format is.
            let sampler = self.x.res.descriptors.iter().any(|d| {
                d.role == Role::Sampler
                    && d.source.leaves().into_iter().any(
                        |l| matches!(*l, Source::Loaded { load, .. } if load == site),
                    )
            });
            if sampler {
                self.stale.extend(inst.writes.iter().map(|w| w.1));
                return Err("a sampler's descriptor read as data");
            }
            // Code may walk the title address or test a structured view's record count.
            let wants = (0..count.min(4) as u8).any(|k| {
                k < 2 && inst
                    .write(Reg::S(code + k))
                    .is_some_and(|v| self.descriptor_word_used(v))
            });
            // The record count of a statically bound structured buffer the
            // container does not declare is the host's table word, beside
            // the view's address; a declared one is a heap view, which a
            // query of its handle answers instead.
            let declared = self
                .used_kind(&Source::Loaded { load: site, first: 0 })
                .is_some_and(|b| {
                    count == 4
                        && self.x.psv.is_some_and(|part| {
                            super::super::declares_structured_buffer(
                                part,
                                b.lower_bound,
                                b.kind == ResourceType::Uav,
                            )
                        })
                });
            let wants = wants
                || (count == 4
                    && !declared
                    && inst
                        .write(Reg::S(code + 2))
                        .is_some_and(|v| self.descriptor_word_used(v)));
            let mut pointer = if wants { self.view_address(site) } else { None };
            if pointer.is_none() && wants && count == 4 && self.picked_address_words(site, code) {
                pointer = Some(self.picked_address(site)?);
            }
            if count <= 4 && let Some((lo, hi)) = pointer
            {
                let records = inst.write(Reg::S(code + 2))
                    .and_then(|value| self.structured_srv_records(site, value));
                for k in 0..count {
                    let reg = Reg::S(code + k as u8);
                    match (k, records) {
                        (0, _) => self.write(inst, reg, lo, None),
                        (1, _) => self.write(inst, reg, hi, None),
                        (2, Some(n)) => self.write(inst, reg, n, None),
                        _ => {
                            if let Some(v) = inst.write(reg) {
                                self.vals.insert(v, Val::Missing(OTHER));
                                self.stale.insert(v);
                            }
                        }
                    }
                }
                return Ok(());
            }
            // Puddle detection looks up a 2D texture's dimensions by heap index
            // without sampling it. Use that index to query the bound host view.
            // Only the heap supplies this otherwise untyped 2D lookup.
            if count == 8 && address.is_some_and(|a| {
                a.dynamic.is_some()
                    && self.x.native.layout.get(a.param).is_some_and(|p| {
                        p.root == RootType::ResourceHeap
                    })
            }) {
                let src = Source::Loaded { load: site, first: 0 };
                let h = self.handle(site, &src, false, (2, 0x409))?;
                let words = self.image_size_words(h, 2, false, true);
                self.write_descriptor_words(inst, code, count, 0, &words);
                return Ok(());
            }
            // A descriptor nothing uses as one, whose size the code reads all
            // the same (`unused_descriptor_kind` answers): as a used one's.
            if let Some((first, props, image, uav, used)) =
                self.unused_descriptor_kind(inst, site, code, count)
            {
                let src = Source::Loaded { load: site, first };
                return self.write_used_descriptor_words(
                    inst,
                    site,
                    code,
                    count,
                    &src,
                    i32::from(first).wrapping_neg(),
                    props,
                    image,
                    uav,
                    used,
                );
            }
            // An image the program only measures (it reads the size bits and
            // samples the image in other programs): eight dwords are an image
            // descriptor, taken here as a 2D texture.
            // shortcut: a 3D or array image would report depth or layers 1.
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            let used = self.used_kind(&src);
            if count == 8 && !used.is_some_and(|u| u.kind != ResourceType::Srv) {
                (0, src, (2, 0x409), Some(2), false, used)
            } else {
                // A buffer the program only counts the records of (#329):
                // an exact statically addressed four-dword binding of a
                // structured buffer the container declares, never the
                // nearest earlier entry or a dynamically indexed range.
                let uav = used.is_some_and(|u| u.kind == ResourceType::Uav);
                let Some(address) = self.scalar.get(&site).copied() else {
                    if std::env::var_os("VICTORY_DESC_WHY").is_some() {
                        eprintln!(
                            "DESC-WHY {:#x} {} scalar=None lane_copied={}",
                            inst.machine.offset,
                            inst.name(),
                            self.x.res.lane_copied.contains(&site)
                        );
                    }
                    return Err("a descriptor read only as data");
                };
                let Some(binding) = used.filter(|b| {
                    count == 4
                        && address.dynamic.is_none()
                        && b.table_offset_dwords.checked_mul(4) == Some(address.offset)
                        && b.dwords == 4
                        && matches!(b.kind, ResourceType::Srv | ResourceType::Uav)
                        && (4..=2048).contains(&b.stride)
                        && b.stride.is_multiple_of(4)
                        && self.x.psv.is_some_and(|part| {
                            super::super::declares_structured_buffer(part, b.lower_bound, uav)
                        })
                }) else {
                    if std::env::var_os("VICTORY_DESC_WHY").is_some() {
                        eprintln!(
                            "DESC-WHY {:#x} {} scalar={:?} lane_copied={}",
                            inst.machine.offset,
                            inst.name(),
                            Some(address),
                            self.x.res.lane_copied.contains(&site)
                        );
                    }
                    return Err("a descriptor read only as data");
                };
                let u = if uav { 0x1000 } else { 0 };
                (0, src, (12 | u, binding.stride), None, uav, used)
            }
        };
        self.write_used_descriptor_words(
            inst, site, code, count, &src, skew, props, image, uav, used,
        )
    }


    /// What [`Self::descriptor_data`] makes of a descriptor load a known use
    /// (or [`Self::unused_descriptor_kind`]) gives a kind to: the handle for
    /// `src`, then the load's `count` dwords from `first` on — an image's size
    /// fields, or a buffer's record count (and address, where the code computes
    /// with it).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_used_descriptor_words(
        &mut self,
        inst: &Inst,
        site: Site,
        code: u8,
        count: u32,
        src: &Source,
        skew: i32,
        props: (u32, u32),
        image: Option<u32>,
        uav: bool,
        used: Option<&'a UsedResource>,
    ) -> Result<(), Why> {
        const OTHER: Why = "a descriptor's address or format read as data";
        let Source::Loaded { load, .. } = *src else {
            return Err("a descriptor read only as data");
        };
        // A structured buffer's records, of a raw view's bytes.
        let records = (props.0 & 0xfff == 12 && props.1.is_multiple_of(16)).then_some(props.1);
        let props = if records.is_some() {
            (11 | props.0 & 0x1000, 0)
        } else {
            props
        };
        // A typed or structured buffer's element count comes from the host's
        // table of them by slot: the AMD driver's compiler crashes on a size
        // query of a structured buffer from the heap (stride 60, nine
        // compute programs of the world) and takes minutes over hundreds of
        // a typed one's; the table holds the title's own count.
        // (A structured buffer the host views raw keeps the query, of bytes.)
        if image.is_none() && records.is_none() && matches!(props.0 & 0xfff, 10 | 12) {
            let slot = if load == site {
                self.heap_slot(src, false)?
            } else {
                self.heap_index_at(site, skew.wrapping_neg(), false)?
            };
            let n = self.slot_dword(slot, super::super::OWN_SLOTS);
            for k in 0..count {
                if (k as i32).wrapping_add(skew) == 2 {
                    self.write(inst, Reg::S(code + k as u8), n, None);
                } else if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                    self.vals.insert(v, Val::Missing(OTHER));
                }
            }
            return Ok(());
        }
        // A buffer's record count is queried through a handle of its own,
        // never the handle of an access. AMD's RDNA 4 driver (RX 9060 XT
        // and RX 9070 XT, 32.0.320xx) faults inside its shader compiler,
        // ending the process at pipeline creation, when one handle of a raw
        // buffer feeds both a size query whose result is used and a load;
        // with a handle each it compiles. An image's handle stays shared
        // with its accesses.
        let buffer = image.is_none();
        if buffer && load == site {
            self.swap_size_caches();
        }
        // The handle is built from this load's own address (the other
        // load's registers need not hold it here): the descriptor starts
        //  dwords before what this load reads.
        let h = if load == site {
            self.handle(site, src, false, props)
        } else {
            let index = self.heap_index_at(site, skew.wrapping_neg(), false)?;
            Ok(self.s.heap_handle(self.cur, index, false, false, props))
        };
        if buffer && load == site {
            self.swap_size_caches();
        }
        let h = h?;
        // The size words are asked of the host only when the program reads
        // one as data: most loads take the descriptor whole for an image or
        // buffer instruction, and hundreds of size queries of heap handles
        // in one program stop the AMD driver's compiler (minutes, then
        // E_FAIL: nine compute programs of the world).
        let sized = if image.is_some() { 1..=4 } else { 2..=2 };
        let wanted = (0..count).any(|k| {
            let rel = (k as i32).wrapping_add(skew);
            let Some(v) = inst.write(Reg::S(code + k as u8)) else {
                return false;
            };
            (sized.contains(&rel) && self.read_as_data(v))
                || (matches!(rel, 0 | 1) && self.computed_with(v, 6))
        });
        if !wanted {
            for k in 0..count {
                if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                    self.vals.insert(v, Val::Missing(OTHER));
                    self.stale.insert(v);
                }
            }
            return Ok(());
        }
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let reads_word3 = (0..count)
                    .filter(|&k| (k as i32).wrapping_add(skew) == 3)
                    .filter_map(|k| inst.write(Reg::S(code + k as u8)))
                    .any(|v| self.read_as_data(v));
                self.image_size_words(h, kind, uav, reads_word3)
            }
            None => {
                let c = self.cur;
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let mip = self.s.m.undef(i32t);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
                // A raw view counts bytes; the descriptor's record count is in elements.
                let mut n = self.s.m.extract(c, d, 0);
                let stride = records.or_else(|| {
                    (props.0 & 0xFF == 11)
                        .then(|| used.map_or(0, |u| u.stride))
                        .filter(|s| *s > 1)
                });
                if let Some(st) = stride {
                    let k = self.s.m.i32(st);
                    n = self.s.m.bin(c, Bin::UDiv, n, k);
                }
                let mut words = vec![(2, n)];
                // Its address too, where the code computes with it and the
                // host can tell it.
                let computed = skew == 0
                    && (0..2u32).any(|k| {
                        inst.write(Reg::S(code + k as u8))
                            .is_some_and(|v| self.computed_with(v, 6))
                    });
                if computed && let Some((lo, hi)) = self.view_address(site)
                {
                    words.extend([(0, lo), (1, hi)]);
                }
                words
            }
        };
        self.write_descriptor_words(inst, code, count, skew.wrapping_neg() as u32, &words);
        Ok(())
    }



    /// An image descriptor's size fields in the GFX10 layout, from the dimensions of the
    /// host view `h` of `kind` (as `descriptor_data` explains): (dword, value).
    pub(super) fn image_size_words(&mut self, h: V, kind: u32, uav: bool, reads_word3: bool) -> Vec<(u32, V)> {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if uav || matches!(kind, 3 | 8) {
            self.s.m.undef(i32t)
        } else {
            self.s.m.i32(0)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let one = self.s.m.i32(1);
        let w = field(self, 0);
        let (height, depth) = match kind {
            1 => (one, one),
            2 | 3 => (field(self, 1), one),
            6 => (one, field(self, 1)),
            // A cube is six faces (DXIL's dimensions have no depth).
            5 => (field(self, 1), self.s.m.i32(6)),
            _ => (field(self, 1), field(self, 2)),
        };
        // The level count only feeds word 3. A level query of a 3D
        // image inside a loop crashes NVIDIA's Vulkan compiler (vkd3d-
        // proton turns it into OpImageQueryLevels), so query it only
        // when the program reads that word as data.
        let levels = if uav || matches!(kind, 3 | 8) || !reads_word3 {
            one
        } else {
            field(self, 3)
        };
        let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
        let (w, height, depth, levels) = (
            less(self, w),
            less(self, height),
            less(self, depth),
            less(self, levels),
        );
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k30, k2, k14, k16, mask) = (
            k(self, 30),
            k(self, 2),
            k(self, 14),
            k(self, 16),
            k(self, 0x3FFF),
        );
        let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
        let hi = self.s.m.bin(c, Bin::LShr, w, k2);
        let hi = self.s.m.bin(c, Bin::And, hi, mask);
        let ht = self.s.m.bin(c, Bin::Shl, height, k14);
        let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
        let ty = [0, 8, 9, 14, 10, 11, 12, 13, 15][kind as usize];
        let base = k(self, 0xFAC | ty << 28);
        let last = self.s.m.bin(c, Bin::Shl, levels, k16);
        let d3 = self.s.m.bin(c, Bin::Or, base, last);
        vec![(1, d1), (2, d2), (3, d3), (4, depth)]
    }

    /// The `count` dwords an `s_load_dword*` writes from `s[code]` on, the descriptor's
    /// dword `first` first: those of `words` (dword, value) get their value, the others
    /// have none (reading them stops the program).
    pub(super) fn write_descriptor_words(
        &mut self, inst: &Inst, code: u8, count: u32, first: u32, words: &[(u32, V)],
    ) {
        const OTHER: Why = "a descriptor's address or format read as data";
        for k in 0..count {
            let rel = k.wrapping_sub(first);
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                        self.stale.insert(v);
                    }
                }
            }
        }
    }

    /// What [`Self::descriptor_data`] makes of a descriptor load that nothing uses as a
    /// descriptor but whose dwords the code reads: the first dword, the handle properties,
    /// the image kind (none for a buffer), whether it is an unordered view, and its
    /// used-resource entry. An image's eight dwords read as a 2D texture's, a buffer's four
    /// as a raw view's where the code reads its record count (dword 2); none for anything
    /// else (a constant buffer, a sampler, a buffer whose address alone is read).
    // shortcut: an image whose kind nothing tells is taken for a 2D texture (the title's
    // reads are texel sizes of 2D textures: puddle masks, shadow masks); a host view of
    // another kind would answer its own sizes.
    #[allow(clippy::type_complexity)]
    pub(super) fn unused_descriptor_kind(
        &self,
        inst: &Inst,
        site: Site,
        code: u8,
        count: u32,
    ) -> Option<(u8, (u32, u32), Option<u32>, bool, Option<&'a UsedResource>)> {
        // Kept per lane and compared (`Resources::lane_copied`), or a sampler's: the cases
        // `descriptor_data` handles on their own.
        if self.x.res.lane_copied.contains(&site)
            || self.x.res.descriptors.iter().any(|d| {
                d.role == Role::Sampler
                    && d.source.leaves().into_iter().any(
                        |l| matches!(*l, Source::Loaded { load, .. } if load == site),
                    )
            })
        {
            return None;
        }
        let used = self.used_kind(&Source::Loaded {
            load: site,
            first: 0,
        });
        if used.is_some_and(|u| !matches!(u.kind, ResourceType::Srv | ResourceType::Uav)) {
            return None;
        }
        let read = |k: u32| {
            inst.write(Reg::S(code + k as u8)).is_some_and(|v| {
                self.x
                    .p
                    .blocks
                    .iter()
                    .flat_map(|b| &b.insts)
                    .any(|i| i.reads.iter().any(|r| r.1 == v))
            })
        };
        let uav = used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        match count {
            // width (dwords 1-2), height (2), levels (3), depth (4)
            8 if (1..5).any(read) && used.is_none_or(|u| u.dwords == 8) => {
                Some((0, (2 | u, 0x409), Some(2), uav, used))
            }
            // the record count
            4 if read(2) && used.is_none_or(|u| u.dwords == 4) => {
                Some((0, (11 | u, 0), None, uav, used))
            }
            _ => None,
        }
    }

    /// The dword of what the scalar load at `other` reads that `inst` (at
    /// `site`) reads first: the same base and offset registers, immediates
    /// a whole number of dwords apart, or else traced addresses that are
    /// (the same parameter and computed term: the code often builds the
    /// table pointer `{table, 0}` again before the second load); none for
    /// loads that differ otherwise.
    pub(super) fn load_skew(&self, inst: &Inst, site: Site, other: Site) -> Option<i32> {
        if other == site {
            return Some(0);
        }
        if let (Some(a), Some(b)) = (self.scalar.get(&site), self.scalar.get(&other))
            && (a.param, a.dynamic, a.includes_base) == (b.param, b.dynamic, b.includes_base)
            && a.offset.wrapping_sub(b.offset) % 4 == 0
        {
            return Some(a.offset.wrapping_sub(b.offset) as i32 / 4);
        }
        let o: &Inst = &self.x.p.blocks[other.0].insts[other.1];
        let (
            Fields::Smem {
                base: Operand::Scalar { code: b0, .. },
                offset: f0,
                soffset: s0,
                ..
            },
            Fields::Smem {
                base: Operand::Scalar { code: b1, .. },
                offset: f1,
                soffset: s1,
                ..
            },
        ) = (&inst.fields, &o.fields)
        else {
            return None;
        };
        let reads = |i: &Inst, code: u8| (i.read(Reg::S(code)), i.read(Reg::S(code + 1)));
        let same_soffset = match (s0, s1) {
            (Operand::Scalar { code: NULL, .. }, Operand::Scalar { code: NULL, .. }) => true,
            (Operand::Scalar { code: c0, .. }, Operand::Scalar { code: c1, .. }) => {
                inst.read(Reg::S(*c0)).is_some() && inst.read(Reg::S(*c0)) == o.read(Reg::S(*c1))
            }
            _ => false,
        };
        let delta = f0.wrapping_sub(*f1);
        (reads(inst, *b0).0.is_some()
            && reads(inst, *b0) == reads(o, *b1)
            && same_soffset
            && delta % 4 == 0)
            .then_some(delta / 4)
    }

    /// Whether an instruction or phi reads `v` other than as a resource
    /// operand of a memory instruction (which takes the descriptor whole).
    pub(super) fn read_as_data(&self, v: Value) -> bool {
        self.x.p.blocks.iter().any(|b| {
            b.phis.iter().any(|p| p.args.contains(&v))
                || b.insts.iter().any(|i| {
                    !matches!(
                        i.fields,
                        Fields::Image { .. } | Fields::Buffer { .. } | Fields::Smem { .. }
                    ) && i.reads.iter().any(|r| r.1 == v)
                })
        })
    }

    /// `s_load_dword*` from the program's constant table.
    pub(super) fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = inst
            .name()
            .strip_prefix("s_load_dword")
            .and_then(smem_dwords)
            .ok_or("a scalar load from the code")?;
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(value) = a.dynamic {
            let mut d = self.i32_of(value)?;
            // A term held per lane reached the address through
            // `v_readfirstlane_b32`: the first lane's.
            let per_lane = self.def_block.get(&value).is_some_and(|&b| {
                let writes = |i: &Inst| i.writes.iter().any(|w| w.1 == value && matches!(w.0, Reg::V(_)));
                self.x.p.blocks[b].insts.iter().any(writes)
            });
            if per_lane {
                self.s.flags |= flags::WAVE_OPS;
                let i32t = self.s.t.i32;
                d = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[d]);
            }
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    pub(super) fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    pub(super) fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    // DXIL's division may be off by an ulp (65535 / 65535 =
                    // 0.99999994 on AMD); the format gives exactly 1.0 for
                    // the largest value, which an f16 conversion keeps.
                    let top = self.s.m.i32(((1u64 << w) - 1) as u32);
                    let full = self.s.m.cmp(c, pred::EQ, field, top);
                    let one = self.s.m.f32(1.0f32.to_bits());
                    let q = self.s.m.select(c, full, one, q);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// Channels of buffer format `format`, as [`Self::format_load`] gives
    /// them, as d16 data: 16 bits each, two per dword, low half first (an
    /// odd last high half 0). Integer formats keep their low 16 bits; the
    /// others become f16.
    pub(super) fn d16_channels(&mut self, format: u8, channels: Vec<V>) -> Result<Vec<V>, Why> {
        let (_, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let mut halves = Vec::new();
        for x in channels {
            halves.push(if matches!(numeric, Numeric::Uint | Numeric::Sint) {
                let m = self.s.m.i32(0xFFFF);
                self.s.m.bin(c, Bin::And, x, m)
            } else {
                let (f, i32t) = (self.as_float(x), self.s.t.i32);
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            });
        }
        let mut out = Vec::new();
        for pair in halves.chunks(2) {
            out.push(match *pair {
                [lo, hi] => {
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                }
                [lo] => lo,
                _ => unreachable!("chunks of one or two"),
            });
        }
        Ok(out)
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    pub(super) fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let mut x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                if let Some(bias) = self.root_bias(address.param) {
                    x = Some(match x {
                        Some(d) => self.s.m.bin(self.cur, Bin::Add, d, bias),
                        None => bias,
                    });
                }
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// For a root view the host binds as a raw view that may begin before
    /// the title's address (`shader.rs` `USER_VIEW_BIAS`): the bytes to add
    /// to offsets through it. Every root UAV, and the root SRVs of programs
    /// with a spilled user-data table (the others keep the views they
    /// have).
    pub(super) fn root_bias(&mut self, param: usize) -> Option<V> {
        use super::super::shader::USER_VIEW_BIAS;
        let rp = *self.x.native.layout.get(param)?;
        if rp.root != RootType::Descriptor {
            return None;
        }
        let kind = self
            .x
            .native
            .used
            .iter()
            .find(|u| u.user_data == rp.user_data)
            .map(|u| u.kind);
        let biased = match kind {
            Some(ResourceType::Uav) => true,
            Some(ResourceType::Srv) => crate::spill::is_spilled(self.x.native),
            _ => false,
        };
        biased.then(|| self.host_word(RootType::Other(USER_VIEW_BIAS), rp.slot, 0))
    }

    /// The byte offset of element `index` at `off` in a buffer whose
    /// descriptor the code builds around a root descriptor; none for other
    /// descriptors. GFX10 addresses base + offset + stride * index with
    /// swizzling off, the stride in dword 1 bits 16-29 (the constant bits the
    /// code sets there), swizzling in dword 1 bit 31 and the lane id added to
    /// the index in dword 3 bit 23.
    pub(super) fn built_element(&mut self, src: &Source, index: V, off: V) -> Result<Option<V>, Why> {
        let Source::Built { bits, words, .. } = src else {
            return Ok(None);
        };
        if bits >> 31 != 0 {
            return Err("a swizzled buffer");
        }
        if words[1] >> 23 & 1 != 0 {
            return Err("a buffer indexed by lane");
        }
        let stride = bits >> 16 & 0x3FFF;
        let c = self.cur;
        let s = self.s.m.i32(stride);
        let rows = self.s.m.bin(c, Bin::Mul, index, s);
        Ok(Some(self.s.m.bin(c, Bin::Add, rows, off)))
    }

    /// Whether a buffer is accessed by its address (see
    /// [`crate::resource::track_with`]): no root view leads to it, or one
    /// that is written through.
    pub(super) fn by_address(&self, src: &Source) -> bool {
        match src {
            Source::Address => true,
            Source::Built { address, .. } => {
                self.x.res.addresses
                    && address.param != CODE
                    && self
                        .x
                        .native
                        .layout
                        .get(address.param)
                        .is_some_and(|rp| rp.root == RootType::Descriptor)
                    && !self
                        .used_kind(src)
                        .is_some_and(|u| u.kind == ResourceType::Cbv)
            }
            _ => false,
        }
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    pub(super) fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. }
        | Source::Select { args, .. }
        | Source::Lanes { args, .. } = src
        {
            return args.iter().find_map(|a| self.used_kind(a));
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// GFX10 buffer dword 2 (NUM_RECORDS) for an exact, statically declared
    /// structured SRV. The host may represent it with a clamped raw view, so
    /// the count comes from the title's descriptor at bind time.
    pub(super) fn structured_srv_records(&mut self, site: Site, value: Value) -> Option<V> {
        let address = *self.scalar.get(&site)?;
        let rp = *self.x.native.layout.get(address.param)?;
        let dword = address.offset / 4;
        let used = self.used_kind(&Source::Loaded { load: site, first: 0 })?;
        if address.dynamic.is_some() || used.table_offset_dwords != dword
            || used.kind != ResourceType::Srv || used.dwords != 4
            || !(4..=2048).contains(&used.stride) || !used.stride.is_multiple_of(4)
        {
            return None;
        }
        // An address-only shader does not need another root constant.
        self.descriptor_word_used(value).then(|| {
            self.host_word(RootType::Other(super::super::shader::USER_VIEW_RECORDS), rp.slot, dword)
        })
    }

    /// Whether anything reads `value`, a descriptor word a load wrote.
    pub(super) fn descriptor_word_used(&mut self, value: Value) -> bool {
        let uses = self.descriptor_word_uses.get_or_init(|| {
            self.x.p.blocks.iter().flat_map(|block| {
                block.insts.iter().flat_map(|i| i.reads.iter().map(|r| r.1))
                    .chain(block.phis.iter().flat_map(|p| p.args.iter().copied()))
            }).collect()
        });
        uses.contains(&value)
    }

    /// The table descriptor `src` is, as (root parameter, dword in the
    /// table), when the code loads it from a fixed place in a table: the
    /// load's own place, not the start of the range its used-resource
    /// entry describes (a range of several descriptors starts before most
    /// of them). `None` for a descriptor picked at run time, built in the
    /// code or chosen by control flow.
    pub(super) fn table_place(&self, src: &Source) -> Option<(u8, u32)> {
        let Source::Loaded { load, first } = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        (rp.root == RootType::Table && a.dynamic.is_none())
            .then(|| (rp.slot, a.offset / 4 + u32::from(*first)))
    }

    /// Trades the handle cache of accesses for that of buffer size queries,
    /// and back when called again.
    pub(super) fn swap_size_caches(&mut self) {
        std::mem::swap(&mut self.handles, &mut self.size_handles);
    }

    /// A handle for the descriptor `src` read at `site`.
    pub(super) fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        let ordered = super::super::pops::active(self.x.native) && props.0 & 0x1000 != 0;
        let props = (props.0 | (u32::from(ordered) << 13), props.1);
        if ordered {
            self.s.flags |= flags::ROVS;
        }
        match src {
            Source::Address => Err("a buffer descriptor made in code"),
            Source::Loaded { load, first } => {
                // The element type is part of the key: one UAV read as float
                // and used atomically needs two handles (an atomic on a float
                // handle is invalid and crashed the NVIDIA compiler).
                let key = (self.cur, *load, *first, sampler, props);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                let per_lane = self.x.res.waterfall.loads.contains(load);
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                let n = match self.samplers.iter().position(|w| *w == words) {
                    Some(n) => n,
                    None => {
                        self.samplers.push(words);
                        self.samplers.len() - 1
                    }
                };
                if props.0 & 0x8000 != 0 {
                    // shortcut: one sampler used with and without comparison
                    // would need two static samplers; it becomes a comparison one.
                    self.s.cmp_samplers |= 1 << n;
                }
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        let h = self.s.root_srv_handle(rp.slot);
                        // A draw with wild vertex numbers must not read past
                        // the memory behind the view (a page fault).
                        if super::super::limits_indices()
                            && (self.s.threads.is_none() || self.s.mesh.is_some())
                            && !self.s.limits.iter().any(|l| l.0 == h)
                        {
                            let nth = 1 + self.s.limits.len() as u32;
                            let limit = self.host_data(nth, RootType::LIMIT, rp.slot.into());
                            self.s.limits.push((h, limit));
                        }
                        Ok(h)
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav) | None, 0x100B)
                    | (RootType::Descriptor, Some(ResourceType::Uav), 11) => {
                        // Other groups write what a builder reads.
                        let shared = self.ordered();
                        Ok(self.s.root_uav_handle(rp.slot, shared))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 0x100b) => {
                        Err("a write through a root SRV")
                    }
                    _ => Err("a typed root view"),
                }
            }
            // shortcut: of samplers given in the code that a branch picks one
            // of (how they treat coordinates off the texture differs), the
            // first is taken.
            Source::Phi { args, .. }
                if sampler
                    && let Some(arg) = args.iter().find(|a| matches!(a, Source::Constant(_))) =>
            {
                self.handle(site, arg, sampler, props)
            }
            Source::Phi { block, args } => {
                let _ = site;
                let phi = self.descriptor_phi(*block, args, sampler, 8)?;
                Ok(self.s.heap_handle(self.cur, phi, sampler, false, props))
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                let index = self.s.m.select(self.cur, cond, set, clear);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
            Source::Carried { .. } => Err("a per-lane descriptor carried without its join"),
        }
    }

    /// The heap index a descriptor phi picks. Each predecessor computes its
    /// heap index at its end and the join picks one with a phi; a
    /// predecessor whose descriptor is itself chosen by control flow (a
    /// phi of an earlier join, which dominates it) gives that join's phi,
    /// at most `depth` joins deep.
    pub(super) fn descriptor_phi(
        &mut self,
        block: usize,
        args: &[Source],
        sampler: bool,
        depth: u32,
    ) -> Result<V, Why> {
        const WHY: Why = "a descriptor chosen by control flow";
        let p = self.x.p;
        let mut incoming = Vec::new();
        for (&q, arg) in p.blocks[block].preds.iter().zip(args) {
            if self.block[q].is_none() {
                continue;
            }
            let tail = self.tail[q].ok_or(WHY)?;
            let at = self.edges.get(&(q, block)).copied().unwrap_or(tail);
            let index = match arg {
                Source::Loaded { load, first } => {
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = self.heap_index(*load, *first, sampler);
                    self.cur = here;
                    index?
                }
                Source::Phi { block: inner, args: inner_args } if depth > 0 => {
                    self.descriptor_phi(*inner, inner_args, sampler, depth - 1)?
                }
                _ => return Err(WHY),
            };
            incoming.push((index, at));
        }
        let join = self.block[block].ok_or(WHY)?;
        let phi = self.s.m.phi(join, self.s.t.i32);
        self.s.m.set_phi(phi, incoming);
        Ok(phi)
    }

    /// The bytes per element of the buffer view `src` names, if the program
    /// picks it from the view heap at an index it computes (root type
    /// `ResourceHeap`): read from the host's table by heap slot in the
    /// current block.
    pub(super) fn slot_stride(&mut self, src: &Source) -> Result<Option<V>, Why> {
        let Source::Loaded { load, first } = src else {
            return Ok(None);
        };
        let heap = self
            .scalar
            .get(load)
            .and_then(|a| self.x.native.layout.get(a.param))
            .is_some_and(|p| p.root == RootType::ResourceHeap);
        if !heap {
            return Ok(None);
        }
        let key = (self.cur, *load, *first);
        if let Some(&v) = self.strides.get(&key) {
            return Ok(Some(v));
        }
        let c = self.cur;
        let slot = self.heap_index(*load, *first, false)?;
        // Inside the table whatever the index (lanes that do not run
        // compute one too): a root view has no bounds of its own.
        let last = self.s.m.i32(super::super::shader::SLOT_STRIDES_COUNT - 1);
        let i32t = self.s.t.i32;
        let slot = self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, last]);
        self.s.m.tag(slot, Tag::Clamp);
        let two = self.s.m.i32(2);
        let byte = self.s.m.bin(c, Bin::Shl, slot, two);
        let h = self.s.root_srv_handle(super::super::shader::SLOT_STRIDES);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, byte, iu, mask, align],
        );
        let f = self.s.m.extract(c, r, 0);
        let v = self.as_int(f);
        self.strides.insert(key, v);
        Ok(Some(v))
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block.
    pub(super) fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0. A carried edge (a
                // loop's back edge) keeps the join's own value. A predecessor
                // lowered later (the join's own block reaches it, as a loop's
                // back edge does) fills its edge in `fill_paths`.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.lane_paths.push((*block, phi));
                let here = self.cur;
                let mut incoming = Vec::new();
                let mut result = Ok(());
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let Some(tail) = self.tail[q] else {
                        incoming.push((q, Err(arg.clone())));
                        continue;
                    };
                    self.cur = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => self.lane_heap_index(arg, sampler),
                        None => Ok(self.s.m.i32(0)),
                    };
                    match index {
                        Ok(x) => incoming.push((q, Ok(x))),
                        Err(why) => {
                            result = Err(why);
                            break;
                        }
                    }
                }
                self.cur = here;
                let outer = self.lane_paths.clone();
                self.lane_paths.pop();
                result?;
                if incoming.iter().all(|i| i.1.is_ok()) {
                    let done = incoming
                        .into_iter()
                        .map(|(q, x)| (x.unwrap(), self.edge_block(q, *block)));
                    self.s.m.set_phi(phi, done.collect());
                } else {
                    self.pending_paths.push(PendingPath {
                        phi,
                        block: *block,
                        sampler,
                        outer,
                        incoming,
                    });
                }
                Ok(phi)
            }
            Source::Carried { block } => self
                .lane_paths
                .iter()
                .rev()
                .find(|p| p.0 == *block)
                .map(|p| p.1)
                .ok_or("a per-lane descriptor carried out of its loop"),
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The DXIL block that ends the edge from IR block `q` (lowered) to `to`.
    pub(super) fn edge_block(&self, q: usize, to: usize) -> usize {
        let tail = self.tail[q].expect("a lowered predecessor");
        self.edges.get(&(q, to)).copied().unwrap_or(tail)
    }

    /// Fills the heap index phis [`Source::LanePhi`] joins left for later.
    pub(super) fn fill_paths(&mut self) -> Result<(), Why> {
        while let Some(pending) = self.pending_paths.pop() {
            let saved = std::mem::replace(&mut self.lane_paths, pending.outer);
            let mut incoming = Vec::new();
            for (q, x) in pending.incoming {
                let at = self.edge_block(q, pending.block);
                let x = match x {
                    Ok(x) => x,
                    Err(arg) => {
                        self.cur = at;
                        match arg {
                            Some(arg) => self.lane_heap_index(&arg, pending.sampler)?,
                            None => self.s.m.i32(0),
                        }
                    }
                };
                incoming.push((x, at));
            }
            self.lane_paths = saved;
            self.s.m.set_phi(pending.phi, incoming);
        }
        Ok(())
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    pub(super) fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        if self.x.st.guards[load.0][load.1].is_some() {
            // Lanes outside EXEC there keep their old dwords.
            return Err("a per-lane descriptor loaded under a partial EXEC");
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        if matches!(li.fields, Fields::Buffer { .. }) {
            // A buffer over the heap or a table: its address (the table's
            // byte offset, or 0 for a heap) plus each lane's offset.
            let c = self.cur;
            let mut at = match self.user(a.param, 0) {
                Val::I32(x) => x,
                _ => return Err(WHY),
            };
            let k = self.s.m.i32(a.offset.wrapping_add(4 * u32::from(first)));
            at = self.s.m.bin(c, Bin::Add, at, k);
            if let Some(d) = self.dynamic_term(&a)? {
                at = self.s.m.bin(c, Bin::Add, at, d);
            }
            let shift = self.s.m.i32(if sampler { 4 } else { 5 });
            return Ok(self.s.m.bin(c, Bin::LShr, at, shift));
        }
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The view-heap slot of `src` when it comes from a descriptor table or
    /// heap (`Loaded` or `Phi` of `Loaded`).
    pub(super) fn heap_slot(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let Some(tail) = self.tail[q] else {
                        return Err("a descriptor chosen by control flow");
                    };
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let index = match arg {
                        Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
                        Source::Phi { .. } | Source::Select { .. } => self.heap_slot(arg, sampler),
                        _ => {
                            if std::env::var_os("VICTORY_WHY").is_some() {
                                eprintln!("phi of {arg:?}");
                            }
                            Err("a descriptor chosen by control flow")
                        }
                    };
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            // A descriptor chosen per lane or by SCC: its index in each lane.
            _ => self.lane_heap_index(src, sampler),
        }
    }

    /// Byte offset `index * stride + off` for a structured buffer whose
    /// stride was not recovered statically, reading `stride` from the host's
    /// runtime stride table (`t0`, [`super::super::STRIDE_SPACE`]).
    pub(super) fn runtime_stride_offset(&mut self, src: &Source, index: V, off: V) -> Result<V, Why> {
        let slot = self
            .heap_slot(src, false)
            .map_err(|_| "a structured buffer of unknown stride")?;
        let c = self.cur;
        let stride = self.slot_dword(slot, 0);
        let elem_off = self.s.m.bin(c, Bin::Mul, index, stride);
        Ok(self.s.m.bin(c, Bin::Add, elem_off, off))
    }

    /// What the host's table by heap slot (`t0`, [`super::super::STRIDE_SPACE`])
    /// holds for `slot`, `from` dwords on: the element stride of its buffer
    /// view from 0, the element count from [`super::super::OWN_SLOTS`].
    pub(super) fn slot_dword(&mut self, slot: V, from: u32) -> V {
        let c = self.cur;
        let table = self.s.stride_table_handle();
        // A root view has no bounds: a wild index would read past the table.
        let slot = if super::super::limits_indices() || from != 0 {
            let (i32t, last) = (self.s.t.i32, self.s.m.i32(super::super::OWN_SLOTS - 1));
            self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[slot, last])
        } else {
            slot
        };
        let slot = if from != 0 {
            let from = self.s.m.i32(from);
            self.s.m.bin(c, Bin::Add, slot, from)
        } else {
            slot
        };
        let four = self.s.m.i32(4);
        let addr = self.s.m.bin(c, Bin::Mul, slot, four);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        let res = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, addr, iu, mask, align],
        );
        let x = self.s.m.extract(c, res, 0);
        self.as_int(x)
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    pub(super) fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        self.heap_index_at(load, i32::from(first), sampler)
    }

    /// The heap index of the descriptor `dwords` dwords from the first one
    /// the scalar load at `load` reads (before it if negative), computed
    /// in the current block.
    pub(super) fn heap_index_at(&mut self, load: Site, dwords: i32, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4u32.wrapping_mul(dwords as u32)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }
}
