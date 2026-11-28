import re
src = open('crates/victory-shader/src/emit/wave.rs', encoding='utf-8').read()
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
hunks = list(pat.finditer(src))
assert len(hunks) == 7, len(hunks)

# hunk0: ours' lanes field + lane_shape, plus theirs' reads field
m0 = '''    /// What each lane holds of vector values.
    lanes: HashMap<Value, Vec<Sym>>,
    /// What a `v_readlane_b32` of one lane takes, later in the block, from a
    /// VGPR the section leaves (by the scalar it writes): the last lanes of
    /// a reduction hold it before the first do.
    pub reads: HashMap<Value, Shape>,
}

impl Section {
    /// What lane `lane` of `v` holds, as a value any lane can make.
    pub fn lane_shape(&self, v: Value, lane: u32) -> Option<Shape> {
        let lanes = self.lanes.get(&v)?;
        let all = u64::MAX >> (64 - lanes.len().min(64));
        match *lanes.get(lane as usize)? {
            Sym::K(k) => Some(Shape::K(k)),
            Sym::Uni(x) => Some(Shape::Value(x)),
            Sym::Red {
                op: Some(op),
                src,
                mask,
                ident,
                lanes,
            } if lanes == all => Some(Shape::Active(op, src, mask, ident)),
            Sym::Red64F {
                lo,
                hi,
                mask,
                lanes,
                high,
            } if lanes == all => Some(Shape::ActiveF64(lo, hi, mask, high)),
            _ => None,
        }
    }

'''

# hunk1: theirs' reads computation + ours' lanes field in the literal
m1 = '''        let (vec, uni) = simulate(&insts[i + 1..end], wave, saved);
        let mut reads = HashMap::new();
        for later in insts[end + 1..].iter().filter(|i| i.name() == "v_readlane_b32") {
            if let Fields::Valu(v) = &later.fields
                && let (Some(Operand::Scalar { code, .. }), [Operand::Vector { reg, .. }, lane]) =
                    (v.dst, &v.src[..])
                && let Some(left) = later.read(Reg::V(*reg)).and_then(|x| vec.get(&x))
                && let Some(w) = later.write(Reg::S(code))
                && let Some(l) = lane_number(lane)
                && let Some(&sym) = left.get(l as usize)
                // (a lane's own value stays a lane read)
                && !matches!(sym, Sym::Own(_))
                && let Some(shape) = single(sym, wave)
            {
                reads.insert(w, shape);
            }
        }
        let lanes = vec.clone();
        let shapes = shapes(vec, uni, wave);
        out.insert(i, Section { end, shapes, lanes, reads });
'''

# hunks 2..6: theirs'
m2 = hunks[2].group(2)
m3 = hunks[3].group(2)
m4 = hunks[4].group(2)
m5 = hunks[5].group(2)
m6 = hunks[6].group(2)

out = (src[:hunks[0].start()] + m0 + src[hunks[0].end():hunks[1].start()] + m1
       + src[hunks[1].end():hunks[2].start()] + m2 + src[hunks[2].end():hunks[3].start()]
       + m3 + src[hunks[3].end():hunks[4].start()] + m4 + src[hunks[4].end():hunks[5].start()]
       + m5 + src[hunks[5].end():hunks[6].start()] + m6 + src[hunks[6].end():])
open('crates/victory-shader/src/emit/wave.rs', 'w', encoding='utf-8').write(out)
print('done', '<<<<<<<' in out)
