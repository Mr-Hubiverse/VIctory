//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, Bin, Cast, Rmw, Tag, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, flags};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Export, Input, Interp, Io, Target};
use crate::ir::{Block, EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, MEMORY, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};

/// The ray-tracing node test (`image_bvh_intersect_ray`), a child module so
/// it shares the emitter's state.
#[path = "intersect.rs"]
pub(super) mod intersect;

#[path = "half_sum.rs"]
mod half_sum;
#[path = "pair.rs"]
mod pair;
#[path = "packed_mask.rs"]
mod packed_mask;
#[path = "guest_address.rs"]
mod guest_address;
#[path = "typed_views.rs"]
mod typed_views;
/// Format loads that move 16-bit halves without the d16 form, a child
/// module so it shares the emitter's state.
#[path = "halves.rs"]
mod halves;

/// Threads past a `DispatchX` thread count (`USER_DISPATCH_THREADS`), a
/// child module so it shares the emitter's state (fork pull request #646).
#[path = "dispatch_threads.rs"]
mod dispatch_threads;

/// Loads and stores through descriptors, group-shared memory, and the
/// descriptor heap tables, a child module so it shares the emitter's state.
mod descriptors;
/// Vector ALU, comparison, and interpolation instruction emission, a child
/// module so it shares the emitter's state.
mod valu;
/// Image, sampler, and buffer accesses, a child module so it shares the
/// emitter's state.
mod image;
/// Exports, block exits, branches, and the joins between them, a child
/// module so it shares the emitter's state.
mod control;

type Why = &'static str;

/// How an interpolation evaluates an input channel.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Loaded with this DXIL interpolation mode (1: constant, P0).
    Mode(u8),
    /// From the vertex values of a constant element: P0 when `p0`, plus
    /// i (P1 - P0), plus j (P2 - P0), weights as floats.
    Weights {
        p0: bool,
        i: Option<V>,
        j: Option<V>,
    },
}

/// Updates which have no native DXIL atomic instruction.
#[derive(Clone, Copy)]
enum AtomicUpdate {
    Inc,
    Dec,
    ReverseSub,
    MaskOr,
    Wrap,
    ClampSub,
}

/// An unformatted byte or halfword load. LLVM's DSInstructions.td and
/// BUFInstructions.td model the D16 destinations as tied to their old value.
fn subword_load(rest: &str) -> Option<(u32, bool, Option<u32>)> {
    let (rest, half) = if let Some(n) = rest.strip_suffix("_d16_hi") {
        (n, Some(16))
    } else if let Some(n) = rest.strip_suffix("_d16") {
        (n, Some(0))
    } else {
        (rest, None)
    };
    let (width, signed) = match rest {
        "i8" | "sbyte" => (8, true),
        "u8" | "ubyte" => (8, false),
        "i16" | "sshort" => (16, true),
        "u16" | "ushort" | "short" => (16, false),
        _ => return None,
    };
    Some((width, signed, half))
}

fn ds_integer_atomic(name: &str) -> Option<(&str, bool, u8)> {
    let (op, ty) = name.strip_prefix("ds_")?.rsplit_once('_')?;
    let op = op.strip_suffix("_rtn").unwrap_or(op);
    let wide = match ty {
        "b32" | "u32" | "i32" => false,
        "b64" | "u64" | "i64" => true,
        _ => return None,
    };
    match op {
        "add" | "sub" | "and" | "or" | "xor" | "min" | "max" | "wrxchg" | "wrxchg2"
        | "wrxchg2st64" | "rsub" | "inc" | "dec" | "mskor" | "wrap" | "cmpst" => {
            Some((op, wide, ty.as_bytes()[0]))
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    pub written_root_uavs: Vec<u8>,
    /// A translated store or atomic can write through the guest page directory.
    pub writes_guest_memory: bool,
    /// See [`super::HostShader::half_views`].
    pub half_views: Vec<(u8, u32)>,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// The wave size the shader requires of the host, if any.
    pub wave_size: Option<u32>,
    /// The runtime stride table is used (see [`super::HostShader::stride_table`]).
    pub stride_table: bool,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Test builds: the property words of each handle annotation, in order.
    #[cfg(test)]
    pub annotations: Vec<(u32, u32)>,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    #[allow(dead_code)]
    pub psv: Option<&'a [u8]>,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    #[allow(dead_code)]
    /// The wave sizes the host runs.
    pub target: super::Target,
    /// Run the program as the console does (see [`CONSOLE_FLOW`]); `st`
    /// is then [`crate::structure::structure_console`]'s.
    pub console_cf: bool,
    /// Never ask for the console flow: the per-lane form or nothing (what
    /// a program falls back to when the console flow cannot lower it).
    pub per_lane_only: bool,
}

#[cfg(test)]
thread_local! {
    /// The opcode audit: the instruction (by code offset) whose refusal is
    /// reported even when none of its results are used.
    pub(crate) static AUDIT_OFFSET: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// What [`lower`] stops with when a program should run as the console runs
/// it: lower it again from [`crate::structure::structure_console`] with
/// [`Parts::console_cf`]. Every lane of a console wave then runs every
/// block, branches are the console's (taken wave-wide) and EXEC guards the
/// vector instructions. Two kinds of program ask for it:
///
/// * a wave64 program whose two 32-lane halves must exchange values on a
///   host without 64-lane waves (a group of one console wave): the wave-wide
///   tests come from masks both halves exchange through group-shared
///   memory, and both halves reach every exchange, so a group barrier
///   orders it;
/// * on any host, a program that reads or writes a named lane
///   (`v_readlane`, `v_writelane`) and has branches its lanes take
///   differently. Those instructions ignore EXEC on the console, and a wave
///   runs a loop while any lane does; in the per-lane form a lane that left
///   a loop (or skipped a branch) is not there to be read, and never sees
///   the scalars set inside. One host wave is the console wave here, so
///   nothing is exchanged. If the console flow cannot lower such a program
///   it keeps the per-lane form ([`Parts::per_lane_only`]).
pub(crate) const CONSOLE_FLOW: &str = "a wave64 program whose halves exchange lanes (console flow)";

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

const UNTRACKED_SELECT: Why = "a select on an untracked mask";

#[cfg(test)]
pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    lower_with_thread_limit(parts, false)
}

#[cfg(test)]
pub(super) fn lower_with_thread_limit(parts: &Parts, partial: bool) -> Result<Lowered, ShaderError> {
    lower_program(parts, partial, &[])
}

pub(super) fn lower_program(
    parts: &Parts, partial: bool, uint_srvs: &'static [(u8, u32, u32)],
) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    let mut numeric_selects = false;
    loop {
        match Lower::new(parts, &bad, uint_srvs, numeric_selects).run(partial) {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(ShaderError::Unsupported { what, .. }))
                if what == UNTRACKED_SELECT
                    && !numeric_selects
                    && parts.native.stage == HwStage::Cs =>
            {
                numeric_selects = true;
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

/// The f16 bits of inline float constant `c` (240..=248: 0.5, -0.5, 1.0,
/// -1.0, 2.0, -2.0, 4.0, -4.0, 1/(2 pi)).
fn inline_f16(c: u8) -> u32 {
    [
        0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
    ][usize::from(c - 240)]
}

/// Private memory per lane when the code does not show how far its
/// addresses reach (an index loaded or carried around a loop).
// shortcut: the console's size is set at pipeline creation (the program
// metadata read so far does not hold it); accesses past this read 0 and
// drop writes, as accesses past the console's size do.
pub(super) const SCRATCH_CAP: u32 = 4 << 10;
/// The most private memory per lane, the D3D11 limit of indexable
/// temporaries (4096 four-dword registers).
// shortcut: accesses past it read 0 and drop writes.
const SCRATCH_MAX: u32 = 64 << 10;

/// Bytes a `scratch_*` opcode moves (per lane), from the name after
/// `scratch_load_` or `scratch_store_`.
fn scratch_width(rest: &str) -> Option<u32> {
    Some(match rest.split('_').next()? {
        "dword" => 4,
        "dwordx2" => 8,
        "dwordx3" => 12,
        "dwordx4" => 16,
        "byte" | "ubyte" | "sbyte" => 1,
        "short" | "ushort" | "sshort" => 2,
        _ => return None,
    })
}

/// Private memory per lane the program needs: past the highest byte its
/// scratch accesses can reach where the code computing their addresses
/// bounds them, else [`SCRATCH_CAP`]; at most [`SCRATCH_MAX`].
fn scratch_bytes(p: &Program) -> u32 {
    let mut defs = HashMap::new();
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        for &(_, v) in &i.writes {
            defs.insert(v, i);
        }
    }
    let mut memo = HashMap::new();
    let mut end = 0u64;
    for i in p.blocks.iter().flat_map(|b| &b.insts) {
        let name = i.name();
        let Fields::Flat {
            addr,
            saddr,
            offset,
            ..
        } = &i.fields
        else {
            continue;
        };
        let rest = name
            .strip_prefix("scratch_load_")
            .or_else(|| name.strip_prefix("scratch_store_"));
        let Some(width) = rest.and_then(scratch_width) else {
            continue;
        };
        let base = match (addr, saddr) {
            (Some(o), _) | (None, Some(o)) => operand_bound(&defs, &mut memo, i, o, 0),
            (None, None) => Some(0),
        };
        // A negative offset only lowers the address (or wraps it out of
        // range, checked when the program runs).
        let off = u64::from((*offset).max(0).unsigned_abs());
        end = end.max(base.map_or(u64::from(SCRATCH_CAP), |b| b + off + u64::from(width)));
    }
    end.min(u64::from(SCRATCH_MAX)) as u32
}

/// An upper bound of operand `o` of `inst` as an unsigned number.
fn operand_bound(
    defs: &HashMap<Value, &Inst>,
    memo: &mut HashMap<Value, Option<u64>>,
    inst: &Inst,
    o: &Operand,
    depth: u8,
) -> Option<u64> {
    match *o {
        Operand::Scalar { code: NULL, .. } => Some(0),
        Operand::Scalar { code, .. } => value_bound(defs, memo, inst.read(Reg::S(code))?, depth),
        Operand::Vector { reg, .. } => value_bound(defs, memo, inst.read(Reg::V(reg))?, depth),
        Operand::Const(c) => crate::resource::inline(c).map(u64::from),
        Operand::Literal(x) => Some(u64::from(x)),
        Operand::Attr { .. } => None,
    }
}

/// An upper bound of 32-bit value `v` as an unsigned number, where the
/// instructions computing it show one (constants combined by masks, shifts,
/// sums, products, selects); `None` where it depends on anything else
/// (inputs, memory, a phi) or may wrap.
fn value_bound(
    defs: &HashMap<Value, &Inst>,
    memo: &mut HashMap<Value, Option<u64>>,
    v: Value,
    depth: u8,
) -> Option<u64> {
    if let Some(&b) = memo.get(&v) {
        return b;
    }
    let inst = *defs.get(&v)?;
    if depth > 32 {
        return None;
    }
    let (dst, src) = match &inst.fields {
        Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            imm: None,
        } => (Reg::S(*code), src),
        Fields::Valu(x)
            if x.sdwa.is_none()
                && x.dpp.is_none()
                && x.neg | x.abs | x.opsel == 0
                && !x.clamp
                && x.omod == 0 =>
        {
            match x.dst {
                Some(Operand::Vector { reg, .. }) => (Reg::V(reg), &x.src),
                _ => return None,
            }
        }
        _ => return None,
    };
    if inst.write(dst) != Some(v) {
        return None;
    }
    let mut arg =
        |k: usize| -> Option<u64> { operand_bound(defs, memo, inst, src.get(k)?, depth + 1) };
    // The smallest all-ones number at least `x` (bounds `|` and `^`).
    let ones = |x: u64| {
        if x == 0 {
            0
        } else {
            u64::MAX >> x.leading_zeros()
        }
    };
    let u24 = |x: Option<u64>| x.map(|x| x.min(0xFF_FFFF));
    let shl = |x: Option<u64>, n: Option<u64>| -> Option<u64> {
        x?.checked_shl(u32::try_from(n.filter(|&n| n < 32)?).ok()?)
    };
    let b = match inst.name() {
        "v_mov_b32" | "s_mov_b32" => arg(0),
        "v_and_b32" | "s_and_b32" => match (arg(0), arg(1)) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        },
        "v_or_b32" | "s_or_b32" | "v_xor_b32" | "s_xor_b32" => Some(ones(arg(0)?.max(arg(1)?))),
        "v_lshlrev_b32" => {
            let n = arg(0);
            shl(arg(1), n)
        }
        "s_lshl_b32" => {
            let x = arg(0);
            shl(x, arg(1))
        }
        "v_lshrrev_b32" => arg(1),
        "s_lshr_b32" => arg(0),
        "v_add_nc_u32" | "v_add_u32" | "v_add_co_u32" | "s_add_u32" | "s_add_i32" => {
            Some(arg(0)? + arg(1)?)
        }
        "v_add3_u32" => Some(arg(0)? + arg(1)? + arg(2)?),
        "v_mul_u32_u24" => Some(u24(arg(0))? * u24(arg(1))?),
        "v_mul_lo_u32" | "s_mul_i32" => arg(0)?.checked_mul(arg(1)?),
        "v_mad_u32_u24" => Some(u24(arg(0))? * u24(arg(1))? + arg(2)?),
        "v_lshl_add_u32" | "v_lshl_or_b32" => {
            let x = arg(0);
            let n = arg(1);
            Some(shl(x, n)? + arg(2)?)
        }
        "v_add_lshl_u32" => {
            let x = Some(arg(0)? + arg(1)?);
            shl(x, arg(2))
        }
        "v_and_or_b32" => {
            let x = match (arg(0), arg(1)) {
                (Some(x), Some(y)) => x.min(y),
                (x, y) => x.or(y)?,
            };
            Some(x + arg(2)?)
        }
        "v_bfe_u32" => match arg(2) {
            Some(w) if w < 32 => Some((1 << w) - 1),
            _ => arg(0),
        },
        "v_cndmask_b32" | "v_max_u32" | "s_max_u32" | "s_cselect_b32" => Some(arg(0)?.max(arg(1)?)),
        "v_min_u32" | "s_min_u32" => match (arg(0), arg(1)) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        },
        _ => None,
    }
    .filter(|&b| b <= u64::from(u32::MAX));
    memo.insert(v, b);
    b
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

#[allow(dead_code)]
struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    /// SGPR spills to VGPR lanes and their reloads that nothing reads any
    /// more (see [`crate::sgpr_spill`]): not lowered.
    dead: HashSet<Site>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    /// The DXIL block the program's body begins in: `s.entry`, or after
    /// the threads past a `DispatchX` thread count have left
    /// ([`Self::dispatch_threads_prologue`]). EXEC on entry is read there.
    body_entry: usize,
    /// Whether this thread is within a `DispatchX` thread count, where its
    /// threads past the count stay for the group's barriers or the halves'
    /// exchange: memory accesses, ballots and wave reductions leave the
    /// others out ([`Self::dispatch_threads_prologue`]).
    partial_active: Option<V>,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    /// Descriptor dwords with no host value (addresses, formats, all of a constant
    /// buffer's): no program reads them as numbers ([`Self::dead`]).
    stale: HashSet<Value>,
    /// Only computed when an unused-as-a-view structured SRV is read as data.
    descriptor_word_uses: std::cell::OnceCell<HashSet<Value>>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    /// Numeric joins a packed mask word already keeps live.
    packed_phis: HashSet<Value>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    /// Table descriptors read or written sixteen bits a channel
    /// ([`super::HostShader::half_views`]): root parameter, dword offset.
    half_views: std::collections::BTreeSet<(u8, u32)>,
    /// [`Self::places_read_as_halves`], found before the code is lowered.
    halves_only: std::collections::BTreeSet<(u8, u32)>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    /// (root slot, table offset in dwords, DXIL resource kind).
    uint_srvs: &'static [(u8, u32, u32)],
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<(usize, Site, u8, bool, (u32, u32)), V>,
    /// [`Lower::handles`] of buffer size queries, which share no handle
    /// with an access ([`Lower::swap_size_caches`]).
    size_handles: HashMap<(usize, Site, u8, bool, (u32, u32)), V>,
    /// Element strides read from the host's table by heap slot
    /// ([`super::shader::SLOT_STRIDES`]), by block and descriptor load.
    strides: HashMap<(usize, Site, u8), V>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// Threads of different groups take turns through atomics (a tree built
    /// from its leaves up): what they wrote before one must be seen after
    /// it. The console's program waits for its writes there.
    ordered: Option<bool>,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    #[allow(dead_code)]
    /// Retry a compute shader with numeric selects only when the normal
    /// translation needs one. Unused selects need no fixed wave size.
    numeric_selects: bool,
    /// A console wave64 compute program run as two 32-lane host waves
    /// (each lane's reads stay in its half; see `Lower::new`).
    split: bool,
    /// Split, and run as the console runs it ([`CONSOLE_FLOW`]).
    console_cf: bool,
    #[allow(dead_code)]
    /// Split without [`Self::console_cf`] cannot run it, the console flow can.
    wants_console_flow: bool,
    /// The console thread's flat id in the group (relabelled; see
    /// [`Self::thread_id`]).
    console_flat: Option<V>,
    /// Group-shared slots the two halves exchange a dword through.
    xchg: Option<V>,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// IR blocks whose `s_barrier` not every lane is sure to reach (a lane
    /// branch lies between the entry and the block, or the program has a
    /// loop a lane can be cut out of): the barrier there keeps its fences
    /// and does not wait for the group ([`super::group_sync`]).
    fence_only: HashSet<usize>,
    /// A group barrier before every LDS access (a console wave run as two host waves).
    split_lds: bool,
    /// LDS holds i64 elements (a ds_* atomic needs them): every access is
    /// through the one i64 array DXIL allows ([`Self::lds_ptr`]).
    wide_lds: bool,
    /// With `split_lds`, LDS accesses in lane branches: per IR block whether every lane runs
    /// it, and whether the host waves meet at its start and before its exit ([`lds_meetings`]).
    lds_meet: Option<(Vec<bool>, Vec<(bool, bool)>)>,
    /// A console wave64 program run as two host waves of 32 that exchange through LDS: the first
    /// of two exchange dwords, whether the group has no upper half (32 threads or fewer), and per
    /// IR block whether every lane runs it (exchanges need both halves there).
    exchange: Option<(u32, bool, Vec<bool>)>,
    /// With `exchange`: nothing is exchanged, each host wave acts as a wave
    /// of its own ([`with_local_waves`]).
    local: bool,
    /// The instruction being lowered may read a lane mask as a number under
    /// `local` (a bit count, a first set bit, `v_mbcnt`).
    local_ok: bool,
    /// With `local`: every mask read and lane read is taken per host wave
    /// ([`with_local_waves_any`]).
    local_any: bool,
    /// With `exchange`: VGPR values every lane holds alike ([`lane_uniform`]); a first-lane
    /// read of one needs nothing from the other host wave.
    uniform: HashSet<Value>,
    /// With `exchange`: the wave ORs that only steer a loop over each lane's own bits, which
    /// a host wave takes over its own lanes wherever they stand ([`super::own_bits`]).
    own_bits: super::own_bits::OwnBits,
    /// The console lane (0..63) of this thread, made in the entry block.
    clane: Option<V>,
    /// Whether this thread's console wave has an upper half on a second
    /// host wave (the last console wave of a group whose thread count is
    /// not a multiple of 64 meets no one), made in the entry block.
    paired: Option<V>,
    /// This host wave's active lanes and the other host wave of this
    /// console wave's (made in the entry block): the ticket counts of
    /// `pair_meet` count a round in active lanes, and a group whose thread
    /// count is not a multiple of 64 gives the last console wave's halves
    /// different counts.
    pair_lanes: Option<(V, V)>,
    /// The first flag dword of this thread's console wave ([`Self::pair_flags`]).
    flag_pair: Option<V>,
    /// The first exchange dword of this thread's console wave ([`Self::exchange_pair`]).
    pair: Option<V>,
    /// A console wave64 program in host waves of 32, in a group of several
    /// console waves: four flag dwords per console wave (each half's
    /// "arrived" and "done"), through which a console wave's two host waves
    /// meet by themselves ([`Self::pair_meet`]). A group barrier would wait
    /// for the other console wave's host waves too, which inside a branch
    /// only this console wave takes never come.
    pair_flags: Option<u32>,
    /// A console wave64 program in host waves of 32: the LDS dword counting the console threads
    /// handed to host waves (`wave_base`), and the base made in the entry block.
    halves: Option<u32>,
    wave_base: Option<V>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// Pixel inputs holding 16-bit values: row, whether two per dword, and
    /// the row packed after it.
    half_inputs: HashMap<u8, (bool, Option<u8>)>,
    /// Multi-element 16-bit rows whose trailing element overflows a 4-dword
    /// `pack2` attribute and spills into a free dword of a later `pack2`
    /// attribute: `(attr_row, half_index) -> (sig_row, col)`.
    half_input_spills: HashMap<(u8, u8), (u8, u8)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// SCC of a scalar mask operation ("some lane is set"), with the low
    /// dword of the mask it tests: read as data, the ballot is not zero.
    mask_scc: HashMap<Value, Value>,
    /// Lane masks read as numbers (see [`mask_numbers`]).
    mask_numbers: HashSet<Value>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// Values of whole-wave sections holding one lane's, the one read after.
    one_lane: HashSet<Value>,
    /// Memory by address is written too, for other groups to read.
    coherent: Option<bool>,
    /// Threads of different groups take turns through atomics (a tree built
    /// from its leaves up): what they wrote before one must be seen after
    /// Views of buffers accessed by address, by instruction.
    made: HashMap<Site, V>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Private (scratch) memory: the lane's array and its dwords.
    scratch: Option<(V, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// Loops that lanes leave through one block, by that block: the header
    /// and the scalar values the loop writes that code after it reads (see
    /// [`Lower::leave_loops`]).
    loop_exits: HashMap<usize, Vec<(usize, Vec<Value>)>>,
    /// Loop bodies (IR blocks) by header.
    loop_bodies: HashMap<usize, HashSet<usize>>,
    /// Scalar values as the wave's last iteration of a divergent loop left
    /// them, with the loop's header: what code after the loop reads.
    finals: HashMap<Value, (Val, usize)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// Per-lane descriptor joins being lowered: IR block and the heap index
    /// phi there (what [`Source::Carried`] reads).
    lane_paths: Vec<(usize, V)>,
    /// Heap index phis of per-lane joins with predecessors lowered later.
    pending_paths: Vec<PendingPath>,
}

/// A heap index phi of a [`Source::LanePhi`], filled once every predecessor is
/// lowered.
struct PendingPath {
    phi: V,
    block: usize,
    sampler: bool,
    /// [`Lower::lane_paths`] where it was made, itself included.
    outer: Vec<(usize, V)>,
    /// Per predecessor: the index when it was lowered already, else its
    /// source.
    incoming: Vec<(usize, Result<V, Option<Source>>)>,
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
pub(super) const LANE_MASK: Why = "a lane mask read as data";
/// A mask with lanes outside EXEC where it is written, read as a number where the lanes that
/// run that block are not all of the wave.
pub(super) const MASK_ELSEWHERE: Why = "a lane mask holding lanes elsewhere read as data";

/// Scalar set operations, which keep a lane mask a lane mask: their result
/// as a number is the same operation on their sources' numbers.
fn mask_set_op(name: &str) -> bool {
    let Some(op) = name
        .strip_prefix("s_")
        .and_then(|n| n.strip_suffix("_b32").or_else(|| n.strip_suffix("_b64")))
    else {
        return false;
    };
    matches!(
        op,
        "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "not" | "mov"
    )
}

#[allow(dead_code)]
/// Samples whose level comes from derivatives across each quad of lanes.
fn implicit_derivatives(name: &str) -> bool {
    sample_form(name).is_some_and(|f| !f.gather && !f.c && !f.d && !f.l && !f.lz)
}

/// The lane masks whose value as a number the program reads: those an
/// instruction reads as data ([`reads_mask_as_data`]) or a join that is no
/// mask takes, and the masks the set operations and joins writing them
/// read in turn. Only these get a number where a set operation writes them
/// (the others would take ballots nothing reads).
fn mask_numbers(st: &Structured, p: &crate::ir::Program) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut out: HashSet<Value> = HashSet::new();
    for b in &p.blocks {
        for phi in b.phis.iter().filter(|phi| !is_mask(phi.value)) {
            out.extend(phi.args.iter().copied().filter(|&a| is_mask(a)));
        }
        for i in b.insts.iter().filter(|i| reads_mask_as_data(st, i)) {
            let src: &[Operand] = match &i.fields {
                Fields::Sop { src, .. } => src,
                Fields::Valu(v) => &v.src,
                _ => &[],
            };
            for op in src {
                if let Operand::Scalar { code, dwords } = *op
                    && code != NULL
                {
                    let read = (0..dwords).filter_map(|k| i.read(Reg::S(code + k)));
                    out.extend(read.filter(|&v| is_mask(v)));
                }
            }
        }
    }
    loop {
        let before = out.len();
        for b in &p.blocks {
            for phi in &b.phis {
                if out.contains(&phi.value) {
                    out.extend(phi.args.iter().copied().filter(|&a| is_mask(a)));
                }
            }
            for i in &b.insts {
                if matches!(i.fields, Fields::Sop { .. })
                    && mask_set_op(i.name())
                    && i.writes.iter().any(|w| out.contains(&w.1))
                {
                    let read = i.reads.iter().filter(|r| matches!(r.0, Reg::S(_)));
                    out.extend(read.map(|r| r.1).filter(|&v| is_mask(v)));
                }
            }
        }
        if out.len() == before {
            return out;
        }
    }
}

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
/// Per IR block: every lane of the wave that started the program runs it – no per-lane branch
/// separates it from the entry (it lies in no lane branch's region: the blocks reachable from
/// the branch that do not post-dominate it). None with an indirect exit (not followed).
fn all_lane_blocks(st: &Structured) -> Option<Vec<bool>> {
    let n = st.exits.len();
    let mut all = vec![true; n];
    for (_, region) in lane_regions(st)? {
        for b in (0..n).filter(|&b| region[b]) {
            all[b] = false;
        }
    }
    Some(all)
}

/// Every per-lane branch (by its block) with its region: the blocks reachable from it that do
/// not post-dominate it, which only some lanes run. Not the branches the wave takes together
/// ([`Structured::wide`]), nor those on scalar conditions. None with an indirect exit.
fn lane_regions(st: &Structured) -> Option<Vec<(usize, Vec<bool>)>> {
    let n = st.exits.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, e) in st.exits.iter().enumerate() {
        match *e {
            LaneExit::Jump(t) => succ[b].push(t),
            LaneExit::Branch { taken, not_taken, .. } => {
                succ[b].push(taken);
                succ[b].push(not_taken);
            }
            LaneExit::End => {}
            LaneExit::Indirect => return None,
        }
    }
    // post-dominators (bitsets; a virtual exit after every End)
    let words = n.div_ceil(64);
    let full = vec![u64::MAX; words];
    let mut pdom: Vec<Vec<u64>> = (0..n).map(|b| if succ[b].is_empty() { let mut v = vec![0; words]; v[b / 64] |= 1 << (b % 64); v } else { full.clone() }).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            if succ[b].is_empty() {
                continue;
            }
            let mut v = full.clone();
            for &t in &succ[b] {
                for w in 0..words {
                    v[w] &= pdom[t][w];
                }
            }
            v[b / 64] |= 1 << (b % 64);
            if v != pdom[b] {
                pdom[b] = v;
                changed = true;
            }
        }
    }
    let mut regions = Vec::new();
    for (d, e) in st.exits.iter().enumerate() {
        let LaneExit::Branch { when, .. } = *e else { continue };
        if st.wide.contains(&d) || st.bools.support(when).iter().all(|a| matches!(a, Atom::Uniform(_))) {
            continue; // a wave-wide or uniform branch: all lanes go the same way
        }
        // blocks reachable from the lane branch that do not post-dominate it
        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = succ[d].clone();
        while let Some(b) = stack.pop() {
            if seen[b] || pdom[d][b / 64] >> (b % 64) & 1 == 1 {
                continue;
            }
            seen[b] = true;
            stack.extend(succ[b].iter().copied());
        }
        regions.push((d, seen));
    }
    Some(regions)
}

/// The per-lane branches (by block) whose regions hold a block of `needs` (blocks where two
/// host waves must meet, [`take_needs`]): the ones to take wave-wide
/// ([`crate::structure::structure_wide`]) so that every lane runs those blocks. None if one
/// of them is a branch whose wave-wide meaning is not known ([`Structured::wave_any`]).
pub(super) fn branches_around(st: &Structured, needs: &[usize]) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for (d, region) in lane_regions(st)? {
        if needs.iter().any(|&b| region.get(b).copied().unwrap_or(false)) {
            st.wave_any.get(d).copied().flatten()?;
            out.push(d);
        }
    }
    Some(out)
}

/// Whether a refusal is of code that needs every lane of the wave where it stands (two host
/// waves meeting there), which the other forms of the program may give it.
pub(super) fn needs_every_lane(what: &str) -> bool {
    what.ends_with("where not every lane runs") || what == MASK_ELSEWHERE
}

/// The blocks where the last lowering on this thread wanted every lane and did not have it
/// (its "... where not every lane runs" refusals), which it forgets.
pub(super) fn take_needs() -> Vec<usize> {
    NEEDS.take()
}

fn need(block: usize) {
    NEEDS.with_borrow_mut(|n| {
        if !n.contains(&block) {
            n.push(block);
        }
    });
}

/// VGPR values that are the same in every lane holding them (`all` is [`all_lane_blocks`]):
/// written by an instruction every arriving lane runs (no guard), lane by lane, from scalars,
/// constants and such values; or loaded from an address made of them. The compiler keeps
/// wave-uniform data in VGPRs and fetches it with `v_readfirstlane`.
/// shortcut: a value joined by a phi counts only where every predecessor is a block all lanes
/// run (no lane branch can have split the arriving lanes); loop-carried values never do.
fn lane_uniform(p: &Program, st: &Structured, all: &[bool]) -> HashSet<Value> {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let mut same: HashSet<Value> = HashSet::new();
    for b in crate::ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        let whole = block.preds.iter().all(|&q| all.get(q).copied().unwrap_or(false));
        for phi in &block.phis {
            if matches!(phi.reg, Reg::V(_)) && whole && phi.args.iter().all(|a| same.contains(a)) {
                same.insert(phi.value);
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            let name = inst.name();
            let ops: Vec<Operand> = match &inst.fields {
                Fields::Valu(v)
                    if v.dpp.is_none()
                        && v.sdwa.is_none()
                        && !name.starts_with("v_mbcnt")
                        && !name.starts_with("v_cmpx")
                        && !name.contains("lane") =>
                {
                    v.src.clone()
                }
                _ => continue,
            };
            if st.guards[b][i].is_some() {
                continue; // lanes outside EXEC keep another value
            }
            let alike = ops.iter().all(|op| match *op {
                Operand::Vector { reg, .. } => {
                    // every dword of the operand the instruction reads from that register on
                    let read: Vec<Value> =
                        inst.reads.iter().filter(|r| matches!(r.0, Reg::V(x) if x >= reg)).map(|r| r.1).collect();
                    !read.is_empty() && read.iter().all(|x| same.contains(x))
                }
                // a scalar is one value; a lane mask is not, unless a compare of such values made it
                Operand::Scalar { code, .. } if code != NULL => {
                    inst.read(Reg::S(code)).is_none_or(|x| !is_mask(x) || same.contains(&x))
                }
                _ => true,
            });
            if alike {
                same.extend(inst.writes.iter().filter(|w| w.0 != Reg::Scc).map(|w| w.1));
            }
        }
    }
    same
}

/// What a register value holds, for [`wave_local`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wave {
    /// Nothing that depends on how the threads are divided into waves.
    Plain,
    /// The number of the wave's first lane in a mask (`s_ff1`).
    First,
    /// A reduction over the wave's lanes: a count, a sum, a minimum...
    Total(super::wave::Op),
    /// A count or sum over the lanes below this one.
    Prefix,
    /// Derived from what a first-lane atomic returned: the slots the wave reserved.
    Base,
    /// A mix the idiom does not make.
    Mixed,
}

/// Whether a console wave64 program may run with each host wave as a wave of its own
/// ([`with_local_waves`]): its wave-wide values are used only as the allocation idiom uses
/// them, which does not care how the threads are divided into waves. A first-lane number only
/// picks the first lane; a count or sum only goes into that lane's atomic on a shared counter;
/// a prefix only goes onto the base that atomic returned; the first lane does nothing else.
/// Anything else done with them takes one wave to be the whole thread group (a tile's result,
/// say) and is refused, with the offset and what was found.
/// Lane masks read as numbers by other instructions and lane reads outside whole-wave sections
/// are stopped by the lowering itself.
// shortcut: `v_readfirstlane` of a value that differs between lanes is taken as it is (this
// host wave's first lane), and a guarded write of a plain value over a wave-dependent one is
// the only mix of old and new lanes that is looked at.
pub(super) fn wave_local(p: &Program, st: &Structured) -> Result<(), (usize, &'static str)> {
    use super::wave::{Op, Shape};
    const FIRST: Why = "a wave's first-lane number used as data";
    const TOTAL: Why = "a count or sum over the wave used as data";
    const PREFIX: Why = "a count or sum over the lanes below used without a reserved base";
    const MIXED: Why = "a value that depends on the wave on some paths only";
    const EFFECT: Why = "a write by the wave's first lane other than its reservation";
    fn of(kind: &HashMap<Value, Wave>, v: Option<Value>) -> Wave {
        v.and_then(|v| kind.get(&v).copied()).unwrap_or(Wave::Plain)
    }
    fn put(kind: &mut HashMap<Value, Wave>, v: Value, k: Wave) {
        if k == Wave::Plain {
            kind.remove(&v);
        } else {
            kind.insert(v, k);
        }
    }
    let wild = |k: Wave| !matches!(k, Wave::Plain | Wave::Base);
    let why = |k: Wave| match k {
        Wave::First => FIRST,
        Wave::Total(_) => TOTAL,
        Wave::Prefix => PREFIX,
        _ => MIXED,
    };
    // a mask (by its low dword) under which only a wave's first lane runs
    let first_mask = |firsts: &HashSet<Value>, m: Option<Value>| {
        m.and_then(|m| st.masks.get(&m)).is_some_and(|&b| {
            st.bools
                .support(b)
                .iter()
                .any(|a| matches!(a, Atom::Lane(v) if firsts.contains(v)))
        })
    };
    // One instruction outside a section: the kind of what it writes, and whether it is a
    // first lane's reservation.
    let step = |inst: &Inst, kind: &HashMap<Value, Wave>, firsts: &mut HashSet<Value>| -> Result<(Wave, bool), Why> {
        let name = inst.name();
        let val = |op: &Operand| match *op {
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
            _ => None,
        };
        let ops: Vec<Operand> = match &inst.fields {
            Fields::Valu(v) => v.src.clone(),
            Fields::Sop { dst, src, .. } => {
                let mut o = src.clone();
                if matches!(name, "s_addk_i32" | "s_mulk_i32") {
                    o.extend(dst.iter().copied());
                }
                o
            }
            Fields::Buffer { data, .. } if name.contains("_atomic_") || name.contains("_store_") => {
                data.iter().copied().collect()
            }
            _ => Vec::new(),
        };
        let kinds: Vec<Wave> = ops.iter().map(|o| of(kind, val(o))).collect();
        // Everything else it reads (addresses, descriptors, SCC, the high dwords of wide
        // operands) must not depend on the wave. Not EXEC, and not the old value of a VGPR it
        // writes, which only the lanes outside EXEC keep.
        let named: Vec<Value> = ops.iter().filter_map(&val).collect();
        for &(r, v) in &inst.reads {
            let kept = matches!(r, Reg::V(_)) && inst.writes.iter().any(|w| w.0 == r);
            if matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1) || kept || named.contains(&v) {
                continue;
            }
            let k = of(kind, Some(v));
            if wild(k) {
                return Err(why(k));
            }
        }
        let w = kinds.iter().copied().find(|&k| wild(k)).unwrap_or(Wave::Mixed);
        let sum = |k: Wave| k == Wave::Total(Op::AddU);
        let is_mask = |o: &Operand| {
            val(o).is_some_and(|v| st.masks.contains_key(&v) || st.highs.contains_key(&v))
        };
        let mut reserved = false;
        let out = match (name, &kinds[..]) {
            ("s_ff1_i32_b64" | "s_ff1_i32_b32", _) if ops.first().is_some_and(is_mask) => Wave::First,
            ("s_bcnt1_i32_b64" | "s_bcnt1_i32_b32", _) if ops.first().is_some_and(is_mask) => {
                Wave::Total(Op::AddU)
            }
            // bits of a mask below the lane; with -1 for the mask, the lane's number
            ("v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32", [_, b]) => {
                match (matches!(ops[0], Operand::Scalar { .. }), *b) {
                    (_, Wave::Prefix) | (true, Wave::Plain | Wave::Base) => Wave::Prefix,
                    (false, Wave::Plain | Wave::Base) => Wave::Plain,
                    (_, k) => return Err(why(k)),
                }
            }
            // nothing wave-dependent goes in; what derives from a reserved base stays marked
            _ if !kinds.iter().any(|&k| wild(k)) => {
                let derives = matches!(inst.fields, Fields::Valu(_) | Fields::Sop { .. });
                if derives && kinds.contains(&Wave::Base) { Wave::Base } else { Wave::Plain }
            }
            ("v_mov_b32" | "s_mov_b32", [k]) => *k,
            ("v_cmp_eq_i32" | "v_cmp_eq_u32" | "v_cmpx_eq_i32" | "v_cmpx_eq_u32", [a, b]) => {
                // the lane whose own number is the first lane's
                let lane = match (*a, *b) {
                    (Wave::First, k) if !wild(k) => &ops[1],
                    (k, Wave::First) if !wild(k) => &ops[0],
                    _ => return Err(why(w)),
                };
                if !matches!(lane, Operand::Vector { .. }) {
                    return Err(FIRST);
                }
                let mask = match &inst.fields {
                    _ if name.starts_with("v_cmpx") => inst.write(Reg::S(EXEC)),
                    Fields::Valu(v) => match v.dst {
                        Some(Operand::Scalar { code, .. }) => inst.write(Reg::S(code)),
                        _ => None,
                    },
                    _ => None,
                };
                firsts.extend(mask);
                Wave::Plain
            }
            ("v_add_nc_u32" | "s_add_i32", [a, b]) => match (*a, *b) {
                (Wave::Prefix, Wave::Base) | (Wave::Base, Wave::Prefix) => Wave::Base,
                (Wave::Prefix, Wave::Plain | Wave::Prefix) | (Wave::Plain, Wave::Prefix) => Wave::Prefix,
                (x, y) if sum(x) && sum(y) => x,
                _ => return Err(why(w)),
            },
            ("v_sub_nc_u32" | "s_sub_i32", [a, b]) => match (*a, *b) {
                (Wave::Base, Wave::Prefix) => Wave::Base,
                (Wave::Prefix, Wave::Plain) => Wave::Prefix,
                (Wave::Plain, y) if sum(y) && matches!(ops[0], Operand::Const(128)) => y,
                _ => return Err(why(w)),
            },
            ("s_mul_i32" | "v_mul_lo_u32" | "v_mul_u32_u24", [a, b]) => match (*a, *b) {
                (x, Wave::Plain) | (Wave::Plain, x) if sum(x) || x == Wave::Prefix => x,
                _ => return Err(why(w)),
            },
            ("s_lshl_b32", [x, Wave::Plain]) | ("v_lshlrev_b32", [Wave::Plain, x])
                if sum(*x) || *x == Wave::Prefix =>
            {
                *x
            }
            ("v_lshl_add_u32", [Wave::Prefix, Wave::Plain, c]) if !wild(*c) => {
                if *c == Wave::Base { Wave::Base } else { Wave::Prefix }
            }
            // the reservation: the first lane adds its wave's total to a shared counter
            (_, [Wave::Total(op)]) if name.contains("_atomic_") => {
                let want = match op {
                    Op::AddU => "buffer_atomic_add",
                    Op::MinU => "buffer_atomic_umin",
                    Op::MaxU => "buffer_atomic_umax",
                    Op::MinI => "buffer_atomic_smin",
                    Op::MaxI => "buffer_atomic_smax",
                    Op::Or => "buffer_atomic_or",
                    Op::And => "buffer_atomic_and",
                    Op::Xor => "buffer_atomic_xor",
                    _ => "",
                };
                if name != want || !first_mask(firsts, inst.read(Reg::S(EXEC))) {
                    return Err(TOTAL);
                }
                reserved = true;
                Wave::Base
            }
            _ => return Err(why(w)),
        };
        Ok((out, reserved))
    };
    let sections: Vec<_> = (0..p.blocks.len()).map(|b| super::wave::sections(p, b)).collect();
    let order = crate::ir::reverse_postorder(&p.blocks);
    let pass = |kind: &mut HashMap<Value, Wave>, firsts: &mut HashSet<Value>, check: bool| -> Result<(), (usize, Why)> {
        for &b in &order {
            let block = &p.blocks[b];
            for phi in &block.phis {
                let mut k = Wave::Plain;
                for (n, &a) in phi.args.iter().enumerate() {
                    let x = of(kind, Some(a));
                    k = match (k, x) {
                        _ if n == 0 => x,
                        (k, x) if k == x => k,
                        (Wave::Plain, m @ (Wave::Base | Wave::Prefix))
                        | (m @ (Wave::Base | Wave::Prefix), Wave::Plain) => m,
                        _ => Wave::Mixed,
                    };
                }
                put(kind, phi.value, k);
            }
            for (i, inst) in block.insts.iter().enumerate() {
                // inside a whole-wave section: what the simulation found each value to be
                if let Some((_, sec)) = sections[b].iter().find(|(s, sec)| i > **s && i < sec.end) {
                    for &(_, w) in &inst.writes {
                        let k = match sec.shapes.get(&w) {
                            None => Wave::Mixed,
                            Some(&Shape::K(_)) => Wave::Plain,
                            Some(&Shape::Value(v)) => of(kind, Some(v)),
                            Some(&Shape::Masked(src, _, _)) => of(kind, Some(src)),
                            Some(&Shape::Active(op, src, mask, _)) => match of(kind, Some(src)) {
                                Wave::Plain | Wave::Base => Wave::Total(op),
                                // the first lane's own sum of a total is that total
                                Wave::Total(o) if o == op && first_mask(firsts, Some(mask)) => Wave::Total(op),
                                _ => Wave::Mixed,
                            },
                            Some(&Shape::Prefix(Op::AddU, src, mask, _, _)) => match of(kind, Some(src)) {
                                Wave::Plain | Wave::Base => Wave::Prefix,
                                Wave::Total(Op::AddU) if first_mask(firsts, Some(mask)) => Wave::Prefix,
                                _ => Wave::Mixed,
                            },
                            Some(_) => Wave::Mixed,
                        };
                        put(kind, w, k);
                    }
                    continue;
                }
                let name = inst.name();
                if name.starts_with("s_cbranch") {
                    continue; // with the block's exit, below
                }
                let (k, reserved) = match step(inst, kind, firsts) {
                    Ok(r) => r,
                    Err(why) if check => return Err((inst.machine.offset, why)),
                    Err(_) => (Wave::Mixed, false),
                };
                // the same test as `Lower::inst`'s `effect`
                let effect = matches!(inst.fields, Fields::Exp { .. })
                    || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
                    || name.contains("store")
                    || name.contains("atomic")
                    || matches!(inst.fields, Fields::Ds { .. })
                        && !(name.starts_with("ds_read")
                            || name.starts_with("ds_swizzle")
                            || name.contains("permute"));
                if check && effect && !reserved && first_mask(firsts, inst.read(Reg::S(EXEC))) {
                    return Err((inst.machine.offset, EFFECT));
                }
                let guarded = st.guards[b][i].is_some();
                for &(r, w) in &inst.writes {
                    let mut kw = match r {
                        // "some lane is counted" keeps the count's kind; any other SCC is one bit of data
                        Reg::Scc if !name.starts_with("s_bcnt1") => {
                            if wild(k) { Wave::Mixed } else { Wave::Plain }
                        }
                        _ => k,
                    };
                    // a guarded write leaves the old value in the lanes outside EXEC
                    if guarded
                        && matches!(r, Reg::V(_))
                        && kw == Wave::Plain
                        && matches!(of(kind, inst.read(r)), Wave::First | Wave::Total(_) | Wave::Prefix)
                    {
                        kw = Wave::Mixed;
                    }
                    put(kind, w, kw);
                }
            }
            // a branch on "some lane is counted": fine where the structure made it the lane's
            // own bit (the mask is EXEC under a scalar condition)
            if check
                && let Some(last) = block.insts.last()
                && last.name().starts_with("s_cbranch_scc")
                && let Some(scc) = last.read(Reg::Scc)
                && wild(of(kind, Some(scc)))
            {
                let own = matches!(st.exits[b], LaneExit::Branch { when, .. }
                    if !st.bools.support(when).iter().any(|a| matches!(a, Atom::Uniform(v) if *v == scc)));
                if !own {
                    return Err((last.machine.offset, TOTAL));
                }
            }
        }
        Ok(())
    };
    let mut kind: HashMap<Value, Wave> = HashMap::new();
    let mut firsts: HashSet<Value> = HashSet::new();
    // to a fixpoint (values carried around loops), then once more to say what is wrong
    let mut settled = false;
    for _ in 0..16 {
        let before = (kind.clone(), firsts.len());
        pass(&mut kind, &mut firsts, false)?;
        if (&before.0, before.1) == (&kind, firsts.len()) {
            settled = true;
            break;
        }
    }
    if !settled {
        return Err((0, MIXED));
    }
    pass(&mut kind, &mut firsts, true)
}

/// The mode of a group barrier the translation adds (host waves meeting):
/// a wait for the group with a shared-memory fence; the fence alone with
/// `VICTORY_GROUP_SYNC=0` ([`super::group_sync`]).
fn meeting_mode() -> u32 {
    if super::group_sync() == 0 { 8 } else { 1 | 8 }
}

/// LDS accesses of a console wave run as two host waves, where some sit in blocks that only
/// some lanes run (`all` is [`all_lane_blocks`]): per IR block, whether the host waves meet (a
/// group barrier) at its start and before its exit. `None` if barriers cannot keep the
/// console's order.
///
/// The console runs the wave's lanes in step, so an LDS write is seen by every later read. Two
/// host waves keep that order if they meet between any two accesses of which one writes. A
/// barrier must be run by every thread, so it cannot stand in a lane branch's region; but
/// every way into and out of such a region passes a block all lanes run, and the waves meet
/// there: before the exit of an all-lane block into a region with LDS accesses, and at the
/// start of an all-lane block entered from one. Accesses in all-lane blocks get their own
/// barrier as before. What that cannot order is a write and a read inside one region (blocks
/// joined without an all-lane block between them): refused.
/// shortcut: two writes in one region are taken not to hit the same dword from both halves.
fn lds_meetings(p: &crate::ir::Program, st: &Structured, all: &[bool]) -> Option<Vec<(bool, bool)>> {
    let n = st.exits.len();
    let succ = |b: usize| -> Vec<usize> {
        match st.exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch { taken, not_taken, .. } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    // regions: blocks not every lane runs, joined by the edges between them
    let mut region: Vec<usize> = (0..n).collect();
    fn find(region: &mut [usize], mut b: usize) -> usize {
        while region[b] != b {
            region[b] = region[region[b]];
            b = region[b];
        }
        b
    }
    for b in 0..n {
        for t in succ(b) {
            if !all[b] && !all[t] {
                let (x, y) = (find(&mut region, b), find(&mut region, t));
                region[x] = y;
            }
        }
    }
    // per region: LDS reads, LDS writes
    let mut reads = vec![false; n];
    let mut writes = vec![false; n];
    for (b, block) in p.blocks.iter().enumerate().take(n) {
        if all[b] {
            continue;
        }
        let r = find(&mut region, b);
        for i in &block.insts {
            let name = i.name();
            if !matches!(i.fields, Fields::Ds { .. }) || name.starts_with("ds_swizzle") || name.contains("permute") {
                continue;
            }
            if name.starts_with("ds_read") {
                reads[r] = true;
            } else {
                writes[r] = true;
            }
        }
    }
    if (0..n).any(|r| reads[r] && writes[r]) {
        if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
            for r in (0..n).filter(|&r| reads[r] && writes[r]) {
                let mut list = Vec::new();
                for (b, block) in p.blocks.iter().enumerate().take(n) {
                    if all[b] || find(&mut region, b) != r {
                        continue;
                    }
                    for i in &block.insts {
                        let name = i.name();
                        if !matches!(i.fields, Fields::Ds { .. }) || name.starts_with("ds_swizzle") || name.contains("permute") {
                            continue;
                        }
                        // the address register and the instruction that wrote it
                        let addr = match &i.fields {
                            Fields::Ds { addr: Some(Operand::Vector { reg, .. }), offset0, offset1, .. } => {
                                let made = i.read(Reg::V(*reg)).and_then(|v| {
                                    p.blocks
                                        .iter()
                                        .flat_map(|bl| &bl.insts)
                                        .find(|x| x.writes.iter().any(|w| w.1 == v))
                                        .map(|x| format!("{:#x} {}", x.machine.offset, x.name()))
                                });
                                format!("v{reg} off {offset0}/{offset1} made by {}", made.as_deref().unwrap_or("an input or a phi"))
                            }
                            _ => "no address".to_string(),
                        };
                        list.push(format!("{:#x} {name} [{addr}]", i.machine.offset));
                    }
                }
                eprintln!("WAVE64-LDS region with reads and writes: {}", list.join("; "));
            }
        }
        return None;
    }
    let mut meet = vec![(false, false); n];
    for b in 0..n {
        for t in succ(b) {
            if all[b] && !all[t] {
                let r = find(&mut region, t);
                meet[b].1 |= reads[r] || writes[r];
            } else if !all[b] && all[t] {
                let r = find(&mut region, b);
                meet[t].0 |= reads[r] || writes[r];
            }
        }
    }
    Some(meet)
}

/// Masks held as numbers or constant lane sets in guards or lane branches: the lane tests its
/// own bit by its console lane.
fn lane_constant_masks(st: &Structured) -> bool {
    let whens = st.exits.iter().filter_map(|e| match *e {
        LaneExit::Branch { when, .. } => Some(when),
        _ => None,
    });
    let mut work: Vec<Bool> = st.guards.iter().flatten().flatten().copied().chain(whens).collect();
    // Through lane-mask joins too: an edge that brings a number or a constant lane set needs
    // the lane's own bit of it.
    // shortcut: masks only selects read (`v_cndmask_b32`, carries in) are not followed; such
    // a program still stops with "a constant lane mask" or "an opaque lane mask".
    let (mut seen, mut joins) = (HashSet::new(), HashSet::new());
    while let Some(g) = work.pop() {
        if !seen.insert(g) {
            continue;
        }
        for a in st.bools.support(g) {
            match a {
                Atom::Opaque(_) | Atom::Constant(_) => return true,
                Atom::Phi(id) if joins.insert(id) => {
                    if let Some(edges) = st.phi_masks.get(&id) {
                        work.extend(edges.iter().copied());
                    }
                }
                _ => {}
            }
        }
    }
    false
}

/// The widest wave the host GPU runs (0: not known – no host-specific forms); see
/// [`set_host_max_wave`].
static HOST_MAX_WAVE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The host GPU's `WaveLaneCountMax`: 32 lets console wave64 compute programs whose cross-lane
/// code stays inside 32-lane halves run in waves of 32 instead of being refused.
pub fn set_host_max_wave(lanes: u32) {
    HOST_MAX_WAVE.store(lanes, std::sync::atomic::Ordering::Relaxed);
}

thread_local! {
    /// See [`take_needs`].
    static NEEDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    /// See [`with_local_waves`].
    static LOCAL_WAVES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// See [`with_local_waves_any`].
    static LOCAL_ANY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// As [`with_local_waves`], with every other use of a lane mask as a number
/// and every lane read and write taken per host wave too: the program runs
/// as it would on a console of 32-lane waves, lanes numbered 0..31 or
/// 32..63 by half. Right for code that only relies on what HLSL's wave
/// operations promise; a program that takes one wave to be its whole
/// thread group gets per-half results. The last resort for programs that
/// do not translate otherwise.
pub(super) fn with_local_waves_any<R>(f: impl FnOnce() -> R) -> R {
    let was = LOCAL_ANY.replace(true);
    let r = with_local_waves(f);
    LOCAL_ANY.set(was);
    r
}

/// Runs `f` with console wave64 programs on 32-lane hosts lowered with each
/// host wave as a wave of its own ("local"): no exchange between the two
/// halves of a console wave, so no barriers and no need for every lane to
/// reach the code. What HLSL's wave operations promise holds for any
/// division of the threads into waves, and the idiom the title's compute
/// programs use them for does not care which wave a thread is in: count or
/// sum over the wave, the first lane reserves that many slots with an atomic
/// add, every lane takes the base (`WaveReadLaneFirst`) plus its prefix.
/// Each lane still gets a slot of its own; only the order differs.
/// Only that idiom's operations are lowered this way: a lane mask read as
/// a number by anything but a bit count, a first-set-bit or `v_mbcnt`, and
/// lane reads and writes outside whole-wave sections, stop the translation.
/// shortcut: where a count or sum goes is not checked. A program that uses
/// a wave-wide result as data for the whole console wave (a tile's maximum,
/// say, with one wave per thread group) gets its half's result. It is used
/// only for programs that do not translate otherwise.
pub(super) fn with_local_waves<R>(f: impl FnOnce() -> R) -> R {
    let was = LOCAL_WAVES.replace(true);
    let r = f();
    LOCAL_WAVES.set(was);
    r
}

/// `VICTORY_F16_NEAREST=1`: the results of 16-bit float operations (`v_add_f16`,
/// `v_mul_f16`, `v_cvt_f16_f32` ...) are rounded to nearest as the console rounds them
/// ([`Lower::f16_nearest`]) instead of as the host's `f32tof16` does (toward zero on NVIDIA:
/// the last place of a result that is not exact is then one lower than the console's).
/// Off by default: it is exact but untried in a game run, and adds instructions to every
/// such operation. The wave sum of 16-bit floats ([`Lower::half_sum`]) is always exact.
fn f16_nearest_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_F16_NEAREST.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_F16_NEAREST").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`f16_nearest_on`], for its thread only.
    pub(super) static TEST_F16_NEAREST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// `VICTORY_WAVE_OWN_BITS=0`: a wave OR where not every lane runs is refused as before
/// ([`super::own_bits`]).
fn own_bits_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAVE_OWN_BITS").is_none_or(|v| v != "0"))
}

/// The refusal of a first half of an interpolation whose result is read
/// as data ([`Lower::interp`]).
pub(super) const FIRST_HALF: &str = "the first half of an interpolation";

thread_local! {
    /// Whether first halves of interpolations are evaluated as the pair
    /// they start ([`with_first_half_pairs`]).
    static FIRST_HALF_PAIRS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `translate` with first halves of interpolations evaluated as the
/// pair they start: the translation of a program refused with
/// [`FIRST_HALF`] (`VICTORY_FIRST_HALF=0`: refused again, as before).
pub(super) fn with_first_half_pairs<R>(translate: impl FnOnce() -> R) -> R {
    let before = FIRST_HALF_PAIRS.replace(true);
    let result = translate();
    FIRST_HALF_PAIRS.set(before);
    result
}

/// What [`set_host_max_wave`] was told (0: nothing).
pub fn host_max_wave() -> u32 {
    #[cfg(test)]
    if let Some(lanes) = TEST_HOST_WAVE.get() {
        return lanes;
    }
    HOST_MAX_WAVE.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// A test's host wave size, for its thread only ([`set_host_max_wave`] is the process's).
    pub(super) static TEST_HOST_WAVE: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

pub(super) fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// The parameters packing 16-bit rows two per dword and where each lays
/// its elements out: (first row, [(dword, register, column, columns)]).
///
/// The console gives every 16-bit element whole dwords. A parameter leads
/// with the first element of its first row, then the first element of its
/// second row. An element that shares a register behind another one (the
/// signature packs a short element into the free columns of an earlier
/// register) is NOT in that row's halves: it follows in the first
/// parameter, in row order, that has whole dwords left.
type PackedLayout = Vec<(u8, Vec<(u8, u8, u8, u8)>)>;

fn packed_layout(elements: &[SigElement], packs: &[(u8, Option<u8>)]) -> PackedLayout {
    let of_row = |r: u8| {
        let mut of: Vec<&SigElement> = elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .collect();
        of.sort_by_key(|e| e.col);
        of
    };
    let mut packs = packs.to_vec();
    packs.sort_unstable();
    packs.dedup_by_key(|p| p.0);
    // Per parameter: its first row, the dwords used, the elements.
    let mut layout: Vec<(u8, u8, Vec<(u8, u8, u8, u8)>)> = Vec::new();
    let mut behind = Vec::new();
    for &(a, b) in &packs {
        let mut placed = Vec::new();
        let mut at = 0u8;
        if !of_row(a).is_empty() {
            for r in [Some(a), b].into_iter().flatten() {
                let of = of_row(r);
                let Some((lead, more)) = of.split_first() else {
                    continue;
                };
                placed.push((at, r, lead.col, lead.cols));
                at += lead.cols.div_ceil(2);
                behind.extend(more.iter().map(|e| (r, e.col, e.cols)));
            }
        }
        layout.push((a, at, placed));
    }
    behind.sort_unstable();
    for (r, col, cols) in behind {
        let need = cols.div_ceil(2);
        if let Some(p) = layout.iter_mut().find(|p| p.1 + need <= 4) {
            p.2.push((p.1, r, col, cols));
            p.1 += need;
        }
    }
    layout.into_iter().map(|(a, _, of)| (a, of)).collect()
}
/// SCC of each scalar operation writing a tracked lane mask, with the
/// mask's low dword (EXEC's new value for the `saveexec` forms).
fn mask_sccs(st: &Structured, p: &crate::ir::Program) -> HashMap<Value, Value> {
    let mut out = HashMap::new();
    for inst in p.blocks.iter().flat_map(|b| &b.insts) {
        let Fields::Sop { dst, .. } = &inst.fields else {
            continue;
        };
        let Some(scc) = inst.write(Reg::Scc) else {
            continue;
        };
        let lo = if inst.name().contains("saveexec") {
            inst.write(Reg::S(EXEC))
        } else {
            match *dst {
                Some(Operand::Scalar { code, .. }) => inst.write(Reg::S(code)),
                _ => None,
            }
        };
        if let Some(lo) = lo.filter(|lo| st.masks.contains_key(lo)) {
            out.insert(scc, lo);
        }
    }
    out
}

/// Whether `inst` gives each lane what it would under any EXEC: scalar
/// code and loads that do not touch EXEC, and vector code that reads no
/// other lane (no DPP, permutes or lane reads and writes).
fn lane_local(inst: &Inst) -> bool {
    // EXEC as an operand, or EXECZ.
    let exec = |o: &Operand| {
        matches!(*o, Operand::Scalar { code, .. } if code >= EXEC) || *o == Operand::Const(252)
    };
    let name = inst.name();
    match &inst.fields {
        Fields::Sop { .. } => !inst
            .reads
            .iter()
            .chain(&inst.writes)
            .any(|&(r, _)| r == Reg::S(EXEC) || r == Reg::S(EXEC + 1)),
        Fields::Smem { .. } => name.starts_with("s_load_") || name.starts_with("s_buffer_load_"),
        Fields::Valu(v) => {
            v.dpp.is_none()
                && !name.starts_with("v_interp_")
                && !matches!(
                    name,
                    "v_readlane_b32"
                        | "v_readfirstlane_b32"
                        | "v_writelane_b32"
                        | "v_permlane16_b32"
                        | "v_permlanex16_b32"
                )
                && !v.src.iter().chain(&v.dst).chain(&v.carry).any(exec)
        }
        _ => false,
    }
}

/// Dwords a scalar load reads, by the suffix of its name after
/// `s_load_dword` or `s_buffer_load_dword`.
fn smem_dwords(suffix: &str) -> Option<u32> {
    match suffix {
        "" => Some(1),
        "x2" => Some(2),
        "x4" => Some(4),
        "x8" => Some(8),
        "x16" => Some(16),
        _ => None,
    }
}

#[allow(dead_code)]
/// A select that needs the actual scalar mask words and console lane index.
fn numeric_select(st: &Structured, inst: &Inst) -> bool {
    inst.name() == "v_cndmask_b32"
        && matches!(&inst.fields, Fields::Valu(v)
            if matches!(v.src.get(2), Some(Operand::Scalar { code, .. }) if *code != NULL))
        && inst
            .writes
            .first()
            .is_some_and(|&(_, v)| !st.select_masks.contains_key(&v))
}



/// Where half `h` (dword h / 2, the high half when odd) of the parameter
/// whose first 16-bit row is `a` comes from: a register and column of
/// `elements`, none for padding ([`packed_layout`]; `packs` lists every
/// parameter packing two 16-bit rows per dword).
fn packed_half(
    elements: &[SigElement],
    packs: &[(u8, Option<u8>)],
    a: u8,
    h: u8,
) -> Option<(u8, u8)> {
    let layout = packed_layout(elements, packs);
    let (_, of) = layout.iter().find(|p| p.0 == a)?;
    of.iter().find_map(|&(dword, row, col, cols)| {
        let k = h.checked_sub(2 * dword)?;
        (k < cols).then_some((row, col + k))
    })
}

/// The pixel inputs packing two 16-bit rows per dword, as (first row,
/// second row) ([`packed_layout`]).
fn input_packs(io: &StageIo, inputs: &[SigElement]) -> Vec<(u8, Option<u8>)> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return Vec::new();
    };
    let partners = half_input_partners(io, inputs);
    entries
        .iter()
        .filter(|i| i.fp16 && i.pack2)
        .map(|i| {
            let b = if i.row2 != i.row {
                Some(i.row2)
            } else {
                partners.get(&i.row).copied()
            };
            (i.row, b)
        })
        .collect()
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]):
/// a packed row's partner is the next 16-bit row of the signature that no
/// output entry names, taken in row order.
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o
                .pack2
                .then(|| free.iter().position(|&r| r > o.row))
                .flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

/// Pixel inputs packing two 16-bit rows per dword whose second row the
/// metadata leaves out (`row2` = `row`): as for outputs
/// ([`half_outputs`]), the next 16-bit row of the signature that no input
/// entry names, taken in row order (rows the metadata does give are taken
/// first).
fn half_input_partners(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, u8> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    // Keep the first fp16 entry per row (matching the previous `.find()`
    // behaviour) so duplicate rows neither overwrite earlier ones nor consume
    // an extra free partner row.
    let mut seen = HashSet::new();
    let mut halves: Vec<_> = entries
        .iter()
        .filter(|i| i.fp16 && seen.insert(i.row))
        .collect();
    halves.sort_by_key(|i| i.row);
    // In the console's PsInput table, `row2` repeats `row` when no distinct
    // partner row is named; exclude both listed rows and explicitly claimed
    // partners (`row2 != row`) from the fallback pool.
    let packed = || entries.iter().filter(|i| i.fp16 && i.pack2);
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !entries.iter().any(|i| i.row == r))
        .filter(|&r| !packed().any(|i| i.row2 != i.row && i.row2 == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut missing: Vec<_> = packed().filter(|i| i.row2 == i.row).collect();
    missing.sort_by_key(|i| i.row);
    let mut partners = HashMap::new();
    for i in missing {
        if let Some(k) = free.iter().position(|&r| r > i.row) {
            partners.insert(i.row, free.remove(k));
        }
    }
    partners
}

/// What a sample or gather opcode adds to the plain form, by its name:
/// comparison (`_c`), explicit derivatives (`_d`, coarse `_cd`, 16-bit
/// `_g16`), a level (`_l`) or level 0 (`_lz`), a bias (`_b`), a level
/// clamp (`_cl`), a texel offset (`_o`).
#[derive(Debug, Clone, Copy, Default)]
struct SampleForm {
    gather: bool,
    c: bool,
    d: bool,
    g16: bool,
    l: bool,
    lz: bool,
    b: bool,
    cl: bool,
    o: bool,
}

fn sample_form(name: &str) -> Option<SampleForm> {
    let (gather, rest) = match name.strip_prefix("image_sample") {
        Some(rest) => (false, rest),
        None => (true, name.strip_prefix("image_gather4")?),
    };
    let mut f = SampleForm {
        gather,
        ..SampleForm::default()
    };
    for token in rest.split('_').skip(1) {
        let flag = match token {
            "c" => &mut f.c,
            "d" | "cd" => &mut f.d,
            "g16" => &mut f.g16,
            "l" => &mut f.l,
            "lz" => &mut f.lz,
            "b" => &mut f.b,
            "cl" => &mut f.cl,
            "o" => &mut f.o,
            _ => return None,
        };
        *flag = true;
    }
    Some(f)
}

/// Pixel inputs of 16-bit values (see [`Lower::half_inputs`]): a packed row's
/// partner is `row2` when distinct from `row`, else the next unlisted 16-bit
/// row of the signature before the next `fp16` input row, taken in row order.
fn half_inputs(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    // Keep the first fp16 entry per row (matching the previous `.find()`
    // behaviour) so duplicate rows neither overwrite earlier ones nor consume
    // an extra free partner row.
    let mut seen = HashSet::new();
    let mut halves: Vec<_> = entries
        .iter()
        .filter(|i| i.fp16 && seen.insert(i.row))
        .collect();
    halves.sort_by_key(|i| i.row);
    // In the console's PsInput table, `row2` repeats `row` when no distinct
    // partner row is named; exclude both listed rows and explicitly claimed
    // partners (`row2 != row`) from the fallback pool.
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| {
            !entries.iter().any(|i| i.row == r)
                && !halves
                    .iter()
                    .any(|i| i.pack2 && i.row2 != i.row && i.row2 == r)
        })
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut out = HashMap::with_capacity(halves.len());
    for (idx, i) in halves.iter().enumerate() {
        let next_fp16 = halves.get(idx + 1).map_or(u8::MAX, |n| n.row);
        // Only take a fallback partner before the next fp16 input row so an
        // earlier unpaired input cannot steal a later fp16 input's partner.
        let b = if !i.pack2 {
            None
        } else if i.row2 != i.row {
            Some(i.row2)
        } else {
            free.iter()
                .position(|&r| r > i.row && r < next_fp16)
                .map(|p| free.remove(p))
        };
        out.insert(i.row, (i.pack2, b));
    }
    out
}

/// Elements from multi-element 16-bit signature rows that overflow a 4-dword
/// `pack2` attribute (because each element starts on a whole dword) and spill
/// into free dwords of a subsequent `pack2` attribute.
fn half_input_spills(
    half_inputs: &HashMap<u8, (bool, Option<u8>)>,
    inputs: &[SigElement],
) -> HashMap<(u8, u8), (u8, u8)> {
    let row_elems = |r: u8| -> Vec<&SigElement> {
        let mut v: Vec<_> = inputs
            .iter()
            .filter(|e| e.half && e.row == Some(u32::from(r)))
            .collect();
        v.sort_by_key(|e| e.col);
        v
    };
    let extent = |r: u8| -> u8 {
        inputs
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
            .unwrap_or(0)
    };
    let mut rows: Vec<(u8, Option<u8>)> = half_inputs
        .iter()
        .filter_map(|(&a, &(pack2, b))| pack2.then_some((a, b)))
        .collect();
    rows.sort_by_key(|&(a, _)| a);
    let mut pending: Vec<(u8, u8, u8)> = Vec::new();
    let mut out = HashMap::new();
    for (a, b) in rows {
        let mut elems_a = row_elems(a);
        let dwords_b: u8 = b
            .map(|r| row_elems(r).iter().map(|e| e.cols.div_ceil(2)).sum())
            .unwrap_or(0);
        let mut dwords_a: u8 = elems_a.iter().map(|e| e.cols.div_ceil(2)).sum();
        while elems_a.len() > 1 && dwords_a + dwords_b > 4 {
            let spill = elems_a.pop().unwrap();
            dwords_a -= spill.cols.div_ceil(2);
            pending.push((a, spill.col, spill.cols));
        }
        let mut used = extent(a).div_ceil(2) + b.map_or(0, |r| extent(r).div_ceil(2));
        if used < 4 && !pending.is_empty() {
            let mut idx = 0;
            while idx < pending.len() && used < 4 {
                let (sr, sc, scols) = pending[idx];
                let need = scols.div_ceil(2);
                if used + need <= 4 {
                    for k in 0..scols {
                        out.insert((a, used * 2 + k), (sr, sc + k));
                    }
                    used += need;
                    pending.remove(idx);
                } else {
                    idx += 1;
                }
            }
        }
    }
    out
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

#[allow(dead_code)]
/// A byte or halfword LDS access: `ds_read_{u,i}{8,16}[_d16[_hi]]` and
/// `ds_write_b{8,16}[_d16_hi]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SubDword {
    write: bool,
    bits: u32,
    /// Reads: sign-extended.
    signed: bool,
    /// Reads: the result is 16 bits wide and goes into one half of the VGPR,
    /// the other half kept (`Some(true)`: the high half). Writes: the data
    /// comes from the high half (`Some(true)`).
    d16: Option<bool>,
}

impl SubDword {
    #[allow(dead_code)]
    fn of(name: &str) -> Option<Self> {
        let (write, rest) = match name.strip_prefix("ds_read_") {
            Some(r) => (false, r),
            None => (true, name.strip_prefix("ds_write_")?),
        };
        let (ty, d16) = match rest.split_once("_d16") {
            None => (rest, None),
            Some((t, "")) if !write => (t, Some(false)),
            Some((t, "_hi")) => (t, Some(true)),
            Some(_) => return None,
        };
        let (signed, bits) = match (write, ty) {
            (false, "u8") => (false, 8),
            (false, "i8") => (true, 8),
            (false, "u16") => (false, 16),
            (false, "i16") => (true, 16),
            (true, "b8") => (false, 8),
            (true, "b16") => (false, 16),
            _ => return None,
        };
        Some(Self {
            write,
            bits,
            signed,
            d16,
        })
    }
}

#[allow(dead_code)]
impl<'a> Lower<'a> {
    fn new(
        x: &'a Parts<'a>, bad: &'a HashMap<Value, Why>,
        uint_srvs: &'static [(u8, u32, u32)],
        numeric_selects: bool,
    ) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let dead = crate::sgpr_spill::dead(p);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let wide_lds =
            insts().any(|i| ds_integer_atomic(i.name()).is_some_and(|(_, wide, _)| wide));
        let mut console_lanes = false;
        // LDS accesses get a group barrier each (a console wave split in two host waves)
        let mut split_lds = false;
        // with `split_lds`, LDS accesses in lane branches: the blocks every lane runs, and
        // where the host waves meet around those branches (`lds_meetings`)
        let mut lds_meet: Option<(Vec<bool>, Vec<(bool, bool)>)> = None;
        // a console wave64 program in two host waves of 32 (`exchange`)
        let mut exchange: Option<(u32, bool, Vec<bool>)> = None;
        // a group of several console waves: the flag dwords a console wave's two host waves
        // meet through (`pair_meet`)
        let mut pair_flags: Option<u32> = None;
        // a console wave64 program in host waves of 32: the LDS dword of `wave_base`'s counter
        let mut halves: Option<u32> = None;
        // a program the per-lane form cannot lower but the console flow can
        // ([`CONSOLE_FLOW`]): lane reads and writes whose lanes branch differently
        let mut wants_console_flow = false;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            // Masks held across a region where lanes part ways (a mask the code
            // builds under its own lane tests and carries in scalar registers):
            // the lane tests its own bit by its console lane, which a host wave
            // of half the lanes cannot answer for the whole.
            // Masks built across a loop's rounds (a mask the code accumulates
            // where lanes leave at different times): each lane's bit stands for
            // its console lane, which a host wave of half the lanes cannot
            // answer for the whole – and each half's accumulation rounds its
            // own lanes' departures, so the halves' exchange cannot join them.
            // A scalar written on a loop and read off it counts the turns the
            // loop took, which each half takes for itself; masks and scalars a
            // loop uses and leaves alone (a waterfall's worklist, the lane it
            // picks) stay each half's own.
            let masks_held = x.native.wave64 && {
                // The blocks on a cycle with `b`: reached from it and reaching
                // it back (its natural loop's blocks).
                let loop_set = |b: usize| -> Vec<bool> {
                    let n = p.blocks.len();
                    let mut fwd = vec![false; n];
                    let mut stack = vec![b];
                    while let Some(q) = stack.pop() {
                        for s in p.blocks[q].exit.successors() {
                            if !fwd[s] {
                                fwd[s] = true;
                                stack.push(s);
                            }
                        }
                    }
                    let mut back = vec![false; n];
                    let mut stack = vec![b];
                    while let Some(q) = stack.pop() {
                        for &q in &p.blocks[q].preds {
                            if !back[q] {
                                back[q] = true;
                                stack.push(q);
                            }
                        }
                    }
                    (0..n).map(|q| fwd[q] && back[q]).collect()
                };
                let s_written = |blk: &Block| {
                    blk.insts
                        .iter()
                        .flat_map(|i| i.writes.iter().map(|&(r, _)| r))
                        .chain(blk.phis.iter().map(|phi| phi.reg))
                        .filter_map(|r| match r {
                            Reg::S(r) if r != EXEC && r != EXEC + 1 => Some(r),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                };
                p.blocks.iter().enumerate().any(|(b, _)| {
                    let set = loop_set(b);
                    if !set[b] {
                        return false;
                    }
                    let mut held = std::collections::HashSet::new();
                    for (q, blk) in p.blocks.iter().enumerate() {
                        if set[q] {
                            held.extend(s_written(blk));
                        }
                    }
                    !held.is_empty()
                        && p.blocks.iter().enumerate().any(|(c, blk)| {
                            !set[c]
                                && blk.insts.iter().any(|i| {
                                    i.reads.iter().any(|&(r, _)| {
                                        matches!(r, Reg::S(r) if held.contains(&r))
                                    })
                                })
                        })
                })
            };
            let all_lane = all_lane_blocks(x.st);
            console_lanes = x.console_cf
                || masks_held
                || insts().any(|i| {
                let n = i.name();
                n.starts_with("ds_swizzle")
                    || n.contains("permute")
                    || n.starts_with("v_permlane")
                    || n.starts_with("v_mbcnt")
                    || implicit_derivatives(n)
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
                    || reads_mask_as_data(x.st, i)
                    || numeric_selects && numeric_select(x.st, i)
            }) || lane_constant_masks(x.st)
                // (a branch the wave takes together asks the wave)
                || !x.st.wide.is_empty();
            let mut wave = if x.native.wave64 { 64 } else { 32 };
            let group = cs.threads.iter().product::<u32>();
            let one_wave = group <= wave;
            // A host whose waves are at most 32 lanes (NVIDIA) refuses `WaveSize(64)`. A console
            // wave64 program whose cross-lane code stays inside 32-lane halves – DPP rows,
            // `ds_swizzle`, `v_permlane16` / `v_permlanex16`, which the ISA confines there – runs
            // exactly in waves of 32: the thread relabelling makes each host wave one half of a
            // console wave (lanes 32k..32k+31). Not when it crosses halves (`v_mbcnt`,
            // `v_readlane`, `v_writelane`, `ds_(b)permute`, masks as numbers or lane-constant
            // masks), nor when LDS may be shared within the console wave without barriers (a
            // group over 32 threads with LDS): those still need 64-lane waves.
            let crosses = masks_held
                || insts().any(|i| {
                let n = i.name();
                n.starts_with("v_mbcnt")
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || n.starts_with("ds_permute")
                    || n == "image_sample"
                    || reads_mask_as_data(x.st, i)
                    || numeric_selects && numeric_select(x.st, i)
            }) || lane_constant_masks(x.st)
                || !x.st.wide.is_empty();
            let no_64 = host_max_wave() == 32 || !x.target.runs(64) && x.target.runs(32);
            if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
                eprintln!("WAVE64-GATE wave={wave} console_lanes={console_lanes} masks_held={masks_held} crosses={crosses} no_64={no_64} lds={} group={group}", s.lds_dwords);
                for e in &x.st.exits {
                    if let LaneExit::Branch { when, .. } = *e {
                        eprintln!("WAVE64-EXIT when={when:?} support={:?}", x.st.bools.support(when));
                    }
                }
                for (b, g) in x.st.guards.iter().enumerate() {
                    for (i, g) in g.iter().enumerate() {
                        if let Some(g) = g {
                            eprintln!("WAVE64-GUARD b{b} i{i} support={:?}", x.st.bools.support(*g));
                        }
                    }
                }
                for (v, b) in &x.st.masks {
                    eprintln!("WAVE64-MASK {v:?} -> {b:?} support={:?}", x.st.bools.support(*b));
                }
                for b in 0..p.blocks.len() {
                    eprintln!("WAVE64-SECTIONS b{b} = {}", super::wave::sections(p, b).len());
                }
            }
            if wave == 64 && no_64 && (console_lanes || one_wave && s.lds_dwords > 0) {
                if !crosses && (s.lds_dwords == 0 || group <= 32) {
                    wave = 32;
                } else {
                    // (Groups of several console waves too: each console wave is two host
                    // waves, which exchange through their own pair of dwords.)
                    // LDS the console shares within its one wave without barriers: split into
                    // two waves of 32, a group barrier before every LDS access keeps their order
                    // – where every thread of the group gets there (no lane branch around it,
                    // no guard), else a barrier would hang the group
                    let lds = |i: &Inst| {
                        matches!(i.fields, Fields::Ds { .. })
                            && !(i.name().starts_with("ds_swizzle") || i.name().contains("permute"))
                    };
                    if let Some(all) = all_lane.clone() {
                        let ok = p.blocks.iter().enumerate().all(|(b, blk)| {
                            blk.insts.iter().all(|inst| {
                                !lds(inst) || all.get(b).copied().unwrap_or(false)
                            })
                        });
                        let mut lds_ok = !(s.lds_dwords > 0 && group > 32) || ok;
                        if !lds_ok && let Some(meet) = lds_meetings(p, x.st, &all) {
                            lds_meet = Some((all.clone(), meet));
                            lds_ok = true;
                        }
                        if !lds_ok {
                            // (the blocks whose LDS accesses not every lane runs: with the
                            // branches around them taken wave-wide they get their barriers)
                            for (b, blk) in p.blocks.iter().enumerate() {
                                if !all.get(b).copied().unwrap_or(false) && blk.insts.iter().any(lds) {
                                    need(b);
                                }
                            }
                        }
                        // A mask held where not every lane runs stays with
                        // the console wave: the halves' exchange reaches only
                        // where both halves run, not the mask a lane branch made.
                        if lds_ok && !masks_held {
                            wave = 32;
                            split_lds = s.lds_dwords > 0 && group > 32;
                            if crosses && !x.console_cf {
                                // lanes 32..63 are the other host wave: masks, lane reads and lane
                                // numbers go through two exchange dwords past the program's LDS
                                exchange = Some((s.lds_dwords, group <= 32, all));
                                s.lds_dwords += 2 * group.div_ceil(64);
                            }
                        }
                    }
                }
                // `VICTORY_WAVE_WHY=1`: why a program keeps 64-lane waves (one line per reason)
                if wave == 64 && std::env::var_os("VICTORY_WAVE_WHY").is_some() {
                    let mut why: Vec<&str> = Vec::new();
                    for i in insts() {
                        let n = i.name();
                        let r = if n.starts_with("v_mbcnt") {
                            "v_mbcnt"
                        } else if n == "v_readlane_b32" {
                            "v_readlane"
                        } else if n == "v_writelane_b32" {
                            "v_writelane"
                        } else if n.starts_with("ds_permute") {
                            "ds_permute"
                        } else if reads_mask_as_data(x.st, i) {
                            "mask as data"
                        } else {
                            continue;
                        };
                        if !why.contains(&r) {
                            why.push(r);
                        }
                    }
                    if lane_constant_masks(x.st) {
                        why.push("lane-constant mask");
                    }
                    if s.lds_dwords > 0 && group > 32 {
                        why.push("LDS in a group over 32");
                        match all_lane_blocks(x.st) {
                            None => why.push("an indirect exit"),
                            Some(all) if lds_meetings(p, x.st, &all).is_none() => {
                                why.push("an LDS read and write in one lane-branch region")
                            }
                            Some(_) => {}
                        }
                    }
                    why.sort_unstable();
                    eprintln!("WAVE64-WHY group={group} lds={} [{}]", s.lds_dwords > 0, why.join(", "));
                }
            }
            if wave == 32 && x.native.wave64 && group > 64 {
                // A group of several console waves: four flag dwords per
                // console wave (each half's "arrived" and "done"), past the
                // program's LDS and any exchange dwords.
                pair_flags = Some(s.lds_dwords);
                s.lds_dwords += 4 * group.div_ceil(64);
            }
            if wave == 32 && x.native.wave64 && !x.console_cf {
                // console threads go to host waves in the order the waves take them (`wave_base`).
                // (Not in the console flow: it numbers lanes in group-flat order, `console_flat`.)
                halves = Some(s.lds_dwords);
                s.lds_dwords += 1;
            }
            if console_lanes || one_wave && s.lds_dwords > 0 {
                s.wave_size = Some(wave);
            }
            if x.console_cf && x.native.wave64 && !x.target.runs(64) {
                // In halves only where the host cannot run the console wave.
                s.wave_size = Some(32);
            }
            // Instructions that run (not the spills forwarding leaves dead).
            let live = || {
                p.blocks.iter().enumerate().flat_map(|(b, block)| {
                    let dead = &dead;
                    block
                        .insts
                        .iter()
                        .enumerate()
                        .filter(move |&(i, _)| !dead.contains(&(b, i)))
                        .map(|(_, inst)| inst)
                })
            };
            // The console flow ([`CONSOLE_FLOW`]), for a wave64 program whose
            // lanes read their whole console wave on a host without 64-lane
            // waves: a group of one console wave, no LDS, no wave-wide sections.
            wants_console_flow = console_lanes
                && crosses
                && x.native.wave64
                && !x.target.runs(64)
                && x.target.runs(32)
                && cs.lds_bytes() == 0
                && group <= 64
                && !insts().any(|i| i.name().contains("permute"))
                && (0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty());
            // On a host that runs the console's wave size: lane reads and
            // writes in a program whose lanes branch differently (see
            // [`CONSOLE_FLOW`]).
            let uniform = |when: Bool| {
                x.st.bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(_)))
            };
            wants_console_flow |= x.target.runs(if x.native.wave64 { 64 } else { 32 })
                && live().any(|i| matches!(i.name(), "v_readlane_b32" | "v_writelane_b32"))
                && x.st
                    .exits
                    .iter()
                    .any(|e| matches!(*e, LaneExit::Branch { when, .. } if !uniform(when)))
                && !insts().any(|i| i.name().contains("permute"))
                && (0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty());
        }
        // The title's barriers a thread of the group may never reach, which
        // leaves the others waiting on the host: those in blocks some lanes
        // do not run, and all of them in a program with a loop closed by a
        // plain jump (a lane still in it at the iteration limit leaves the
        // program there, `abort_block`).
        let fence_only: HashSet<usize> = if insts().any(|i| i.name() == "s_barrier") {
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut place = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                place[b] = n;
            }
            let back = |from: usize, to: usize| {
                let at = |b: usize| place.get(b).copied().unwrap_or(usize::MAX);
                at(from) != usize::MAX && at(to) <= at(from)
            };
            let cut_short = x
                .st
                .exits
                .iter()
                .enumerate()
                .any(|(b, e)| matches!(*e, LaneExit::Jump(t) if back(b, t)));
            let all = all_lane_blocks(x.st);
            (0..p.blocks.len())
                .filter(|&b| p.blocks[b].insts.iter().any(|i| i.name() == "s_barrier"))
                .filter(|&b| {
                    cut_short
                        || all
                            .as_ref()
                            .is_none_or(|all| !all.get(b).copied().unwrap_or(false))
                })
                .collect()
        } else {
            HashSet::new()
        };
        let mut no_barrier = HashSet::new();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        let entry = s.entry;
        Lower {
            s,
            x,
            bad,
            uint_srvs,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            body_entry: entry,
            partial_active: None,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            packed_phis: HashSet::new(),
            stale: HashSet::new(),
            descriptor_word_uses: Default::default(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            half_views: Default::default(),
            halves_only: Default::default(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: {
                let mut m: HashMap<Site, &Interp> = x.io.interps.iter().map(|(s, i)| (*s, i)).collect();
                // A first half whose result is also read as data (an f16 second half keeps
                // the other half of its dst, PS_DeferredTextured): evaluated as the pair it
                // starts. Approximate for those other bits (P0 + i P10 on the console); a
                // first half only the second half reads is gone before lowering.
                // Only in the second translation of a program the first
                // one refused for it ([`with_first_half_pairs`]).
                if FIRST_HALF_PAIRS.get() && std::env::var_os("VICTORY_FIRST_HALF").is_none_or(|v| v != "0") {
                    for (_, i) in &x.io.interps {
                        if let Interp::Interpolated { first, .. } = i {
                            for s in first {
                                m.entry(*s).or_insert(i);
                            }
                        }
                    }
                }
                m
            },
            handles: HashMap::new(),
            size_handles: HashMap::new(),
            strides: HashMap::new(),
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            ordered: None,
            src0: None,
            console_lanes,
            numeric_selects,
            split: exchange.is_some()
                || halves.is_some()
                || (x.console_cf && x.native.wave64 && !x.target.runs(64)),
            console_cf: x.console_cf,
            wants_console_flow,
            console_flat: None,
            xchg: None,
            split_lds,
            wide_lds,
            lds_meet,
            local: exchange.is_some() && LOCAL_WAVES.get(),
            local_ok: false,
            local_any: LOCAL_ANY.get(),
            uniform: match &exchange {
                Some((_, false, all)) if !LOCAL_WAVES.get() => lane_uniform(p, x.st, all),
                _ => HashSet::new(),
            },
            own_bits: match &exchange {
                Some((_, false, _)) if !LOCAL_WAVES.get() && own_bits_on() => {
                    super::own_bits::find(p, x.st, std::env::var_os("VICTORY_WAVE_WHY").is_some())
                }
                _ => Default::default(),
            },
            exchange,
            pair_flags,
            clane: None,
            paired: None,
            pair_lanes: None,
            flag_pair: None,
            pair: None,
            halves,
            wave_base: None,
            flat: None,
            no_barrier,
            fence_only,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            half_inputs: {
                
                half_inputs(&x.native.io, &x.inputs)
            },
            half_input_spills: half_input_spills(&half_inputs(&x.native.io, &x.inputs), &x.inputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs | HwStage::Ms => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            mask_scc: mask_sccs(x.st, p),
            mask_numbers: mask_numbers(x.st, p),
            balloting: HashSet::new(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            one_lane: Default::default(),
            coherent: None,
            made: Default::default(),
            code_table: None,
            scratch: None,
            counters: HashMap::new(),
            loop_exits: HashMap::new(),
            loop_bodies: HashMap::new(),
            finals: HashMap::new(),
            abort: None,
            lane_paths: Vec::new(),
            pending_paths: Vec::new(),
            dead,
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| if a.includes_base { 0 } else { a.offset & !3 })
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// Private (scratch) memory, `scratch_load_*` and `scratch_store_*`.
    /// Each lane has its own: the address (VGPR, SGPR or none, plus the
    /// offset) is a byte offset into the lane's memory, and the hardware
    /// interleaves the lanes' dwords itself (RDNA 2 ISA, "Scratch
    /// (Private)"; LLVM AMDGPUUsage, "Private Segment"). On the host it is
    /// an array of dwords per lane (`alloca`, as DXC makes local arrays).
    // shortcut: a dword access is taken as aligned, a byte or short one
    // not to cross a dword (console compilers keep private data aligned).
    fn scratch(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            lds,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *lds {
            return Err("a scratch load into LDS");
        }
        let name = inst.name();
        let (load, rest) = match name.strip_prefix("scratch_load_") {
            Some(rest) => (true, rest),
            None => (false, name.strip_prefix("scratch_store_").ok_or(name)?),
        };
        let width = scratch_width(rest).ok_or(name)?;
        let (signed, hi) = (rest.starts_with('s'), rest.ends_with("_hi"));
        let guard = self.guard(b, i)?;
        let base = match (addr, saddr) {
            (Some(o), _) => self.vgprs(inst, o, 1)?[0],
            (None, Some(o)) => self.operand(inst, o)?,
            (None, None) => self.s.m.i32(0),
        };
        let (arr, dwords) = match self.scratch {
            Some(a) => a,
            None => {
                let dwords = scratch_bytes(self.x.p).div_ceil(4);
                let ty = self.s.m.array(dwords, self.s.t.i32);
                let a = (self.s.m.alloca(self.s.entry, ty), dwords);
                self.scratch = Some(a);
                a
            }
        };
        let c = self.cur;
        let off = self.s.m.i32(i32::from(*offset) as u32);
        let at = self.s.m.bin(c, Bin::Add, base, off);
        let two = self.s.m.i32(2);
        let first = self.s.m.bin(c, Bin::LShr, at, two);
        // Dword `k` of the access: its element (the first when out of
        // range) and whether it is in range.
        let element = |l: &mut Self, k: u32| -> (V, V) {
            let c = l.cur;
            let k = l.s.m.i32(k);
            let index = l.s.m.bin(c, Bin::Add, first, k);
            let n = l.s.m.i32(dwords);
            let inside = l.s.m.cmp(c, pred::IULT, index, n);
            let zero = l.s.m.i32(0);
            let index = l.s.m.select(c, inside, index, zero);
            (l.s.m.element(c, arr, index), inside)
        };
        let three = self.s.m.i32(3);
        let byte = self.s.m.bin(c, Bin::And, at, three);
        let shift = self.s.m.bin(c, Bin::Shl, byte, three);
        if load {
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a scratch load result");
            };
            let zero = self.s.m.i32(0);
            if width >= 4 {
                for k in 0..width / 4 {
                    let (p, inside) = element(self, k);
                    let x = self.s.m.load(c, p);
                    let x = self.s.m.select(c, inside, x, zero);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let (p, inside) = element(self, 0);
            let w = self.s.m.load(c, p);
            let w = self.s.m.select(c, inside, w, zero);
            let x = self.s.m.bin(c, Bin::LShr, w, shift);
            let left = self.s.m.i32(32 - 8 * width);
            let x = self.s.m.bin(c, Bin::Shl, x, left);
            let mut x = self
                .s
                .m
                .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left);
            if rest.contains("d16") {
                // Into one half of the VGPR (extended to 16 bits); the
                // other half keeps its bits.
                let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    _ => self.s.m.i32(0),
                };
                let low = self.s.m.i32(0xFFFF);
                let h = self.s.m.bin(c, Bin::And, x, low);
                let (h, keep) = if hi {
                    let sixteen = self.s.m.i32(16);
                    (self.s.m.bin(c, Bin::Shl, h, sixteen), low)
                } else {
                    (h, self.s.m.i32(0xFFFF_0000))
                };
                let o = self.s.m.bin(c, Bin::And, old, keep);
                x = self.s.m.bin(c, Bin::Or, o, h);
            }
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        let data = data.as_ref().ok_or("scratch store data")?;
        let vals = self.vgprs(inst, data, width.div_ceil(4) as u8)?;
        for (k, mut x) in vals.into_iter().enumerate() {
            let (p, inside) = element(self, k as u32);
            let g = match guard {
                Some(g) => self.s.m.bin(self.cur, Bin::And, g, inside),
                None => inside,
            };
            let at = self.begin_guarded(Some(g));
            let c = self.cur;
            if width < 4 {
                // The byte or short within its dword (`_d16_hi`: from the
                // high half of the VGPR).
                if hi {
                    let sixteen = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, sixteen);
                }
                let mask = self.s.m.i32((1 << (8 * width)) - 1);
                let mask = self.s.m.bin(c, Bin::Shl, mask, shift);
                let x = self.s.m.bin(c, Bin::Shl, x, shift);
                let x = self.s.m.bin(c, Bin::And, x, mask);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, mask, ones);
                let old = self.s.m.load(c, p);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let new = self.s.m.bin(c, Bin::Or, old, x);
                self.s.m.store(c, p, new);
            } else {
                self.s.m.store(c, p, x);
            }
            self.end_guarded(at, &[]);
        }
        Ok(())
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, lds, .. } = &inst.fields else {
            unreachable!()
        };
        if *lds {
            return Err("a global load directly into LDS");
        }
        let name = inst.name();
        if self.scalar.get(&(b, i)).is_some_and(|a| a.param == MEMORY) {
            return self.memory_flat(inst, b, i);
        }
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global store or atomic without root UAV address support")?;
        let sub = subword_load(rest);
        let n = match rest {
            "dword" => 1,
            "dwordx2" => 2,
            "dwordx3" => 3,
            "dwordx4" => 4,
            "dword_addtid" => return Err("a global load with a console thread ID offset"),
            _ if sub.is_some() => 1,
            _ => return Err(name),
        };
        let _a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.effect_guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        let Some(&a) = self.scalar.get(&(b, i)) else {
            if sub.is_none() && crate::resource::traces_rays(self.x.p) {
                // shortcut: out of the acceleration structure.
                let z = self.s.m.i32(0);
                for k in 0..n {
                    self.write(inst, Reg::V(reg + k as u8), z, guard);
                }
                return Ok(());
            }
            return Err("an untracked global address");
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = self.dynamic_term(&a)? {
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            if let Some(bias) = self.root_bias(a.param) {
                off = self.s.m.bin(c, Bin::Add, off, bias);
            }
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let u = l.s.m.undef(l.s.t.i32);
                    Ok(l.raw_dword(h, [byte, u]))
                }
            }
        };
        if let (None, Some(h), true) = (sub, h, n > 1 && Self::wide_loads()) {
            for (k, x) in self.raw_dwords(h, off, n).into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if let Some((width, signed, half)) = sub {
            let x = self.read_subword(off, width, signed, dword)?;
            self.write_subword_result(inst, reg, x, half, guard)?;
        } else {
            for k in 0..n {
                let four = self.s.m.i32(4 * k);
                let at = self.s.m.bin(c, Bin::Add, off, four);
                let x = dword(self, at)?;
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if self.dead.contains(&(b, i)) {
                    continue;
                }
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// The instructions of the whole-wave section from `start` to `end`:
    /// the values that leave it are made from what the simulation found
    /// them to be. A value it found no shape for, written by a step that
    /// does not look at other lanes (scalar code the compiler scheduled
    /// into the section, or lane-local vector code), is lowered as outside
    /// a section: the lanes active on entry compute it as they would under
    /// their own EXEC, and only they run on the host.
    fn section(&mut self, b: usize, start: usize, end: usize) -> Result<(), Stop> {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let insts = &self.x.p.blocks[b].insts;
        let inside: HashSet<Value> = insts[start + 1..end]
            .iter()
            .flat_map(|inst| inst.writes.iter().map(|w| w.1))
            .collect();
        let mut need: HashSet<Value> = inside
            .iter()
            .copied()
            .filter(|w| self.section_out.contains(w))
            .collect();
        let mut plain = vec![false; end - start];
        for j in (start + 1..end).rev() {
            let inst = &insts[j];
            let shapes = &self.sections[&b][&start].shapes;
            let wanted = inst
                .writes
                .iter()
                .any(|&(_, w)| need.contains(&w) && !shapes.contains_key(&w));
            if wanted && lane_local(inst) {
                plain[j - start] = true;
                need.extend(
                    inst.reads
                        .iter()
                        .map(|r| r.1)
                        .filter(|r| inside.contains(r)),
                );
            }
        }
        for j in start + 1..end {
            if plain[j - start] {
                self.inst(b, j)?;
            }
            for &(_, w) in &insts[j].writes {
                let shapes = &self.sections[&b][&start].shapes;
                let mut shape = need.contains(&w).then(|| shapes.get(&w).copied()).flatten();
                // Lanes that do not agree, of which one is read after.
                if shape.is_none()
                    && need.contains(&w)
                    && let Some(lane) = self.lane_read(w, b, start, end)
                    && let Some(sh) = self.sections[&b][&start].lane_shape(w, lane)
                {
                    shape = Some(sh);
                    self.one_lane.insert(w);
                }
                let own = self.own_bits.totals.contains(&w);
                let val = match (shape.map(|sh| self.shape(sh, own)), plain[j - start]) {
                    (Some(Ok(x)), _) => Val::I32(x),
                    (_, true) => continue,
                    (Some(Err(why)), false) => Val::Missing(why),
                    (None, false) => Val::Missing(UNKNOWN),
                };
                self.vals.insert(w, val);
            }
        }
        Ok(())
    }

    /// The one lane of `w` read outside the section from `start` to `end` of
    /// block `b`, if nothing else reads it there.
    fn lane_read(&self, w: Value, b: usize, start: usize, end: usize) -> Option<u32> {
        let mut found = None;
        for (ob, block) in self.x.p.blocks.iter().enumerate() {
            if block.phis.iter().any(|phi| phi.args.contains(&w)) {
                return None;
            }
            for (i, inst) in block.insts.iter().enumerate() {
                if ob == b && i > start && i < end || !inst.reads.iter().any(|r| r.1 == w) {
                    continue;
                }
                let Fields::Valu(v) = &inst.fields else {
                    return None;
                };
                let lane = match (inst.name(), v.src.get(1)) {
                    ("v_readlane_b32", Some(&Operand::Const(c))) => crate::resource::inline(c)?,
                    _ => return None,
                };
                if found.is_some_and(|f| f != lane) {
                    return None;
                }
                found = Some(lane);
            }
        }
        found
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    ///
    /// `own`: a wave OR that only steers a loop over each lane's own bits
    /// ([`super::own_bits`]); where two host waves cannot exchange their halves' ORs it is
    /// this host wave's.
    fn shape(&mut self, shape: super::wave::Shape, own: bool) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::ActiveF64(lo, hi, mask, high) => {
                return self.active_f64_sum(lo, hi, mask, high);
            }
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let mut on = self.boolean(m, c)?;
                // (EXEC on entry leaves out the threads past a DispatchX count)
                if let Some(active) = self.partial_active
                    && self.x.st.active.contains(&mask)
                {
                    on = self.s.m.bin(c, Bin::And, on, active);
                }
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        // Mesh: no host wave size can be declared, so a reduction is over this host wave
        // only. A minimum or maximum is taken that way: the compiler's "next distinct
        // value" loop (the wave's minimum, then the lanes that hold it) serves every
        // lane however the lanes are divided into waves. A sum, a bit reduction or a
        // prefix would be of part of a console wave: still refused.
        if self.mask_data == MaskData::None
            && (prefix.is_some()
                || !matches!(
                    op,
                    Op::MinU | Op::MaxU | Op::MinI | Op::MaxI | Op::MinF | Op::MaxF
                ))
        {
            return Err("a wave reduction in a mesh shader");
        }
        if self.split && self.exchange.is_none() {
            return Err("a wave reduction in a wave64 program split in halves");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let mut on = self.boolean(m, c)?;
        if let Some(active) = self.partial_active
            && self.x.st.active.contains(&mask)
        {
            on = self.s.m.bin(c, Bin::And, on, active);
        }
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        if op == Op::AddH {
            return match prefix {
                Some(_) => Err("a wave prefix of an operation other than a sum"),
                None => self.half_sum(x),
            };
        }
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF | Op::AddH, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF | Op::AddH => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        // two host waves of 32 hold the console's 64 lanes: their results combine
        let r = if self.local || self.exchange.as_ref().is_some_and(|e| e.1) {
            // this host wave's own result (and prefix): each wave on its own, or a group of
            // 32 threads or fewer, which one host wave holds whole (combining with the 0 a
            // missing half "gives" broke AND, MIN and signed MAX)
            r
        } else if own && prefix.is_none() && op == Op::Or && !self.all_lanes(self.cur_ir) {
            // each thread walks the bits of this value and works only at its own, which the
            // host wave's OR holds: the same run as with the console wave's OR
            r
        } else if self.exchange.is_some() {
            if !self.all_lanes(self.cur_ir) {
                need(self.cur_ir);
                return Err("a wave reduction where not every lane runs");
            }
            let bits = |l: &mut Self, v: V| if float { l.as_int(v) } else { v };
            let back = |l: &mut Self, v: V| if float { l.as_float(v) } else { v };
            match prefix {
                Some(_) => {
                    // the upper half adds the lower half's total to its prefix
                    let (k0, ku) = (self.s.m.i8(0), self.s.m.i8(u8::from(!float)));
                    let total = self.s.op(c, &format!("waveActiveOp.{sfx}"), Attr::NoUnwind, ty, 119, &[arg, k0, ku]);
                    let tb = bits(self, total);
                    let c = self.cur;
                    let (c, (lo, _)) = self.exchange_halves(c, tb);
                    self.cur = c;
                    let lo = back(self, lo);
                    let cl = self.console_lane();
                    let k32 = self.s.m.i32(32);
                    let upper = self.s.m.cmp(c, pred::IUGE, cl, k32);
                    let zero = if float { self.s.m.f32(0) } else { self.s.m.i32(0) };
                    let add = self.s.m.select(c, upper, lo, zero);
                    self.s.m.bin(c, Bin::Add, r, add)
                }
                None => {
                    let rb = bits(self, r);
                    let c = self.cur;
                    let (c, (lo, hi)) = self.exchange_halves(c, rb);
                    self.cur = c;
                    let (lo, hi) = (back(self, lo), back(self, hi));
                    match op {
                        Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
                        Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
                        Op::And => self.s.m.bin(c, Bin::And, lo, hi),
                        Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
                        _ => {
                            // FMax 35, FMin 36, IMax 37, IMin 38, UMax 39, UMin 40
                            let code = match op {
                                Op::MaxF => 35,
                                Op::MinF => 36,
                                Op::MinU => 40,
                                Op::MaxU => 39,
                                Op::MinI => 38,
                                _ => 37,
                            };
                            self.s.op(c, &format!("binary.{sfx}"), Attr::ReadNone, ty, code, &[lo, hi])
                        }
                    }
                }
            }
        } else {
            r
        };
        Ok(if float { self.as_int(r) } else { r })
    }

    /// `WaveActiveSum` of 64-bit floats (`f64(lo, hi)` in active lanes, `+0.0` elsewhere),
    /// returning the low (`!high`) or high (`high`) 32-bit word.
    fn active_f64_sum(
        &mut self,
        lo: Value,
        hi: Value,
        mask: Value,
        high: bool,
    ) -> Result<V, Why> {
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let mut on = self.boolean(m, c)?;
        if let Some(active) = self.partial_active
            && self.x.st.active.contains(&mask)
        {
            on = self.s.m.bin(c, Bin::And, on, active);
        }
        let lo = self.i32_of(lo)?;
        let hi = self.i32_of(hi)?;
        let z = self.s.m.i32(0);
        let lo = self.s.m.select(c, on, lo, z);
        let hi = self.s.m.select(c, on, hi, z);
        self.s.flags |= flags::WAVE_OPS | flags::DOUBLES;
        let f64t = self.s.m.double();
        let i32t = self.s.t.i32;
        let split_ty = self
            .s
            .m
            .named_struct("dx.types.splitdouble", vec![i32t, i32t]);
        let arg = self
            .s
            .op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[lo, hi]);
        let (k0, k0b) = (self.s.m.i8(0), self.s.m.i8(0));
        let sum = self
            .s
            .op(c, "waveActiveOp.f64", Attr::NoUnwind, f64t, 119, &[arg, k0, k0b]);
        let split = self
            .s
            .op(c, "splitDouble.f64", Attr::ReadNone, split_ty, 102, &[sum]);
        let mut r_lo = self.s.m.extract(c, split, 0);
        let mut r_hi = self.s.m.extract(c, split, 1);
        if self.exchange.is_some() {
            if !self.all_lanes(self.cur_ir) {
                need(self.cur_ir);
                return Err("a wave reduction where not every lane runs");
            }
            let (c, (lo0, lo1)) = self.exchange_halves(self.cur, r_lo);
            let (c, (hi0, hi1)) = self.exchange_halves(c, r_hi);
            self.cur = c;
            let d0 = self
                .s
                .op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[lo0, hi0]);
            let d1 = self
                .s
                .op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[lo1, hi1]);
            let total = self.s.m.bin(c, Bin::Add, d0, d1);
            let split = self
                .s
                .op(c, "splitDouble.f64", Attr::ReadNone, split_ty, 102, &[total]);
            r_lo = self.s.m.extract(c, split, 0);
            r_hi = self.s.m.extract(c, split, 1);
        }
        Ok(if high { r_hi } else { r_lo })
    }

    /// The sum of two 16-bit floats (the low halves of `a` and `b`) as the console's
    /// `v_add_f16` gives it: added in f32 and rounded to 16 bits, to nearest
    /// ([`Self::f16_nearest`]).
    fn add_halves(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let f32t = self.s.t.f32;
        let low = self.s.m.i32(0xFFFF);
        let wide = |l: &mut Self, x: V| {
            let h = l.s.m.bin(c, Bin::And, x, low);
            l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h])
        };
        let (fa, fb) = (wide(self, a), wide(self, b));
        let f = self.s.m.bin(c, Bin::Add, fa, fb);
        self.f16_nearest(f)
    }

    fn run(mut self, partial: bool) -> Result<Lowered, Stop> {
        if partial {
            self.enter_partial_dispatch();
        }
        self.lower_ir()?;
        self.fill_phis()?;
        if self.s.pixel {
            settle_position_mode(&mut self.s.inputs);
            settle_row_modes(&mut self.s.inputs, true).map_err(|w| unsupported(0, w))?;
        } else {
            settle_row_modes(&mut self.s.outputs, false).map_err(|w| unsupported(0, w))?;
        }
        let user_data = self.user_reads.values().copied().collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let half_views = std::mem::take(&mut self.half_views).into_iter().collect();
        let cmp_samplers = self.s.cmp_samplers;
        let wave_size = self.s.wave_size;
        Ok(Lowered {
            wave_size,
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            written_root_uavs: std::mem::take(&mut self.s.written_root_uavs),
            writes_guest_memory: self.s.writes_guest_memory,
            half_views,
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            stride_table: self.s.stride_table.is_some(),
            #[cfg(test)]
            annotations: std::mem::take(&mut self.s.annotations),
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
        })
    }

    fn lower_ir(&mut self) -> Result<(), Stop> {
        if self.wants_console_flow && !self.x.console_cf && !self.x.per_lane_only {
            return Err(unsupported(0, CONSOLE_FLOW));
        }
        if let Some(n) = self.s.wave_size
            && !self.x.target.runs(n)
        {
            // shortcut: no form of these lanes on the host's wave size yet
            // (two 32-lane waves standing for one console wave64 need the
            // halves to exchange masks and lane values).
            return Err(unsupported(
                0,
                match (n, self.console_lanes) {
                    (64, true) => "console wave64 lanes on a host without 64-lane waves",
                    (64, false) => {
                        "a wave64 group sharing LDS without barriers on a host without 64-lane waves"
                    }
                    _ => "a console wave size the host does not run",
                },
            ));
        }
        let p = self.x.p;
        self.halves_only = self.places_read_as_halves();
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.find_loop_exits();
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        // A compute program that reads its group id x may run spread across
        // z (`USER_DISPATCH_SPREAD`). Not a spilled one (the builder's
        // stages, near the root signature's 64 dwords), nor one whose first
        // block is a loop header (its counter sits in the entry block).
        if self.x.native.stage == HwStage::Cs
            && self.user_first() == 0
            && !self.counters.contains_key(&order[0])
            && self.x.io.inputs.iter().any(|&(_, i)| i == Input::GroupId(0))
        {
            self.cs_spread_prologue();
            self.block[order[0]] = Some(self.cur);
        }
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        } else if self.x.native.stage == HwStage::Cs && crate::spill::is_spilled(self.x.native) {
            // The structure builder's stages: a DispatchX may name threads,
            // not groups (`USER_DISPATCH_THREADS`; fork pull request #646).
            // They never take the spread prologue above (`user_first`).
            self.dispatch_threads_prologue(&order);
        }
        // Dispatches count threads too: the last group's others do not run.
        if self.x.res.addresses
            && self.s.mesh.is_none()
            && let Some([x, ..]) = self.s.threads
        {
            let e = self.s.entry;
            let zero = self.s.m.i32(0);
            let i32t = self.s.t.i32;
            let id = self.s.op(e, "threadId.i32", Attr::ReadNone, i32t, 93, &[zero]);
            let start = self.host_data(1, RootType::GROUPS, 0);
            let group = self.s.m.i32(x);
            let start = self.s.m.bin(e, Bin::Mul, start, group);
            let id = self.s.m.bin(e, Bin::Add, id, start);
            let most = self.host_data(2, RootType::THREADS, x);
            let runs = self.s.m.cmp(e, pred::IULT, id, most);
            let stop = self.abort_block();
            let go = self.s.m.new_block();
            self.s.m.terminate(e, Term::CondBr(runs, go, stop));
            self.cur = go;
            self.block[order[0]] = Some(go);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            // Compute programs only: their host waves are the console's, so a
            // wave's value is the console wave's.
            if self.mask_data == MaskData::Console {
                self.leave_loops(b);
                if !self.loop_exits.contains_key(&b) {
                    self.join_scalars(b);
                }
            }
            if self.lds_meet.as_ref().is_some_and(|(_, meet)| meet[b].0) {
                self.lds_meeting();
            }
            let mut i = 0;
            while i < p.blocks[b].insts.len() {
                self.inst(b, i)?;
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    self.section(b, i, end)?;
                    i = end;
                    continue;
                }
                i += 1;
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            if self.lds_meet.as_ref().is_some_and(|(_, meet)| meet[b].1) {
                self.lds_meeting();
            }
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        Ok(())
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        // RDNA2 POPS (ROV) pixel shaders read a hardware wave-ordering SGPR
        // (bit 31 = overlap flag) and spin on `src_pops_exiting_wave_id` (239)
        // when bit 31 is set. Zeroing that SGPR takes the non-overlapping fast
        // path straight past the `s_sleep` spin loop into the critical section.
        let _uses_pops = self.s.pixel
            && self.x.p.blocks.iter().flat_map(|b| &b.insts).any(|inst| {
                matches!(
                    &inst.fields,
                    Fields::Sop { src, .. } if src.contains(&Operand::Const(239))
                )
            });
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                // Most programs only interpolate with them: loaded on first
                // use as a value (`get`).
                Reg::V(_) if matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) => {
                    continue;
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    None if super::pops::collision_register(self.x.native) == Some(n) => {
                        Val::I32(self.s.m.i32(super::pops::NO_COLLISION))
                    }
                    // Treat SGPRs above the hardware inputs like unset VGPRs.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    // A scalar parked in a VGPR lane (`ir::unspill`), read where the
                    // paths only seem to allow it before any save: the lane of a VGPR
                    // nobody wrote, and VGPRs nobody set read as zero.
                    None if n >= crate::ir::PARKED => Val::I32(self.s.m.i32(0)),
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { kind, index } => return self.barycentric(kind, index),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => {
                let id = self.compute_id("groupId.i32", 94, c);
                if !self.one_dimensional() {
                    return id;
                }
                let Val::I32(id) = id else { unreachable!() };
                let base = self.host_word(RootType::Other(super::shader::USER_GROUP_BASE), 0, 0);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Add, id, base));
            }
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    /// Whether the program is a compute program that reads its group's
    /// number in x alone, with room for a dword of the host's among its user data
    /// (the host's root signature holds 64 dwords, views included).
    fn one_dimensional(&self) -> bool {
        const MOST: u32 = 20;
        let reads = |c| self.x.io.inputs.iter().any(|i| i.1 == Input::GroupId(c));
        let words = self.x.native.layout.iter().map(|e| u32::from(e.user_data) + e.sgprs).max().unwrap_or(0);
        matches!(self.x.native.io, StageIo::Cs(_)) && reads(0) && !reads(1) && !reads(2) && words <= MOST && !crate::spill::is_spilled(self.x.native)
    }

    /// A dispatch the host spread across z (`USER_DISPATCH_SPREAD`): the
    /// group id as the title asked it, `(x + z * row, y, 0)`, for every input
    /// that reads it, and the groups past the asked count leave before the
    /// body (whole groups: no barrier is split). `row` 0: as dispatched.
    /// The entry block ends here; the first block's code goes to the next.
    /// (After upstream VIctory pull request #2312.)
    fn cs_spread_prologue(&mut self) {
        use super::shader::USER_DISPATCH_SPREAD;
        let row = self.host_word(RootType::Other(USER_DISPATCH_SPREAD), 0, 0);
        let asked = self.host_word(RootType::Other(USER_DISPATCH_SPREAD), 0, 1);
        let Val::I32(gx) = self.compute_id("groupId.i32", 94, 0) else {
            unreachable!()
        };
        let Val::I32(gz) = self.compute_id("groupId.i32", 94, 2) else {
            unreachable!()
        };
        let e = self.s.entry;
        let zero = self.s.m.i32(0);
        let unspread = self.s.m.cmp(e, pred::EQ, row, zero);
        let mul = self.s.m.bin(e, Bin::Mul, gz, row);
        let spread_x = self.s.m.bin(e, Bin::Add, gx, mul);
        let eff_x = self.s.m.select(e, unspread, gx, spread_x);
        let eff_z = self.s.m.select(e, unspread, gz, zero);
        let in_x = self.s.m.cmp(e, pred::IULT, eff_x, asked);
        let keep = self.s.m.bin(e, Bin::Or, unspread, in_x);
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for (&v, &inp) in &io_inputs {
            match inp {
                Input::GroupId(0) => {
                    self.vals.insert(v, Val::I32(eff_x));
                }
                Input::GroupId(2) => {
                    self.vals.insert(v, Val::I32(eff_z));
                }
                _ => {}
            }
        }
        let body = self.s.m.new_block();
        let ret = self.s.m.new_block();
        self.s.m.terminate(ret, Term::Ret);
        self.s.m.terminate(e, Term::CondBr(keep, body, ret));
        self.cur = body;
    }

    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        if self.phi_at.contains_key(&v) {
            self.read_phis.insert(v);
        }
        let is_mask = self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v);
        if let Some(&(val, h)) = self.finals.get(&v)
            && !self.loop_bodies[&h].contains(&self.cur_ir)
        {
            if is_mask && !self.mask_ok {
                return Err(LANE_MASK);
            }
            return Ok(val);
        }
        if is_mask && !self.mask_ok && std::env::var_os("VICTORY_WHY").is_some() {
            let def = self.def_block.get(&v).map(|&b| (b, self.x.p.blocks[b].insts.iter().find(|i| i.writes.iter().any(|w| w.1 == v)).map(|i| (format!("{:#x}", i.machine.offset), i.name().to_owned())), self.x.p.blocks[b].phis.iter().find(|p| p.value == v).map(|p| (p.reg, p.args.clone()))));
            eprintln!("mask data {v:?} {:?} def {def:?} cur {} high {:?} val {:?}", self.x.st.masks.get(&v), self.cur_ir, self.x.st.highs.get(&v), self.vals.get(&v));
        }
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(_)) if self.mask_ok && self.mask_scc.contains_key(&v) => {
                let lo = self.mask_scc[&v];
                return self.mask_any(lo).map(Val::Bit);
            }
            // A mask's number (from a set operation) only for the uses
            // `MaskData` allows.
            Some(&Val::I32(_)) if is_mask && !self.mask_ok && !self.phi_at.contains_key(&v) => {
                return Err(LANE_MASK);
            }
            // shortcut: descriptors a program that traces rays reads as data
            // (a hit's material, say): no ray hits anything.
            Some(&Val::Missing(
                "a descriptor's address or format read as data"
                | "a descriptor read only as data"
                | "a constant buffer's descriptor read as data",
            )) if crate::resource::traces_rays(self.x.p) => {
                return Ok(Val::I32(self.s.m.i32(0)));
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => return Ok(val),
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        if let Some(loaded) = self.lazy_barycentric(v) {
            loaded?;
            return Ok(self.vals[&v]);
        }
        Err("a value defined after its use")
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one): DXIL allows two.
        let bary = |e: &SigElement| e.kind == 28;
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    /// The computed byte offset `a` adds to its parameter's address: its
    /// computed term, less the parameter's address when that term is a
    /// pointer that holds it (see [`Address::includes_base`]).
    fn dynamic_term(&mut self, a: &Address) -> Result<Option<V>, Why> {
        let Some(d) = a.dynamic else { return Ok(None) };
        let d = self.i32_of(d)?;
        if !a.includes_base {
            return Ok(Some(d));
        }
        if a.param == CODE {
            return Ok(Some(d));
        }
        let base = match self.user(a.param, 0) {
            Val::I32(x) => x,
            _ => return Err("an advanced pointer into a root descriptor"),
        };
        Ok(Some(self.s.m.bin(self.cur, Bin::Sub, d, base)))
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            // The dispatch's thread count, which the host sets (`spill::thread_limited`).
            RootType::Other(super::shader::USER_DISPATCH_THREADS_X) => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword - self.user_first();
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        Val::I32(self.user_at(index))
    }

    /// The user-data index that is dword 0 of the host's user data: 0, or
    /// 16 for a program with a spilled user-data table, whose parameters
    /// all sit at 16 and above. (The host's root signature has 64 dwords:
    /// a builder stage with constants up to index 45, eight root views,
    /// their biases and the memory directory would not fit with 16 dwords
    /// nothing reads in front.)
    fn user_first(&self) -> u32 {
        if crate::spill::is_spilled(self.x.native) {
            16
        } else {
            0
        }
    }

    /// A user-data dword no register of the console's holds: one the host
    /// computes when it binds (`root` says what, an `Other` code of
    /// `shader.rs`), placed after the layout's own dwords.
    fn host_word(&mut self, root: RootType, slot: u8, dword: u32) -> V {
        let known = self
            .user_reads
            .values()
            .find(|u| u.root == root && u.slot == slot && u.dword == dword)
            .map(|u| u.index);
        let index = match known {
            Some(index) => index,
            None => {
                let first = self.user_first();
                let layout = &self.x.native.layout;
                let mut index = layout
                    .iter()
                    .map(|e| u32::from(e.user_data) + e.sgprs)
                    .max()
                    .unwrap_or(0)
                    .saturating_sub(first);
                while self.user_reads.contains_key(&index) {
                    index += 1;
                }
                self.user_reads.insert(
                    index,
                    UserData {
                        index,
                        slot,
                        root,
                        dword,
                    },
                );
                index
            }
        };
        self.user_at(index)
    }

    /// User-data dword `index`, loaded in the entry block.
    fn user_at(&mut self, index: u32) -> V {
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        self.s.m.extract(entry, r, index % 4)
    }

    /// Whether some instruction computes with `v` (scalar or vector
    /// arithmetic), rather than only naming it in a descriptor operand,
    /// moving it (at most `depth` moves are followed) or comparing it: a
    /// waterfall loop keeps a descriptor's dwords per lane, reads the first
    /// lane's back and compares the lanes' with them.
    fn computed_with(&self, v: Value, depth: u32) -> bool {
        self.x.p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
            if !i.reads.iter().any(|r| r.1 == v) {
                return false;
            }
            let name = i.name();
            let moves = matches!(
                name,
                "s_mov_b32"
                    | "s_mov_b64"
                    | "v_mov_b32"
                    | "v_readfirstlane_b32"
                    | "v_readlane_b32"
                    | "v_writelane_b32"
                    | "v_cndmask_b32"
            );
            match &i.fields {
                Fields::Sop { .. } | Fields::Valu(_) if moves => {
                    depth > 0 && i.writes.iter().any(|w| self.computed_with(w.1, depth - 1))
                }
                Fields::Sop { .. } | Fields::Valu(_) => {
                    !(name.starts_with("v_cmp") || name.starts_with("s_cmp"))
                }
                _ => false,
            }
        })
    }

    /// The title address (low dword, bits from 32 up) of the buffer view
    /// whose descriptor the scalar load at `site` reads from the start of
    /// a table entry, as the host tells it when it binds (`shader.rs`
    /// `USER_VIEW_ADDRESS`); none for anything else. (The console's
    /// descriptor holds more in the second dword, the stride from bit 16
    /// up; code that takes the address masks those away.)
    fn view_address(&mut self, site: Site) -> Option<(V, V)> {
        use super::shader::USER_VIEW_ADDRESS;
        let a = self.scalar.get(&site).copied()?;
        let rp = *self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table || a.dynamic.is_some() || a.offset % 32 != 0 {
            return None;
        }
        let dword = a.offset / 4;
        let used = self
            .x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)?;
        if used.dwords != 4 || !matches!(used.kind, ResourceType::Srv | ResourceType::Uav) {
            return None;
        }
        let lo = self.host_word(RootType::Other(USER_VIEW_ADDRESS), rp.slot, dword);
        let hi = self.host_word(RootType::Other(USER_VIEW_ADDRESS + 1), rp.slot, dword);
        Some((lo, hi))
    }

    /// `lo`, `hi` plus the instruction's byte offset `off`, with the carry.
    fn address_plus(&mut self, lo: V, hi: V, off: i32) -> (V, V) {
        if off == 0 {
            return (lo, hi);
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let k = self.s.m.i32(off as u32);
        let sum = self.s.m.bin(c, Bin::Add, lo, k);
        // Up: a carry when the sum wrapped below `lo`. Down: a borrow when
        // it wrapped above.
        let wrapped = self
            .s
            .m
            .cmp(c, if off > 0 { pred::IULT } else { pred::IUGT }, sum, lo);
        let one = self.s.m.cast(c, Cast::ZExt, wrapped, i32t);
        let hi = self.s.m.bin(c, if off > 0 { Bin::Add } else { Bin::Sub }, hi, one);
        (sum, hi)
    }

    /// Title memory at the 40-bit address `lo`, `hi` (`shader.rs`
    /// `SLOT_MEMORY`): the handle of the raw view of the host copy holding
    /// it, the byte in that view, and whether the page is mapped. A page without a copy reads zeros
    /// (and drops writes) through the null view at the first slot, and its
    /// number is left for the host, for lanes that run (`guard`).
    fn memory(&mut self, lo: V, hi: V, guard: Option<V>) -> (V, V, V) {
        let (slot, byte, mapped) = self.memory_location(lo, hi, guard);
        (self.memory_handle(slot), byte, mapped)
    }

    fn memory_location(&mut self, lo: V, hi: V, guard: Option<V>) -> (V, V, V) {
        use super::shader::{MEMORY_FAULTS, MEMORY_PAGES, SLOT_FAULTS, SLOT_MEMORY, USER_MEMORY_BASE};
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k16, k8) = (k(self, 16), k(self, 0xFF));
        let low = self.s.m.bin(c, Bin::LShr, lo, k16);
        let high = self.s.m.bin(c, Bin::And, hi, k8);
        let high = self.s.m.bin(c, Bin::Shl, high, k16);
        let page = self.s.m.bin(c, Bin::Or, low, high);
        // Inside the directory whatever the address (lanes that do not run
        // compute one too); its last entry is never filled.
        let last = k(self, MEMORY_PAGES - 1);
        let page = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[page, last]);
        self.s.m.tag(page, Tag::Clamp);
        let k3 = k(self, 3);
        let at = self.s.m.bin(c, Bin::Shl, page, k3);
        let directory = self.s.root_srv_handle(SLOT_MEMORY);
        let (iu, both, align) = (self.s.m.undef(i32t), self.s.m.i8(3), k(self, 4));
        let entry = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            res,
            139,
            &[directory, at, iu, both, align],
        );
        let (copy, start) = (self.s.m.extract(c, entry, 0), self.s.m.extract(c, entry, 1));
        let (copy, start) = (self.as_int(copy), self.as_int(start));
        // A page without a copy: its number, at a place by the number, only
        // when an active lane misses; mapped accesses skip the modulo and store.
        let zero = k(self, 0);
        // A newer command list can publish more pages after this one has
        // recorded. Only reference descriptors prepared for this particular bind.
        let count = self.host_word(RootType::Other(USER_MEMORY_BASE), 0, 1);
        let in_range = self.s.m.cmp(c, pred::IULE, copy, count);
        let copy = self.s.m.select(c, in_range, copy, zero);
        let mut missing = self.s.m.cmp(c, pred::EQ, copy, zero);
        if let Some(g) = guard {
            missing = self.s.m.bin(c, Bin::And, missing, g);
        }
        let fault_gate = self.begin_guarded(Some(missing));
        let fc = self.cur;
        let places = k(self, MEMORY_FAULTS - 1);
        let place = self.s.m.bin(fc, Bin::URem, page, places);
        let one = k(self, 1);
        let place = self.s.m.bin(fc, Bin::Add, place, one);
        let k2 = k(self, 2);
        let place = self.s.m.bin(fc, Bin::Shl, place, k2);
        let faults = self.s.root_uav_handle(SLOT_FAULTS, false);
        let void = self.s.t.void;
        let mask = self.s.m.i8(1);
        self.s.op(
            fc,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[faults, place, iu, page, iu, iu, iu, mask, align],
        );
        self.end_guarded(fault_gate, &[]);
        let c = self.cur;
        let base = self.host_word(RootType::Other(USER_MEMORY_BASE), 0, 0);
        let slot = self.s.m.bin(c, Bin::Add, base, copy);
        let h = self.s.heap_handle(c, slot, false, true, (11 | 0x1000, 0));
        // Pages written here are copied back as title memory's declared
        // ranges say, not by view marks.
        self.s.view_index.remove(&h);
        let k64k = k(self, 0xFFFF);
        let within = self.s.m.bin(c, Bin::And, lo, k64k);
        let byte = self.s.m.bin(c, Bin::Add, start, within);
        let mapped = self.s.m.cmp(c, pred::NE, copy, zero);
        (slot, byte, mapped)
    }

    fn memory_handle(&mut self, slot: V) -> V {
        self.s.heap_handle(self.cur, slot, false, true, (11 | 0x1000, 0))
    }

    /// `global_load_*`, `global_store_*` and `global_atomic_*` on title
    /// memory by address (`resource::MEMORY`): the address is the VGPR
    /// pair, each lane its own.
    fn memory_flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat {
            dst,
            addr,
            data,
            saddr,
            offset,
            lds,
            cache,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *lds || !matches!(saddr, None | Some(Operand::Scalar { code: NULL, .. })) {
            return Err("a global address with a scalar base");
        }
        let Some(Operand::Vector { reg: areg, .. }) = *addr else {
            return Err("a global address not in VGPRs");
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a global address")?;
            l.i32_of(v)
        };
        let (lo, hi) = (word(self, 0)?, word(self, 1)?);
        let (lo, hi) = self.address_plus(lo, hi, i32::from(*offset));
        let guard = self.effect_guard(b, i)?;
        let c = self.cur;
        let (i32t, res, void) = (self.s.t.i32, self.s.t.res_ret, self.s.t.void);
        let iu = self.s.m.undef(i32t);
        let align = self.s.m.i32(4);
        let name = inst.name();
        let count = |rest: &str| match rest {
            "dword" => Ok(1u8),
            "dwordx2" => Ok(2),
            "dwordx3" => Ok(3),
            "dwordx4" => Ok(4),
            _ => Err(name),
        };
        if let Some(rest) = name.strip_prefix("global_load_") {
            let n = count(rest)?;
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a global load result");
            };
            let (h, byte, _) = self.memory(lo, hi, guard);
            let mask = self.s.m.i8((1u8 << n) - 1);
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                res,
                139,
                &[h, byte, iu, mask, align],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, u32::from(k));
                let x = self.as_int(x);
                self.write(inst, Reg::V(reg + k), x, guard);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("global_store_") {
            let n = count(rest)?;
            let d = data.as_ref().ok_or("global store data")?;
            let vals = self.vgprs(inst, d, n)?;
            let (h, byte, _) = self.memory(lo, hi, guard);
            let mut args = vec![h, byte, iu];
            args.extend(&vals);
            args.resize(3 + 4, iu);
            args.push(self.s.m.i8((1u8 << n) - 1));
            args.push(align);
            let at = self.begin_guarded(guard);
            self.s
                .op(self.cur, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
            self.end_guarded(at, &[]);
            self.s.writes_guest_memory = true;
            return Ok(());
        }
        let Some(op) = name.strip_prefix("global_atomic_") else {
            return Err(name);
        };
        let d = data.as_ref().ok_or("global atomic data")?;
        let v = self.vgprs(inst, d, 1)?[0];
        // With `glc` the value from before comes back, in the result
        // register the instruction names.
        let back = match (*dst, cache.glc) {
            (Some(Operand::Vector { reg, .. }), true) => Some(reg),
            _ => None,
        };
        let old = match back.and_then(|reg| inst.read(Reg::V(reg))) {
            Some(x) => self.i32_of(x)?,
            None => self.s.m.i32(0),
        };
        let (h, byte, _) = self.memory(lo, hi, guard);
        // DXIL's atomic operations: add 0, and 1, or 2, xor 3, signed
        // minimum 4 and maximum 5, unsigned 6 and 7, exchange 8.
        let simple = match op {
            "add" => Some((0, v)),
            "sub" => {
                let zero = self.s.m.i32(0);
                Some((0, self.s.m.bin(c, Bin::Sub, zero, v)))
            }
            "and" => Some((1, v)),
            "or" => Some((2, v)),
            "xor" => Some((3, v)),
            "smin" => Some((4, v)),
            "smax" => Some((5, v)),
            "umin" => Some((6, v)),
            "umax" => Some((7, v)),
            "swap" => Some((8, v)),
            "fmin" | "fmax" => None,
            _ => return Err(name),
        };
        let at = self.begin_guarded(guard);
        let was = match simple {
            Some((code, x)) => {
                let code = self.s.m.i32(code);
                self.s.op(
                    self.cur,
                    "atomicBinOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    78,
                    &[h, code, byte, iu, iu, x],
                )
            }
            None => {
                // A float minimum or maximum has no atomic operation on
                // the host: the dword is exchanged for the smaller (larger)
                // of itself and the value until no other thread got in
                // between. Each failed pass means another one's succeeded;
                // the limit only keeps a broken card from hanging.
                let head = self.cur;
                let mask = self.s.m.i8(1);
                let first = self.s.op(
                    head,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    res,
                    139,
                    &[h, byte, iu, mask, align],
                );
                let first = self.s.m.extract(head, first, 0);
                let first = self.as_int(first);
                let (pass, done) = (self.s.m.new_block(), self.s.m.new_block());
                self.s.m.terminate(head, Term::Br(pass));
                self.cur = pass;
                let seen = self.s.m.phi(pass, i32t);
                let n = self.s.m.phi(pass, i32t);
                let (f32t, fv) = (self.s.t.f32, self.as_float(v));
                let fseen = self.as_float(seen);
                // `dx.op.binary`: FMax 35, FMin 36.
                let pick = if op == "fmin" { 36 } else { 35 };
                let wanted = self
                    .s
                    .op(pass, "binary.f32", Attr::ReadNone, f32t, pick, &[fseen, fv]);
                let wanted = self.as_int(wanted);
                let found = self.s.op(
                    pass,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &[h, byte, iu, iu, seen, wanted],
                );
                let one = self.s.m.i32(1);
                let n1 = self.s.m.bin(pass, Bin::Add, n, one);
                let lost = self.s.m.cmp(pass, pred::NE, found, seen);
                let limit = self.s.m.i32(1 << 16);
                let more = self.s.m.cmp(pass, pred::IULT, n1, limit);
                let again = self.s.m.bin(pass, Bin::And, lost, more);
                self.s.m.terminate(pass, Term::CondBr(again, pass, done));
                let zero = self.s.m.i32(0);
                self.s.m.set_phi(seen, vec![(first, head), (found, pass)]);
                self.s.m.set_phi(n, vec![(zero, head), (n1, pass)]);
                self.cur = done;
                found
            }
        };
        let was = self.end_guarded(at, &[(was, old)])[0];
        self.s.writes_guest_memory = true;
        if let Some(reg) = back {
            self.write(inst, Reg::V(reg), was, None);
        }
        Ok(())
    }

    /// `s_load_dword*` from title memory by address (`resource::MEMORY`):
    /// the SGPR pair plus the instruction's offsets.
    fn memory_smem(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            base: Operand::Scalar { code: base, .. },
            offset,
            soffset,
            ..
        } = inst.fields
        else {
            return Err("a scalar load from memory");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u8,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory access by address"),
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::S(base + k)).ok_or("an address not in SGPRs")?;
            l.i32_of(v)
        };
        let (mut lo, mut hi) = (word(self, 0)?, word(self, 1)?);
        let c = self.cur;
        let (i32t, res) = (self.s.t.i32, self.s.t.res_ret);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            // An offset register counts up from the address.
            let so = self.operand(inst, &soffset)?;
            let sum = self.s.m.bin(c, Bin::Add, lo, so);
            let carry = self.s.m.cmp(c, pred::IULT, sum, lo);
            let carry = self.s.m.cast(c, Cast::ZExt, carry, i32t);
            hi = self.s.m.bin(c, Bin::Add, hi, carry);
            lo = sum;
        }
        let (iu, align) = (self.s.m.undef(i32t), self.s.m.i32(4));
        for first in (0..count).step_by(4) {
            let n = (count - first).min(4);
            let (lo, hi) = self.address_plus(lo, hi, offset + 4 * i32::from(first));
            let (h, byte, _) = self.memory(lo, hi, self.partial_active);
            let mask = self.s.m.i8((1u8 << n) - 1);
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                res,
                139,
                &[h, byte, iu, mask, align],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, u32::from(k));
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + first + k), x, None);
            }
        }
        Ok(())
    }

    /// A scalar move copies all 32 bits of each word, regardless of EXEC.
    /// Follow its same-half SSA input to an existing numeric word. A phi
    /// ends the search and is kept live by the caller's normal numeric read.
    fn copied_mask_word(&self, mut v: Value) -> Option<Value> {
        let low = self.x.st.highs.get(&v).copied().unwrap_or(v);
        if self.mask_data != MaskData::Console || self.s.wave_size != Some(64)
            || !self.x.st.masks.contains_key(&low) || self.x.st.active.contains(&low)
        {
            return None;
        }
        loop {
            let block = &self.x.p.blocks[*self.def_block.get(&v)?];
            let inst = block.insts.iter().find(|i| i.writes.iter().any(|&(_, out)| out == v))?;
            if !matches!(inst.name(), "s_mov_b32" | "s_mov_b64") {
                return None;
            }
            let Fields::Sop { dst: Some(Operand::Scalar { code: dst, dwords }), ref src, .. } =
                inst.fields else { return None };
            let [Operand::Scalar { code: src, .. }] = src.as_slice() else { return None };
            let word = (0..dwords).find(|half| inst.write(Reg::S(dst + half)) == Some(v))?;
            v = inst.read(Reg::S(src + word))?;
            if matches!(self.vals.get(&v), Some(Val::I32(_))) {
                return Some(v);
            }
        }
    }

    /// The heap index of the host's first view of memory by address: the
    /// map, then the memory it names.
    fn memory_base(&mut self) -> V {
        self.host_data(0, RootType::MEMORY, 0)
    }

    /// User data the host adds behind the title's: the `nth` such dword.
    fn host_data(&mut self, nth: u32, root: RootType, dword: u32) -> V {
        let layout = &self.x.native.layout;
        let index = layout
            .iter()
            .map(|p| u32::from(p.user_data) + p.sgprs.max(1))
            .max()
            .unwrap_or(0)
            + nth;
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: u8::MAX,
                root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        self.s.m.extract(entry, r, index % 4)
    }

    /// The host's view of the title memory at `hi:lo` (40 bits) and the
    /// byte in it. The map holds two dwords per MiB: which view, and the
    /// byte in it of that MiB's start.
    fn memory_raw(&mut self, lo: V, hi: V, nonuniform: bool) -> (V, V) {
        let c = self.cur;
        let first = self.memory_base();
        let map = self.s.handle_at(c, first, false, false, (11, 0));
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let (k255, k12, k20, k3, low) =
            (k(self, 255), k(self, 12), k(self, 20), k(self, 3), k(self, 0xF_FFFF));
        let h = self.s.m.bin(c, Bin::And, hi, k255);
        let h = self.s.m.bin(c, Bin::Shl, h, k12);
        let l = self.s.m.bin(c, Bin::LShr, lo, k20);
        let mib = self.s.m.bin(c, Bin::Or, h, l);
        let at = self.s.m.bin(c, Bin::Shl, mib, k3);
        let (mask, align, u) = (self.s.m.i8(3), self.s.m.i32(4), self.s.m.undef(self.s.t.i32));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[map, at, u, mask, align],
        );
        let view = self.s.m.extract(c, r, 0);
        let view = self.as_int(view);
        let start = self.s.m.extract(c, r, 1);
        let start = self.as_int(start);
        // Never past the views of ours, whatever the map holds.
        let (i32t, most) = (self.s.t.i32, self.s.m.i32(super::VIEW_SLOTS - super::OWN_SLOTS - 1));
        let view = self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[view, most]);
        let index = self.s.m.bin(c, Bin::Add, first, view);
        // Builders read what other groups of the dispatch wrote.
        let p = self.x.p;
        let shared = *self.coherent.get_or_insert_with(|| {
            p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                let n = i.name();
                n.starts_with("global_store") || n.starts_with("global_atomic")
            })
        });
        let shared = if shared { 0x4000 } else { 0 };
        let handle = self.s.handle_at(c, index, false, nonuniform, (11 | 0x1000 | shared, 0));
        let within = self.s.m.bin(c, Bin::And, lo, low);
        (handle, self.s.m.bin(c, Bin::Add, start, within))
    }

    /// `hi:lo` plus `x` (sign-extended).
    fn add_address(&mut self, (lo, hi): (V, V), x: V, signed: bool) -> (V, V) {
        let c = self.cur;
        let (r, carry) = self.add_sub_carry(c, true, lo, x, None);
        let up = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
        let mut hi = self.s.m.bin(c, Bin::Add, hi, up);
        if signed {
            let k31 = self.s.m.i32(31);
            let s = self.s.m.bin(c, Bin::AShr, x, k31);
            hi = self.s.m.bin(c, Bin::Add, hi, s);
        }
        (r, hi)
    }

    /// `count` dwords of title memory at `hi:lo`.
    fn memory_dwords(&mut self, at: (V, V), count: u32, nonuniform: bool) -> Vec<V> {
        let (h, off) = self.memory_raw(at.0, at.1, nonuniform);
        let mut out = Vec::new();
        let mut done = 0;
        while done < count {
            let n = (count - done).min(4);
            let c = self.cur;
            let d = self.s.m.i32(4 * done);
            let a = self.s.m.bin(c, Bin::Add, off, d);
            let (mask, align, u) =
                (self.s.m.i8((1u8 << n) - 1), self.s.m.i32(4), self.s.m.undef(self.s.t.i32));
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                139,
                &[h, a, u, mask, align],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, k);
                let x = self.as_int(x);
                out.push(x);
            }
            done += n;
        }
        out
    }

    #[allow(dead_code)]
    /// The address a global memory instruction gives each lane.
    fn flat_at(&mut self, inst: &Inst) -> Result<(V, V), Why> {
        let Fields::Flat {
            addr, saddr, offset, ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = *addr else {
            return Err("a global address");
        };
        let lo = inst.read(Reg::V(reg)).ok_or("a global address")?;
        let lo = self.i32_of(lo)?;
        let mut at = match *saddr {
            Some(Operand::Scalar { code, .. }) if code != NULL => {
                let (a, b) = (
                    inst.read(Reg::S(code)).ok_or("a global address")?,
                    inst.read(Reg::S(code + 1)).ok_or("a global address")?,
                );
                let base = (self.i32_of(a)?, self.i32_of(b)?);
                self.add_address(base, lo, false)
            }
            _ => {
                let hi = inst.read(Reg::V(reg + 1)).ok_or("a global address")?;
                (lo, self.i32_of(hi)?)
            }
        };
        if *offset != 0 {
            let x = self.s.m.i32(i32::from(*offset) as u32);
            at = self.add_address(at, x, true);
        }
        Ok(at)
    }

    #[allow(dead_code)]
    /// A global memory access by address.
    fn flat_memory(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, data, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let at = self.flat_at(inst)?;
        let dwords = |s: &str| match s {
            "dword" => Some(1u32),
            "dwordx2" => Some(2),
            "dwordx3" => Some(3),
            "dwordx4" => Some(4),
            _ => None,
        };
        if let Some(rest) = name.strip_prefix("global_load_") {
            let Some(Operand::Vector { reg, .. }) = *dst else {
                return Err("a global load result");
            };
            if let Some((width, signed, half)) = subword_load(rest) {
                let (h, off) = self.memory_raw(at.0, at.1, true);
                let x = self.read_subword(off, width, signed, |l, byte| {
                    let u = l.s.m.undef(l.s.t.i32);
                    Ok(l.raw_dword(h, [byte, u]))
                })?;
                self.write_subword_result(inst, reg, x, half, guard)?;
                return Ok(());
            }
            let n = dwords(rest).ok_or(name)?;
            let xs = self.memory_dwords(at, n, true);
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let data = data.as_ref().ok_or("global data")?;
        if let Some(n) = name.strip_prefix("global_store_").and_then(dwords) {
            let vals = self.vgprs(inst, data, n as u8)?;
            let place = self.begin_guarded(guard);
            let (h, off) = self.memory_raw(at.0, at.1, true);
            let iu = self.s.m.undef(self.s.t.i32);
            let mut args = vec![h, off, iu];
            args.extend(&vals);
            args.resize(3 + 4, iu);
            args.push(self.s.m.i8((1u8 << n) - 1));
            args.push(self.s.m.i32(4));
            let void = self.s.t.void;
            self.s.op(self.cur, "rawBufferStore.i32", Attr::NoUnwind, void, 140, &args);
            self.end_guarded(place, &[]);
            return Ok(());
        }
        let op = name.strip_prefix("global_atomic_").ok_or(name)?;
        let code = match op {
            "add" | "sub" => Some(0),
            "and" => Some(1),
            "or" => Some(2),
            "xor" => Some(3),
            "smin" => Some(4),
            "smax" => Some(5),
            "umin" => Some(6),
            "umax" => Some(7),
            "swap" => Some(8),
            "cmpswap" | "fmin" | "fmax" => None,
            _ => return Err(name),
        };
        let vals = self.vgprs(inst, data, if op == "cmpswap" { 2 } else { 1 })?;
        let zero = self.s.m.i32(0);
        let place = self.begin_guarded(guard);
        let (h, off) = self.memory_raw(at.0, at.1, true);
        let iu = self.s.m.undef(self.s.t.i32);
        let coords = [off, iu, iu];
        let r = match (op, code) {
            ("cmpswap", _) => self.buffer_compare_exchange(h, coords, vals[1], vals[0], false),
            // shortcut: no denormal or NaN rules of the console's.
            ("fmin" | "fmax", _) => {
                // Not exchanged where nothing changes: a whole scene's lanes
                // meet at its bounds.
                let head = self.cur;
                let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
                let ret = self.s.t.res_ret;
                let first = self.s.op(
                    head,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    ret,
                    139,
                    &[h, off, iu, mask, align],
                );
                let first = self.s.m.extract(head, first, 0);
                let first = self.as_int(first);
                let retry = self.s.m.new_block();
                let change = self.s.m.new_block();
                let done = self.s.m.new_block();
                self.s.m.terminate(head, Term::Br(retry));
                self.cur = retry;
                let i32t = self.s.t.i32;
                let expected = self.s.m.phi(retry, i32t);
                let (a, b) = (self.as_float(expected), self.as_float(vals[0]));
                let x = self.binary(if op == "fmin" { 36 } else { 35 }, a, b);
                let x = self.as_int(x);
                let same = self.s.m.cmp(retry, pred::EQ, x, expected);
                self.s.m.terminate(retry, Term::CondBr(same, done, change));
                self.cur = change;
                let old = self.buffer_compare_exchange(h, coords, expected, x, false);
                let ok = self.s.m.cmp(change, pred::EQ, old, expected);
                self.s.m.set_phi(expected, vec![(first, head), (old, change)]);
                self.s.m.terminate(change, Term::CondBr(ok, done, retry));
                self.cur = done;
                expected
            }
            (_, Some(code)) => {
                let code = self.s.m.i32(code);
                let data = if op == "sub" {
                    self.s.m.bin(self.cur, Bin::Sub, zero, vals[0])
                } else {
                    vals[0]
                };
                let i32t = self.s.t.i32;
                self.s.op(
                    self.cur,
                    "atomicBinOp.i32",
                    Attr::NoUnwind,
                    i32t,
                    78,
                    &[h, code, off, iu, iu, data],
                )
            }
            _ => return Err(name),
        };
        let r = self.end_guarded(place, &[(r, zero)])[0];
        if let Some(Operand::Vector { reg, .. }) = *dst {
            self.write(inst, Reg::V(reg), r, guard);
        }
        Ok(())
    }

    /// `image_bvh64_intersect_ray`: the ray `words` (node pointer, extent,
    /// origin, direction, inverse direction) against a node of the
    /// acceleration structure, as the console's ray accelerator answers.
    ///
    /// A node pointer is an address over 8 with the node's kind in its low
    /// bits: 0 to 3 a triangle of a node of 5 vertices, 4 a box of 4
    /// children with 16-bit floats, 5 with 32-bit ones (4 child pointers,
    /// then each child's minimum and maximum); the rest is the title's own.
    /// A box gives its children hit, nearest first, then -1; a triangle the
    /// distance and barycentrics i and j, each over the second value
    /// (negative from behind), as the title divides them.
    // shortcut: no watertight edges, no box growing.
    fn intersect(&mut self, words: &[V; 12], guard: Option<V>) -> [V; 4] {
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let kf = |l: &mut Self, x: f32| l.s.m.f32(x.to_bits());
        let [lo, hi, rest @ ..] = *words;
        let f: Vec<V> = rest.iter().map(|&x| self.as_float(x)).collect();
        let extent = f[0];
        let (o, d, inv) = ([f[1], f[2], f[3]], [f[4], f[5], f[6]], [f[7], f[8], f[9]]);
        let c = self.cur;
        let (k3, k7, k29, not7, ones) = (k(self, 3), k(self, 7), k(self, 29), k(self, !7), k(self, u32::MAX));
        let kind = self.s.m.bin(c, Bin::And, lo, k7);
        let base = self.s.m.bin(c, Bin::And, lo, not7);
        let alo = self.s.m.bin(c, Bin::Shl, base, k3);
        let ahi = self.s.m.bin(c, Bin::Shl, hi, k3);
        let up = self.s.m.bin(c, Bin::LShr, lo, k29);
        let ahi = self.s.m.bin(c, Bin::Or, ahi, up);
        let (k4, k5, k6) = (k(self, 4), k(self, 5), k(self, 6));
        let and = |l: &mut Self, a: V, b: V| {
            let c = l.cur;
            l.s.m.bin(c, Bin::And, a, b)
        };
        let boxes = self.s.m.bin(c, Bin::And, kind, k6);
        let is_box = self.s.m.cmp(c, pred::EQ, boxes, k4);
        let is_tri = self.s.m.cmp(c, pred::IULT, kind, k4);
        let (is_box, is_tri) = match guard {
            Some(g) => (and(self, is_box, g), and(self, is_tri, g)),
            None => (is_box, is_tri),
        };

        // Boxes.
        let place = self.begin_guarded(Some(is_box));
        let node = self.memory_dwords((alo, ahi), 28, true);
        let c = self.cur;
        let wide = self.s.m.cmp(c, pred::EQ, kind, k5);
        let (k16, kffff) = (k(self, 16), k(self, 0xFFFF));
        let f32t = self.s.t.f32;
        let mut bounds = Vec::new();
        for n in 0..24usize {
            let full = self.as_float(node[4 + n]);
            let w = node[4 + n / 2];
            let h = if n % 2 == 1 { self.s.m.bin(c, Bin::LShr, w, k16) } else { w };
            let h = self.s.m.bin(c, Bin::And, h, kffff);
            let half = self.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
            bounds.push(self.s.m.select(c, wide, full, half));
        }
        let (zero, far) = (kf(self, 0.0), kf(self, f32::INFINITY));
        let mut hits: Vec<(V, V)> = Vec::new();
        for child in 0..4usize {
            let (mut near_all, mut far_all) = (zero, extent);
            for axis in 0..3usize {
                let side = |l: &mut Self, which: usize| {
                    let x = bounds[6 * child + 3 * which + axis];
                    let x = l.s.m.bin(c, Bin::Sub, x, o[axis]);
                    l.s.m.bin(c, Bin::Mul, x, inv[axis])
                };
                let (a, b) = (side(self, 0), side(self, 1));
                let (n, x) = (self.binary(36, a, b), self.binary(35, a, b));
                near_all = self.binary(35, near_all, n);
                far_all = self.binary(36, far_all, x);
            }
            let pointer = node[child];
            let there = self.s.m.cmp(c, pred::NE, pointer, ones);
            let meets = self.s.m.cmp(c, pred::OLE, near_all, far_all);
            let hit = self.s.m.bin(c, Bin::And, there, meets);
            let p = self.s.m.select(c, hit, pointer, ones);
            let t = self.s.m.select(c, hit, near_all, far);
            hits.push((p, t));
        }
        for (a, b) in [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)] {
            let ((pa, ta), (pb, tb)) = (hits[a], hits[b]);
            // Misses last, also behind hits infinitely far.
            let later = self.s.m.cmp(c, pred::OGT, ta, tb);
            let (pn, pf) = (self.s.m.select(c, later, pb, pa), self.s.m.select(c, later, pa, pb));
            let (tn, tf) = (self.s.m.select(c, later, tb, ta), self.s.m.select(c, later, ta, tb));
            hits[a] = (pn, tn);
            hits[b] = (pf, tf);
        }
        let found: Vec<(V, V)> = hits.iter().map(|h| (h.0, ones)).collect();
        let r = self.end_guarded(place, &found);

        // Triangles.
        let place = self.begin_guarded(Some(is_tri));
        let node = self.memory_dwords((alo, ahi), 16, true);
        let c = self.cur;
        let vertex = |l: &mut Self, v: usize, axis: usize| l.as_float(node[3 * v + axis]);
        // The vertices of each of the node's triangles.
        const VERTICES: [[usize; 3]; 4] = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [2, 4, 0]];
        let is: Vec<V> = (0..3u32)
            .map(|n| {
                let n = k(self, n);
                self.s.m.cmp(c, pred::EQ, kind, n)
            })
            .collect();
        let mut v = [[zero; 3]; 3];
        for (corner, row) in v.iter_mut().enumerate() {
            for (axis, x) in row.iter_mut().enumerate() {
                let mut at = vertex(self, VERTICES[3][corner], axis);
                for n in (0..3).rev() {
                    let other = vertex(self, VERTICES[n][corner], axis);
                    at = self.s.m.select(c, is[n], other, at);
                }
                *x = at;
            }
        }
        let sub = |l: &mut Self, a: [V; 3], b: [V; 3]| [0, 1, 2].map(|i| l.s.m.bin(c, Bin::Sub, a[i], b[i]));
        let cross = |l: &mut Self, a: [V; 3], b: [V; 3]| {
            [(1, 2), (2, 0), (0, 1)].map(|(i, j)| {
                let x = l.s.m.bin(c, Bin::Mul, a[i], b[j]);
                let y = l.s.m.bin(c, Bin::Mul, a[j], b[i]);
                l.s.m.bin(c, Bin::Sub, x, y)
            })
        };
        let dot = |l: &mut Self, a: [V; 3], b: [V; 3]| {
            let x = l.s.m.bin(c, Bin::Mul, a[0], b[0]);
            let x = l.fmad(a[1], b[1], x);
            l.fmad(a[2], b[2], x)
        };
        let (e1, e2) = (sub(self, v[1], v[0]), sub(self, v[2], v[0]));
        let p = cross(self, d, e2);
        let det = dot(self, e1, p);
        let tv = sub(self, o, v[0]);
        let b1 = dot(self, tv, p);
        let q = cross(self, tv, e1);
        let b2 = dot(self, d, q);
        let t = dot(self, e2, q);
        let rest = self.s.m.bin(c, Bin::Sub, det, b1);
        let b0 = self.s.m.bin(c, Bin::Sub, rest, b2);
        // All on the side of the determinant.
        let back = self.s.m.cmp(c, pred::OLT, det, zero);
        let mut inside = self.s.m.cmp(c, pred::ONE, det, zero);
        for x in [b0, b1, b2, t] {
            let neg = self.s.m.bin(c, Bin::Sub, zero, x);
            let x = self.s.m.select(c, back, neg, x);
            let ok = self.s.m.cmp(c, pred::OGE, x, zero);
            inside = self.s.m.bin(c, Bin::And, inside, ok);
        }
        // Which barycentrics i and j are: 8 bits per triangle.
        let shift = self.s.m.bin(c, Bin::Shl, kind, k3);
        let id = self.s.m.bin(c, Bin::LShr, node[15], shift);
        let (one, two) = (k(self, 1), k(self, 2));
        let pick = |l: &mut Self, id: V| {
            let n = l.s.m.bin(c, Bin::And, id, k3);
            let first = l.s.m.cmp(c, pred::EQ, n, one);
            let second = l.s.m.cmp(c, pred::EQ, n, two);
            let x = l.s.m.select(c, second, b2, b0);
            l.s.m.select(c, first, b1, x)
        };
        let i = pick(self, id);
        let id = self.s.m.bin(c, Bin::LShr, id, two);
        let j = pick(self, id);
        let unit = kf(self, 1.0);
        let out = [(t, far), (det, unit), (i, zero), (j, zero)].map(|(x, miss)| {
            let x = self.s.m.select(c, inside, x, miss);
            self.as_int(x)
        });
        let found: Vec<(V, V)> = out.iter().zip(&r).map(|(&x, &before)| (x, before)).collect();
        let r = self.end_guarded(place, &found);
        [r[0], r[1], r[2], r[3]]
    }

    /// The host's view of a buffer whose descriptor the code makes, the byte
    /// its base is at, and its stride.
    fn made_buffer(&mut self, inst: &Inst, rsrc: &Operand) -> Result<(V, V, V), Why> {
        let Operand::Scalar { code, .. } = *rsrc else {
            return Err("a buffer descriptor");
        };
        let word = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::S(code + k)).ok_or("a buffer descriptor")?;
            l.i32_of(v)
        };
        let (w0, w1) = (word(self, 0)?, word(self, 1)?);
        let c = self.cur;
        let (k16, k3fff) = (self.s.m.i32(16), self.s.m.i32(0x3FFF));
        let s = self.s.m.bin(c, Bin::LShr, w1, k16);
        let stride = self.s.m.bin(c, Bin::And, s, k3fff);
        let (h, base) = self.memory_raw(w0, w1, false);
        Ok((h, base, stride))
    }

    /// Both dwords `(lo, hi)` of a lane mask's ballot, taken in the block
    /// that writes the mask (see [`MaskData`]).
    fn mask_ballot(&mut self, lo: Value) -> Result<(V, V), Why> {
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if self.console_cf {
            // Every lane runs every block: the mask's bits are the lanes'
            // own, both halves' words exchanged.
            if let Some(&xy) = self.ballots.get(&lo) {
                return Ok(xy);
            }
            let at = match self.def_block.get(&lo) {
                None => self.s.entry,
                Some(&b) if b == self.cur_ir => self.cur,
                Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
            };
            let bit = self.boolean(m, at)?;
            let xy = self.console_ballot(at, bit);
            self.ballots.insert(lo, xy);
            return Ok(xy);
        }
        if self.split {
            return Err("a lane mask read whole in a wave64 program split in halves");
        }
        if !self.x.st.active.contains(&lo) {
            if std::env::var_os("VICTORY_WHY").is_some() {
                let def = self.def_block.get(&lo).map(|&b| (b, self.x.p.blocks[b].insts.iter().find(|i| i.writes.iter().any(|w| w.1 == lo)).map(|i| (i.machine.offset, i.name().to_owned())), self.x.p.blocks[b].phis.iter().find(|p| p.value == lo).map(|p| (p.reg, p.args.clone()))));
                eprintln!("mask {lo:?} {m:?} def {def:?} cur {}", self.cur_ir);
            }
            return Err("a lane mask holding lanes elsewhere read as data");
        }
        if let Some(&xy) = self.ballots.get(&lo) {
            return Ok(xy);
        }
        let at = match self.def_block.get(&lo) {
            // Program inputs (EXEC on entry).
            None => self.s.entry,
            Some(&b) if b == self.cur_ir => self.cur,
            Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
        };
        if !self.balloting.insert(lo) {
            return Err("a lane mask whose lanes depend on its own value");
        }
        let bit = self.boolean(m, at);
        self.balloting.remove(&lo);
        let bit = bit?;
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
        self.ballots.insert(lo, xy);
        Ok(xy)
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if self.console_cf {
            // Every lane runs every block: both halves' words exchanged.
            let (x, y) = self.mask_ballot(lo)?;
            return Ok(if high { y } else { x });
        }
        // A mask that holds lanes outside EXEC where it is written (the complement of one,
        // say) is the ballot of the lanes that are there: all of them, where two exchanging
        // host waves know every lane runs that block.
        let made = self.def_block.get(&lo).copied();
        let whole = self.exchange.is_some() && !self.local && made.is_none_or(|b| self.all_lanes(b));
        if !self.x.st.active.contains(&lo) && !whole {
            if let Some(source) = self.copied_mask_word(v) {
                return self.i32_of(source);
            }
            if self.exchange.is_some() && !self.local {
                need(made.unwrap_or(0));
            }
            if let Some(word) = self.packed_mask_word(v) {
                return Ok(word);
            }
            return Err(MASK_ELSEWHERE);
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.body_entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let mut bit = bit?;
                if let Some(active) = self.partial_active
                    && self.x.st.active.contains(&lo)
                {
                    bit = self.s.m.bin(at, Bin::And, bit, active);
                }
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let mut xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                if self.exchange.is_some() {
                    // a host wave's ballot is its half's 32 bits: the halves exchange them
                    let b = self.def_block.get(&lo).copied();
                    if self.local {
                        if !self.local_ok && !self.local_any {
                            return Err("a lane mask read as data in a half-wave");
                        }
                    } else if !self.exchange.as_ref().is_some_and(|e| e.1) && !b.is_none_or(|b| self.all_lanes(b)) {
                        // (a group of 32 or fewer has no other half to exchange with)
                        need(b.unwrap_or(0));
                        return Err("a lane mask exchanged where not every lane runs");
                    }
                    let (cont, x) = self.exchange_halves(at, xy.0);
                    match self.def_block.get(&lo).copied() {
                        None if self.s.entry == at => self.s.entry = cont,
                        Some(b) if b == self.cur_ir => self.cur = cont,
                        Some(b) => self.tail[b] = Some(cont),
                        None => self.s.entry = cont,
                    }
                    xy = x;
                }
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high { y } else { x })
    }

    /// Whether some lane of the mask with low dword `lo` is set, as data
    /// (SCC of the operation writing it): its ballot is not zero.
    fn mask_any(&mut self, lo: Value) -> Result<V, Why> {
        // When `lo` is the new EXEC written in block `b` (e.g. `s_*_wrexec_*` or
        // `s_*_saveexec_*`) and `b` exits via `s_cbranch_execz`/`execnz` without
        // rewriting EXEC, a block entered solely through one branch edge knows
        // whether `lo != 0` without balloting lanes from elsewhere.
        if let Some(&b) = self.def_block.get(&lo) {
            let blk = &self.x.p.blocks[b];
            if let crate::ir::Exit::Branch { taken, not_taken } = blk.exit
                && taken != not_taken
                && self.x.p.blocks[self.cur_ir].preds.as_slice() == [b]
                && let Some((idx, def)) = blk
                    .insts
                    .iter()
                    .enumerate()
                    .find(|(_, inst)| inst.writes.iter().any(|&(_, w)| w == lo))
                && (def.write(Reg::S(EXEC)) == Some(lo) || def.name().contains("_wrexec_"))
                && !blk.insts[idx + 1..]
                    .iter()
                    .any(|inst| inst.write(Reg::S(EXEC)).is_some())
                && let Some(term) = blk.insts.last()
            {
                let nonzero = match term.name() {
                    "s_cbranch_execnz" => Some(self.cur_ir == taken),
                    "s_cbranch_execz" => Some(self.cur_ir == not_taken),
                    _ => None,
                };
                if let Some(bit) = nonzero {
                    return Ok(self.s.m.int_const(self.s.t.i1, i64::from(bit)));
                }
            }
        }
        let x = self.mask_dword(lo)?;
        let (_, y) = self.ballots[&lo];
        let c = self.cur;
        let any = if self.x.p.wave64 {
            self.s.m.bin(c, Bin::Or, x, y)
        } else {
            x
        };
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, any, z))
    }

    /// For a wave32 program under [`MaskData::Host`], a 32-bit scalar operand
    /// holding a tracked mask still represents a host-wave ballot, so its
    /// `(lo, hi)` pair must be read to preserve lanes 32-63 on wave64 hosts.
    fn host_mask_pair(&mut self, inst: &Inst, op: &Operand) -> Result<Option<(V, V)>, Why> {
        if self.mask_data != MaskData::Host || self.x.native.wave64 {
            return Ok(None);
        }
        let Operand::Scalar { code, .. } = *op else {
            return Ok(None);
        };
        if code == NULL {
            return Ok(None);
        }
        let Some(v) = inst.read(Reg::S(code)) else {
            return Ok(None);
        };
        if matches!(self.vals.get(&v), Some(Val::I32(_) | Val::Bit(_)))
            || !self.x.st.masks.contains_key(&v)
        {
            return Ok(None);
        }
        self.mask_ballot(v).map(Some)
    }

    fn is_zero_operand(op: &Operand) -> bool {
        matches!(op, Operand::Scalar { code: NULL, .. } | Operand::Literal(0))
            || matches!(op, Operand::Const(c) if crate::resource::inline(*c) == Some(0))
    }

    fn bcnt1_64(&mut self, c: usize, a: V, b: V) -> V {
        let i32t = self.s.t.i32;
        let na = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
        let nb = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
        self.s.m.bin(c, Bin::Add, na, nb)
    }

    fn ff1_64(&mut self, c: usize, a: V, b: V) -> V {
        // The lowest set bit; -1 for none (as DXIL's first bit low).
        let i32t = self.s.t.i32;
        let fa = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
        let fb = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
        let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
        let none = self.s.m.cmp(c, pred::EQ, b, z);
        let fb = self.s.m.select(c, none, fb, fb32);
        let low = self.s.m.cmp(c, pred::NE, a, z);
        self.s.m.select(c, low, fa, fb)
    }

    fn eq_64(&mut self, c: usize, (a0, a1): (V, V), (b0, b1): (V, V)) -> V {
        let e0 = self.s.m.cmp(c, pred::EQ, a0, b0);
        let e1 = self.s.m.cmp(c, pred::EQ, a1, b1);
        self.s.m.bin(c, Bin::And, e0, e1)
    }

    /// `(1 << below) - 1`, all ones from 32 on.
    fn below_mask(&mut self, c: usize, below: V) -> V {
        let (one, k31, k32, ones) = (
            self.s.m.i32(1),
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(u32::MAX),
        );
        let sh = self.s.m.bin(c, Bin::And, below, k31);
        let m = self.s.m.bin(c, Bin::Shl, one, sh);
        let m = self.s.m.bin(c, Bin::Sub, m, one);
        let full = self.s.m.cmp(c, pred::IUGE, below, k32);
        self.s.m.select(c, full, ones, m)
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            super::pops::EXITING_WAVE_ID if super::pops::active(self.x.native) => Ok(self.s.m.i32(0)),
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// An operand read as f16 data: GFX10 puts an inline f16 constant in
    /// the low half of the operand (upstream PR #637, checked there against
    /// the instruction encodings; the half-and-half reading taken from PR
    /// #458 gave the high half the constant too).
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => Ok(self.s.m.i32(inline_f16(c))),
            _ => self.operand(inst, op),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    fn sign_extend(&mut self, x: V, bits: u32) -> V {
        let n = self.s.m.i32(32 - bits);
        let x = self.s.m.bin(self.cur, Bin::Shl, x, n);
        self.s.m.bin(self.cur, Bin::AShr, x, n)
    }

    fn pack_halves(&mut self, lo: V, hi: V) -> V {
        let mask = self.s.m.i32(0xFFFF);
        let k16 = self.s.m.i32(16);
        let lo = self.s.m.bin(self.cur, Bin::And, lo, mask);
        let hi = self.s.m.bin(self.cur, Bin::Shl, hi, k16);
        self.s.m.bin(self.cur, Bin::Or, lo, hi)
    }

    fn int_pick(&mut self, p: u8, a: V, b: V) -> V {
        let choose = self.s.m.cmp(self.cur, p, a, b);
        self.s.m.select(self.cur, choose, a, b)
    }

    fn clamp_int(&mut self, x: V, bits: u32, min: i64, max: i64, signed: bool) -> V {
        let t = self.s.m.int(bits);
        let (min, max) = (self.s.m.int_const(t, min), self.s.m.int_const(t, max));
        let (ge, le) = if signed { (pred::SGE, pred::SLE) } else { (pred::IUGE, pred::IULE) };
        let x = self.int_pick(ge, x, min);
        self.int_pick(le, x, max)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                match self.floats.get(&val) {
                    Some(&f) => f,
                    None => {
                        let x = self.i32_of(val)?;
                        self.as_float(x)
                    }
                }
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `f` as an f16 in the low bits, rounded to nearest, ties to even, as the console's
    /// 16-bit float operations round, whatever the host's own conversion does: D3D lets
    /// `f32tof16` round toward zero, and the NVIDIA driver's does (measured on an RTX 4070:
    /// `wave64_tests`). The conversion toward zero, or the next half away from zero where
    /// that one is nearer, or as near and the first one odd. Past the largest half the next
    /// one counts as 2^16, the bound IEEE rounds by, so 65520 and up become infinity.
    /// (`f` is an f32 result of two halves here; rounding it again to 16 bits gives the
    /// rounding of the exact result, 24 bits being more than twice 11 and two.)
    fn f16_nearest(&mut self, f: V) -> V {
        let c = self.cur;
        let f32t = self.s.t.f32;
        let lo = self.f16_toward_zero(f);
        let (one, zero) = (self.s.m.i32(1), self.s.m.i32(0));
        let hi = self.s.m.bin(c, Bin::Add, lo, one);
        let size = |l: &mut Self, h: V| {
            let w = l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
            l.unary(6, w)
        };
        let (below, above) = (size(self, lo), size(self, hi));
        let (magnitude, infinity) = (self.s.m.i32(0x7FFF), self.s.m.i32(0x7C00));
        let m = self.s.m.bin(c, Bin::And, hi, magnitude);
        let top = self.s.m.cmp(c, pred::EQ, m, infinity);
        let bound = self.s.m.f32(65536.0f32.to_bits());
        let above = self.s.m.select(c, top, bound, above);
        let fa = self.unary(6, f);
        // (both differences are exact: the three values lie within one half's step)
        let under = self.s.m.bin(c, Bin::Sub, fa, below);
        let over = self.s.m.bin(c, Bin::Sub, above, fa);
        // Ordered: never for NaN, which stays what the conversion made of it.
        let nearer = self.s.m.cmp(c, pred::OLT, over, under);
        let tie = self.s.m.cmp(c, pred::OEQ, over, under);
        let odd = self.s.m.bin(c, Bin::And, lo, one);
        let odd = self.s.m.cmp(c, pred::NE, odd, zero);
        let tie = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, nearer, tie);
        self.s.m.select(c, up, hi, lo)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        self.bitfield(a, off, width, signed, 32)
    }

    fn bitfield(&mut self, a: V, off: V, width: V, signed: bool, bits: u32) -> V {
        let c = self.cur;
        let ty = self.s.m.int(bits);
        let last = self.s.m.i32(bits - 1);
        let size = self.s.m.i32(bits);
        let zero = self.s.m.i32(0);
        let one = self.s.m.int_const(ty, 1);
        let ones = self.s.m.int_const(ty, -1);
        let off = self.s.m.bin(c, Bin::And, off, last);
        let off = if bits == 64 { self.s.m.cast(c, Cast::ZExt, off, ty) } else { off };
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, size);
        let w = self.s.m.bin(c, Bin::And, width, last);
        let count = if bits == 64 { self.s.m.cast(c, Cast::ZExt, w, ty) } else { w };
        let mask = self.s.m.bin(c, Bin::Shl, one, count);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than the register (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, size, w);
        let left = self.s.m.bin(c, Bin::And, left, last);
        let left = if bits == 64 { self.s.m.cast(c, Cast::ZExt, left, ty) } else { left };
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// Whether an f32 value is NaN, tested in integer arithmetic so host
    /// drivers (such as NVIDIA) do not fold `fcmp uno` away under fast-math.
    fn is_nan_f32(&mut self, f: V) -> V {
        let c = self.cur;
        let bits = self.as_int(f);
        let mask = self.s.m.i32(0x7FFF_FFFF);
        let abs = self.s.m.bin(c, Bin::And, bits, mask);
        let inf = self.s.m.i32(0x7F80_0000);
        self.s.m.cmp(c, pred::IUGT, abs, inf)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    ///
    /// Uses `dx.op.isSpecialFloat.f32` (`IsNaN`) instead of `fcmp uno`: some
    /// D3D12 drivers (observed on Intel Iris Xe) fold `fcmp uno x, x` to false
    /// under default shader fast-math, returning the wrong operand when the
    /// third input is NaN.
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let nan_a = self.is_nan_f32(a);
        let nan_b = self.is_nan_f32(b);
        let nan_3 = self.is_nan_f32(third);
        let nan_ab = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let nan = self.s.m.bin(c, Bin::Or, nan_ab, nan_3);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    /// Flushes f32 subnormals to signed zero, keeping all other values and
    /// the sign of zero intact.
    fn ftz(&mut self, x: V) -> V {
        let c = self.cur;
        let bits = self.as_int(x);
        let exp_mask = self.s.m.i32(0x7F80_0000);
        let sign_mask = self.s.m.i32(0x8000_0000);
        let zero = self.s.m.i32(0);
        let exp = self.s.m.bin(c, Bin::And, bits, exp_mask);
        let sub = self.s.m.cmp(c, pred::EQ, exp, zero);
        let signed_zero = self.s.m.bin(c, Bin::And, bits, sign_mask);
        let flushed = self.s.m.select(c, sub, signed_zero, bits);
        self.as_float(flushed)
    }

    /// A legacy multiply-add as RDNA 2 runs it: fused (`v_fma_legacy_f32`),
    /// with the legacy rule that a zero factor makes the product a +0 rather
    /// than turning an infinite or NaN factor into a NaN.
    fn legacy_fmad(&mut self, a: V, b: V, c: V) -> V {
        let cur = self.cur;
        let fused = self.fmad(a, b, c);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(cur, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(cur, pred::OEQ, b, zero);
        let z = self.s.m.bin(cur, Bin::Or, az, bz);
        // +0 + c, so the sign of a zero addend follows the addition.
        let addend = self.s.m.bin(cur, Bin::Add, zero, c);
        self.s.m.select(cur, z, addend, fused)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            // A partial dispatch's lanes past the title's thread count are in
            // no mask, not even the unconditional one.
            return Ok(match (b, self.partial_active) {
                (Bool::TRUE, Some(a)) => a,
                (b, _) => self.s.m.bool(b == Bool::TRUE),
            });
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        let v = match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        };
        // A lane past a partial dispatch's thread count is in no mask at all.
        v.map(|v| match self.partial_active {
            Some(a) => self.s.m.bin(at, Bin::And, v, a),
            None => v,
        })
    }

    /// The console's decision of the branch closing IR block `b`, which the wave takes
    /// together ([`Structured::wide`]): `when` is the lane's own bit, and the branch is taken
    /// when some lane's `when` holds (`execnz`, `vccnz`, SCC of a mask operation set) or
    /// when every lane's does (`execz`, `vccz`, that SCC clear), [`Structured::wave_any`].
    /// Each host wave asks its lanes and the halves exchange the answers, so both go the
    /// same way and can meet again inside the branch.
    /// SCC of a mask operation as data ([`Structured::scc_some`]): whether some lane of the
    /// console wave has its bit of the result, asked where the operation stands. The lanes
    /// that are elsewhere have no bit (the result is ANDed with EXEC or is EXEC), so with
    /// the console's waves the host wave's answer is the console's; two host waves of 32
    /// exchange theirs, which needs every lane there.
    fn scc_some(&mut self, scc: Value) -> Result<V, Why> {
        let &m = self.x.st.scc_some.get(&scc).ok_or(LANE_MASK)?;
        // (not for a console wave cut into host waves that do not exchange)
        if !self.console_lanes || self.local || self.exchange.is_none() && self.halves.is_some() {
            return Err(LANE_MASK);
        }
        let made = self.def_block.get(&scc).copied();
        let at = match made {
            None => self.body_entry,
            Some(b) if b == self.cur_ir => self.cur,
            Some(b) => self.tail[b].ok_or("a lane mask written after its use")?,
        };
        let mut bit = self.boolean(m, at)?;
        if let Some(active) = self.partial_active {
            bit = self.s.m.bin(at, Bin::And, bit, active);
        }
        let (i1t, i32t) = (self.s.t.i1, self.s.t.i32);
        let some = self.s.op(at, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        if !self.exchange.as_ref().is_some_and(|e| !e.1) {
            return Ok(some);
        }
        if !made.is_none_or(|b| self.all_lanes(b)) {
            need(made.unwrap_or(0));
            return Err("a lane mask exchanged where not every lane runs");
        }
        let some = self.s.m.cast(at, Cast::ZExt, some, i32t);
        let (cont, (lo, hi)) = self.exchange_halves(at, some);
        match made {
            None if self.s.entry == at => self.s.entry = cont,
            Some(b) if b == self.cur_ir => self.cur = cont,
            Some(b) => self.tail[b] = Some(cont),
            None => self.s.entry = cont,
        }
        let both = self.s.m.bin(cont, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(cont, pred::NE, both, z))
    }

    fn wave_wide(&mut self, b: usize, when: Bool) -> Result<V, Why> {
        const WHY: Why = "a wave-wide branch where not every lane runs";
        let any = self.x.st.wave_any.get(b).copied().flatten().ok_or("a wave-wide branch on a scalar condition")?;
        if self.exchange.is_none() && self.halves.is_none() && self.console_lanes && self.mask_data == MaskData::Console {
            // The host's waves are the console's: the wave's own answer. The lanes that are
            // elsewhere have no bit set, as on the console, so "some lane" needs no one but
            // the lanes here, and "every lane's bit is clear" is true of the others anyway.
            let c = self.cur;
            let mut mine = self.boolean(when, c)?;
            // Threads past a DispatchX count speak neither for nor against.
            if let Some(active) = self.partial_active {
                mine = if any {
                    self.s.m.bin(c, Bin::And, mine, active)
                } else {
                    let t = self.s.m.bool(true);
                    let off = self.s.m.bin(c, Bin::Xor, active, t);
                    self.s.m.bin(c, Bin::Or, mine, off)
                };
            }
            let i1t = self.s.t.i1;
            self.s.flags |= flags::WAVE_OPS;
            let (name, code) = if any { ("waveAnyTrue", 113) } else { ("waveAllTrue", 114) };
            return Ok(self.s.op(c, name, Attr::NoUnwind, i1t, code, &[mine]));
        }
        if self.local || !self.exchange.as_ref().is_some_and(|e| !e.1) {
            return Err(WHY);
        }
        if !self.all_lanes(b) {
            need(b);
            return Err(WHY);
        }
        let c = self.cur;
        let mine = self.boolean(when, c)?;
        // what the halves exchange is "some lane speaks against" either way (for `any`:
        // some lane's `when` holds; else: some lane's does not), so a half with no lane says
        // nothing
        let mut set = if any {
            mine
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, mine, t)
        };
        if let Some(active) = self.partial_active {
            set = self.s.m.bin(c, Bin::And, set, active);
        }
        let (i1t, i32t) = (self.s.t.i1, self.s.t.i32);
        let some = self.s.op(c, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[set]);
        self.s.flags |= flags::WAVE_OPS;
        let some = self.s.m.cast(c, Cast::ZExt, some, i32t);
        let (c, (lo, hi)) = self.exchange_halves(c, some);
        self.cur = c;
        let both = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, if any { pred::NE } else { pred::EQ }, both, z))
    }

    /// Whether every lane runs IR block `b` (always true without `exchange`).
    fn all_lanes(&self, b: usize) -> bool {
        self.exchange.as_ref().is_none_or(|(_, _, all)| all.get(b).copied().unwrap_or(false))
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64; host lane = console
    /// lane, so not in a program split in halves).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let lane = if self.exchange.is_some() || self.console_cf {
            self.console_lane()
        } else {
            self.s.op(at, "waveGetLaneIndex", Attr::ReadOnly, self.s.t.i32, 111, &[])
        };
        self.mask_lane_bit(at, lane, lo, hi)
    }

    fn mask_lane_bit(&mut self, at: usize, lane: V, lo: V, hi: Option<V>) -> V {
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    fn select_condition(&mut self, inst: &Inst, mask: Option<&Operand>) -> Result<V, Why> {
        if let Some(&m) = inst
            .writes
            .first()
            .and_then(|&(_, v)| self.x.st.select_masks.get(&v))
        {
            return self.boolean(m, self.cur);
        }
        let Some(&Operand::Scalar { code, .. }) = mask else {
            return Err(UNTRACKED_SELECT);
        };
        if code == NULL {
            return Ok(self.s.m.bool(false));
        }
        if !self.numeric_selects || !self.console_lanes {
            return Err(UNTRACKED_SELECT);
        }
        // A scalar operation can define either dword independently. The SSA
        // operands at this use give the actual pair, even across a join.
        let lo = self.i32_of(inst.read(Reg::S(code)).ok_or(UNTRACKED_SELECT)?)?;
        let hi = if self.x.p.wave64 {
            Some(self.i32_of(inst.read(Reg::S(code + 1)).ok_or(UNTRACKED_SELECT)?)?)
        } else {
            None
        };
        self.s.flags |= flags::WAVE_OPS;
        Ok(self.lane_bit(self.cur, lo, hi))
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        if self.console_cf {
            // Every lane is here: EXEC is the guard. Lane reads and writes
            // ignore EXEC on the console.
            let inst = &self.x.p.blocks[b].insts[i];
            let ignores = matches!(
                inst.name(),
                "v_readlane_b32" | "v_writelane_b32" | "v_readfirstlane_b32"
            ) || matches!(
                inst.fields,
                Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }
            );
            let e = self.x.st.execs[b].get(i).copied().flatten();
            return match e {
                Some(e) if !ignores && e != Bool::TRUE => Ok(Some(self.boolean(e, self.cur)?)),
                _ => Ok(None),
            };
        }
        let g = match self.x.st.guards[b][i] {
            None => None,
            Some(g) => Some(self.boolean(g, self.cur)?),
        };
        Ok(match g {
            Some(g) => Some(self.with_partial(self.cur, g)),
            None => self.partial_active,
        })
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let old = match inst.read(reg).map(|o| self.i32_of(o)) {
                    Some(Ok(o)) => o,
                    // The validator refuses reads of undef.
                    _ => self.s.m.i32(0),
                };
                let v = self.s.m.select(self.cur, g, x, old);
                self.s.m.tag(v, Tag::Guard);
                v
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = match inst.read(reg).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(self.cur, Bin::Mul, f, c);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        let head = self.s.m.cont(self.cur);
        self.s.m.terminate(head, Term::CondBr(g, then, join));
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.s.m.cont(self.cur);
        let head = self.s.m.cont(head);
        self.s.m.terminate(then, Term::Br(join));
        self.s.m.close(then, join);
        self.s.m.close(head, join);
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// A dword of the raw buffer `h` at byte offset `at`.
    fn raw_dword_u(&mut self, h: V, at: V) -> V {
        let c = self.cur;
        let iu = self.s.m.undef(self.s.t.i32);
        let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, at, iu, mask, align],
        );
        let f = self.s.m.extract(c, r, 0);
        self.as_int(f)
    }

    /// The byte stride of the buffer view at heap index `index`, from the
    /// view-stride table (`t`[`super::shader::SLOT_STRIDES`]).
    fn stride_at(&mut self, index: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let at = self.s.m.bin(c, Bin::Mul, index, four);
        let table = self.s.root_srv_handle(super::shader::SLOT_STRIDES);
        self.raw_dword_u(table, at)
    }

    /// Marks the pages a write of `bytes` bytes at byte `x` of the unordered
    /// buffer view `h` touches, in the lanes `guard` lets through
    /// (`shader.rs` `SLOT_VIEW_MARKS`): the bits of its first and last byte,
    /// each set only where still clear. Nothing for handles that are no
    /// heap buffer view (title memory's, say, or a root view's).
    fn mark_view_write(&mut self, h: V, x: V, bytes: u32, guard: Option<V>) {
        let Some(&index) = self.s.view_index.get(&h) else {
            return;
        };
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let table = self.s.root_srv_handle(super::shader::SLOT_VIEW_MARKS);
        let c = self.cur;
        let (eight, four) = (k(self, 8), k(self, 4));
        let at = self.s.m.bin(c, Bin::Mul, index, eight);
        let base = self.raw_dword_u(table, at);
        let at = self.s.m.bin(c, Bin::Add, at, four);
        let phase = self.raw_dword_u(table, at);
        let none = k(self, super::shader::VIEW_MARK_NONE);
        let valid = self.s.m.cmp(c, pred::NE, base, none);
        let active = match guard {
            Some(g) => self.s.m.bin(c, Bin::And, g, valid),
            None => valid,
        };
        let first = self.s.m.bin(c, Bin::Add, phase, x);
        let span = k(self, bytes.max(1) - 1);
        let last = self.s.m.bin(c, Bin::Add, first, span);
        let (k16, cap) = (k(self, 16), k(self, super::shader::VIEW_WRITE_BITS - 1));
        let mut bits = Vec::new();
        for byte in [first, last] {
            let page = self.s.m.bin(c, Bin::LShr, byte, k16);
            let n = self.s.m.bin(c, Bin::Add, base, page);
            let over = self.s.m.cmp(c, pred::IUGT, n, cap);
            bits.push(self.s.m.select(c, over, cap, n));
        }
        let bitmap = self.s.root_uav_handle(super::shader::SLOT_VIEW_WRITES, false);
        let outer = self.begin_guarded(Some(active));
        for (i, &n) in bits.iter().enumerate() {
            let c = self.cur;
            let (k5, k2, k31, one, zero) = (k(self, 5), k(self, 2), k(self, 31), k(self, 1), k(self, 0));
            let word = self.s.m.bin(c, Bin::LShr, n, k5);
            let word = self.s.m.bin(c, Bin::Shl, word, k2);
            let bit = self.s.m.bin(c, Bin::And, n, k31);
            let bit = self.s.m.bin(c, Bin::Shl, one, bit);
            // Most writes find their page marked: a load, not an atomic.
            let old = self.raw_dword_u(bitmap, word);
            let set = self.s.m.bin(c, Bin::And, old, bit);
            let mut clear = self.s.m.cmp(c, pred::EQ, set, zero);
            if i == 1 {
                let other = self.s.m.cmp(c, pred::NE, n, bits[0]);
                clear = self.s.m.bin(c, Bin::And, clear, other);
            }
            let inner = self.begin_guarded(Some(clear));
            let iu = self.s.m.undef(self.s.t.i32);
            // Operation 2 is OR.
            self.s.op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, self.s.t.i32, 78, &[bitmap, k2, word, iu, iu, bit]);
            self.end_guarded(inner, &[]);
        }
        self.end_guarded(outer, &[]);
    }

    /// Atomically updates one location. The first compare also obtains the
    /// initial value, so every observation is atomic, even when it is i64.
    fn atomic_update(&mut self, zero: V, exchange: impl FnOnce(&mut Self, V) -> V) -> V {
        let head = self.cur;
        let retry = self.s.m.new_block();
        let done = self.s.m.new_block();
        self.s.m.terminate(head, Term::Br(retry));
        self.cur = retry;
        let ty = self.s.m.type_of(zero);
        let expected = self.s.m.phi(retry, ty);
        let old = exchange(self, expected);
        let success = self.s.m.cmp(retry, pred::EQ, expected, old);
        self.s.m.set_phi(expected, vec![(zero, head), (old, retry)]);
        self.s.m.terminate(retry, Term::CondBr(success, done, retry));
        self.cur = done;
        old
    }

    /// AMD RDNA 2 ISA sections 12.13–12.14 (including DS_WRAP).
    fn atomic_new_value(&mut self, op: AtomicUpdate, old: V, data: V, data2: V) -> V {
        use AtomicUpdate::*;
        let c = self.cur;
        let ty = self.s.m.type_of(old);
        let zero = self.s.m.int_const(ty, 0);
        let one = self.s.m.int_const(ty, 1);
        match op {
            Inc => {
                let wrap = self.s.m.cmp(c, pred::IUGE, old, data);
                let next = self.s.m.bin(c, Bin::Add, old, one);
                self.s.m.select(c, wrap, zero, next)
            }
            Dec => {
                let at_zero = self.s.m.cmp(c, pred::EQ, old, zero);
                let above = self.s.m.cmp(c, pred::IUGT, old, data);
                let wrap = self.s.m.bin(c, Bin::Or, at_zero, above);
                let next = self.s.m.bin(c, Bin::Sub, old, one);
                self.s.m.select(c, wrap, data, next)
            }
            ReverseSub => self.s.m.bin(c, Bin::Sub, data, old),
            MaskOr => {
                let ones = self.s.m.int_const(ty, -1);
                let keep = self.s.m.bin(c, Bin::Xor, data, ones);
                let kept = self.s.m.bin(c, Bin::And, old, keep);
                self.s.m.bin(c, Bin::Or, kept, data2)
            }
            Wrap => {
                let wrap = self.s.m.cmp(c, pred::IUGE, old, data);
                let low = self.s.m.bin(c, Bin::Sub, old, data);
                let high = self.s.m.bin(c, Bin::Add, old, data2);
                self.s.m.select(c, wrap, low, high)
            }
            ClampSub => {
                let underflow = self.s.m.cmp(c, pred::IULT, old, data);
                let diff = self.s.m.bin(c, Bin::Sub, old, data);
                self.s.m.select(c, underflow, zero, diff)
            }
        }
    }

    fn byte_shift(&mut self, address: V) -> V {
        let three = self.s.m.i32(3);
        let byte = self.s.m.bin(self.cur, Bin::And, address, three);
        self.s.m.bin(self.cur, Bin::Shl, byte, three)
    }

    fn write_subword_result(
        &mut self,
        inst: &Inst,
        reg: u8,
        x: V,
        half: Option<u32>,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let reg = Reg::V(reg);
        let x = if let Some(shift) = half {
            let old = self.i32_of(inst.read(reg).ok_or("a D16 load's previous register value")?)?;
            let c = self.cur;
            let (mask, keep, at) = (
                self.s.m.i32(0xFFFF),
                self.s.m.i32(!(0xFFFF << shift)),
                self.s.m.i32(shift),
            );
            let field = self.s.m.bin(c, Bin::And, x, mask);
            let field = self.s.m.bin(c, Bin::Shl, field, at);
            let old = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, old, field)
        } else {
            x
        };
        self.write(inst, reg, x, guard);
        Ok(())
    }

    /// Loads a byte field via aligned dword accesses. A half starting at
    /// byte 3 reads the first byte of the next dword.
    fn read_subword(
        &mut self,
        address: V,
        width: u32,
        signed: bool,
        dword: impl Fn(&mut Self, V) -> Result<V, Why>,
    ) -> Result<V, Why> {
        let c = self.cur;
        let shift = self.byte_shift(address);
        let not3 = self.s.m.i32(!3);
        let aligned = self.s.m.bin(c, Bin::And, address, not3);
        let w = dword(self, aligned)?;
        let low = self.s.m.bin(c, Bin::LShr, w, shift);
        let x = if width == 16 {
            let k24 = self.s.m.i32(24);
            let split = self.s.m.cmp(c, pred::EQ, shift, k24);
            let at = self.begin_guarded(Some(split));
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(self.cur, Bin::Add, aligned, four);
            let high = dword(self, next)?;
            let eight = self.s.m.i32(8);
            let high = self.s.m.bin(self.cur, Bin::Shl, high, eight);
            let both = self.s.m.bin(self.cur, Bin::Or, low, high);
            self.end_guarded(at, &[(both, low)])[0]
        } else {
            low
        };
        let left = self.s.m.i32(32 - width);
        let x = self.s.m.bin(self.cur, Bin::Shl, x, left);
        Ok(self.s.m.bin(self.cur, if signed { Bin::AShr } else { Bin::LShr }, x, left))
    }

    /// Masks the affected bytes in each dword. A CAS keeps concurrent
    /// writes to other bytes in the same dword from being lost.
    fn write_subword(
        &mut self,
        address: V,
        width: u32,
        data: V,
        store: impl Fn(&mut Self, V, V, V),
    ) {
        let c = self.cur;
        let shift = self.byte_shift(address);
        let not3 = self.s.m.i32(!3);
        let aligned = self.s.m.bin(c, Bin::And, address, not3);
        let mask = self.s.m.i32((1 << width) - 1);
        let data = self.s.m.bin(c, Bin::And, data, mask);
        let bits = self.s.m.bin(c, Bin::Shl, data, shift);
        let mask = self.s.m.bin(c, Bin::Shl, mask, shift);
        store(self, aligned, mask, bits);
        if width == 16 {
            let c = self.cur;
            let k24 = self.s.m.i32(24);
            let split = self.s.m.cmp(c, pred::EQ, shift, k24);
            let at = self.begin_guarded(Some(split));
            let four = self.s.m.i32(4);
            let next = self.s.m.bin(self.cur, Bin::Add, aligned, four);
            let (eight, byte_mask) = (self.s.m.i32(8), self.s.m.i32(255));
            let bits = self.s.m.bin(self.cur, Bin::LShr, data, eight);
            store(self, next, byte_mask, bits);
            self.end_guarded(at, &[]);
        }
    }

    fn lds_ptr(&mut self, byte: V) -> V {
        let shift = self.s.m.i32(if self.wide_lds { 3 } else { 2 });
        let index = self.s.m.bin(self.cur, Bin::LShr, byte, shift);
        let lds = self.s.lds(self.wide_lds);
        self.s.m.element(self.cur, lds, index)
    }

    // DXIL forbids reinterpreting TGSM through a differently sized pointer.
    // When a shader needs i64 atomics, every access uses the same i64 array.
    fn lds_half_shift(&mut self, byte: V) -> V {
        let (four, eight) = (self.s.m.i32(4), self.s.m.i32(8));
        let half = self.s.m.bin(self.cur, Bin::And, byte, four);
        let shift = self.s.m.bin(self.cur, Bin::Mul, half, eight);
        let ty = self.s.m.int(64);
        self.s.m.cast(self.cur, Cast::ZExt, shift, ty)
    }

    fn lds_load_dword(&mut self, byte: V) -> V {
        if self.wide_lds {
            let end = self.s.m.i32(self.s.lds_dwords * 4);
            let in_bounds = self.s.m.cmp(self.cur, pred::IULT, byte, end);
            let at = self.begin_guarded(Some(in_bounds));
            let p = self.lds_ptr(byte);
            let ty = self.s.m.int(64);
            let zero = self.s.m.int_const(ty, 0);
            // Atomic access avoids racing with a lane using the other half.
            let pair = self.s.m.atomic(self.cur, Rmw::Or, p, zero);
            let shift = self.lds_half_shift(byte);
            let x = self.s.m.bin(self.cur, Bin::LShr, pair, shift);
            let x = self.s.m.cast(self.cur, Cast::Trunc, x, self.s.t.i32);
            let zero = self.s.m.i32(0);
            return self.end_guarded(at, &[(x, zero)])[0];
        }
        let p = self.lds_ptr(byte);
        self.s.m.load(self.cur, p)
    }

    fn lds_update_dword(&mut self, byte: V, update: impl FnOnce(&mut Self, V) -> V) -> V {
        let p = self.lds_ptr(byte);
        let zero = if self.wide_lds {
            let ty = self.s.m.int(64);
            self.s.m.int_const(ty, 0)
        } else {
            self.s.m.i32(0)
        };
        let shift = self.wide_lds.then(|| self.lds_half_shift(byte));
        let mut result = None;
        self.atomic_update(zero, |l, old| {
            let c = l.cur;
            let field = if let Some(shift) = shift {
                let field = l.s.m.bin(c, Bin::LShr, old, shift);
                l.s.m.cast(c, Cast::Trunc, field, l.s.t.i32)
            } else {
                old
            };
            result = Some(field);
            let next = update(l, field);
            let next = if let Some(shift) = shift {
                let ty = l.s.m.int(64);
                let next = l.s.m.cast(c, Cast::ZExt, next, ty);
                let next = l.s.m.bin(c, Bin::Shl, next, shift);
                let mask = l.s.m.int_const(ty, i64::from(u32::MAX));
                let mask = l.s.m.bin(c, Bin::Shl, mask, shift);
                l.atomic_new_value(AtomicUpdate::MaskOr, old, mask, next)
            } else {
                next
            };
            l.s.m.cmpxchg(c, p, old, next)
        });
        result.expect("the compare-exchange body was emitted")
    }

    fn lds_store_dword(&mut self, byte: V, data: V) {
        if self.wide_lds {
            self.lds_update_dword(byte, |_, _| data);
        } else {
            let p = self.lds_ptr(byte);
            self.s.m.store(self.cur, p, data);
        }
    }

    fn atomic_data(&mut self, inst: &Inst, data: &Operand, wide: bool) -> Result<V, Why> {
        let words = self.vgprs(inst, data, if wide { 2 } else { 1 })?;
        Ok(if wide { self.join64(words[0], words[1]) } else { words[0] })
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        if self.halves.is_some() {
            let f = self.console_thread();
            self.s.flags |= flags::WAVE_OPS;
            let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
            return Val::I32(match c {
                0 => self.s.m.bin(e, Bin::URem, f, xk),
                1 => {
                    let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                    self.s.m.bin(e, Bin::URem, t, yk)
                }
                _ => self.s.m.bin(e, Bin::UDiv, f, xy),
            });
        }
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.lane_index();
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mut mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        // D3D12 caps the outputs of the declared maxima at 32 KiB. Lane n
        // writes only vertex n and primitive n, so maxima past the lanes are
        // never reached: they come down to the lane count when the outputs
        // do not fit.
        use super::container::{MESH_OUTPUT_LIMIT, mesh_output_bytes};
        let (outputs, prims) = (&self.s.outputs, &self.s.prims);
        let fits = |m: super::shader::Mesh| {
            mesh_output_bytes(outputs, prims, m.max_vertices, m.max_primitives) <= MESH_OUTPUT_LIMIT
        };
        if !fits(mesh) {
            mesh.max_vertices = mesh.max_vertices.min(threads);
            mesh.max_primitives = mesh.max_primitives.min(threads);
            // shortcut: outputs past the cap even at the lane count are
            // still emitted with the lower maxima (drivers take them);
            // refuse them here if one hangs.
            // Experiment (`VICTORY_MESH_OUTPUT_STRICT=1`): they are refused.
            if !fits(mesh) && std::env::var_os("VICTORY_MESH_OUTPUT_STRICT").is_some() {
                return Err("a mesh shader whose outputs pass the host's 32 KiB");
            }
            self.s.mesh = Some(mesh);
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let z = self.s.m.i32(0);
        // Every primitive of the declared count starts culled, also those
        // past the lanes (a count of 256 with 128 lanes): lane n culls
        // primitives n, n + lanes, ... The host draws all it was promised,
        // and one never written has whatever indices its memory held: long
        // triangles over the whole target, each pixel of them shaded.
        let passes = mesh.max_primitives.div_ceil(threads.max(1));
        for pass in 0..passes {
            let index = if pass == 0 {
                flat
            } else {
                let past = self.s.m.i32(pass * threads);
                self.s.m.bin(self.cur, Bin::Add, flat, past)
            };
            let guard = ((pass + 1) * threads > mesh.max_primitives)
                .then(|| self.s.m.cmp(self.cur, pred::IULT, index, p));
            let at = self.begin_guarded(guard);
            self.s.op(
                self.cur,
                "emitIndices",
                Attr::NoUnwind,
                void,
                169,
                &[index, z, z, z],
            );
            self.end_guarded(at, &[]);
        }
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// Read this quad, including pixel helpers across a triangle edge.
    /// Common broadcasts and exchanges need only one native quad operation.
    fn quad_pick(&mut self, x: V, src: V, pattern: Option<u8>) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        match pattern {
            Some(0xe4) => return x,
            Some(p @ (0x00 | 0x55 | 0xaa | 0xff)) => {
                let lane = self.s.m.i32(u32::from(p & 3));
                return self.s.op(c, "quadReadLaneAt.i32", Attr::NoUnwind, self.s.t.i32, 122, &[x, lane]);
            }
            Some(p @ (0xb1 | 0x4e | 0x1b)) => {
                // The low two bits select lane 0's partner: XOR 1, 2, or 3.
                let direction = self.s.m.i8((p & 3) - 1);
                return self.s.op(c, "quadOp.i32", Attr::NoUnwind, self.s.t.i32, 123, &[x, direction]);
            }
            _ => {}
        }
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// `x` in the first lane of this host wave where `guard` holds, and whether it has such a
    /// lane: the console reads the first lane of EXEC, and a lane that got here with EXEC clear
    /// (a guard, not a branch) is not one. With none, `x` of the wave's first lane here.
    fn first_lane(&mut self, x: V, guard: Option<V>) -> (V, V) {
        let c = self.cur;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let first = self.s.op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]);
        let Some(g) = guard else {
            return (first, self.s.m.bool(true));
        };
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[g]);
        let (lo, hi) = (self.s.m.extract(c, r, 0), self.s.m.extract(c, r, 1));
        let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
        // FirstbitLo (dx.op 32): the lowest set bit
        let flo = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let fhi = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
        let in_lo = self.s.m.cmp(c, pred::NE, lo, z);
        let in_hi = self.s.m.cmp(c, pred::NE, hi, z);
        let any = self.s.m.bin(c, Bin::Or, in_lo, in_hi);
        let lane = self.s.m.select(c, in_lo, flo, fhi);
        // no lane in EXEC: read this lane (one that is here); the read is not used
        let me = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let lane = self.s.m.select(c, any, lane, me);
        let at = self.read_lane(x, lane);
        (self.s.m.select(c, any, at, first), any)
    }

    /// Console flow: `x` of console lane `lane` (uniform), from whichever
    /// half holds it.
    fn console_read_lane(&mut self, x: V, lane: V) -> V {
        if !self.split {
            // One host wave is the console wave, all of its lanes here.
            return self.read_lane(x, lane);
        }
        let c = self.cur;
        let (k31, k5, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(5),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let at = self.s.m.bin(c, Bin::And, lane, k31);
        let mine = self.read_lane(x, at);
        let (x0, x1) = self.exchange(c, mine);
        let h = self.s.m.bin(c, Bin::LShr, lane, k5);
        let h = self.s.m.bin(c, Bin::And, h, one);
        let low = self.s.m.cmp(c, pred::EQ, h, z);
        self.s.m.select(c, low, x0, x1)
    }

    /// Whether a read of another lane of this lane's quad must reach helper
    /// lanes: in a pixel shader, always. At a triangle's edge the quad's
    /// other pixels are helpers (and so are discarded ones), which only quad
    /// reads see: wave reads give 0 from them on AMD, so a derivative taken
    /// by hand (`quad_perm`) was the whole value there, the texture read
    /// that used it took its smallest level, and the triangles' edges showed
    /// as lines on car paint, glass and skin. Upstream does this after
    /// discards only (77053a0).
    fn helpers_read(&self, quad: bool) -> bool {
        quad && self.s.pixel
    }

    /// `v_readfirstlane_b32`'s result `first` of `own` in a pixel shader:
    /// a helper lane takes its own value instead (upstream 0601955).
    ///
    /// On the console, whole-quad mode puts helper lanes in EXEC, so a
    /// waterfall loop (read the first lane's value, run the lanes with that
    /// value, take them out of EXEC) serves helper lanes too. On the host,
    /// wave operations leave helper lanes out: once only helper lanes are
    /// left in such a loop, `WaveReadLaneFirst` reads no lane, no lane
    /// matches, and the loop runs to its limit in every such lane, with
    /// whatever descriptor the read gave. With its own value a helper lane
    /// matches in the first pass and leaves; what it computes only feeds
    /// derivatives (those of a texture read's dependent coordinates, at
    /// triangle edges), and its stores are dropped.
    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// Coarse derivatives of `coords` across this lane's quad, as DXIL
    /// gradients (x for up to three coordinates, then y), each times
    /// `scale` if given.
    fn quad_gradients(&mut self, coords: &[V], scale: Option<V>) -> Vec<V> {
        let c = self.cur;
        let lane = self.lane_index();
        let top = self.s.m.i32(!3);
        let base = self.s.m.bin(c, Bin::And, lane, top);
        let lanes: Vec<V> = (1..3u32)
            .map(|k| {
                let k = self.s.m.i32(k);
                self.s.m.bin(c, Bin::Or, base, k)
            })
            .collect();
        let fu = self.s.m.undef(self.s.t.f32);
        let mut g = vec![fu; 6];
        for (k, &x) in coords.iter().enumerate().take(3) {
            let xi = self.as_int(x);
            let pixel = self.s.pixel;
            let at = |l: &mut Self, lane: V| {
                // A pixel's quad through quad reads (see `helpers_read`).
                let r = if pixel { l.quad_pick(xi, lane, None) } else { l.read_lane(xi, lane) };
                l.as_float(r)
            };
            let origin = at(self, base);
            for (d, &lane) in lanes.iter().enumerate() {
                let there = at(self, lane);
                let mut diff = self.s.m.bin(c, Bin::Sub, there, origin);
                if let Some(s) = scale {
                    diff = self.s.m.bin(c, Bin::Mul, diff, s);
                }
                g[3 * d + k] = diff;
            }
        }
        g
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// A byte or halfword LDS read or write at byte address `a + offset`,
    /// through the dword holding it.
    #[allow(clippy::too_many_arguments)]
    fn ds_sub_dword(
        &mut self,
        inst: &Inst,
        sub: SubDword,
        a: V,
        offset: u32,
        dst_reg: Option<u8>,
        data: &[Operand],
        guard: Option<V>,
    ) -> Result<(), Why> {
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        // shortcut: as for dwords, in bounds; a halfword is taken as
        // 2-aligned (the bytes never straddle two dwords).
        let o = k(self, offset);
        let at = self.s.m.bin(c, Bin::Add, a, o);
        let not3 = k(self, !3);
        let aligned = self.s.m.bin(c, Bin::And, at, not3);
        let three = k(self, 3);
        let byte = self.s.m.bin(c, Bin::And, at, three);
        let shift = self.s.m.bin(c, Bin::Shl, byte, three);
        let mask = k(self, (1 << sub.bits) - 1);
        if !sub.write {
            let reg = dst_reg.ok_or("an LDS read result")?;
            let x = self.lds_load_dword(aligned);
            let c = self.cur;
            let x = self.s.m.bin(c, Bin::LShr, x, shift);
            let x = if sub.signed {
                let up = k(self, 32 - sub.bits);
                let x = self.s.m.bin(c, Bin::Shl, x, up);
                self.s.m.bin(c, Bin::AShr, x, up)
            } else {
                self.s.m.bin(c, Bin::And, x, mask)
            };
            let x = match sub.d16 {
                None => x,
                Some(high) => {
                    let old = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                    let old = self.i32_of(old)?;
                    let low16 = k(self, 0xFFFF);
                    let half = self.s.m.bin(c, Bin::And, x, low16);
                    if high {
                        let keep = self.s.m.bin(c, Bin::And, old, low16);
                        let sixteen = k(self, 16);
                        let half = self.s.m.bin(c, Bin::Shl, half, sixteen);
                        self.s.m.bin(c, Bin::Or, keep, half)
                    } else {
                        let high16 = k(self, 0xFFFF_0000);
                        let keep = self.s.m.bin(c, Bin::And, old, high16);
                        self.s.m.bin(c, Bin::Or, keep, half)
                    }
                }
            };
            self.write(inst, Reg::V(reg), x, guard);
            return Ok(());
        }
        let d = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
        let d = if sub.d16 == Some(true) {
            let sixteen = k(self, 16);
            self.s.m.bin(c, Bin::LShr, d, sixteen)
        } else {
            d
        };
        let d = self.s.m.bin(c, Bin::And, d, mask);
        let set = self.s.m.bin(c, Bin::Shl, d, shift);
        let bits = self.s.m.bin(c, Bin::Shl, mask, shift);
        let ones = k(self, !0);
        let clear = self.s.m.bin(c, Bin::Xor, bits, ones);
        // Lanes write other bytes of the same dword at the same time, so
        // the bytes go in atomically: cleared, then set. shortcut: lanes
        // writing the same bytes at once leave both values or-ed together
        // where the console keeps one of them.
        let at = self.begin_guarded(guard);
        if self.wide_lds {
            // The LDS is an i64 array (64-bit atomics): update the dword
            // in place through its compare-exchange loop.
            self.lds_update_dword(aligned, |l, old| {
                let c = l.cur;
                let kept = l.s.m.bin(c, Bin::And, old, clear);
                l.s.m.bin(c, Bin::Or, kept, set)
            });
        } else {
            let p = self.lds_ptr(aligned);
            self.s.m.atomic(self.cur, Rmw::And, p, clear);
            self.s.m.atomic(self.cur, Rmw::Or, p, set);
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        if name == "ds_nop" {
            return Ok(());
        }
        // Lanes past a DispatchX count neither read nor write LDS; a
        // swizzle or permute moves registers only, under its own guard.
        let guard = if name == "ds_swizzle_b32" || name.contains("permute") {
            self.guard(b, i)?
        } else {
            self.effect_guard(b, i)?
        };
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            let use_quad = quad && (self.s.pixel || self.s.mesh.is_some());
            if !use_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if use_quad {
                self.quad_pick(x, src, (offset & 0x8000 != 0).then_some(offset as u8))
            } else {
                self.read_lane(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if name == "ds_bpermute_b32" {
            // RDNA2 ISA §10.4.4: wave64 permutes each 32-lane half
            // independently. Use the host lane's half, not the address.
            self.cross_lane(false)?;
            let at = self.vgprs(inst, addr.as_ref().ok_or("a permute address")?, 1)?[0];
            let x = self.vgprs(inst, data.first().ok_or("a permute source")?, 1)?[0];
            let o = self.s.m.i32(offset);
            let at = self.s.m.bin(c, Bin::Add, at, o);
            let two = self.s.m.i32(2);
            let src = self.s.m.bin(c, Bin::LShr, at, two);
            let mask = self.s.m.i32(31);
            let src = self.s.m.bin(c, Bin::And, src, mask);
            let here = self.lane_index();
            let native_64 = self.s.wave_size == Some(64);
            let src = if native_64 {
                let not31 = self.s.m.i32(!31);
                let half = self.s.m.bin(c, Bin::And, here, not31);
                self.s.m.bin(c, Bin::Or, half, src)
            } else {
                src
            };
            // Never ask the host to read a lane missing from EXEC or absent
            // from a partial wave. It leaves such reads undefined.
            let active = guard.unwrap_or_else(|| self.s.m.bool(true));
            let four = self.s.m.named_struct("dx.types.fouri32", vec![self.s.t.i32; 4]);
            let ballot = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[active]);
            let lo = self.s.m.extract(c, ballot, 0);
            let hi = native_64.then(|| self.s.m.extract(c, ballot, 1));
            let source_active = self.mask_lane_bit(c, src, lo, hi);
            let lane = self.s.m.select(c, source_active, src, here);
            let value = self.read_lane(x, lane);
            let zero = self.s.m.i32(0);
            let r = self.s.m.select(c, source_active, value, zero);
            let reg = dst_reg.ok_or("a permute result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        if name.contains("addtid") {
            return Err("an LDS address based on the console lane ID and M0");
        }
        if name.contains("_src2_") {
            return Err("an LDS operation with a second memory source");
        }
        if name.starts_with("ds_gws") || matches!(name, "ds_append" | "ds_consume" | "ds_ordered_count") {
            return Err("an LDS or GDS queue or global wave synchronization operation");
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        let byte = |l: &mut Self, bytes: u32| {
            let o = l.s.m.i32(bytes);
            l.s.m.bin(l.cur, Bin::Add, a, o)
        };
        let dword = |l: &mut Self, bytes: u32| -> V {
            let at = byte(l, bytes);
            l.lds_ptr(at)
        };
        // shortcut: whole-dword LDS addresses are taken as aligned and in bounds.
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        // Bytes and halves: the dword holding them, and their bit position
        // in it (naturally aligned, so they never straddle dwords).
        // `_d16` reads fill a register's low half (`_d16_hi` its high half)
        // and keep the other; `_d16_hi` writes take the data's high half.
        let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
        let sub = match base {
            "ds_read_u8" | "ds_read_i8" | "ds_write_b8" => Some(8u32),
            "ds_read_u16" | "ds_read_i16" | "ds_write_b16" => Some(16),
            _ => None,
        };
        let sub = sub.filter(|_| {
            base == name
                || name.ends_with("_d16") && base.starts_with("ds_read")
                || name.ends_with("_d16_hi")
        });
        if let Some(bits) = sub {
            let p = dword(self, offset);
            let o = self.s.m.i32(offset);
            let at = self.s.m.bin(c, Bin::Add, a, o);
            let three = self.s.m.i32(3);
            let byte = self.s.m.bin(c, Bin::And, at, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let mask = self.s.m.i32((1u32 << bits) - 1);
            if name.starts_with("ds_read") {
                let reg = dst_reg.ok_or("an LDS read result")?;
                let w = self.s.m.load(c, p);
                let x = self.s.m.bin(c, Bin::LShr, w, sh);
                let x = if base.contains("_i") {
                    let k = self.s.m.i32(32 - bits);
                    let x = self.s.m.bin(c, Bin::Shl, x, k);
                    self.s.m.bin(c, Bin::AShr, x, k)
                } else {
                    self.s.m.bin(c, Bin::And, x, mask)
                };
                let x = if base == name {
                    x
                } else {
                    let old = match inst.read(Reg::V(reg)) {
                        Some(v) => self.i32_of(v)?,
                        None => self.s.m.i32(0),
                    };
                    let half = self.s.m.i32(0xFFFF);
                    let x = self.s.m.bin(c, Bin::And, x, half);
                    let (x, keep) = if name.ends_with("_hi") {
                        let k16 = self.s.m.i32(16);
                        (self.s.m.bin(c, Bin::Shl, x, k16), half)
                    } else {
                        (x, self.s.m.i32(0xFFFF_0000))
                    };
                    let old = self.s.m.bin(c, Bin::And, old, keep);
                    self.s.m.bin(c, Bin::Or, old, x)
                };
                self.write(inst, Reg::V(reg), x, guard);
            } else {
                // Clear, then set: two atomics, so lanes writing other
                // bytes of the dword keep theirs.
                let mut x = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                if name != base {
                    // The data's high half.
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let x = self.s.m.bin(c, Bin::And, x, mask);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let m = self.s.m.bin(c, Bin::Shl, mask, sh);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, m, ones);
                use super::bitcode::Rmw;
                let at = self.begin_guarded(guard);
                self.s.m.atomic(self.cur, Rmw::And, p, keep);
                self.s.m.atomic(self.cur, Rmw::Or, p, x);
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let reg = dst_reg.ok_or("an LDS read result")?;
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS read of bytes or halves");
            };
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let address = byte(self, off + 4 * k);
                    let x = self.lds_load_dword(address);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS write of bytes or halves");
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let address = byte(self, off);
                self.lds_store_dword(address, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        if let Some(max) = match name {
            "ds_max_f32" | "ds_max_rtn_f32" => Some(true),
            "ds_min_f32" | "ds_min_rtn_f32" => Some(false),
            _ => None,
        } {
            let v0 = self.atomic_data(inst, data.first().ok_or("LDS atomic data")?, false)?;
            let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
            let outside = match old_reg {
                Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
                None => self.s.m.i32(0),
            };
            let at = byte(self, offset);
            let p = self.lds_ptr(at);
            let at = self.begin_guarded(guard);
            let r = self.lds_float_min_max(p, v0, max);
            let r = self.end_guarded(at, &[(r, outside)])[0];
            if let Some(reg) = dst_reg {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if name.starts_with("ds_condxchg32") {
            return Err("an LDS conditional exchange with undocumented compare operands");
        }
        let (op, wide, kind) = ds_integer_atomic(name).ok_or("an LDS operation")?;
        if matches!(op, "wrxchg2" | "wrxchg2st64") {
            // AMD RDNA 2 ISA §12.13 specifies separate exchanges at the
            // two addresses. It does not specify their order if they alias.
            if offset0 == offset1 {
                return Err("an LDS paired exchange with coincident addresses");
            }
            let reg = dst_reg.ok_or("an LDS exchange result")?;
            let d0 = self.atomic_data(inst, data.first().ok_or("LDS exchange data")?, wide)?;
            let d1 = self.atomic_data(inst, data.get(1).ok_or("LDS exchange data")?, wide)?;
            let ty = self.s.m.type_of(d0);
            let zero = self.s.m.int_const(ty, 0);
            let at = self.begin_guarded(guard);
            let mut old = Vec::new();
            for ((off, _), data) in pair(name).into_iter().zip([d0, d1]) {
                let address = byte(self, off);
                let x = if !wide && self.wide_lds {
                    self.lds_update_dword(address, |_, _| data)
                } else {
                    let p = self.lds_ptr(address);
                    self.s.m.atomic(self.cur, Rmw::Xchg, p, data)
                };
                old.push((x, zero));
            }
            if self.wide_lds {
                self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_GROUP_SHARED;
            }
            let words = if wide { 2 } else { 1 };
            for (k, x) in self.end_guarded(at, &old).into_iter().enumerate() {
                self.write_atomic_result(inst, reg + k as u8 * words, x, wide, guard);
            }
            return Ok(());
        }
        let rmw = match (op, kind) {
            ("add" | "sub", _) => Some(Rmw::Add),
            ("and", _) => Some(Rmw::And),
            ("or", _) => Some(Rmw::Or),
            ("xor", _) => Some(Rmw::Xor),
            ("min", b'i') => Some(Rmw::Min),
            ("max", b'i') => Some(Rmw::Max),
            ("min", _) => Some(Rmw::UMin),
            ("max", _) => Some(Rmw::UMax),
            ("wrxchg", _) => Some(Rmw::Xchg),
            _ => None,
        };
        let update = match op {
            "rsub" => Some(AtomicUpdate::ReverseSub),
            "inc" => Some(AtomicUpdate::Inc),
            "dec" => Some(AtomicUpdate::Dec),
            "mskor" => Some(AtomicUpdate::MaskOr),
            "wrap" => Some(AtomicUpdate::Wrap),
            _ if rmw.is_some() || op == "cmpst" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.atomic_data(inst, data.first().ok_or("LDS atomic data")?, wide)?;
        let ty = self.s.m.type_of(v0);
        let zero = self.s.m.int_const(ty, 0);
        let v1 = if matches!(op, "mskor" | "wrap" | "cmpst") {
            self.atomic_data(inst, data.get(1).ok_or("a second LDS atomic operand")?, wide)?
        } else {
            zero
        };
        let address = byte(self, offset);
        let at = self.begin_guarded(guard);
        if wide || self.wide_lds {
            self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_GROUP_SHARED;
        }
        let r = if !wide && self.wide_lds {
            self.lds_update_dword(address, |l, old| {
                if let Some(update) = update {
                    return l.atomic_new_value(update, old, v0, v1);
                }
                let c = l.cur;
                match op {
                    "add" => l.s.m.bin(c, Bin::Add, old, v0),
                    "sub" => l.s.m.bin(c, Bin::Sub, old, v0),
                    "and" => l.s.m.bin(c, Bin::And, old, v0),
                    "or" => l.s.m.bin(c, Bin::Or, old, v0),
                    "xor" => l.s.m.bin(c, Bin::Xor, old, v0),
                    "wrxchg" => v0,
                    "cmpst" => {
                        let same = l.s.m.cmp(c, pred::EQ, old, v0);
                        l.s.m.select(c, same, v1, old)
                    }
                    _ => {
                        let pred = match (op, kind) {
                            ("min", b'i') => pred::SLT,
                            ("max", b'i') => pred::SGT,
                            ("min", _) => pred::IULT,
                            ("max", _) => pred::IUGT,
                            _ => unreachable!(),
                        };
                        let take = l.s.m.cmp(c, pred, v0, old);
                        l.s.m.select(c, take, v0, old)
                    }
                }
            })
        } else if let Some(rmw) = rmw {
            let p = self.lds_ptr(address);
            // DXIL supports atomic add, with subtraction encoded as -DATA.
            let data = if op == "sub" { self.s.m.bin(self.cur, Bin::Sub, zero, v0) } else { v0 };
            self.s.m.atomic(self.cur, rmw, p, data)
        } else if let Some(update) = update {
            let p = self.lds_ptr(address);
            self.atomic_update(zero, |l, old| {
                let x = l.atomic_new_value(update, old, v0, v1);
                l.s.m.cmpxchg(l.cur, p, old, x)
            })
        } else {
            // Compare with data0, store data1.
            let p = self.lds_ptr(address);
            self.s.m.cmpxchg(self.cur, p, v0, v1)
        };
        let r = self.end_guarded(at, &[(r, zero)])[0];
        if let Some(reg) = dst_reg {
            self.write_atomic_result(inst, reg, r, wide, guard);
        }
        Ok(())
    }

    /// `ds_max_f32` (`max`) or `ds_min_f32` on the LDS dword at `p` with
    /// the float bits `data`; the old bits return.
    ///
    /// DXIL has no float atomics on group-shared memory, so this is a
    /// compare-exchange loop around [`Lower::lds_float_pick`], which gives
    /// what RDNA2 writes (measured on gfx1036, reported on the pull
    /// request): an sNaN, data first, then memory, is written quieted;
    /// otherwise a NaN loses to a number (two quiet NaNs give the data);
    /// numbers compare as floats with -0 below +0.
    // shortcut: f32 denormals compare as kept; with the console's MODE
    // flushing them they would order as ±0 first (MODE is not modelled).
    fn lds_float_min_max(&mut self, p: V, data: V, max: bool) -> V {
        let pre = self.cur;
        let init = self.s.m.load(pre, p);
        let (lp, exit) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(pre, Term::Br(lp));
        let i32t = self.s.t.i32;
        let old = self.s.m.phi(lp, i32t);
        self.cur = lp;
        let new = self.lds_float_pick(old, data, max);
        let prev = self.s.m.cmpxchg(lp, p, old, new);
        let done = self.s.m.cmp(lp, pred::EQ, prev, old);
        self.s.m.set_phi(old, vec![(init, pre), (prev, lp)]);
        self.s.m.terminate(lp, Term::CondBr(done, exit, lp));
        self.cur = exit;
        old
    }

    /// What `ds_max_f32` (`max`) or `ds_min_f32` writes over the dword
    /// `mem` with `data` (float bits; see [`Lower::lds_float_min_max`]).
    fn lds_float_pick(&mut self, mem: V, data: V, max: bool) -> V {
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (abs_mask, inf, quiet, zero) = (
            k(self, 0x7FFF_FFFF),
            k(self, 0x7F80_0000),
            k(self, 0x40_0000),
            k(self, 0),
        );
        let nan = |l: &mut Self, x: V| {
            let m = l.s.m.bin(c, Bin::And, x, abs_mask);
            l.s.m.cmp(c, pred::IUGT, m, inf)
        };
        let snan = |l: &mut Self, x: V| {
            let n = nan(l, x);
            let q = l.s.m.bin(c, Bin::And, x, quiet);
            let sig = l.s.m.cmp(c, pred::EQ, q, zero);
            l.s.m.bin(c, Bin::And, n, sig)
        };
        // A total order of the numbers as signed integers, -0 below +0:
        // negative floats' magnitude bits reversed.
        let key = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            let sign = l.s.m.bin(c, Bin::AShr, x, k31);
            let flip = l.s.m.bin(c, Bin::And, sign, abs_mask);
            l.s.m.bin(c, Bin::Xor, x, flip)
        };
        let (dk, mk) = (key(self, data), key(self, mem));
        let take = self
            .s
            .m
            .cmp(c, if max { pred::SGT } else { pred::SLT }, dk, mk);
        let r = self.s.m.select(c, take, data, mem);
        let (dn, mn) = (nan(self, data), nan(self, mem));
        // One NaN: the number; two quiet NaNs: the data.
        let r = self.s.m.select(c, mn, data, r);
        let both = self.s.m.bin(c, Bin::And, dn, mn);
        let only_data = {
            let t = self.s.m.bool(true);
            let not_both = self.s.m.bin(c, Bin::Xor, both, t);
            self.s.m.bin(c, Bin::And, dn, not_both)
        };
        let r = self.s.m.select(c, only_data, mem, r);
        // Signalling NaNs win, quieted: the data's first.
        let qm = self.s.m.bin(c, Bin::Or, mem, quiet);
        let sm = snan(self, mem);
        let r = self.s.m.select(c, sm, qm, r);
        let qd = self.s.m.bin(c, Bin::Or, data, quiet);
        let sd = snan(self, data);
        self.s.m.select(c, sd, qd, r)
    }

    // ---- instructions ----
    fn write_atomic_result(&mut self, inst: &Inst, reg: u8, old: V, wide: bool, guard: Option<V>) {
        if wide {
            let (lo, hi) = self.split64(old);
            self.write(inst, Reg::V(reg), lo, guard);
            self.write(inst, Reg::V(reg + 1), hi, guard);
        } else {
            self.write(inst, Reg::V(reg), old, guard);
        }
    }

    /// See [`Lower::ordered`]: programs that reach memory by address or
    /// write through root views, which are the builders of ray tracing's
    /// structures.
    fn ordered(&mut self) -> bool {
        let x = self.x;
        *self.ordered.get_or_insert_with(|| {
            super::rays()
                && (x.res.scalar.iter().any(|(_, a)| a.param == MEMORY)
                    || x.native.used.iter().any(|u| {
                        u.root == RootType::Descriptor && u.kind == ResourceType::Uav
                    }))
        })
    }

    /// Writes to buffers made before are seen before those made after, by
    /// every group.
    fn uav_fence(&mut self) {
        let (mode, void) = (self.s.m.i32(2), self.s.t.void);
        self.s
            .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        if self.dead.contains(&(b, i)) {
            for &(_, v) in &inst.writes {
                self.vals
                    .insert(v, Val::Missing("a spilled SGPR's lane read as data"));
            }
            return Ok(());
        }
        let offset = inst.machine.offset;
        let name = inst.name();
        self.local_ok = name.starts_with("s_bcnt1_i32_")
            || name.starts_with("s_ff1_i32_")
            || name.starts_with("v_mbcnt_");
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            // Mesh: only "do all lanes hold the first lane's value?" (`uniform_test`), which
            // each host wave answers for itself.
            MaskData::None => self.uniform_test(inst),
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. })
                && !(name.starts_with("global_load_") || name.starts_with("scratch_load_"))
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        if self.split_lds
            && matches!(inst.fields, Fields::Ds { .. })
            && !(name.starts_with("ds_swizzle") || name.contains("permute"))
            && self.lds_meet.as_ref().is_none_or(|(all, _)| all[b])
        {
            // the two host waves of a console wave meet before each LDS access: what the other
            // half wrote is visible, and what it still reads is not overwritten
            self.lds_meeting();
        }
        let fence = (name.starts_with("buffer_atomic") || name.starts_with("global_atomic"))
            && self.ordered();
        if fence {
            self.uav_fence();
        }
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                let mut mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                // Without the wait for the group (bit 0) where not every
                // lane arrives, or where the switch says so.
                let waits = match super::group_sync() {
                    0 => false,
                    1 => !self.fence_only.contains(&b),
                    _ => true,
                };
                if !waits {
                    mode &= !1;
                }
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            // The ray and node test. A failure is a refusal at once: an
            // instruction without a side effect otherwise only stops the
            // program where its results are needed, and two trace programs
            // came out without their test that way.
            Fields::Image { .. } if name.starts_with("image_bvh") => {
                return self
                    .bvh(inst, b, i)
                    .map_err(|why| unsupported(offset, why));
            }
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } if name.starts_with("scratch_") => self.scratch(inst, b, i),
            // Generic addresses: private, shared or global memory by aperture.
            Fields::Flat { .. } => Err("flat memory"),
        };
        if fence && r.is_ok() {
            self.uav_fence();
        }
        if let Err(w) = &r && std::env::var_os("VICTORY_WHY").is_some() {
            eprintln!("missing {offset:#x} {name} {w}");
        }
        #[cfg(test)]
        if let Err(why) = r
            && AUDIT_OFFSET.get() == Some(offset)
        {
            return Err(unsupported(offset, why));
        }
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    /// Whether `v` is a register no program reads as a number: a descriptor dword without
    /// a host value ([`Self::stale`]), a lane mask's dword (outside compute, where a ballot
    /// gives the mask as data), or a join of only such values. Compiled code joins numbers
    /// with numbers: where one of these meets numbers at a join, the register is dead on
    /// that path (a variable set in one arm of a branch, its use behind a flag kept in a
    /// VGPR).
    fn dead(&self, v: Value, seen: &mut Vec<Value>) -> bool {
        if self.stale.contains(&v) {
            return true;
        }
        let st = self.x.st;
        if self.mask_data != MaskData::Console
            && (st.masks.contains_key(&v) || st.highs.contains_key(&v))
        {
            return true;
        }
        let Some(&(b, i)) = self.phi_at.get(&v) else {
            return false;
        };
        if seen.contains(&v) {
            return true;
        }
        seen.push(v);
        self.x.p.blocks[b].phis[i]
            .args
            .iter()
            .all(|&a| self.dead(a, seen))
    }

    /// `s_bcnt1_i32_b*` of the lanes that differ from the first lane's value:
    /// `v_readfirstlane_b32 s, v`, `v_cmp_eq_u32 m, s, v`, `s_xor_b* d, m, exec`. The
    /// compiler tests a value for being the same in the whole wave this way, to take a
    /// scalar path; both paths give each lane the same result, so the lanes of a host
    /// wave may ask it of their own first lane, whatever the wave's size.
    fn uniform_test(&self, inst: &Inst) -> bool {
        let p = self.x.p;
        let writer = |v: Value| -> Option<&'a Inst> {
            let b = *self.def_block.get(&v)?;
            p.blocks[b]
                .insts
                .iter()
                .find(|i| i.writes.iter().any(|w| w.1 == v))
        };
        let scalar = |i: &Inst, op: &Operand| match *op {
            Operand::Scalar { code, .. } if code != NULL && code != EXEC => i.read(Reg::S(code)),
            _ => None,
        };
        if !inst.name().starts_with("s_bcnt1_i32_") {
            return false;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return false;
        };
        let Some(xor) = src.first().and_then(|o| scalar(inst, o)).and_then(writer) else {
            return false;
        };
        let Fields::Sop { src, .. } = &xor.fields else {
            return false;
        };
        if !matches!(xor.name(), "s_xor_b64" | "s_xor_b32")
            || !src
                .iter()
                .any(|o| matches!(o, Operand::Scalar { code: EXEC, .. }))
        {
            return false;
        }
        let Some(cmp) = src.iter().find_map(|o| scalar(xor, o)).and_then(writer) else {
            return false;
        };
        let Fields::Valu(f) = &cmp.fields else {
            return false;
        };
        if !matches!(cmp.name(), "v_cmp_eq_u32" | "v_cmp_eq_i32") || f.sdwa.is_some() || f.dpp.is_some() {
            return false;
        }
        let lanes = f.src.iter().find_map(|o| match *o {
            Operand::Vector { reg, .. } => cmp.read(Reg::V(reg)),
            _ => None,
        });
        let first = f.src.iter().find_map(|o| scalar(cmp, o)).and_then(writer);
        lanes.is_some()
            && first.is_some_and(|fl| {
                fl.name() == "v_readfirstlane_b32"
                    && matches!(&fl.fields, Fields::Valu(g)
                        if matches!(g.src.first(), Some(&Operand::Vector { reg, .. })
                            if fl.read(Reg::V(reg)) == lanes))
            })
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.x.native.stage == HwStage::Ps && super::pops::skip_setreg(self.x.native, inst) {
            return Ok(());
        }
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        // A lane mask written by a set operation also gets its value as a
        // number, from its sources' numbers. A ballot where the mask is
        // written misses the lanes it holds that do not run there: a loop's
        // exit mask gathers lanes that left in earlier iterations. Only the
        // masks something reads as numbers get one.
        let number = self.mask_ok
            && mask_set_op(name)
            && inst.writes.iter().any(|w| self.mask_numbers.contains(&w.1));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) && !number {
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        if name == "s_getpc_b64" {
            // The address of the next instruction is no number the host has: tables in
            // the code are read by their tracked address. As a number the counter is
            // that instruction's offset in the code, so an address paths join before
            // one load is the byte in the code, whichever table each path took
            // (resource tracking gives such a join as the computed term, with no base).
            let z = self.s.m.i32(0);
            let at = self.s.m.i32(inst.machine.offset as u32 + 4);
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            self.set(lo, Val::I32(at));
            self.set(hi, Val::I32(z));
            return Ok(());
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                Some(Operand::Literal(x)) if name == "s_bfe_i64" => {
                    // AMDGPUOperandSyntax: a literal in a signed i64 operand sign-extends.
                    Ok(l.s.m.i32(if (*x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        // See AMD's RDNA 2 ISA, sections 12.1–12.4 (SOP2/K/1/C).
        let old = |l: &mut Self, h: u8| -> Result<V, Why> {
            let reg = dst_code.ok_or("a missing destination")?;
            if reg == NULL {
                return Ok(l.s.m.i32(0));
            }
            let value = inst.read(Reg::S(reg + h)).ok_or("an unread destination")?;
            l.i32_of(value)
        };
        if name.starts_with("s_bitcmp") {
            let index = src32(self, 1)?;
            let a = half(self, 0, 0)?;
            let x = if name.ends_with("64") {
                let b = half(self, 0, 1)?;
                let k32 = self.s.m.i32(32);
                let hi = self.s.m.bin(c, Bin::And, index, k32);
                let zero = self.s.m.i32(0);
                let lo = self.s.m.cmp(c, pred::EQ, hi, zero);
                self.s.m.select(c, lo, a, b)
            } else {
                a
            };
            let last = self.s.m.i32(31);
            let index = self.s.m.bin(c, Bin::And, index, last);
            let bit = self.s.m.bin(c, Bin::LShr, x, index);
            let one = self.s.m.i32(1);
            let bit = self.s.m.bin(c, Bin::And, bit, one);
            let k = self.s.m.i32(u32::from(name.starts_with("s_bitcmp1")));
            let test = self.s.m.cmp(c, pred::EQ, bit, k);
            self.set(scc, Val::Bit(test));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            if matches!(name, "s_cmp_eq_u32" | "s_cmp_lg_u32") {
                let s0 = src.first().ok_or("a missing operand")?;
                let s1 = src.get(1).ok_or("a missing operand")?;
                let p0 = self.host_mask_pair(inst, s0)?;
                let p1 = self.host_mask_pair(inst, s1)?;
                let eq = match (p0, p1) {
                    (Some(a), Some(b)) => Some(self.eq_64(c, a, b)),
                    (Some(a), None) if Self::is_zero_operand(s1) => {
                        let z = self.s.m.i32(0);
                        Some(self.eq_64(c, a, (z, z)))
                    }
                    (None, Some(b)) if Self::is_zero_operand(s0) => {
                        let z = self.s.m.i32(0);
                        Some(self.eq_64(c, (z, z), b))
                    }
                    _ => None,
                };
                if let Some(mut r) = eq {
                    if op.starts_with("lg") {
                        let t = self.s.m.bool(true);
                        r = self.s.m.bin(c, Bin::Xor, r, t);
                    }
                    self.set(scc, Val::Bit(r));
                    return Ok(());
                }
            }
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64")
            || name == "s_bitreplicate_b64_b32";
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let r = self.bcnt1_64(c, a, b);
                    let r = if name == "s_bcnt0_i32_b64" {
                        let size = self.s.m.i32(64);
                        self.s.m.bin(c, Bin::Sub, size, r)
                    } else {
                        r
                    };
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff0_i32_b64" | "s_ff1_i32_b64" | "s_flbit_i32_b64" | "s_flbit_i32_i64" => {
                    let (mut a, mut b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    if name == "s_ff0_i32_b64" {
                        let ones = self.s.m.i32(u32::MAX);
                        a = self.s.m.bin(c, Bin::Xor, a, ones);
                        b = self.s.m.bin(c, Bin::Xor, b, ones);
                    } else if name == "s_flbit_i32_i64" {
                        let k31 = self.s.m.i32(31);
                        let sign = self.s.m.bin(c, Bin::AShr, b, k31);
                        a = self.s.m.bin(c, Bin::Xor, a, sign);
                        b = self.s.m.bin(c, Bin::Xor, b, sign);
                    }
                    let op = if name.starts_with("s_flbit") {
                        (a, b) = (b, a);
                        33 // first bit high, counted from the MSB
                    } else {
                        32 // first bit low
                    };
                    let first = self.s.op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                    let second = self.s.op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let offset = self.s.m.bin(c, Bin::Add, second, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let second = self.s.m.select(c, none, second, offset);
                    let found = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, found, first, second);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_brev_b64" => {
                    for (h, dst) in [(0, hi), (1, lo)] {
                        let x = half(self, 0, h)?;
                        let x = self.s.op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[x]);
                        self.set(dst, Val::I32(x));
                    }
                    Ok(())
                }
                "s_bitreplicate_b64_b32" => {
                    let x = src32(self, 0)?;
                    let k16 = self.s.m.i32(16);
                    let upper = self.s.m.bin(c, Bin::LShr, x, k16);
                    for (mut x, dst) in [(x, lo), (upper, hi)] {
                        let mask = self.s.m.i32(0xFFFF);
                        x = self.s.m.bin(c, Bin::And, x, mask);
                        // Spread the 16 source bits onto even positions.
                        for (shift, mask) in [(8, 0x00FF_00FF), (4, 0x0F0F_0F0F),
                                              (2, 0x3333_3333), (1, 0x5555_5555)] {
                            let sh = self.s.m.i32(shift);
                            let m = self.s.m.i32(mask);
                            let t = self.s.m.bin(c, Bin::Shl, x, sh);
                            let t = self.s.m.bin(c, Bin::Or, t, x);
                            x = self.s.m.bin(c, Bin::And, t, m);
                        }
                        let one = self.s.m.i32(1);
                        let other = self.s.m.bin(c, Bin::Shl, x, one);
                        let r = self.s.m.bin(c, Bin::Or, x, other);
                        self.set(dst, Val::I32(r));
                    }
                    Ok(())
                }
                "s_bfe_u64" | "s_bfe_i64" | "s_bfm_b64" | "s_bitset0_b64" | "s_bitset1_b64" => {
                    let ty = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let one = self.s.m.int_const(ty, 1);
                    let k63 = self.s.m.i32(63);
                    let count = |l: &mut Self, x: V| {
                        let x = l.s.m.bin(c, Bin::And, x, k63);
                        l.s.m.cast(c, Cast::ZExt, x, ty)
                    };
                    let r = match name {
                        "s_bfe_u64" | "s_bfe_i64" => {
                            let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                            let a = self.join64(a, b);
                            let off = src32(self, 1)?;
                            let k16 = self.s.m.i32(16);
                            let k127 = self.s.m.i32(127);
                            let w = self.s.m.bin(c, Bin::LShr, off, k16);
                            let w = self.s.m.bin(c, Bin::And, w, k127);
                            self.bitfield(a, off, w, name == "s_bfe_i64", 64)
                        }
                        "s_bfm_b64" => {
                            let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                            let (width, off) = (count(self, a), count(self, b));
                            let m = self.s.m.bin(c, Bin::Shl, one, width);
                            let m = self.s.m.bin(c, Bin::Sub, m, one);
                            self.s.m.bin(c, Bin::Shl, m, off)
                        }
                        _ => {
                            let (a, b) = (old(self, 0)?, old(self, 1)?);
                            let x = self.join64(a, b);
                            let bit = src32(self, 0)?;
                            let bit = count(self, bit);
                            let bit = self.s.m.bin(c, Bin::Shl, one, bit);
                            if name == "s_bitset0_b64" {
                                let ones = self.s.m.int_const(ty, -1);
                                let mask = self.s.m.bin(c, Bin::Xor, bit, ones);
                                self.s.m.bin(c, Bin::And, x, mask)
                            } else {
                                self.s.m.bin(c, Bin::Or, x, bit)
                            }
                        }
                    };
                    let (a, b) = self.split64(r);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    let nz = nonzero(self, a, b);
                    self.set(scc, Val::Bit(nz));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_getpc_b64" => {
                    // Addresses in the code are offsets in it.
                    let next = self.s.m.i32(inst.machine.offset as u32 + 4);
                    let zero = self.s.m.i32(0);
                    self.set(lo, Val::I32(next));
                    self.set(hi, Val::I32(zero));
                    Ok(())
                }
                "s_mov_b64" => {
                    // Each half by itself: two numbers that travel as a pair (two scalars
                    // parked in neighbouring VGPR lanes, say) need not both have a value.
                    for (h, v) in [(0, lo), (1, hi)] {
                        let val = match half(self, 0, h) {
                            Ok(a) => Val::I32(a),
                            Err(why) => Val::Missing(why),
                        };
                        self.set(v, val);
                    }
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                "s_cmov_b64" => {
                    // Keeps the old destination when SCC is clear.
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let a = half(self, 0, h)?;
                        let old = dst_code
                            .and_then(|d| inst.read(Reg::S(d + h)))
                            .ok_or("an unread destination")?;
                        let old = self.i32_of(old)?;
                        let r = self.s.m.select(c, cond, a, old);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_cmov_b32" | "s_cmovk_i32" => {
                // Keeps the old destination when SCC is clear.
                let cond = self.scc_in(inst)?;
                let a = if name == "s_cmovk_i32" {
                    sext16(self)
                } else {
                    src32(self, 0)?
                };
                let d = dst.as_ref().ok_or("a missing operand")?;
                let old = self.operand(inst, d)?;
                (self.s.m.select(c, cond, a, old), None)
            }
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff0_i32_b32" | "s_ff1_i32_b32" | "s_bcnt0_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let op0 = src.first().ok_or("a missing operand")?;
                let pair = if matches!(name, "s_ff1_i32_b32" | "s_bcnt1_i32_b32") {
                    self.host_mask_pair(inst, op0)?
                } else {
                    None
                };
                let r = if let Some((a, b)) = pair {
                    if name == "s_ff1_i32_b32" {
                        self.ff1_64(c, a, b)
                    } else {
                        self.bcnt1_64(c, a, b)
                    }
                } else {
                    let mut a = src32(self, 0)?;
                    if matches!(name, "s_ff0_i32_b32" | "s_bcnt0_i32_b32") {
                        let ones = self.s.m.i32(u32::MAX);
                        a = self.s.m.bin(c, Bin::Xor, a, ones);
                    }
                    let op = if name.starts_with("s_ff") { 32 } else { 31 };
                    self.s
                        .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a])
                };
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                // SCC: the sum overflows 32 bits, through bits shifted out
                // of src0 or the add's carry.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let n = u32::from(name.as_bytes()[6] - b'0');
                let k = self.s.m.i32(n);
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                let r = self.s.m.bin(c, Bin::Add, t, b);
                let (rest, z) = (self.s.m.i32(32 - n), self.s.m.i32(0));
                let out = self.s.m.bin(c, Bin::LShr, a, rest);
                let shifted_out = self.s.m.cmp(c, pred::NE, out, z);
                let carry = self.s.m.cmp(c, pred::IULT, r, t);
                (r, Some(self.s.m.bin(c, Bin::Or, shifted_out, carry)))
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                // SCC: src0 is the one chosen.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), Some(t))
            }
            "s_abs_i32" | "s_absdiff_i32" => {
                let mut a = src32(self, 0)?;
                if name == "s_absdiff_i32" {
                    let b = src32(self, 1)?;
                    a = self.s.m.bin(c, Bin::Sub, a, b);
                }
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_sext_i32_i8" | "s_sext_i32_i16" => {
                let a = src32(self, 0)?;
                (self.sign_extend(a, if name.ends_with("i8") { 8 } else { 16 }), None)
            }
            "s_pack_ll_b32_b16" | "s_pack_lh_b32_b16" | "s_pack_hh_b32_b16" => {
                let (mut a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let k16 = self.s.m.i32(16);
                if name == "s_pack_hh_b32_b16" {
                    a = self.s.m.bin(c, Bin::LShr, a, k16);
                }
                if name != "s_pack_ll_b32_b16" {
                    b = self.s.m.bin(c, Bin::LShr, b, k16);
                }
                (self.pack_halves(a, b), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cmov_b32"
                        | "s_cmovk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    /// Whether the dwords of one console load are read by one call for up
    /// to four of them (`bitcode::opt::WIDE` of `VICTORY_SHADER_OPT`), not
    /// by a call each.
    fn wide_loads() -> bool {
        super::shader_opt() & super::bitcode::opt::WIDE != 0
    }

    /// `count` dwords from byte `addr` on of the raw view `h`, as integers,
    /// read in the current block by one call for every four of them
    /// ([`Self::wide_loads`]). They are the dwords of ONE console load,
    /// which the console reads as one (address plus 4, 8, ... without
    /// wrapping at 32 bits): the same bytes as a call a dword reads. Loads
    /// of different console instructions are never joined: the program
    /// computes their offsets in 32-bit registers, and `x - 4` and `x` at
    /// `x = 0` are one dword out of the view and its first one, where one
    /// call from `x - 4` on would read both out of it.
    fn raw_dwords(&mut self, h: V, addr: V, count: u32) -> Vec<V> {
        let c = self.cur;
        let (u, al) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let mut out = Vec::with_capacity(count as usize);
        for first in (0..count).step_by(4) {
            let n = (count - first).min(4);
            let off = self.s.m.i32(4 * first);
            let a = self.s.m.bin(c, Bin::Add, addr, off);
            let mask = self.s.m.i8((1u8 << n) - 1);
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                139,
                &[h, a, u, mask, al],
            );
            for k in 0..n {
                let x = self.s.m.extract(c, r, k);
                out.push(self.as_int(x));
            }
        }
        out
    }

}

#[allow(dead_code)]
/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
/// The host's limit on a mesh shader's vertex plus primitive outputs
/// (DXC's `kMaxMSOutputTotalBytes`).
const MAX_MESH_OUTPUT_BYTES: u32 = 32768;

#[allow(dead_code)]
/// A mesh shader's output size as the validator counts it: every element's
/// components times the maximum vertex or primitive count, each rounded up
/// to 32, four bytes each.
fn mesh_output_bytes(
    outputs: &[SigElement],
    prims: &[SigElement],
    mesh: super::shader::Mesh,
) -> u32 {
    let scalars = |els: &[SigElement], n: u32| -> u32 {
        els.iter().map(|e| u32::from(e.cols)).sum::<u32>() * n.next_multiple_of(32)
    };
    4 * (scalars(outputs, mesh.max_vertices) + scalars(prims, mesh.max_primitives))
}

/// The coordinates `calculateLOD` takes for a texture of DXIL resource
/// kind `kind` (DXC's `GetNumDimensionsForCalcLOD`): the size's dimensions,
/// without an array slice. The validator wants the rest undef ("out of
/// bound coord must be undef").
fn lod_dims(kind: u32) -> usize {
    match kind {
        1 | 6 => 1,
        2 | 7 => 2,
        _ => 3,
    }
}

/// Gives the elements of each packed row one interpolation mode, as DXIL
/// requires ("interpolation mode that differs from another element packed
/// into the same row"). Elements whose mode is fixed (pixel inputs that
/// are read, integers and `SV_IsFrontFace`, which are constant, and the
/// position) set the row's mode; the others, floats nobody reads (vertex
/// outputs: any float), take it. Rows stay where the console put them:
/// the host links stages by register. A pixel row read two ways is an
/// error rather than a shader the validator rejects; `per_vertex` makes
/// rows with a constant element constant beforehand.
pub(super) fn settle_row_modes(elements: &mut [SigElement], pixel: bool) -> Result<(), Why> {
    // Packed elements only: barycentrics, sample index and coverage are not.
    let packed = |e: &SigElement| e.row.is_some() && !matches!(e.kind, 12 | 14 | 15 | 28 | 30);
    let fixed = |e: &SigElement| e.comp != 9 || e.kind == 3 || (pixel && e.usage != 0);
    let mut rows: Vec<u32> = elements
        .iter()
        .filter(|e| packed(e))
        .filter_map(|e| e.row)
        .collect();
    rows.sort_unstable();
    rows.dedup();
    for row in rows {
        let in_row = |e: &SigElement| packed(e) && e.row == Some(row);
        let mut modes: Vec<u8> = elements
            .iter()
            .filter(|e| in_row(e) && fixed(e))
            .map(|e| e.interp)
            .collect();
        modes.sort_unstable();
        modes.dedup();
        let mode = match modes[..] {
            [m] => m,
            // Nothing fixed: the first element's.
            [] => match elements.iter().find(|e| in_row(e)) {
                Some(e) => e.interp,
                None => continue,
            },
            _ if pixel => return Err("input elements of one row interpolated two ways"),
            // Vertex outputs: the host interpolates by the pixel inputs' modes.
            _ => continue,
        };
        for e in elements.iter_mut().filter(|e| in_row(e) && !fixed(e)) {
            e.interp = mode;
        }
    }
    Ok(())
}

/// A pixel program under conservative depth declares its position centroid
/// (`emit`); one that runs per sample (an `SV_SampleIndex` input or a
/// per-sample interpolation) declares it per sample instead.
pub(super) fn settle_position_mode(inputs: &mut [SigElement]) {
    let per_sample = inputs
        .iter()
        .any(|e| e.kind == 12 || matches!(e.interp, 6 | 7));
    for e in inputs.iter_mut().filter(|e| e.kind == 3 && e.interp == 5) {
        if per_sample {
            e.interp = 7;
        }
    }
}

fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}


#[cfg(test)]
mod tests;

#[cfg(test)]
mod select_tests;
