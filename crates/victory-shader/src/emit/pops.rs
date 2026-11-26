//! GFX10 primitive-ordered pixel shading (POPS) before host lowering.

use crate::ShaderError;
use crate::gfx10::{Fields, Operand, Valu};
use crate::ir::{Exit, Inst, Program};
use crate::native::{NativeProgram, StageIo};

pub(super) const EXITING_WAVE_ID: Operand = Operand::Const(239);
pub(super) const NO_COLLISION: u32 = 1 << 16;
const POPS_PACKER: u16 = 25;
const ORDERED_PS_DONE: u16 = 7;

pub(super) fn active(native: &NativeProgram) -> bool {
    matches!(native.io, StageIo::Ps { db_shader_control, .. } if db_shader_control & (1 << 16) != 0)
}

/// The collision input follows the user data and the primitive mask.
/// The host orders UAV accesses with ROVs. Seed distinct synthetic wave IDs
/// with overlap clear so native polling does not wait on an AMD wave.
pub(super) fn collision_register(native: &NativeProgram) -> Option<u8> {
    if !active(native) {
        return None;
    }
    let user = native
        .layout
        .iter()
        .map(|r| u32::from(r.user_data) + r.sgprs)
        .max();
    Some(user.unwrap_or(0).clamp(8, 104) as u8 + 1)
}

pub(super) fn skip_setreg(native: &NativeProgram, inst: &Inst) -> bool {
    disables_ordering(inst) || (active(native) && packer_write(inst).is_some())
}

fn packer_write(inst: &Inst) -> Option<(u16, &[Operand])> {
    match (inst.name(), &inst.fields) {
        (
            "s_setreg_b32" | "s_setreg_imm32_b32",
            Fields::Sop { imm: Some(imm), src, .. },
        ) if imm & 0x3f == POPS_PACKER => Some((*imm, src)),
        _ => None,
    }
}

/// With POPS inactive, explicitly clearing its enable bit changes no shader
/// value. Any other use of the packer or ordered protocol still needs `check`.
pub(super) fn disables_ordering(inst: &Inst) -> bool {
    matches!(packer_write(inst), Some((imm, [Operand::Literal(0)])) if imm >> 6 & 0x1f == 0)
}

fn completes_ordering(inst: &Inst) -> bool {
    match (inst.name(), &inst.fields) {
        ("s_endpgm_ordered_ps_done", _) => true,
        ("s_sendmsg", Fields::Sopp { imm }) => imm & 0xf == ORDERED_PS_DONE,
        _ => false,
    }
}

fn uses_ordering(inst: &Inst) -> bool {
    if packer_write(inst).is_some() {
        return !disables_ordering(inst);
    }
    completes_ordering(inst)
        || match &inst.fields {
            Fields::Sop { src, .. } | Fields::Valu(Valu { src, .. }) => src.contains(&EXITING_WAVE_ID),
            _ => false,
        }
}

pub(super) fn check(p: &Program) -> Result<(), ShaderError> {
    let Some((block, index, inst)) = p.blocks.iter().enumerate().find_map(|(b, block)| {
        block
            .insts
            .iter()
            .enumerate()
            .find(|(_, inst)| uses_ordering(inst))
            .map(|(i, inst)| (b, i, inst))
    }) else {
        return Ok(());
    };

    // Visit all successors after the packer write to find paths that bypass
    // the explicit completion. A rasterizer-ordered view translation needs
    // to account for these paths too.
    if packer_write(inst).is_some()
        && let Some(exit) = exit_before_completion(p, block, index)
    {
        return Err(ShaderError::Unsupported {
            offset: exit,
            what: "primitive-ordered pixel shading (POPS) with a path that bypasses explicit completion",
        });
    }
    Err(ShaderError::Unsupported {
        offset: inst.machine.offset,
        what: "primitive-ordered pixel shading (POPS)",
    })
}

fn exit_before_completion(p: &Program, start: usize, first: usize) -> Option<usize> {
    let mut seen = vec![false; p.blocks.len()];
    let mut pending = vec![(start, first)];
    while let Some((b, index)) = pending.pop() {
        // A back edge to the first block enters it from the beginning.
        if index == 0 && std::mem::replace(&mut seen[b], true) {
            continue;
        }
        let block = &p.blocks[b];
        if block.insts[index..].iter().any(completes_ordering) {
            continue;
        }
        match block.exit {
            Exit::End => {
                return Some(block.insts.last().map_or(block.offset, |i| i.machine.offset));
            }
            exit => pending.extend(exit.successors().into_iter().map(|b| (b, 0))),
        }
    }
    None
}
