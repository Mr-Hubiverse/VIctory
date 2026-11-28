import re
src = open('crates/victory-loader/src/console_insn.rs', encoding='utf-8').read()
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
hunks = list(pat.finditer(src))
assert len(hunks) == 4, len(hunks)

# hunk0: ours' counters/wait machinery + theirs' patch machinery
m0 = hunks[0].group(1) + hunks[0].group(2)
# hunk1: ours only (theirs is empty)
m1 = hunks[1].group(1)
# hunk2: union of the match arms
m2 = '''        ConsoleOnly::Monitorx => {
            KINDS[K_MONITORX].fetch_add(1, Relaxed);
            MONITOR.with(|m| m.set((ctx.Rax as usize, ctx.Rsi)));
            // Without a prefix, so 3 bytes: replaced, the next pass does not fault.
            if len == 3
                && patches_waits()
                && let Some(new) = wait_patch(what)
                // SAFETY: the title's code, where the instruction just faulted; both
                // replacements are waits that wait for nothing, as the emulation is.
                && unsafe { patch(rip, [code[0], code[1], code[2]], new) }
            {
                #[cfg(feature = "diagnostics")]
                sites::patched();
            }
        }
        ConsoleOnly::Mwaitx => {
            KINDS[K_MWAITX].fetch_add(1, Relaxed);
            wait_step();
            if len == 3
                && patches_waits()
                && let Some(new) = wait_patch(what)
                // SAFETY: as for `monitorx` above.
                && unsafe { patch(rip, [code[0], code[1], code[2]], new) }
            {
                #[cfg(feature = "diagnostics")]
                sites::patched();
            }
            std::hint::spin_loop()
        }
        ConsoleOnly::Clzero | ConsoleOnly::ClzeroStep { .. } => {
            // The loop `mov rax, ptr; add ptr, 0x40; clzero; cmp ptr, end; jb` (the title's
            // +0x4e8ab4c and three more like it zero 896 to 7,040 bytes each, a line an
            // exception) is taken whole from the line it stands at.
            if let ConsoleOnly::ClzeroStep { ptr_reg, end_reg } = what
                && loop15()
                && let Some(done) = unsafe { clzero_loop15(ctx, rip, ptr_reg, end_reg) }
            {
                if !done {
                    return 0;
                }
                ctx.Rip += len as u64;
                EMULATED.fetch_add(1, Relaxed);
                KINDS[K_LOOP15].fetch_add(1, Relaxed);
                line();
                return CONTINUE_EXECUTION;
            }
            KINDS[K_CLZERO].fetch_add(1, Relaxed);
'''
# hunk3: both test sets
m3 = hunks[3].group(1) + hunks[3].group(2)

out = (src[:hunks[0].start()] + m0 + src[hunks[0].end():hunks[1].start()] + m1
       + src[hunks[1].end():hunks[2].start()] + m2 + src[hunks[2].end():hunks[3].start()]
       + m3 + src[hunks[3].end():])
open('crates/victory-loader/src/console_insn.rs', 'w', encoding='utf-8').write(out)
print('done', '<<<<<<<' in out)
