# Recurring crash forensics

Working notes on the three failure modes seen in benchmark runs on the
wave32 NVIDIA host (Series X profile, `GTA6 Xbox v1.0.2.1`).
They are ordered by how early they kill a run.

## 1. Translator panic — FIXED (`cf8d3d6`)

`victory-pipelines-N` panicked in `emit/bitcode.rs` with *"forward
reference outside a phi"* during boot warm-up of the saved pipeline
library (~1 s in, several workers at once). Nondeterministic — roughly
one chance in eight per file; `02384a60fda6f2d0` (NumThreads 128),
`0873f80dd84a3c5c`, `0c028454b58543f0`, `0dd520d405264736` all
reproduced. Never seen at `VICTORY_HOST_WAVE=64` (0/15 runs) — the bug
was in the wave32 split path only.

Root cause: `Lower::sop` binds `let c = self.cur`, then resolving the
operands can lazily run `mask_dword`'s ballot + `exchange_halves`, which
splits `self.cur` and joins it at fresh blocks. `sop` then emitted
`unaryBits`/`Add` (the 64-bit popcount of the exchanged ballot halves)
into the stale captured block — a use of the join's phis before the
branch. `self.cur`/`tail[]`/`s.entry` bookkeeping is kept current by the
mask code itself; only caller-side captures went stale, so the fix is in
`Module`, not at each callsite:

- `Module::close(at, into)` records that `at` was split into paths that
  merge at `into`, with the instruction count at the split.
- `Module::push` resolves a stale target by its operands: operands made
  before the split keep the instruction before the branch — it
  dominates the most, which the pair handshake's entry-block flag
  addressing needs (`element(e = s.entry, ...)` used by atomics on the
  handshake path) — while an operand made at the merge or after sends
  the instruction to the merge.
- `Module::cont`/`phi`/`terminate` resolve through the same map;
  `inst_block[]` tracks each instruction's resolved block for `ready`.
- `begin_guarded`/`end_guarded`/`exchange_halves`/`pair_meet` resolve
  the blocks they carry so a phi's incoming labels stay the real
  predecessors.
- Second bug found in-game afterwards (*"unterminated block"* on
  `e95858837a9aa22d`): an exchange at `tail[b]` — a block `exit(b)`
  already terminated — replaced the IR block's exit branch and left the
  join with no way out. `exchange_halves`/`pair_meet` now save `at`'s
  terminator and re-issue it on the merge.

Verified: `cargo test -p victory-shader` 259/259; serial sweep of all
~3,420 saved `.cdxil` shaders (compute, mesh, vertex) panics nowhere;
the two `wave64` unit tests that caught the secondary failure pass.

## 2. Guest script-VM crash (`VCRUNTIME140!+0x1cc66` memmove, `0xc0000005`)

Deterministic: every run that survives failure mode 3 dies here at
10–17 min. The guest's own crash report (`t:\victory-*.crash.log`, on
the host `AppData\Local\victory\data\7952987A\temp`) shows every crash
in a day of runs at **`Game Physical phys=94.6–94.7%`** of a
2,305,818,624-byte (`0x89700000`, 2199 MiB) pool — a memory-threshold
pass, not a timer. Breakpad writes a ~35 MB minidump beside the log.

- Fault chain in `game_scarlett_master.exe` (image base `0x140000000`,
  `.text` RVA `0x1000`–`0x5f40966`, disassemblable with `objdump -d
  --start-address`): frame `+0x3de6491` calls `memmove` with `rdx=0`,
  `r8=0x10`. The function at `0x143de6466` is a **script-VM opcode
  handler**: `r13` is the VM context, `[r13]` its stack pointer;
  handlers dispatch `movzbl 0x30(%rsp)` ×136 into a table at
  `0x14abc3cb0`. The opcode pops a slot expecting a struct pointer,
  computes `src = popped + operandA + 0x18` and `memmove`s `operandB`
  bytes, then `SP += align8(len)`. The popped value was a small negative
  (`-0x18` at operandA 0) — a corrupted/type-confused VM stack slot.
- Outer frames are the guest job system: ConCRT chores
  (`Concurrency::details`, `concrt140.dll` in the game dir). A TLS
  current-context swap wraps the dispatch.
- `SCRIPT TRACE` block in every crash log: the same three-script chain
  `0x72e73a29 → 0xb63fe7d7 → 0xbab1b2fb` at varying PCs; the hashes are
  not resolvable from `script.rpf` directly.
- The minidump also contains an *earlier different* crash — a wild
  pointer `0x34067557b8fd` read at `+0x4ede5a0`.
- `XMemGetWorkingSetStatistics` is never called by the title; its
  budget is derived at boot from `XMemQueryPhysicalMemoryStatistics`
  (10 GiB + 3.5 GiB on a Series X). No xmem failure path hands the guest
  a sentinel — the corruption more likely comes from the title's own
  pool compaction or an HLE native returning a value the script pushes.

Instrumentation committed in `crash.rs` (`cf8d3d6`): on a reported
near-null fault inside the title image, the handler dereferences `r13`
(the SP field) and prints the VM stack slots around the popped position
plus the caller's opcode bytes, via a VirtualQuery-guarded `read_u64`.
**Still untested** — run 7 died to failure mode 3 before reaching the
94.6% window.

## 3. Host device removal (`0x887A0006` / `0x887A0001`)

Stochastic, minutes-scale. Run history on the perf env set:

| Run | Config | Outcome |
| --- | --- | --- |
| 1 | window+gather | device-removed ~2.5 min |
| 2 | window+gather+DRED | device-removed ~4.5 min |
| 3 | both off | ~8.3 min clean exit |
| 4 | window only | no DR; VCRUNTIME ~12.5 min |
| 5 | window+gather | no DR; VCRUNTIME ~17.5 min |
| 6 | window+gather, instrumented | translator panic ~1 s |
| 7 | window+gather, fixed translator | device-removed ~6.5 min ("no host list for a state prelude", `0x887A0001`) |

DRED (run 2) named the stopped command twice: `indirect kind 2`
(`ExecuteIndirect` DISPATCH, max 1) on the ptfx light-atlas kernels
`2561af9577c47289` / `cabb87599722e59a` — wave64→32 single-wave kernels
binding `u4=0xa2db00 [heap 333528: null]` and a null-resource UAV. But
the window/gather path only touches *draw* kinds, so the stopped
dispatch is where the GPU sat at detection; several other lists had
incomplete breadcrumbs (`0/16`, `0/2`, `0/4`) — the remover was an
earlier in-flight command.

The bisect cleared the indirect switches (run 5 ran 17.5 min with both
on). The honest read: **correlates with pipeline-compile churn** —
crashes land mid compile-burst (run 7 died with ~1,690 pipelines in
flight; runs 4–5 cruised past on a warm 835 MB pipeline library). Either
a burst of new PSO creation races in-flight work, or a half-made
pipeline gets bound. The `0x80070057` refusals of `NumThreads>64`
wave64 kernels on the wave32 host are a separate pre-existing emit
behaviour, not the remover.

## Benchmark

~29.8–30.0 fps during early gameplay — the title caps presentation at
33.333 ms and the run hits the cap (README baseline: 1–3 fps).

## Runbook

- Launch: `powershell -File %TEMP%\launch-benchN.ps1` — sets the perf
  env (`VICTORY_WAVE_WIDE`, `INDIRECT_WINDOW/GATHER`, `BVH_BUILDER`,
  `ASYNC_SUBMIT`, `QUEUE2`, `MERGE_*`, `D3D12_DRED`, `SHADER_SAVE`, …)
  and starts `target\diag\victory.exe run <exe> --execute --console
  series-x`, stderr → `ztrx-benchN-err.log`.
- Diag build: `cargo build --profile diag -p victory --features
  diagnostics` → `target\diag\victory.exe` (release codegen +
  diagnostics features).
- Offline reproduce: `target\debug\victory.exe translate <file.cdxil>`;
  saved shader dir `%TEMP%\victory-shaders` (~3,420 files).
- Guest dumps: `T:` maps to
  `%LOCALAPPDATA%\victory\data\7952987A\temp`.
- Guest image disasm: `objdump -d --start-address=0x<va>
  --stop-address=0x<va+N> game_scarlett_master.exe` (image base
  `0x140000000`).
- Shell gotcha: the exec wrapper eats `$VAR` — escape `\$` in loops;
  PowerShell NativeCommandError on stderr gives false FAIL flags.

## Open questions

- What pushes the `-0x18` sentinel onto the VM stack — the 94.6%
  compaction pass itself, or an HLE native returning a bad value the
  script pushes? The committed `crash.rs` dump should name the opcode
  and the neighbours on the next long run.
- Is failure mode 3 a PSO-creation race (churn) or a specific
  command/resource? A pipeline-count-at-crash metric would test the
  churn hypothesis; `VICTORY_D3D12_DEBUG=1` + DRED on a debug-layer-capable
  box may name the call (the layer failed to install here, `0x887A002D`).
