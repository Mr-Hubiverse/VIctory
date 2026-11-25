# RTX 5060 wave-size diagnosis

The Series X profile reached the opening credits on an RTX 5060 (Windows driver `32.0.16.2124`, Windows build `26300.9974`). The original session ran DexNom commit `2e01ab4`. D3D12 reported 32 lanes for both `WaveLaneCountMin` and `WaveLaneCountMax`. [Microsoft's D3D12 options reference](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_d3d12_options1)

## Compute pipeline refusals

The third session logged `0x80070057` for these six compute pipelines. A separate run of `victory translate --pipelines --list --validate` on the same PC validated the DXIL for all six with the Windows SDK validator. All six still declare a fixed wave size of 64 after translation. The tool now detects the host's maximum automatically, reports the wave-size mismatch, and created pipelines for all 12 comparison shaders selected from successful creations in the same log.

| Compute shader hash | Refused in session 3 (UTC) |
| --- | --- |
| `369cbe906ea72764` | 18:12:02 |
| `02384a60fda6f2d0` | 18:13:26 |
| `2da4beb386509b77` | 18:13:54 |
| `2cc4c197ee29e291` | 18:13:58 |
| `5dc3b25986622949` | 18:19:27 |
| `6f225b641c5b0708` | 18:19:28 |

The runtime now records the adapter name and PCI IDs and skips creation of compute pipelines that declare a wave size above the device's reported maximum. This makes skipped dispatches easier to trace. A wave32 implementation still needs to preserve the order of shared-memory reads and writes inside branches. For groups of multiple console waves, any synchronization that uses a group barrier also needs every thread in the group to participate. Earlier experiments with `VICTORY_WAVE_WIDE=1` reported device loss on another NVIDIA system. [RTX 3060 Ti report](https://github.com/DexNom/VIctory/pull/14) · [RTX 4070 report](https://github.com/DexNom/VIctory/pull/15)

To repeat the offline check from a diagnostics build, substitute the archive from your local game installation:

```powershell
$env:VICTORY_TRANSLATE_ONLY = '369cbe906ea72764,02384a60fda6f2d0,2da4beb386509b77,2cc4c197ee29e291,5dc3b25986622949,6f225b641c5b0708,13dfdd3b1136e407'
& .\target\x86_64-pc-windows-msvc\release\victory.exe translate --pipelines --validate --list --jobs 1 'D:\Path\To\Game\shaders.rpf'
```

## Frame time and termination

The third session started with an existing pipeline library of about 87 MB. The screenshots show roughly 1–2 FPS during the opening credits. One late log sample recorded 1,633.9 ms per frame, including 1,035.5 ms of preparation, 253.8 ms of readback, and 17.0 ms of GPU execution. The runtime also reached its texture budget and reported recurring failures to rebase 12-byte typed buffer views. The effect of those view failures on the image remains unmeasured.

At 18:21:19 UTC, about 9 min 19 s after launch, multiple threads in the title hit the same breakpoint and the process exited with `0x80000003`. The causal link between the missing compute pipelines and the breakpoint remains unverified.

For a controlled test of upload workers and parallel preludes, use the same save and scene with equally warmed caches. Set both variables before starting the launcher so its child process inherits them.

```powershell
$env:VICTORY_PARALLEL_PRELUDES = '1'
$env:VICTORY_FILL_WORKERS = '4'
& "$env:LOCALAPPDATA\DexNom Runtime\victory-play.exe"
```

## Independent validation on RTX 4060

On 25 November 2026, GitHub contributor `molodayamama` reported a clean test merge into `2e01ab4` on a GeForce RTX 4060 8 GB with Windows 10 22H2 and Agility SDK 1.614.1 from PR #12. The contributor reported passing the fixed-wave-size regression, all seven wave64 tests, and all 188 `victory-gpu` library tests. The four `xtask` tests also passed, and Clippy reported zero warnings with both default and all features. The contributor ran these checks without game files. The game-specific wave32 lowering remains open. [GitHub test report](https://github.com/DexNom/VIctory/pull/16#issuecomment-7171554003)

## In-game retest on RTX 5060

The diagnostics build at PR commit `1ae3b85` ran with the Series X profile, experimental RT off, copied saves, and a separate cache. Both upload options were set before launch (`VICTORY_FILL_WORKERS=4` and `VICTORY_PARALLEL_PRELUDES=1`). Scripted Enter presses passed the start screen. The pipeline library was effectively cold at 3,006 bytes, so the loading times are unsuitable for comparison with the earlier warm-cache session.

The runtime identified the NVIDIA GeForce RTX 5060 (PCI `10de:2d05`) and reported a 32-lane maximum. It logged explicit `required=64 max=32` refusals in game for `02384a60fda6f2d0`, `369cbe906ea72764`, and `2da4beb386509b77`. Rendering continued after all three refusals.

The title then terminated during intro prestreaming after about 2 min 21 s from launcher startup (1 min 51 s on the game's clock, frame 943). It attempted a null write and exited with `0xc0000005`. The fault offset `+0x23cf683` and the next two image-relative stack offsets `+0x2215c35` and `+0x2215eca` match the pre-PR crash in the second session. The log contains no DXGI device removal for this run. This test confirms the wave-width diagnostics in game. The cause of the recurring null write remains open.

## Integration run with PRs #1, #7, #9 and #12

The integration build combined `a902405` (AVX2 buffer comparison and merge), `3ac8713` (DXIL handle optimization and a pipeline wait ceiling), `83e78b9` (texture streaming, DirectStorage staging and indirect draws), and `999fd75` (optional Agility SDK) with PR #16 at `1ae3b85`. The combined source built in release with diagnostics. [PR #1](https://github.com/DexNom/VIctory/pull/1) · [PR #7](https://github.com/DexNom/VIctory/pull/7) · [PR #9](https://github.com/DexNom/VIctory/pull/9) · [PR #12](https://github.com/DexNom/VIctory/pull/12)

The test used a separate profile with copied saves, `VICTORY_LOG=trace`, DRED requested, and a foreground PowerShell log. Windows reported that the D3D12 debug layer was unavailable (`0x887A002D`). This run used the system D3D12 runtime on Windows 11. The optional GPU texture detiler from PR #9 and experimental ray tracing were off.

PR #7 reports the same guest stack offsets for the title's 30-second GPU watchdog. It caps pipeline waiting based on the time since the previous Present. The combined run advanced from the menu into the introduction. It recorded a sample with 3,846 draws per frame after 3 min 49 s of wall-clock time without the earlier exception in the observed window. One earlier sample took about 25.2 s, including 8.3 s of preparation, 8.1 s of which was spent on copies. Cold pipeline creation and trace logging make these samples unsuitable for judging frame-rate changes. A single combined run cannot isolate the contribution of any one PR.

The follow-up for PR #16 checks the required fixed wave size against both limits reported by the D3D12 device. Offline translation can still be configured to target another wave width, while the pipeline check uses the actual device's range. [DirectX WaveSize validation rules](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html)
