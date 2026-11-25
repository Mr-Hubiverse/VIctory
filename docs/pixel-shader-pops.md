# Primitive ordering in ten refused pixel shaders

The Windows shader-preload run reported 13 translation refusals. Ten were distinct pixel shaders in one alpha-cutout/tint family. The other three were compute shaders addressed by [PR #35](https://github.com/DexNom/VIctory/pull/35).

The identifiers below are the 16-character BLAKE3 prefixes used in the VIctory log. The instruction offsets are relative to the start of the native shader code.

| Shader container hash | Previously reported branch | POPS setup |
| --- | --- | --- |
| `c1dacd75698a5db8` | `0x2dbc` | `0x2d8c` |
| `f0296ce9fbe481cb` | `0x2dbc` | `0x2d8c` |
| `a76619e2dcb45652` | `0x2dbc` | `0x2d8c` |
| `aafcf40b30b9df18` | `0x2dbc` | `0x2d8c` |
| `8c5ae9ea0f2158eb` | `0x2dbc` | `0x2d8c` |
| `c94ff21793ffd81e` | `0x2dbc` | `0x2d8c` |
| `09cee52e3ebbdf51` | `0x3014` | `0x2fe4` |
| `5ac9e7769a54e90c` | `0x3014` | `0x2fe4` |
| `1b1bee1ab00a82b5` | `0x3014` | `0x2fe4` |
| `bb3f113d13bd2788` | `0x3014` | `0x2fe4` |

All ten decode the overlap bit and wave IDs from `s18`, configure `HW_REG_POPS_PACKER`, poll `src_pops_exiting_wave_id`, and send `MSG_ORDERED_PS_DONE` after the ordered section. Those `s18` fields match the documented `COLLISION_WAVEID` input. The ordered section contains texture reads and writes. [AMD RDNA 2 ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf) · [Mesa driver documentation](https://docs.mesa3d.org/drivers/amd/hw/pops.html)

The DXIL emitter now reports `primitive-ordered pixel shading (POPS)` at the first recognized ordering instruction, before generic scalar lowering. The detection uses decoded GFX10 instructions and applies to the pixel stage. A synthetic test covers the packer write, volatile wave-ID source, completion message, and ordered termination.

On an RTX 5060 with driver 621.24, a filtered release run matched all ten hashes. The installed preload build reported ten generic SGPR errors. The patched build reported ten POPS errors at the setup offsets in the table. Both emitted zero DXIL for this set. All 11 tests in `emit::barycentric_tests` passed. A D3D12 feature query on the RTX 5060 returned `ROVsSupported = true`.

Supporting the full family requires tracing the UAV accesses inside the ordered section, proving the coordinates and ordering are representable by D3D12 rasterizer-ordered views, and accounting for ordering between lanes of the same guest wave before removing its synchronization. An overlapping-fragment test should verify the order-dependent read/modify/write results. [D3D12 rasterizer-ordered views](https://learn.microsoft.com/en-us/windows/win32/direct3d12/rasterizer-order-views)
