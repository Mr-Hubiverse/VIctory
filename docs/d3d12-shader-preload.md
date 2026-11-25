# Direct3D 12 checks and shader preload

Choose **DexNom Runtime (Preload and Play)** from the Start menu after running Setup, or run this command from PowerShell.

```powershell
& "$env:LOCALAPPDATA\DexNom Runtime\victory-warm.exe" --play
```

The console shows progress while the runtime scans the installed archives. It caches supported shader translations and compute pipelines, reusing compatible entries. The game starts once the scan finishes and reports at least one successful translation. If the scan fails or finds no translatable shaders, the console displays the error and the log path. The same launcher lock covers both steps, and both use the same settings and cache.

The preload log is identified by `logs/latest-warm.txt`. Once the game starts, `logs/latest.txt` points to the same log. The offline pass reports individual refusals and graphics pipelines are created as the game supplies their draw state.

Pipeline preload checks for a Direct3D 12 device with Shader Model 6.6 before scanning archives. The runtime also respects the reported shader-visible view heap limit. It uses one million descriptors below Resource Binding Tier 3. A Tier 3 adapter can use a larger heap when the runtime and driver support it. [Direct3D 12 hardware tiers](https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-support) · [Descriptor heap limit query](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/VulkanOn12.md#sampler-descriptor-heap-size-increase)

## Verification

On the RTX 5060, Direct3D 12 reported Resource Binding Tier 3 and a maximum of 1,000,000 shader-visible view descriptors. The runtime selected that size. A test using 52 local compute shaders translated 49, created 47 compute pipelines, and reported the remaining refusals. Launcher tests verified that preloading ran first, that both phases shared the same lock and settings, and that failure or an empty scan stopped the launch.
