# Live progress during shader preload

`victory warm` reports the shaders checked, shaders translated, and compute pipelines created at each 2,000-shader boundary. It also reports the first completion after each 15-second interval. The `warm` summary remains the completion signal for Preload and Play.

## Windows x64 on RTX 5060

The 25 November 2026 test combined the local gameplay integration `00b26e5` with the preload launcher from [PR #47](https://github.com/DexNom/VIctory/pull/47), the POPS follow-up from [PR #50](https://github.com/DexNom/VIctory/pull/50) at `33c05ba`, and the progress change. The runtime used a cold isolated cache and `VICTORY_PIPELINE_WAIT_MS=500`.

| Elapsed | Shaders checked | Translated | Compute pipelines |
| ---: | ---: | ---: | ---: |
| 150 s | 22,005 | 21,995 | 627 |
| 300 s | 22,431 | 22,421 | 1,053 |
| 630 s | 23,937 | 23,927 | 2,539 |
| 653 s | 25,758 | 25,745 | 2,699 |

Preload exited with code 0 and launched the game. There were 13 translation refusals and 32 compute pipeline refusals in preload. Ten of the translation refusals were POPS pixel shaders. The game phase logged seven compute pipeline refusals for a fixed 64-lane wave size on the 32-lane adapter.

The game confirmed the 500 ms base wait and exited with code 0 after a window close roughly 6 min 15 s after launch. The requested suspend exceeded its one-second deadline. The game log contains no instance of the earlier `0x80000003` breakpoint, D3D12 device removal, or any of the ten POPS hashes. The session ended before the 7 min 34 s point at which the previous cold run crashed, so stability beyond that point and interactive gameplay remain unverified on this build.
