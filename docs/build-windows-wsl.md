# Building Windows executables from Linux (WSL)

The Windows x64 binaries can be cross-compiled from a Linux WSL distribution using cargo-xwin. The resulting programs still require a supported native Windows Direct3D 12 environment for gameplay. The regular Windows setup remains supported.

## Prerequisites

Install Rustup, the toolchain specified by `rust-toolchain.toml`, Clang/clang-cl, LLD and LLVM utilities, CMake, Ninja, Git, curl and cargo-xwin. Follow Microsoft's SDK/CRT license terms when cargo-xwin prepares its SDK cache. The verified Arch Linux WSL build used Rust 1.98.1, cargo-xwin 0.24.1, and Clang/LLD 23.1.5. Use the repository's pinned compiler and preserve Cargo.lock. Use a native Linux filesystem for the checkout and Cargo target directory; a Windows-mounted output directory is suitable for staging finished binaries.

## Prepare the verified format tables

The source archive does not distribute the separately supplied BCPack table. Follow `docs/asset-policy.md`. Obtain it via the existing `Initialize-FormatTables.ps1` procedure on Windows, or reproduce its pinned download and SHA-256 verification in Linux. The following values are from the initializer at the tested revision. If that script changes, use its updated pin and digest.

```bash
set -euo pipefail
rel=crates/victory-formats/src/bcpack-tables.bin
expected=1f1d4760419202bb5672f0c428ce51e456b5d08fbaec8f371d043fd94e01c36b
if [[ ! -f "$rel" ]]; then
  temp=$(mktemp)
  trap 'rm -f -- "$temp"' EXIT
  curl --fail --location --retry 2 \
    "https://raw.githubusercontent.com/Marvisak/VIctory/2849477415a46f5a079c85cf3e64a3dff32f713c/$rel" \
    -o "$temp"
  printf '%s  %s\n' "$expected" "$temp" | sha256sum -c -
  cp -- "$temp" "$rel"
fi
printf '%s  %s\n' "$expected" "$rel" | sha256sum -c -
```

Do not add the table to a source contribution. If an existing copy fails verification, investigate its origin before replacing it.

## Compile and stage

From the repository root:

```bash
rustup target add x86_64-pc-windows-msvc --toolchain 1.98.1
cargo +1.98.1 fetch --locked --target x86_64-pc-windows-msvc
cargo +1.98.1 xwin build --locked --release \
  --target x86_64-pc-windows-msvc --jobs 2 \
  -p victory --features diagnostics --bins
```

Two build jobs limit peak compilation memory. The first build needs access to package sources and the Windows SDK/CRT cache. If downloads fail, resolve the specific dependency or SDK error in the normal terminal; do not change Cargo.lock to work around a network error. If the checkout uses a newer pinned toolchain, substitute its version in the commands.

Stage the three programs together in a new directory on a Windows drive. Set `stage` to a directory you own (the example is deliberately generic):

```bash
stage=/mnt/c/Users/Public/Victory-build
mkdir -p "$stage"
for name in victory victory-play victory-warm; do
  exe="target/x86_64-pc-windows-msvc/release/$name.exe"
  llvm-readobj --file-headers --coff-imports "$exe"
  cp -n -- "$exe" "$stage/"
  cmp -- "$exe" "$stage/$name.exe"
done
sha256sum "$stage"/*.exe
```

Verify the PE32+ AMD64 architecture, record the source revision and hashes, and run `victory.exe --version` from native Windows PowerShell. A missing VC runtime requires Microsoft's x64 Visual C++ redistributable, as described in the README. Launch with `victory-play.exe` as documented in the README, using the user's own game installation and separate cache/save directories. Do not distribute game data, generated caches or shader dumps with the build. Cross-compilation and a successful version check establish build and startup compatibility. Gameplay and rendering still require separate native testing.

Validated source revision: `2e01ab4760db91820ef5f9a9f85761f83cd6b5d0`. Tested on the documented toolchain with the locked release diagnostics build and native Windows startup. No Windows Rust or Python installation was required for that workflow.
