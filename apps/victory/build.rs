//! Link settings for the Windows executable.
//!
//! victory.exe exports `D3D12SDKVersion` and `D3D12SDKPath` (main.rs): Windows'
//! Direct3D 12 reads them from the process's executable to load Microsoft's
//! newer runtime from `D3D12\` beside it. An executable exports nothing by
//! itself, so the linker is told to.
//!
//! Direct3D 12 reads `D3D12SDKVersion` when `d3d12.dll` is loaded. Imported the
//! usual way, it is loaded before `main` runs, so the version `main` sets from
//! the `D3D12` folder (or Windows' own, without one) came too late and every
//! start asked for 614. Loaded on its first use instead, it reads that version.

fn main() {
    // Threads created without a stack size take the executable's default.
    // Host graphics drivers compiling the title's large shaders on such
    // threads (vkd3d-proton's and the drivers' own, under Wine too) overflow
    // the linker's default of 1 MiB: reserve 16 MiB instead.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/STACK:16777216");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
        println!("cargo:rustc-link-arg-bins=/DELAYLOAD:d3d12.dll");
        println!("cargo:rustc-link-arg-bins=delayimp.lib");
    }
}
