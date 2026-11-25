//! Optional runtime loading of the DirectX 12 Agility SDK (`D3D12Core.dll`).
//!
//! Windows 10's in-box `d3d12.dll` tops out at Shader Model 6.5 and rejects
//! root signatures with `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`. Rather than
//! baking `D3D12SDKVersion`/`D3D12SDKPath` exports into every binary, we load
//! an opt-in Agility SDK at runtime through `ID3D12SDKConfiguration1` when
//! `VICTORY_D3D12_AGILITY` is set or `D3D12\D3D12Core.dll` sits beside the
//! executable. When neither is present, the system runtime is used unchanged.

use std::ffi::CString;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::{
    CLSID_D3D12SDKConfiguration, D3D12_DEVICE_FACTORY_FLAG_ALLOW_RETURNING_EXISTING_DEVICE,
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC, D3D12GetInterface, D3D12SerializeVersionedRootSignature,
    ID3D12Device, ID3D12DeviceConfiguration, ID3D12DeviceFactory, ID3D12SDKConfiguration1,
};
use windows::core::{GUID, IUnknown, Interface, PCSTR, Param};

/// Environment variable naming an Agility SDK `D3D12Core.dll`, its folder, or
/// an unpacked `Microsoft.Direct3D.D3D12` NuGet directory.
pub const ENV_VAR: &str = "VICTORY_D3D12_AGILITY";

/// Process-wide Agility SDK device factory.
pub struct Factory {
    factory: ID3D12DeviceFactory,
    version: u32,
    core_path: PathBuf,
}

// SAFETY: D3D12 device factories are free-threaded COM objects.
unsafe impl Send for Factory {}
// SAFETY: D3D12 device factories are free-threaded COM objects.
unsafe impl Sync for Factory {}

impl Factory {
    /// Returns the process-wide Agility SDK factory when configured, `Ok(None)`
    /// when no Agility SDK is requested or installed beside the executable, or
    /// `Err` if one was found/requested but could not be loaded.
    pub fn load() -> Result<Option<&'static Self>, String> {
        static FACTORY: OnceLock<Result<Option<Factory>, String>> = OnceLock::new();
        match FACTORY.get_or_init(load_uncached) {
            Ok(Some(factory)) => Ok(Some(factory)),
            Ok(None) => Ok(None),
            Err(e) => Err(e.clone()),
        }
    }

    /// The `D3D12SDKVersion` exported by the loaded `D3D12Core.dll`.
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Path to the loaded `D3D12Core.dll`.
    pub fn core_path(&self) -> &Path {
        &self.core_path
    }

    /// Queries a configuration interface (such as `CLSID_D3D12Debug` or
    /// `CLSID_D3D12DeviceRemovedExtendedData`) from this factory.
    pub fn configuration_interface<T: Interface>(&self, clsid: &GUID) -> windows::core::Result<T> {
        // SAFETY: `self.factory` is a live COM factory and `clsid` is a valid GUID reference.
        unsafe { self.factory.GetConfigurationInterface(clsid) }
    }

    /// Creates a D3D12 device through this Agility SDK factory.
    pub fn create_device<P>(
        &self,
        adapter: P,
        level: D3D_FEATURE_LEVEL,
    ) -> windows::core::Result<ID3D12Device>
    where
        P: Param<IUnknown>,
    {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: `self.factory` is live and `device` is a valid out pointer.
        unsafe { self.factory.CreateDevice(adapter, level, &mut device)? };
        device.ok_or_else(windows::core::Error::empty)
    }
}

/// Loads the process-wide Agility SDK factory, if configured.
pub fn load() -> Result<Option<&'static Factory>, String> {
    Factory::load()
}

/// Serializes a versioned root signature through `ID3D12DeviceConfiguration`
/// when the device supports it (Agility SDK / Windows 11 22H2+), falling back
/// to the in-box `D3D12SerializeVersionedRootSignature`.
///
/// This keeps SM 6.6 heap-indexing root signature flags (`0xc01`) working on
/// Windows 10 where the system `d3d12.dll` serializer rejects them.
///
/// # Safety
/// `desc` must point to valid live root-parameter and static-sampler arrays for
/// the duration of the call, matching `D3D12SerializeVersionedRootSignature`.
pub unsafe fn serialize_versioned_root_signature(
    device: &ID3D12Device,
    desc: &D3D12_VERSIONED_ROOT_SIGNATURE_DESC,
    blob: &mut Option<ID3DBlob>,
    error: Option<&mut Option<ID3DBlob>>,
) -> windows::core::Result<()> {
    let error_ptr = error.map(|e| e as *mut Option<ID3DBlob>);
    if let Ok(config) = device.cast::<ID3D12DeviceConfiguration>() {
        // SAFETY: caller upholds `desc` validity; `blob` and `error_ptr` are valid out pointers.
        return unsafe { config.SerializeVersionedRootSignature(desc, blob, error_ptr) };
    }
    // SAFETY: caller upholds `desc` validity; `blob` and `error_ptr` are valid out pointers.
    unsafe { D3D12SerializeVersionedRootSignature(desc, blob, error_ptr) }
}

fn load_uncached() -> Result<Option<Factory>, String> {
    let configured = std::env::var_os(ENV_VAR)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from);
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let Some(core_path) = discover_core(configured.as_deref(), exe_dir.as_deref())? else {
        return Ok(None);
    };
    let bytes = std::fs::read(&core_path)
        .map_err(|e| format!("failed to read {}: {e}", core_path.display()))?;
    let version = read_sdk_version(&bytes)
        .map_err(|e| format!("invalid Agility SDK {}: {e}", core_path.display()))?;
    let sdk_dir = core_path
        .parent()
        .ok_or_else(|| format!("Agility SDK path has no parent: {}", core_path.display()))?;
    let exe_dir =
        exe_dir.ok_or_else(|| "cannot determine current executable directory".to_owned())?;
    let sdk_path = format_sdk_path(&exe_dir, sdk_dir)?;
    let c_path = CString::new(sdk_path.as_bytes())
        .map_err(|_| format!("Agility SDK path contains NUL byte: {sdk_path}"))?;

    // SAFETY: valid CLSID and out pointer for D3D12 SDK configuration interface.
    let config: ID3D12SDKConfiguration1 = unsafe {
        let mut out: Option<ID3D12SDKConfiguration1> = None;
        D3D12GetInterface(&CLSID_D3D12SDKConfiguration, &mut out).map_err(|e| {
            format!(
                "ID3D12SDKConfiguration1 unavailable (update Windows 10 to 19041+ with Agility SDK support): {}",
                e.message()
            )
        })?;
        out.ok_or_else(|| "D3D12GetInterface returned no SDK configuration".to_owned())?
    };

    // SAFETY: `c_path` is a valid NUL-terminated path string and `version` matches `D3D12Core.dll`.
    let factory: ID3D12DeviceFactory = unsafe {
        config
            .CreateDeviceFactory(version, PCSTR(c_path.as_ptr().cast()))
            .map_err(|e| {
                format!(
                    "CreateDeviceFactory(version={version}, path={sdk_path:?}) failed: {}",
                    e.message()
                )
            })?
    };

    // SAFETY: valid flag bitmask on a newly created device factory.
    unsafe {
        let _ = factory.SetFlags(D3D12_DEVICE_FACTORY_FLAG_ALLOW_RETURNING_EXISTING_DEVICE);
        let _ = factory.ApplyToGlobalState();
    }

    Ok(Some(Factory {
        factory,
        version,
        core_path,
    }))
}

/// Resolves `D3D12Core.dll` from an explicit configuration path or from
/// `D3D12\D3D12Core.dll` beside the executable.
fn discover_core(
    configured: Option<&Path>,
    exe_dir: Option<&Path>,
) -> Result<Option<PathBuf>, String> {
    if let Some(path) = configured {
        let candidates = [
            path.to_path_buf(),
            path.join("D3D12Core.dll"),
            path.join("D3D12").join("D3D12Core.dll"),
            path.join("build")
                .join("native")
                .join("bin")
                .join("x64")
                .join("D3D12Core.dll"),
        ];
        for candidate in candidates {
            if candidate.is_file() {
                return Ok(Some(candidate));
            }
        }
        return Err(format!(
            "{ENV_VAR} does not point to D3D12Core.dll or a directory containing it: {}",
            path.display()
        ));
    }

    if let Some(dir) = exe_dir {
        let local = dir.join("D3D12").join("D3D12Core.dll");
        if local.is_file() {
            return Ok(Some(local));
        }
    }
    Ok(None)
}

/// Formats `sdk_dir` as a relative path from `exe_dir` with a trailing
/// backslash, as `ID3D12SDKConfiguration1::CreateDeviceFactory` expects.
///
/// Falls back to an absolute path with a trailing backslash when the two paths
/// reside on different drives.
fn format_sdk_path(exe_dir: &Path, sdk_dir: &Path) -> Result<String, String> {
    let exe = normalize_path(exe_dir);
    let sdk = normalize_path(sdk_dir);

    let exe_comps: Vec<Component<'_>> = exe.components().collect();
    let sdk_comps: Vec<Component<'_>> = sdk.components().collect();

    let same_prefix = match (exe_comps.first(), sdk_comps.first()) {
        (Some(Component::Prefix(a)), Some(Component::Prefix(b))) => a
            .as_os_str()
            .to_string_lossy()
            .eq_ignore_ascii_case(&b.as_os_str().to_string_lossy()),
        (Some(Component::RootDir), Some(Component::RootDir)) => true,
        _ => false,
    };

    if !same_prefix {
        let mut abs = sdk.to_string_lossy().into_owned();
        if !abs.ends_with('\\') {
            abs.push('\\');
        }
        return Ok(abs);
    }

    let mut common = 0usize;
    while common < exe_comps.len() && common < sdk_comps.len() {
        let eq = match (&exe_comps[common], &sdk_comps[common]) {
            (Component::Prefix(a), Component::Prefix(b)) => a
                .as_os_str()
                .to_string_lossy()
                .eq_ignore_ascii_case(&b.as_os_str().to_string_lossy()),
            (Component::RootDir, Component::RootDir) => true,
            (Component::Normal(a), Component::Normal(b)) => a
                .to_string_lossy()
                .eq_ignore_ascii_case(&b.to_string_lossy()),
            _ => false,
        };
        if !eq {
            break;
        }
        common += 1;
    }

    let mut parts: Vec<String> = Vec::new();
    for comp in &exe_comps[common..] {
        if matches!(comp, Component::Normal(_)) {
            parts.push("..".to_owned());
        }
    }
    for comp in &sdk_comps[common..] {
        if let Component::Normal(seg) = comp {
            let s = seg
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 path component in {}", sdk.display()))?;
            parts.push(s.to_owned());
        }
    }

    if parts.is_empty() {
        Ok(".\\".to_owned())
    } else {
        let mut out = parts.join("\\");
        out.push('\\');
        Ok(out)
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    strip_verbatim_prefix(&resolved)
}

fn strip_verbatim_prefix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        path.to_path_buf()
    }
}

/// Reads the exported `D3D12SDKVersion` `u32` directly from a PE/PE32+
/// `D3D12Core.dll` image without loading or executing the DLL.
pub fn read_sdk_version(bytes: &[u8]) -> Result<u32, String> {
    if bytes.len() < 0x40 || &bytes[0..2] != b"MZ" {
        return Err("missing DOS header".to_owned());
    }
    let pe_off = read_u32(bytes, 0x3c)? as usize;
    if pe_off.checked_add(24).is_none_or(|end| end > bytes.len())
        || &bytes[pe_off..pe_off + 4] != b"PE\0\0"
    {
        return Err("missing PE signature".to_owned());
    }
    let coff = pe_off + 4;
    let num_sections = usize::from(read_u16(bytes, coff + 2)?);
    let opt_size = usize::from(read_u16(bytes, coff + 16)?);
    let opt = coff + 20;
    let sections_off = opt
        .checked_add(opt_size)
        .ok_or_else(|| "optional header overflow".to_owned())?;
    if sections_off > bytes.len() || opt_size < 2 {
        return Err("truncated optional header".to_owned());
    }

    let magic = read_u16(bytes, opt)?;
    let (rva_count_rel, export_dir_rel) = match magic {
        0x20b => (108usize, 112usize), // PE32+
        0x10b => (92usize, 96usize),   // PE32
        _ => return Err(format!("unsupported PE optional header magic {magic:#x}")),
    };
    if opt_size < export_dir_rel + 8 {
        return Err("optional header has no export directory".to_owned());
    }
    let num_dirs = read_u32(bytes, opt + rva_count_rel)?;
    if num_dirs == 0 {
        return Err("PE image has no data directories".to_owned());
    }
    let export_rva = read_u32(bytes, opt + export_dir_rel)?;
    if export_rva == 0 {
        return Err("PE image has no export table".to_owned());
    }

    let mut sections = Vec::with_capacity(num_sections);
    for i in 0..num_sections {
        let base = sections_off
            .checked_add(i * 40)
            .ok_or_else(|| "section table overflow".to_owned())?;
        if base + 40 > bytes.len() {
            return Err("truncated section header".to_owned());
        }
        let virt_size = read_u32(bytes, base + 8)?;
        let virt_addr = read_u32(bytes, base + 12)?;
        let raw_size = read_u32(bytes, base + 16)?;
        let raw_ptr = read_u32(bytes, base + 20)?;
        sections.push((virt_addr, virt_size.max(raw_size), raw_ptr, raw_size));
    }

    let rva_to_off = |rva: u32| -> Result<usize, String> {
        for &(va, span, raw_ptr, raw_size) in &sections {
            if let Some(delta) = rva.checked_sub(va)
                && delta < span
                && delta < raw_size
            {
                let off = (raw_ptr as usize)
                    .checked_add(delta as usize)
                    .ok_or_else(|| "section offset overflow".to_owned())?;
                if off < bytes.len() {
                    return Ok(off);
                }
            }
        }
        Err(format!("RVA {rva:#x} is not backed by file data"))
    };

    let exp_off = rva_to_off(export_rva)?;
    if exp_off + 40 > bytes.len() {
        return Err("truncated export directory".to_owned());
    }
    let num_names = read_u32(bytes, exp_off + 24)? as usize;
    let funcs_rva = read_u32(bytes, exp_off + 28)?;
    let names_rva = read_u32(bytes, exp_off + 32)?;
    let ords_rva = read_u32(bytes, exp_off + 36)?;

    let funcs_off = rva_to_off(funcs_rva)?;
    let names_off = rva_to_off(names_rva)?;
    let ords_off = rva_to_off(ords_rva)?;

    for i in 0..num_names {
        let name_rva = read_u32(bytes, names_off + i * 4)?;
        let name_off = rva_to_off(name_rva)?;
        let tail = &bytes[name_off..];
        let Some(nul) = tail.iter().position(|&b| b == 0) else {
            return Err("unterminated export name".to_owned());
        };
        if &tail[..nul] == b"D3D12SDKVersion" {
            let ordinal = usize::from(read_u16(bytes, ords_off + i * 2)?);
            let sym_rva = read_u32(bytes, funcs_off + ordinal * 4)?;
            let sym_off = rva_to_off(sym_rva)?;
            let version = read_u32(bytes, sym_off)?;
            if version == 0 {
                return Err("D3D12SDKVersion export is 0".to_owned());
            }
            return Ok(version);
        }
    }

    Err("D3D12SDKVersion export not found".to_owned())
}

fn read_u16(bytes: &[u8], off: usize) -> Result<u16, String> {
    let slice = bytes
        .get(off..off + 2)
        .ok_or_else(|| format!("unexpected EOF at {off:#x}"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], off: usize) -> Result<u32, String> {
    let slice = bytes
        .get(off..off + 4)
        .ok_or_else(|| format!("unexpected EOF at {off:#x}"))?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "victory-agility-{label}-{}-{id}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Builds a minimal valid PE32+ image with one `.rdata` section exporting
    /// `OtherSymbol` and (when `version` is `Some`) `D3D12SDKVersion`.
    fn synthetic_pe32_plus(version: Option<u32>) -> Vec<u8> {
        let mut img = vec![0u8; 0x600];
        img[0..2].copy_from_slice(b"MZ");
        let pe_off = 0x80usize;
        img[0x3c..0x40].copy_from_slice(&(pe_off as u32).to_le_bytes());
        img[pe_off..pe_off + 4].copy_from_slice(b"PE\0\0");

        let coff = pe_off + 4;
        img[coff + 2..coff + 4].copy_from_slice(&1u16.to_le_bytes()); // 1 section
        let opt_size = 0xf0u16;
        img[coff + 16..coff + 18].copy_from_slice(&opt_size.to_le_bytes());

        let opt = coff + 20;
        img[opt..opt + 2].copy_from_slice(&0x20bu16.to_le_bytes()); // PE32+
        img[opt + 108..opt + 112].copy_from_slice(&16u32.to_le_bytes()); // NumberOfRvaAndSizes
        // Export directory at RVA 0x1000.
        img[opt + 112..opt + 116].copy_from_slice(&0x1000u32.to_le_bytes());
        img[opt + 116..opt + 120].copy_from_slice(&0x100u32.to_le_bytes());

        // Single section: VA 0x1000, size 0x400, raw ptr 0x200.
        let sec = opt + usize::from(opt_size);
        img[sec..sec + 6].copy_from_slice(b".rdata");
        img[sec + 8..sec + 12].copy_from_slice(&0x400u32.to_le_bytes());
        img[sec + 12..sec + 16].copy_from_slice(&0x1000u32.to_le_bytes());
        img[sec + 16..sec + 20].copy_from_slice(&0x400u32.to_le_bytes());
        img[sec + 20..sec + 24].copy_from_slice(&0x200u32.to_le_bytes());

        // Export directory at file offset 0x200 (RVA 0x1000).
        let exp = 0x200usize;
        let num_names: u32 = if version.is_some() { 2 } else { 1 };
        img[exp + 20..exp + 24].copy_from_slice(&num_names.to_le_bytes());
        img[exp + 24..exp + 28].copy_from_slice(&num_names.to_le_bytes());
        img[exp + 28..exp + 32].copy_from_slice(&0x1040u32.to_le_bytes()); // functions
        img[exp + 32..exp + 36].copy_from_slice(&0x1060u32.to_le_bytes()); // names
        img[exp + 36..exp + 40].copy_from_slice(&0x1080u32.to_le_bytes()); // ordinals

        // Functions: [0] -> 0x1100, [1] -> 0x1104
        img[0x240..0x244].copy_from_slice(&0x1100u32.to_le_bytes());
        img[0x244..0x248].copy_from_slice(&0x1104u32.to_le_bytes());
        // Names: [0] -> 0x1120 ("OtherSymbol"), [1] -> 0x1140 ("D3D12SDKVersion")
        img[0x260..0x264].copy_from_slice(&0x1120u32.to_le_bytes());
        img[0x264..0x268].copy_from_slice(&0x1140u32.to_le_bytes());
        // Ordinals: [0] -> 0, [1] -> 1
        img[0x280..0x282].copy_from_slice(&0u16.to_le_bytes());
        img[0x282..0x284].copy_from_slice(&1u16.to_le_bytes());

        img[0x300..0x304].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        if let Some(v) = version {
            img[0x304..0x308].copy_from_slice(&v.to_le_bytes());
        }
        img[0x320..0x320 + 12].copy_from_slice(b"OtherSymbol\0");
        img[0x340..0x340 + 16].copy_from_slice(b"D3D12SDKVersion\0");
        img
    }

    #[test]
    fn reads_sdk_version_from_pe32_plus_exports() {
        let pe = synthetic_pe32_plus(Some(614));
        assert_eq!(read_sdk_version(&pe).unwrap(), 614);

        let missing = synthetic_pe32_plus(None);
        assert!(read_sdk_version(&missing).is_err());

        let zero = synthetic_pe32_plus(Some(0));
        assert!(read_sdk_version(&zero).is_err());

        assert!(read_sdk_version(b"not a PE").is_err());
    }

    #[test]
    fn discovers_core_across_supported_layouts() {
        let tmp = TempDir::new("discover");

        // No env var and no local D3D12\D3D12Core.dll -> None.
        assert_eq!(discover_core(None, Some(&tmp.0)).unwrap(), None);

        // Local D3D12\D3D12Core.dll beside executable.
        let local = tmp.0.join("D3D12").join("D3D12Core.dll");
        std::fs::create_dir_all(local.parent().unwrap()).unwrap();
        std::fs::write(&local, b"stub").unwrap();
        assert_eq!(
            discover_core(None, Some(&tmp.0)).unwrap(),
            Some(local.clone())
        );

        // Direct DLL path, parent folder, and unpacked NuGet root.
        assert_eq!(
            discover_core(Some(&local), None).unwrap(),
            Some(local.clone())
        );
        assert_eq!(
            discover_core(Some(local.parent().unwrap()), None).unwrap(),
            Some(local.clone())
        );
        assert_eq!(discover_core(Some(&tmp.0), None).unwrap(), Some(local));

        let nuget = TempDir::new("nuget");
        let nuget_dll = nuget
            .0
            .join("build")
            .join("native")
            .join("bin")
            .join("x64")
            .join("D3D12Core.dll");
        std::fs::create_dir_all(nuget_dll.parent().unwrap()).unwrap();
        std::fs::write(&nuget_dll, b"stub").unwrap();
        assert_eq!(
            discover_core(Some(&nuget.0), None).unwrap(),
            Some(nuget_dll)
        );

        let empty = TempDir::new("empty");
        assert!(discover_core(Some(&empty.0), None).is_err());
    }

    #[test]
    fn formats_relative_paths_from_exe_directory() {
        assert_eq!(
            format_sdk_path(Path::new(r"C:\app"), Path::new(r"C:\app")).unwrap(),
            r".\"
        );
        assert_eq!(
            format_sdk_path(Path::new(r"C:\app"), Path::new(r"C:\app\D3D12")).unwrap(),
            r"D3D12\"
        );
        assert_eq!(
            format_sdk_path(
                Path::new(r"\\?\E:\Repo\target\debug\deps"),
                Path::new(r"e:\repo\work\agility\x64"),
            )
            .unwrap(),
            r"..\..\..\work\agility\x64\"
        );
        assert_eq!(
            format_sdk_path(Path::new(r"E:\app"), Path::new(r"C:\sdk\x64")).unwrap(),
            r"C:\sdk\x64\"
        );
    }

    #[test]
    fn configured_agility_sdk_exposes_sm66_and_heap_indexed_root_signature() {
        if std::env::var_os(ENV_VAR).is_none_or(|v| v.is_empty()) {
            return;
        }
        let factory = load()
            .expect("Agility SDK configured via VICTORY_D3D12_AGILITY must load")
            .expect("Agility SDK factory must be present");
        assert!(
            factory.version() >= 600,
            "unexpected SDK version {}",
            factory.version()
        );

        let host = super::super::host::Host::new(false)
            .or_else(|_| super::super::host::Host::new(true))
            .expect("Agility SDK must provide an SM 6.6 device");
        let _root = host
            .root_signature_for(&[])
            .expect("SM 6.6 heap-indexed root signature must serialize and create");
    }
}
