p = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-core/src/config.rs'
s = open(p, encoding='utf-8').read()
old = '''/// The config file explicitly named by the environment, if any. An empty
/// `VICTORY_CONFIG` means unset, like the other path variables.
fn explicit_config_path(env: impl Fn(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
    env("VICTORY_CONFIG")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

'''
assert s.count(old) == 1, s.count(old)
open(p, 'w', encoding='utf-8').write(s.replace(old, ''))
print('config ok')
