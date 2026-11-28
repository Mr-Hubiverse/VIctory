import re

path = 'crates/victory-shader/src/native.rs'
src = open(path, encoding='utf-8').read()

# Hunk 1: ours' container_programs + accc_program  vs  theirs' fspx_program.
# Keep container_programs (callers use it) but route ACCC to fspx_program.
wrapper = '''/// The native programs of a container: in its `DXIL` part, or the one of
/// its `ACCC` part.
pub fn container_programs(c: &crate::Container) -> Result<Vec<NativeProgram>, ShaderError> {
    use crate::FourCC;
    if let Some(part) = c.part(FourCC::DXIL) {
        return native_programs(part.data);
    }
    match c.part(FourCC::ACCC) {
        Some(part) => Ok(vec![fspx_program(part.data)?]),
        None => Err(Malformed("no DXIL part")),
    }
}

'''
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
hunks = list(pat.finditer(src))
assert len(hunks) == 2, len(hunks)

# Rebuild: replace hunk1 with wrapper + theirs; hunk2 with theirs.
out = src[:hunks[0].start()] + wrapper + hunks[0].group(2) + src[hunks[0].end():hunks[1].start()] + hunks[1].group(2) + src[hunks[1].end():]
open(path, 'w', encoding='utf-8').write(out)
print('done')
