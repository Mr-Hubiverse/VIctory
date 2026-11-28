p = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-gpu/src/host.rs'
s = open(p, encoding='utf-8').read()
# The flattened line-continued string: the `\D` escape is invalid in Rust.
old = '"host shader model below 6.6: no translated shader can run here. On Windows 10,                  `victory play` fetches Microsoft\'s newer Direct3D 12 runtime (D3D12\\D3D12Core.dll                  beside victory.exe); otherwise update the graphics driver"'
new = '"host shader model below 6.6: no translated shader can run here. On Windows 10, `victory play` fetches Microsoft\'s newer Direct3D 12 runtime (D3D12\\\\D3D12Core.dll beside victory.exe); otherwise update the graphics driver"'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('string fixed')
