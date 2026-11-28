
import re
src = open('crates/victory-gpu/src/descriptor/host.rs', encoding='utf-8').read()
s = re.sub(r'//[^
]*', '', src)
s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
s = re.sub(r'"(?:[^"\]|\.)*"', '""', s)
lines = s.splitlines()
depth = 0
for i, l in enumerate(lines):
    if re.search(r'fn [a-z_]+', l) and depth > 0:
        print('fn inside depth', depth, 'line', i+1, l.strip()[:80])
    depth += l.count('{') - l.count('}')
    if depth < 0:
        print('negative depth at line', i+1, l.strip()[:80]); depth = 0
print('final depth', depth)
