import re
whole = open('crates/victory-gpu/src/draw.rs').read()
# strip raw strings r#"..."# (any # count), then normal strings, then line comments
whole = re.sub(r'r#+"(?:.*?)"#+', 'RAWSTR', whole, flags=re.S)
lines = whole.split('\n')
def strip(l):
    l = re.sub('//.*', '', l)
    l = re.sub('"(?:[^"\\\\]|\\\\.)*"', '""', l)
    l = re.sub("'(?:[^'\\\\]|\\\\.)'", "''", l)
    l = l.replace("'static", "STATIC")
    return l
depth = 0
bad = []
for i, l in enumerate(lines):
    s = strip(l)
    depth += s.count('{') - s.count('}')
    if depth < 0:
        bad.append(i + 1)
        depth = 0
print('final depth', depth, 'neg at', bad[:10])
depth = 0
for i, l in enumerate(lines):
    s = strip(l)
    if re.match(r'^\s+(pub\(crate\) )?fn |^\s+(pub\(crate\) )?unsafe fn |^\s*fn ', l) and depth == 0:
        print(f'line {i+1}: top-level-free fn? {l.strip()[:90]}')
    depth += s.count('{') - s.count('}')
