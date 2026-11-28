import re
whole = open('crates/victory-gpu/src/draw.rs').read()
whole = re.sub(r'r#+"(?:.*?)"#+', 'RAWSTR', whole, flags=re.S)
lines = whole.split('\n')
def strip(l):
    l = re.sub(r'//.*', '', l)
    l = re.sub(r'"(?:[^"\\]|\\.)*"', '""', l)
    l = re.sub(r"'(?:[^'\\]|\\.)'", "''", l)
    l = l.replace("'static", "STATIC")
    return l
depth = 0
lo, hi = 7430, 8220
for i, l in enumerate(lines):
    if i + 1 < lo:
        s = strip(l)
        depth += s.count('{') - s.count('}')
        continue
    if i + 1 > hi:
        break
    s = strip(l)
    nd = depth + s.count('{') - s.count('}')
    if nd <= 1 or re.search(r'\bfn |\bimpl|\bmod ', l):
        print(f'{i+1}: {depth}->{nd}  {l.strip()[:80]}')
    depth = nd
