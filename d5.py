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
lo, hi = 7640, 8190
for i, l in enumerate(lines):
    s = strip(l)
    nd = depth + s.count('{') - s.count('}')
    if lo <= i + 1 <= hi and nd <= 2:
        print(f'{i+1}: {depth}->{nd}  {l.strip()[:75]}')
    depth = nd
