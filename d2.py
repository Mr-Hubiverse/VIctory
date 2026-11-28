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
for i, l in enumerate(lines):
    s = strip(l)
    nd = depth + s.count('{') - s.count('}')
    if nd == 0 and depth != 0:
        print(f'{i+1}: depth {depth}->0  {l.strip()[:70]}')
    depth = nd
    if depth < 0:
        print(f'{i+1}: NEG  {l.strip()[:70]}')
        depth = 0
print('end depth', depth)
