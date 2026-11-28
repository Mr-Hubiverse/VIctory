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
    if i + 1 > 8850 and re.match(r'^\s{0,4}(pub\(crate\) )?(unsafe )?fn |^\s*impl |^\s*(pub )?mod |^\s*struct |^\s*enum |^\s*#\[cfg\(test\)\]', l):
        print(f'{i+1}: d={depth}  {l.strip()[:80]}')
    depth += s.count('{') - s.count('}')
print('end', depth)
