import re, os, sys
src = open('crates/victory-gpu/src/draw.rs', encoding='utf-8').read()
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
os.makedirs('/tmp/hunks', exist_ok=True)
for i, m in enumerate(pat.finditer(src)):
    o, t = m.group(1), m.group(2)
    ln = src[:m.start()].count('\n') + 1
    open(f'/tmp/hunks/h{i:02d}_o.rs', 'w', encoding='utf-8').write(o)
    open(f'/tmp/hunks/h{i:02d}_t.rs', 'w', encoding='utf-8').write(t)
    print(f'{i:02d} line={ln} ours={len(o)} theirs={len(t)}')
