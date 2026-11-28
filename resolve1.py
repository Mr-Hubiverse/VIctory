import re, sys

def hunks(path):
    src = open(path, encoding='utf-8').read()
    out = []
    pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n?', re.S)
    for m in pat.finditer(src):
        out.append((m.span(), m.group(1), m.group(2)))
    return src, out

def resolve(path, choices):
    """choices: list of 'ours'|'theirs'|'both'|'both_rev' per hunk"""
    src, hs = hunks(path)
    assert len(hs) == len(choices), f"{path}: {len(hs)} hunks, {len(choices)} choices"
    res = []
    last = 0
    for (span, ours, theirs), c in zip(hs, choices):
        res.append(src[last:span[0]])
        if c == 'ours':
            res.append(ours)
        elif c == 'theirs':
            res.append(theirs)
        elif c == 'both':
            res.append(ours + theirs)
        elif c == 'both_rev':
            res.append(theirs + ours)
        elif c == 'none':
            pass
        else:
            res.append(c)
        last = span[1]
    res.append(src[last:])
    open(path, 'w', encoding='utf-8').write(''.join(res))
    print(path, 'resolved', len(hs))
