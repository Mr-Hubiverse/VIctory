import sys, re
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
for path in sys.argv[1:]:
    src = open(path, encoding='utf-8', errors='replace').read()
    print(f"########## {path}")
    for i, m in enumerate(pat.finditer(src)):
        o, t = m.group(1), m.group(2)
        print(f"=== hunk {i} ours {len(o)}b theirs {len(t)}b")
        if len(o) + len(t) < 1800:
            print("OURS:" + o + "\nTHEIRS:" + t)
        else:
            print("OURS head:" + o[:260]); print("OURS tail:" + o[-200:])
            print("THEIRS head:" + t[:260]); print("THEIRS tail:" + t[-200:])
