import re
src = open('crates/victory-gpu/src/primitive.rs', encoding='utf-8').read()
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
hunks = list(pat.finditer(src))
assert len(hunks) == 2, len(hunks)

# Hunk 1: ours' dispatch + fast path + generic signature, then theirs' guards
# (which define in_rect/out_rect the shared loop uses).
ours1 = hunks[0].group(1)
theirs1 = hunks[0].group(2)
merged1 = ours1 + theirs1

# Hunk 2: both tests
merged2 = hunks[1].group(1) + hunks[1].group(2)

out = src[:hunks[0].start()] + merged1 + src[hunks[0].end():hunks[1].start()] + merged2 + src[hunks[1].end():]
open('crates/victory-gpu/src/primitive.rs','w',encoding='utf-8').write(out)
print('done', '<<<<<<<' in out)
