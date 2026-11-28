F = 'crates/victory-gpu/src/watch.rs'
s = open(F, encoding='utf-8').read()
# delete ours' extern block + placeholder (second ntdll link + second placeholder def)
first = s.index('#[link(name = "ntdll"')
second = s.index('#[link(name = "ntdll"', first + 1)
# doc comment just before it ('/// Whether `block` lies in a placeholder reservation...')
doc_start = s.rfind('/// Whether `block` lies in a placeholder', 0, second)
end = s.index('fn forget_hole', second)
# back up to the doc comment of forget_hole
end = s.rfind('/// The note that `block` is uncommitted', second, end)
if doc_start != -1:
    s = s[:doc_start] + s[end:]
else:
    s = s[:second] + s[end:]
# rename ours' look_at_holes(most)
s = s.replace('fn look_at_holes(most: usize)', 'fn look_at_holes_most(most: usize)')
s = s.replace('look_at_holes(HOLES_A_LOOK);', 'look_at_holes_most(HOLES_A_LOOK);')
open(F, 'w', encoding='utf-8').write(s)
print('done')
