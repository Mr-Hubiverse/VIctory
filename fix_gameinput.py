import re
src = open('crates/victory-hle/src/modules/gameinput.rs', encoding='utf-8').read()
pat = re.compile(r'<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> nvidia/main\n', re.S)
hunks = list(pat.finditer(src))
assert len(hunks) == 4, len(hunks)

# hunk0: theirs' one_pad broadcast adapted to the Motors signature
m0 = '''    let motors = Motors {
        low: speed(low),
        high: speed(high),
        left_trigger: speed(left_trigger),
        right_trigger: speed(right_trigger),
    };
    let mut source = lock(&device.hub().source);
    if one_pad {
        // The one pad is every host pad.
        for slot in 0..PADS {
            if source.pad(slot).is_some() {
                source.rumble(slot, motors);
            }
        }
    } else {
        source.rumble(device.index, motors);
    }
'''
m1 = hunks[1].group(1)          # ours' impl Source for Host
m2 = hunks[2].group(2)          # theirs' key_code + from-to range
m3 = hunks[3].group(2)          # theirs' live-key machinery + tests

out = (src[:hunks[0].start()] + m0 + src[hunks[0].end():hunks[1].start()] + m1
       + src[hunks[1].end():hunks[2].start()] + m2 + src[hunks[2].end():hunks[3].start()]
       + m3 + src[hunks[3].end():])
open('crates/victory-hle/src/modules/gameinput.rs', 'w', encoding='utf-8').write(out)
print('done', '<<<<<<<' in out)
