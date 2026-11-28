p = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-gpu/src/host.rs'
s = open(p, encoding='utf-8').read()
# dangling ours' tracing::info! tail inside Meter::frame
old = """        }
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame; {refused:.2} indirect records a frame ran empty for want of their list's budget (2^27), the largest cost of one {biggest} (0: under 2^22)"
        );
        crate::indirect::merge::line(frames);"""
new = """        }
        crate::indirect::merge::line(frames);"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('ok')
