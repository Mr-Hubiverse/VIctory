p = '/mnt/c/Users/Hube/Desktop/XCL/ztrx-main/crates/victory-shader/src/io.rs'
s = open(p, encoding='utf-8').read()

# 1) interpolation_can_take_its_first_half_from_either_branch: restore the
# dropped tail (S_ENDPGM + close + asserts) before theirs' sibling test.
old = """            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
    fn first_halves_read_through_a_phi_keep_their_partial_values() {"""
new = """            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
            S_ENDPGM,
        ]);
        let io = raise(&p, &pixel(vec![TEXCOORD], 2)).unwrap();
        assert_eq!(io.interps.len(), 2);
        for (i, (site, interp)) in io.interps.iter().enumerate() {
            assert_eq!(*site, (3, i));
            let Interp::Interpolated { first, .. } = interp else {
                panic!("{interp:?}");
            };
            assert_eq!(first, &[(1, i), (2, i)]);
        }
    }

    #[test]
    fn first_halves_read_through_a_phi_keep_their_partial_values() {"""
assert s.count(old) == 1
s = s.replace(old, new)

# 2) vertex_sample_needs_no_second_half: close ours' assert_eq! and drop
# theirs' contradicting Weighted assert.
old = """                    vertex: 2,
                },
            )]
        assert!(
            io.interps
                .iter()
                .all(|(_, i)| matches!(i, Interp::Weighted { .. }))
        );
    }
"""
new = """                    vertex: 2,
                },
            )]
        );
    }
"""
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('io.rs fixed')
