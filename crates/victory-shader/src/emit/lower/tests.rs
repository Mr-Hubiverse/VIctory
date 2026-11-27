    use super::{
        Lowered, Numeric, Parts, SigElement, SubDword, buffer_format, half_shift, inline_f16,
        lower,
    };
    use crate::ShaderError;
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    pub(super) const S_ENDPGM: u32 = 0xBF81_0000;

    #[test]
    fn packed_pixel_inputs_find_their_second_row() {
        use super::super::container::SigElement;
        use super::half_input_partners;
        use crate::native::{PsInput, Usage};
        let entry = |row, row2, pack2| PsInput {
            usage: Usage::Generic,
            row,
            mask: 0xF,
            attr: row,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        let element = |row| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: true,
        };
        // Rows 0 and 3 pack and leave their second row out; 5 names its
        // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
        let io = StageIo::Ps {
            inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
        let p = half_input_partners(&io, &sig);
        assert_eq!(p.get(&0), Some(&1));
        assert_eq!(p.get(&3), Some(&4));
        assert_eq!(p.get(&5), None);
    }

    /// The packed pedestrian layouts of the title: an element
    /// behind another one in its register follows in the first parameter
    /// with a dword left, and is not a half of its register's row.
    #[test]
    fn packed_elements_behind_another_take_the_first_free_dword() {
        use super::super::container::SigElement;
        use super::packed_half;
        let element = |index, row, col, cols| SigElement {
            name: "TEXCOORD".into(),
            index,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col,
            cols,
            usage: 0,
            half: true,
        };
        // MS_VS_PedTransformPreSkinD_Packed 005fe88ec1bdc0c7 and its pixel
        // shader 93667197d3d05e1c: TEXCOORD6 is one value at register 4
        // column 3; parameter 3 (rows 6 and 7) carries it in dword 3.
        let sig = vec![
            element(1, 2, 0, 4),
            element(2, 3, 0, 4),
            element(4, 4, 0, 3),
            element(6, 4, 3, 1),
            element(3, 5, 0, 4),
            element(5, 6, 0, 4),
            element(7, 7, 0, 2),
            element(9, 9, 0, 3),
            element(11, 10, 0, 3),
            element(15, 12, 0, 4),
        ];
        let packs = [
            (2, Some(3)),
            (4, Some(5)),
            (6, Some(7)),
            (9, Some(10)),
            (12, None),
        ];
        let at = |a, h| packed_half(&sig, &packs, a, h);
        // Parameter 2: TEXCOORD4's three values, a zero half, TEXCOORD3.
        assert_eq!(at(4, 2), Some((4, 2)));
        assert_eq!(at(4, 3), None);
        assert_eq!(at(4, 4), Some((5, 0)));
        assert_eq!(at(4, 7), Some((5, 3)));
        // Parameter 3: TEXCOORD5, TEXCOORD7, then TEXCOORD6 alone.
        assert_eq!(at(6, 3), Some((6, 3)));
        assert_eq!(at(6, 5), Some((7, 1)));
        assert_eq!(at(6, 6), Some((4, 3)));
        assert_eq!(at(6, 7), None);
        // Parameters without such an element are as before.
        assert_eq!(at(9, 3), None);
        assert_eq!(at(9, 4), Some((10, 0)));
        assert_eq!(at(12, 3), Some((12, 3)));
        assert_eq!(at(12, 4), None);
        // The common layout (29805d23a495cdf8): TEXCOORD7 is two values at
        // register 1 columns 2-3; parameter 0 is TEXCOORD0, TEXCOORD1,
        // TEXCOORD7.
        let sig = vec![
            element(0, 1, 0, 2),
            element(7, 1, 2, 2),
            element(1, 2, 0, 4),
            element(2, 3, 0, 4),
            element(4, 4, 0, 3),
        ];
        let packs = [(3, Some(4)), (1, Some(2))];
        let at = |a, h| packed_half(&sig, &packs, a, h);
        assert_eq!(at(1, 1), Some((1, 1)));
        assert_eq!(at(1, 2), Some((2, 0)));
        assert_eq!(at(1, 5), Some((2, 3)));
        assert_eq!(at(1, 6), Some((1, 2)));
        assert_eq!(at(1, 7), Some((1, 3)));
        assert_eq!(at(3, 6), Some((4, 2)));
        assert_eq!(at(3, 7), None);
        // A row no element holds gives nothing, as before.
        assert_eq!(packed_half(&sig, &[(8, None)], 8, 0), None);
    }

    /// Mesh output size as the validator counts it (53 programs were over
    /// the limit; they now fail as unsupported).
    #[test]
    fn mesh_output_size_counts_aligned_maxima() {
        let el = |cols| SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(0),
            col: 0,
            cols,
            usage: 0,
            half: false,
        };
        let mesh = |max_vertices, max_primitives| crate::emit::shader::Mesh {
            max_vertices,
            max_primitives,
            topology: 2,
        };
        // 4 + 4 columns of 64 vertices, 1 column of 126 primitives (128).
        let size = super::mesh_output_bytes(&[el(4), el(4)], &[el(1)], mesh(64, 126));
        assert_eq!(size, 4 * (8 * 64 + 128));
        // 32 columns of 256 vertices: exactly the limit; one more is over.
        let full: Vec<_> = (0..8).map(|_| el(4)).collect();
        assert_eq!(
            super::mesh_output_bytes(&full, &[], mesh(256, 1)),
            super::MAX_MESH_OUTPUT_BYTES
        );
        assert!(
            super::mesh_output_bytes(&full, &[el(1)], mesh(256, 1)) > super::MAX_MESH_OUTPUT_BYTES
        );
    }

    /// `calculateLOD` takes the size's dimensions only, as DXC's
    /// `GetNumDimensionsForCalcLOD` counts them: an array's slice
    /// coordinate is left undef.
    #[test]
    fn level_queries_leave_the_array_slice_undef() {
        // 1D, 2D, 3D, cube, 1D array, 2D array.
        let dims: Vec<usize> = [1, 2, 4, 5, 6, 7].map(super::lod_dims).to_vec();
        assert_eq!(dims, [1, 2, 3, 3, 1, 2]);
    }

    /// VOP3 `op dst, a, b` (GFX10 encoding); `flags` are the clamp, op_sel
    /// and abs bits of the first dword.
    fn vop3(op: u32, flags: u32, dst: u32, a: u32, b: u32) -> [u32; 2] {
        [0xD400_0000 | op << 16 | flags | dst, a | b << 9]
    }

    /// The DXIL of a pixel shader that exports `body`'s v2, computed from
    /// a per-lane value in v3.
    fn lower_ps(body: [u32; 2]) -> Result<Vec<u8>, ShaderError> {
        let words = [
            // v_mbcnt_lo_u32_b32 v3, -1, 0
            vop3(0x365, 0, 3, 193, 128)[0],
            vop3(0x365, 0, 3, 193, 128)[1],
            body[0],
            body[1],
            0xF800_180F, // exp mrt0, v2, v2, v2, v2 done vm
            0x0202_0202,
            0xBF81_0000, // s_endpgm
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = ir::build(&code, false, 256)?;
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code,
        };
        let io = io::raise(&p, &native)?;
        let res = resource::track(&p, &native.layout, native.stage, None)?;
        let st = structure::structure(&p)?;
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let parts = Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![target],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        };
        lower(&parts).map(|l| l.dxil)
    }

    /// Lowers a compute program of constants: whether it translates.
    fn lowers(words: &[u32]) -> Result<(), crate::ShaderError> {
        lower_with(words, vec![], vec![]).map(|_| ())
    }

    /// [`lowers`] with the input registers `rsrc2` names.
    fn lowers_rsrc2(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        lower_io(words, vec![], vec![], false, Some(rsrc2)).map(|_| ())
    }

    /// [`lower_wave`] with the input registers `rsrc2` names (default: the
    /// layout's user SGPRs, at least two, and 512 bytes of LDS).
    fn lower_io(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
        rsrc2: Option<u32>,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: rsrc2.unwrap_or(1 << 15 | sgprs.max(2) << 1),
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            psv: None,
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: super::super::Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
    }

    /// The view a root descriptor at `user_data` is.
    fn root_view(kind: ResourceType, user_data: u8, slot: u8) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data,
            slot,
        }
    }

    #[test]
    fn scalar_code_inside_a_whole_wave_section_is_lowered() {
        // Assembled with llvm-mc (gfx1030, wave32): scalar code the
        // compiler scheduled into a whole-wave section, branched on after.
        let words = [
            0xBE84_0385, // s_mov_b32 s4, 5
            0xBE80_3DC1, // s_or_saveexec_b32 s0, -1
            0x8005_8104, // s_add_u32 s5, s4, 1
            0x7E02_0280, // v_mov_b32_e32 v1, 0
            0xBEFE_0300, // s_mov_b32 exec_lo, s0
            0xBF06_8605, // s_cmp_eq_u32 s5, 6
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE86_0380, // s_mov_b32 s6, 0
            S_ENDPGM,
        ];
        lowers(&words).unwrap();
    }

    #[test]
    fn constants_loaded_from_a_root_view() {
        // Two dwords at byte 0x10 of the memory of a root descriptor (in
        // s[0:1]) read with s_load and branched on: read through the root
        // CBV, or the root SRV when the root signature says it is one.
        let words = [
            0xF404_0080,
            0xFA00_0010, // s_load_dwordx2 s[2:3], s[0:1], 0x10
            0x8002_0302, // s_add_u32 s2, s2, s3
            0xBF06_8502, // s_cmp_eq_u32 s2, 5
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Descriptor, 0, 3)];
        let cbv = lower_with(&words, layout(), vec![]).unwrap();
        assert_eq!((cbv.root_cbvs, cbv.root_srvs), (vec![3], vec![]));
        let used = vec![root_view(ResourceType::Srv, 0, 3)];
        let srv = lower_with(&words, layout(), used).unwrap();
        assert_eq!((srv.root_cbvs, srv.root_srvs), (vec![], vec![3]));
        // A root UAV's memory has no read-only host view.
        let used = vec![root_view(ResourceType::Uav, 0, 3)];
        assert!(lower_with(&words, layout(), used).is_err());
    }

    #[test]
    fn a_dword_of_a_descriptor_another_load_reads_whole() {
        // A 2D image's descriptor (a table range at s0) loaded whole for
        // image_get_resinfo, and its dword 2 (width and height) loaded
        // again alone and branched on.
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF40C_0100,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], null
            0x7E04_0480, // v_mov_b32 v2, 0
            0xF038_0308,
            0x0001_0402, // image_get_resinfo v[4:5], v2, s[4:11] dim:SQ_RSRC_IMG_2D
            0xF400_0300,
            0xFA00_0008, // s_load_dword s12, s[0:1], 0x8
            0xBF06_850C, // s_cmp_eq_u32 s12, 5
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0)];
        lower_with(&words, layout(), vec![]).unwrap();
        // Dword 2 of the next descriptor: no load of it is used.
        let mut next = words;
        next[7] = 0xFA00_0028;
        let e = lower_with(&next, layout(), vec![]).err().unwrap();
        assert!(
            e.to_string().contains("a descriptor read only as data"),
            "{e}"
        );
    }

    #[test]
    fn a_descriptor_dword_loaded_through_a_rebuilt_table_pointer() {
        // A 2D image's descriptor (a table range at s0) loaded whole for an
        // image_store in an exec-masked tail, and its dword 2 (width and
        // height) loaded alone through another copy of the table pointer
        // {s0, 0} (a copy in other registers, or the high half set to 0
        // again after the registers held scratch), its height bits read.
        let layout = || vec![root(RootType::Table, 0, 0)];
        let copied = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF400_0300,
            0xFA00_0008, // s_load_dword s12, s[0:1], 0x8
            0xBE82_0300, // s_mov_b32 s2, s0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF40C_0101,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[2:3], 0x0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x938D_FF0C,
            0x000E_000E, // s_bfe_u32 s13, s12, 0xe000e
            0x7E0A_020D, // v_mov_b32 v5, s13
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            0x7D88_0084, // v_cmp_gt_u32 vcc_lo, 4, v0
            0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
            0xBF88_0002, // s_cbranch_execz -> s_endpgm
            0xF020_1F08,
            0x0001_0100, // image_store v[1:4], v[0:1], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
            S_ENDPGM,
        ];
        lower_with(&copied, layout(), vec![]).unwrap();
        let rebuilt = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF40C_0100,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], 0x0
            0x7D88_0084, // v_cmp_gt_u32 vcc_lo, 4, v0
            0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
            0xBF88_0002, // s_cbranch_execz -> s_or_b32
            0xF020_1F08,
            0x0001_0100, // image_store v[1:4], v[0:1], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0x7E08_0500, // v_readfirstlane_b32 s4, v0
            0x7E02_0501, // v_readfirstlane_b32 s1, v1
            0x8704_FF04,
            0x0000_01FF, // s_and_b32 s4, s4, 0x1ff
            0x8F01_8901, // s_lshl_b32 s1, s1, 9
            0x8804_0104, // s_or_b32 s4, s4, s1
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF400_0300,
            0xFA00_0008, // s_load_dword s12, s[0:1], 0x8
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x938D_FF0C,
            0x000E_000E, // s_bfe_u32 s13, s12, 0xe000e
            0x8804_0D04, // s_or_b32 s4, s4, s13
            0x7E0A_0204, // v_mov_b32 v5, s4
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            S_ENDPGM,
        ];
        lower_with(&rebuilt, layout(), vec![]).unwrap();
        // Dword 2 of the next descriptor: no load of it is used.
        let mut next = copied;
        next[2] = 0xFA00_0028;
        let e = lower_with(&next, layout(), vec![]).err().unwrap();
        assert!(
            e.to_string().contains("a descriptor read only as data"),
            "{e}"
        );
    }

    #[test]
    fn descriptor_registers_reused_after_a_masked_image_store() {
        // An image_store in an exec-masked tail is the only use of its
        // descriptor load; after the join the descriptor's registers hold
        // scratch (v_readfirstlane, then bit packing), alone or as the low
        // half of a 64-bit shift whose high half is still a descriptor
        // dword.
        let layout = || vec![root(RootType::Table, 0, 0)];
        let head = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF40C_0100,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], 0x0
            0x7D88_0084, // v_cmp_gt_u32 vcc_lo, 4, v0
            0xBE82_3C6A, // s_and_saveexec_b32 s2, vcc_lo
            0xBF88_0002, // s_cbranch_execz -> past the store
            0xF020_1F08,
            0x0001_0100, // image_store v[1:4], v[0:1], s[4:11] dmask:0xf dim:SQ_RSRC_IMG_2D unorm
        ];
        let packed = [
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0x7E08_0500, // v_readfirstlane_b32 s4, v0
            0x7E0A_0501, // v_readfirstlane_b32 s5, v1
            0x8704_FF04,
            0x0000_01FF, // s_and_b32 s4, s4, 0x1ff
            0x8F05_8905, // s_lshl_b32 s5, s5, 9
            0x8804_0504, // s_or_b32 s4, s4, s5
            0x7E0A_0204, // v_mov_b32 v5, s4
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            S_ENDPGM,
        ];
        let wide = [
            0x887E_027E, // s_or_b32 exec_lo, exec_lo, s2
            0x7E08_0500, // v_readfirstlane_b32 s4, v0
            0x8F84_8904, // s_lshl_b64 s[4:5], s[4:5], 9
            0x7E0A_0204, // v_mov_b32 v5, s4
            0xD834_0000,
            0x0000_0500, // ds_write_b32 v0, v5
            S_ENDPGM,
        ];
        lower_with(&[&head[..], &[S_ENDPGM]].concat(), layout(), vec![]).unwrap();
        lower_with(&[&head[..], &packed].concat(), layout(), vec![]).unwrap();
        lower_with(&[&head[..], &wide].concat(), layout(), vec![]).unwrap();
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    pub(super) fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_io(words, layout, used, wave64, None)
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    pub(super) fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    #[test]
    fn buffer_cache_invalidations_lower_as_no_ops() {
        // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
        // host to invalidate its cached view of a buffer.
        assert!(lowers_rsrc2(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl0_inv");
        assert!(lowers_rsrc2(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl1_inv");
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];

        for words in [add, sub] {
            lowers_rsrc2(&words, 0).unwrap();
        }
    }

    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(p.blocks.iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32"));

        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

        let io = io::raise(&p, &native).unwrap();

        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: super::super::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        }).unwrap();

        let container =
            crate::Container::parse(&emitted.dxil).unwrap();

        let features =
            container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    /// A dword load by index through a descriptor built around the root
    /// descriptor in s[0:1] with dword 1 bits `hi` and dword 3 `w3`, stored
    /// to LDS so the load is needed.
    fn indexed_root_load(hi: u32, w3: u32) -> Result<Lowered, crate::ShaderError> {
        let words = [
            0x8805_FF01,
            hi,          // s_or_b32 s5, s1, hi
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE86_03C1, // s_mov_b32 s6, -1
            0xBE87_03FF,
            w3, // s_mov_b32 s7, w3
            0xE030_2000,
            0x8001_0100, // buffer_load_dword v1, v0, s[4:7], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let root = RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 3,
        };
        // The container lists the root SRV without a stride.
        let used = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 3,
        };
        lower_with(&words, vec![root], vec![used])
    }

    #[test]
    fn indexed_loads_through_built_descriptors_use_their_stride() {
        // No used-resource entry gives a stride: the descriptor's own (12,
        // and 1 as vertex fetch builds them) addresses the root view raw.
        for stride in [12, 1] {
            let l = indexed_root_load(stride << 16, 0x1101_4FAC).unwrap();
            assert_eq!(l.root_srvs, [3]);
        }
    }

    #[test]
    fn indexed_loads_through_swizzled_or_lane_indexed_descriptors_stop() {
        let why = |r: Result<Lowered, crate::ShaderError>| match r {
            Err(crate::ShaderError::Unsupported { what, .. }) => what,
            r => panic!("{:?}", r.map(|l| l.root_srvs)),
        };
        assert_eq!(
            why(indexed_root_load(0x8010_0000, 0x1101_4FAC)),
            "a swizzled buffer"
        );
        assert_eq!(
            why(indexed_root_load(0x0010_0000, 0x1181_4FAC)),
            "a buffer indexed by lane"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    /// Tables of 2D textures at s0 and samplers at s1.
    pub(super) fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn a_uniform_branch_can_choose_between_immediate_samplers() {
        // A root constant selects wrap or clamp. Keep the sample live with an LDS store.
        let words = [
            0xBE84_0300, 0xBE85_0380,       // s[4:5] = texture table
            0xF40C_0302, 0xFA00_0000,       // s_load_dwordx8 s[12:19], s[4:5], 0
            0xBE89_0380, 0xBE8A_0380, 0xBE8B_0380, // sampler words 1-3
            0xBF06_8001,                    // s_cmp_eq_u32 s1, 0
            0xBF85_0002,                    // branch to clamp
            0xBE88_0380, 0xBF82_0002,       // wrap, skip clamp
            0xBE88_03FF, 0x0000_0092,       // clamp all axes
            0xF09C_0F08, 0x0043_0C02,       // image_sample_lz v[12:15], ..., s[8:11]
            0xD834_0000, 0x0000_0C00,       // ds_write_b32 v0, v12
            S_ENDPGM,
        ];
        let (mut layout, mut used) = texture_tables();
        layout[1] = root(RootType::Constants, 1, 1);
        used.truncate(1);
        let shader = lower_with(&words, layout, used).unwrap();
        assert_eq!(shader.samplers.len(), 2);
        assert!(shader.samplers.contains(&[0, 0, 0, 0]));
        assert!(shader.samplers.contains(&[0x92, 0, 0, 0]));
    }

    /// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
    pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let (layout, mut used) = texture_tables();
        used.push(UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 16,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        });
        (layout, used)
    }

    #[test]
    fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
        use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
        for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
            let (layout, used) = texture_and_uav_tables();
            let l = lower_wave(words, layout, used, false).unwrap();
            // One image handle, from the join's per-lane heap index.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers_rsrc2(&words, 0).unwrap();
        }
    }

    /// An f16 (sign, 5-bit exponent, 10-bit fraction) as an f32; normal numbers only.
    fn f16_value(h: u32) -> f32 {
        let sign = if h >> 15 != 0 { -1.0 } else { 1.0 };
        let exponent = (h >> 10 & 0x1F) as i32 - 15;
        sign * (1.0 + (h & 0x3FF) as f32 / 1024.0) * 2f32.powi(exponent)
    }

    #[test]
    fn inline_float_constants_mean_the_same_in_f16() {
        // v_fma_mix and v_pack_b32_f16 read these as f16: 1.0 must be 0x3C00,
        // not the low half (0) of the f32 bits 0x3F800000.
        assert_eq!(inline_f16(242), 0x3C00);
        for code in 240..=248u8 {
            let f32_value = f32::from_bits(crate::resource::inline(code).unwrap());
            let f16_value = f16_value(inline_f16(code));
            assert!(
                (f32_value - f16_value).abs() <= f32_value.abs() * 1e-3,
                "inline constant {code}: f32 {f32_value}, f16 {f16_value}"
            );
        }
    }

    #[test]
    fn scalar_conditional_moves_lower() {
        // Each sets SCC, moves under it into a destination that already has a
        // value, and branches on the result (only code with an effect needs
        // its operands).
        let tail = [
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let b32 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xBE82_0501, // s_cmov_b32 s2, s1
        ];
        let k32 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8001, // s_cmp_lg_u32 s1, 0
            0xB102_0005, // s_cmovk_i32 s2, 5
        ];
        let b64 = [
            0xBE82_0387, // s_mov_b32 s2, 7
            0xBE83_0387, // s_mov_b32 s3, 7
            0xBE80_0381, // s_mov_b32 s0, 1
            0xBE81_0381, // s_mov_b32 s1, 1
            0xBF07_8000, // s_cmp_lg_u32 s0, 0
            0xBE82_0600, // s_cmov_b64 s[2:3], s[0:1]
        ];
        for head in [&b32[..], &k32[..], &b64[..]] {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words).unwrap();
        }
    }

    #[test]
    fn signed_names_of_wrapping_integer_operations_match_the_unsigned_ones() {
        // The low 32 bits of a sum, difference or product do not depend on
        // the sign, so the _i32 forms lower exactly as their _u32 twins.
        for (signed, unsigned) in [
            (0x37F, 0x125), // v_add_nc_i32, v_add_nc_u32
            (0x376, 0x126), // v_sub_nc_i32, v_sub_nc_u32
            (0x16B, 0x169), // v_mul_lo_i32, v_mul_lo_u32
        ] {
            let a = lower_ps(vop3(signed, 0, 2, 259, 129)).unwrap();
            let b = lower_ps(vop3(unsigned, 0, 2, 259, 129)).unwrap();
            assert_eq!(a, b, "opcode {signed:#x}");
        }
    }

    #[test]
    fn sixteen_bit_signed_add_and_sub_lower_with_half_selects() {
        // v_add_nc_i16 and v_sub_nc_i16, also reading the high halves.
        for op in [0x30D, 0x30E] {
            for opsel in [0, 1 << 11, 1 << 12 | 1 << 14] {
                lower_ps(vop3(op, opsel, 2, 259, 259)).unwrap();
            }
        }
    }

    #[test]
    fn clamped_signed_operations_are_refused_not_miscompiled() {
        // Clamping saturates; wrapping them would give other values.
        let clamp = 1 << 15;
        for op in [0x37F, 0x376, 0x16B, 0x30D, 0x30E] {
            let err = lower_ps(vop3(op, clamp, 2, 259, 129)).unwrap_err();
            assert!(
                matches!(err, ShaderError::Unsupported { what, .. } if what.contains("modifier")),
                "{op:#x}: {err:?}"
            );
        }
    }

    #[test]
    fn sub_dword_lds_names() {
        let s = |write, bits, signed, d16| {
            Some(SubDword {
                write,
                bits,
                signed,
                d16,
            })
        };
        assert_eq!(SubDword::of("ds_read_i8"), s(false, 8, true, None));
        assert_eq!(SubDword::of("ds_read_u16"), s(false, 16, false, None));
        assert_eq!(
            SubDword::of("ds_read_u8_d16"),
            s(false, 8, false, Some(false))
        );
        assert_eq!(
            SubDword::of("ds_read_i16_d16_hi"),
            s(false, 16, true, Some(true))
        );
        assert_eq!(SubDword::of("ds_write_b8"), s(true, 8, false, None));
        assert_eq!(
            SubDword::of("ds_write_b16_d16_hi"),
            s(true, 16, false, Some(true))
        );
        for name in [
            "ds_read_b32",
            "ds_read2_b32",
            "ds_write_b64",
            "ds_write_b8_d16",
            "ds_read_u8_d32",
            "ds_add_u32",
        ] {
            assert_eq!(SubDword::of(name), None, "{name}");
        }
    }

    #[test]
    fn scalar_pack_extend_and_absdiff_lower() {
        // Each computes into s2 from two constants and branches on it (only
        // code with an effect needs its operands).
        let head = [
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0383, // s_mov_b32 s1, 3
        ];
        let tail = [
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let ops = [
            0x9602_0100, // s_absdiff_i32 s2, s0, s1
            0xBE82_1901, // s_sext_i32_i8 s2, s1
            0xBE82_1A01, // s_sext_i32_i16 s2, s1
            0x9902_0100, // s_pack_ll_b32_b16 s2, s0, s1
            0x9982_0100, // s_pack_lh_b32_b16 s2, s0, s1
            0x9A02_0100, // s_pack_hh_b32_b16 s2, s0, s1
        ];
        for op in ops {
            let words: Vec<u32> = head.iter().chain(&[op]).chain(&tail).copied().collect();
            lowers(&words).unwrap();
        }
    }

    #[test]
    fn cube_descriptor_dwords_read_as_data() {
        // A cube image's descriptor (a table range at s0) is loaded, sized
        // with image_get_resinfo, and its depth dword (4: the six faces, so
        // 5) is read as a number and branched on.
        let table = || vec![root(RootType::Table, 0, 0)];
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF40C_0100,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[0:1], null
            0x7E04_0480, // v_mov_b32 v2, 0
            0xF038_0318,
            0x0001_0402, // image_get_resinfo v[4:5], v2, s[4:11] dim:SQ_RSRC_IMG_CUBE
            0xBF06_8508, // s_cmp_eq_u32 s8, 5
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        lower_with(&words, table(), vec![]).unwrap();
        // The same with a 2D image (dim 1) also lowers.
        let mut flat = words;
        flat[4] = 0xF038_0308;
        lower_with(&flat, table(), vec![]).unwrap();
    }


    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_BRANCHES_WAVE32, WATERFALL_CARRIED_WAVE32, WATERFALL_FETCHED_WAVE32,
            WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32, WATERFALL_REPLACED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
            (WATERFALL_CARRIED_WAVE32, false),
            (WATERFALL_BRANCHES_WAVE32, false),
            (WATERFALL_REPLACED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn waterfall_loops_over_descriptors_from_buffer_loads_sample_with_each_lanes_own() {
        let words = crate::resource::tests::waterfall_buffer_wave32();
        let (layout, used) = texture_tables();
        let l = lower_wave(&words, layout, used, false).unwrap();
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }

    #[test]
    fn per_lane_descriptors_loaded_under_a_partial_exec_fail() {
        // The if/else without its s_cbranch_execz: both branches run, each
        // loading for its lanes only, so v[4:11] after them is the second
        // load only in the else lanes.
        let mut words = crate::resource::tests::WATERFALL_BRANCHES_WAVE32.to_vec();
        for k in [12, 19] {
            assert_eq!(words[k] >> 16, 0xBF88);
            words[k] = 0xBF80_0000; // s_nop 0
        }
        let (layout, used) = texture_tables();
        let Err(e) = lower_wave(&words, layout, used, false) else {
            panic!("lowered");
        };
        assert!(
            matches!(
                e,
                crate::ShaderError::Unsupported {
                    what: "a per-lane descriptor written where some lanes may keep the old one",
                    ..
                }
            ),
            "{e:?}"
        );
    }

    /// A table at s0 holding structured buffers of dwords.
    fn dword_buffers() -> Vec<UsedResource> {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    }

    #[test]
    fn descriptors_loaded_through_pointers_kept_in_phis() {
        // A buffer descriptor loaded from the table at s0 through a pointer
        // a root constant (s1) picks by a branch, then loaded through by
        // index and stored to LDS so the load is needed.
        let picked = [
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF85_0002, // s_cbranch_scc1 -> 0x10
            0x8004_9000, // s_add_u32 s4, s0, 16
            0xBF82_0001, // s_branch -> 0x14
            0x8004_A000, // 0x10: s_add_u32 s4, s0, 32
            0xBE85_0380, // 0x14: s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        // The same through a pointer advanced over four descriptors.
        let advanced = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0380, // s_mov_b32 s6, 0
            0xF408_0202,
            0xFA00_0000, // 0xc: s_load_dwordx4 s[8:11], s[4:5], null
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0x8004_9004, // s_add_u32 s4, s4, 16
            0x8006_8106, // s_add_u32 s6, s6, 1
            0xBF07_8406, // s_cmp_lg_u32 s6, 4
            0xBF85_FFF6, // s_cbranch_scc1 -> 0xc
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        lower_with(&picked, layout(), dword_buffers()).unwrap();
        lower_with(&advanced, layout(), dword_buffers()).unwrap();
    }

    #[test]
    fn data_read_through_a_pointer_into_a_root_descriptor_stops() {
        // A root descriptor's address (s[0:1]) advanced in a loop and read
        // through: its address is no value the translation has, so the
        // offset from it is unknown.
        let words = [
            0xBE82_0300, // s_mov_b32 s2, s0
            0xBE83_0301, // s_mov_b32 s3, s1
            0xF400_0101,
            0xFA00_0000, // 0x8: s_load_dword s4, s[2:3], null
            0x8002_9002, // s_add_u32 s2, s2, 16
            0x8203_8003, // s_addc_u32 s3, s3, 0
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0xBF84_FFFA, // s_cbranch_scc0 -> 0x8
            S_ENDPGM,
        ];
        let e = lower_with(&words, vec![root(RootType::Descriptor, 0, 3)], vec![])
            .err()
            .unwrap();
        assert!(
            e.to_string()
                .contains("an advanced pointer into a root descriptor"),
            "{e}"
        );
    }

    #[test]
    fn scalar_zero_counts_and_bit_scans_lower() {
        // s_bcnt0_i32_b32/_b64, s_ff0_i32_b32/_b64, s_flbit_i32_b64 and
        // s_flbit_i32_i64 of s0 (or the pair s[0:1]), with a branch on the
        // result so the instruction has an effect.
        for op in [0x0D, 0x0E, 0x11, 0x12, 0x16, 0x18] {
            let words = [
                0xBE80_03C1,           // s_mov_b32 s0, -1
                0xBE81_0380,           // s_mov_b32 s1, 0
                0xBE82_0000 | op << 8, // <op> s2, s0 (s[0:1])
                0xBF07_8002,           // s_cmp_lg_u32 s2, 0
                0xBF85_0001,           // s_cbranch_scc1 -> s_endpgm
                0xBE83_0380,           // s_mov_b32 s3, 0
                S_ENDPGM,
            ];
            lowers(&words).unwrap_or_else(|e| panic!("SOP1 {op:#x}: {e}"));
        }
        // The zero counts also leave "result != 0" in SCC.
        for op in [0x0D, 0x0E] {
            let words = [
                0xBE80_03C1,           // s_mov_b32 s0, -1
                0xBE81_0380,           // s_mov_b32 s1, 0
                0xBE82_0000 | op << 8, // s_bcnt0_i32_b32/_b64 s2, s0 (s[0:1])
                0xBF85_0001,           // s_cbranch_scc1 -> s_endpgm
                0xBE83_0380,           // s_mov_b32 s3, 0
                S_ENDPGM,
            ];
            lowers(&words).unwrap_or_else(|e| panic!("SOP1 {op:#x} SCC: {e}"));
        }
    }

    #[test]
    fn private_memory_is_as_large_as_its_addresses_reach() {
        let bytes = |words: &[u32]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            super::scratch_bytes(&ir::build(&code, false, 256).unwrap())
        };
        let store = [
            0x7E02_0280, // v_mov_b32 v1, 0
            0xDC70_401C, // scratch_store_dword v1, v2, off offset:28
            0x007D_0201,
        ];
        let indexed = [
            0x3606_0087, // v_and_b32 v3, 7, v0
            0x3406_0682, // v_lshlrev_b32 v3, 2, v3
            0xDC30_4000, // scratch_load_dword v4, v3, off
            0x047D_0003,
        ];
        assert_eq!(bytes(&[&store[..], &indexed, &[S_ENDPGM]].concat()), 32);
        // An address the code does not bound (the thread id).
        let free = [0xDC30_4000, 0x047D_0000]; // scratch_load_dword v4, v0, off
        assert_eq!(
            bytes(&[&store[..], &free, &[S_ENDPGM]].concat()),
            super::SCRATCH_CAP
        );
        let far = [
            0x7E02_02FF, // v_mov_b32 v1, 0x100000
            0x0010_0000,
            0xDC70_4000, // scratch_store_dword v1, v2, off
            0x007D_0201,
            S_ENDPGM,
        ];
        assert_eq!(bytes(&far), super::SCRATCH_MAX);
    }

    #[test]
    fn generic_flat_memory_is_refused() {
        // flat_store_dword v[0:1], v2: the address may be private, shared
        // or global memory.
        let e = lowers(&[0xDC70_0000, 0x007D_0200, S_ENDPGM]).unwrap_err();
        assert!(
            matches!(
                e,
                crate::ShaderError::Unsupported {
                    what: "flat memory",
                    ..
                }
            ),
            "{e}"
        );
    }

    #[test]
    #[ignore = "ours marks joins read through joins only with VICTORY_STRICT_JOINS (40 programs stop otherwise)"]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers_rsrc2(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers_rsrc2(&words, group_ids_xy).expect("dead timestamp");
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    /// Lowers a pixel program that loads a sampler and an image from two
    /// tables, samples v[0:3] from the address at v2 on with the two-dword
    /// instruction `sample`, and exports the result.
    fn lower_sample(sample: [u32; 2]) -> Result<Vec<u8>, crate::ShaderError> {
        let words = [
            0xBE82_0309, // s_mov_b32 s2, s9
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0204,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
            0xF40C_0001,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
            sample[0],
            sample[1],
            0xF800_180F,
            0x0302_0100, // exp mrt0, v0, v1, v2, v3 done vm
            0xBF81_0000, // s_endpgm
        ];
        let table = |user_data, slot| RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot,
        };
        let used = |kind, dwords, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords,
            user_data,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![table(8, 0), table(9, 1)],
            used: vec![
                used(ResourceType::Sampler, 4, 8, 0),
                used(ResourceType::Srv, 8, 9, 1),
            ],
            io: StageIo::Ps {
                inputs: Vec::new(),
                input_addr: 0,
                db_shader_control: 0,
                patches: Vec::new(),
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs)?;
        let res = crate::resource::track(&p, &native.layout, native.stage, None)?;
        let io = crate::io::raise(&p, &native)?;
        let st = crate::structure::structure(&p)?;
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        };
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: vec![target],
            prims: Vec::new(),
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        })
        .map(|l| l.dxil)
    }

    /// MIMG words: opcode, all four channels, the dimension; data v0,
    /// address v2, image s[0:7], sampler s[8:11].
    fn mimg(opcode: u32, dim: u32) -> [u32; 2] {
        [0xF000_0F00 | opcode << 18 | dim << 3, 0x0040_0002]
    }

    const SAMPLE_D: u32 = 0x22;
    const SAMPLE_D_CL: u32 = 0x23;
    const DIM_2D: u32 = 1;
    const DIM_CUBE: u32 = 3;

    #[test]
    fn gradient_samples_lower_on_planes_and_cubes() {
        for (opcode, dim) in [
            (SAMPLE_D, DIM_2D),
            (SAMPLE_D, DIM_CUBE),
            (SAMPLE_D_CL, DIM_2D),
            (SAMPLE_D_CL, DIM_CUBE),
        ] {
            if let Err(e) = lower_sample(mimg(opcode, dim)) {
                panic!("opcode {opcode:#x} on dimension {dim}: {e:?}");
            }
        }
    }

    #[test]
    fn host_wave32_mask_ops_use_both_ballot_halves() {
        use std::collections::HashMap;
        let words = [
            0xBE80_0F7E, // s_bcnt1_i32_b32 s0, exec_lo
            0xBE81_137E, // s_ff1_i32_b32 s1, exec_lo
            0xD765_0002,
            0x0001_007E, // v_mbcnt_lo_u32_b32 v2, exec_lo, 0
            0xD765_0003,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v3, -1, 0
            0xBF06_807E, // s_cmp_eq_u32 exec_lo, 0
            0x8504_0100, // s_cselect_b32 s4, s0, s1
            0x7E00_0204, // v_mov_b32 v0, s4
            0x7E02_0201, // v_mov_b32 v1, s1
            0xF800_180F,
            0x0302_0100, // exp mrt0, v0, v1, v2, v3 done vm
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code,
        };
        let p = ir::build(&native.code, false, 8).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let parts = Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        };
        let bad = HashMap::new();
        let mut l = super::Lower::new(&parts, &bad, &[], false);
        assert!(l.lower_ir().is_ok());
        assert_eq!(l.mask_data, super::MaskData::Host);
        assert_eq!(l.s.wave_size, None);
        let &(_lo, hi) = l.ballots.values().next().unwrap();
        assert!(
            l.s.m.uses_of(hi) >= 4,
            "expected high ballot half to be used by s_bcnt1_i32_b32, s_ff1_i32_b32, v_mbcnt_lo_u32_b32, and s_cmp_eq_u32 (>= 4 uses), got {}",
            l.s.m.uses_of(hi)
        );
    }

    /// Lowers a one-thread compute program that copies four root constants
    /// (s0..s3) into v1..v4, runs `body`, and stores v0 to a raw buffer from a
    /// descriptor table, so the vector instructions in `body` have an effect.
    fn lowers_valu(body: &[u32]) -> Result<(), crate::ShaderError> {
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xFA00_0000,
        ];
        // v_mov_b32 v1..v4, s0..s3
        words.extend((0..4u32).map(|i| 0x7E00_0200 | (i + 1) << 17 | i));
        words.extend_from_slice(body);
        words.extend_from_slice(&[
            0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
            0x8002_0000,
            S_ENDPGM,
        ]);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let param = |root, sgprs, user_data, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![
                param(RootType::Constants, 4, 0, 0),
                param(RootType::Table, 1, 4, 1),
            ],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1, // five user SGPRs, s0..s4
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        })
        .map(|_| ())
    }

    #[test]
    fn the_24_bit_multiplies_lower_both_halves() {
        // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low
        // dword) and v_mul_hi_i32_i24 and v_mul_hi_u32_u24 (the high one).
        for op in [0x09u32, 0x0A, 0x0B, 0x0C] {
            let word = op << 25 | 2 << 9 | 0x101;
            lowers_valu(&[word]).unwrap_or_else(|e| panic!("VOP2 {op:#x}: {e}"));
        }
    }

    #[test]
    fn sums_of_absolute_differences_lower() {
        // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16,
        // v_sad_u32 and v_msad_u8.
        for op in [0x15Au32, 0x15B, 0x15C, 0x15D, 0x171] {
            let words = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
            lowers_valu(&words).unwrap_or_else(|e| panic!("VOP3 {op:#x}: {e}"));
        }
    }

    /// MIMG `op vdata, v[vaddr..], s[srsrc..srsrc+7]` on a 2D image.
    const fn mimg_2d(op: u8, dmask: u8, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            0x3C << 26 | (op as u32 & 0x7F) << 18 | (dmask as u32) << 8 | 1 << 3,
            ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    }

    #[test]
    fn an_image_read_as_float_and_by_an_atomic_is_annotated_twice() {
        use crate::native::{ResourceType, RootParam, RootType, UsedResource};

        // One 2D image UAV from the descriptor table at s4, loaded (four
        // float channels) and then atomically minimised in the same block.
        let mut words = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            // s_load_dwordx8 s[8:15], s[4:5], 0
            0x3D << 26 | 0x03 << 18 | 8 << 6 | 4 >> 1,
            125 << 25,
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0280, // v_mov_b32 v1, 0
        ];
        words.extend(mimg_2d(0x00, 0xF, 2, 0, 8)); // image_load v[2:5], v[0:1], s[8:15]
        words.extend(mimg_2d(0x15, 0x1, 2, 0, 8)); // image_atomic_umin v2, v[0:1], s[8:15]
        words.push(S_ENDPGM);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 4,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let lowered = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        })
        .unwrap();

        // DXIL resource properties: word 0 is the kind (Texture2D = 2) with
        // the UAV bit (0x1000); word 1 is the component type in the low byte
        // (U32 = 5, F32 = 9, DxilConstants.h `ComponentType`) and the
        // component count above it. D3D12 allows atomics (Interlocked*) on
        // integer UAVs only, so the atomic needs its own one-U32 annotation
        // next to the load's four-float one.
        const TEXTURE_2D_UAV: u32 = 0x1002;
        assert!(
            lowered.annotations.contains(&(TEXTURE_2D_UAV, 0x409)),
            "the load's annotation: {:x?}",
            lowered.annotations
        );
        assert!(
            lowered.annotations.contains(&(TEXTURE_2D_UAV, 0x105)),
            "the atomic's annotation: {:x?}",
            lowered.annotations
        );
    }

    #[test]
    fn packed_half_inputs_pair_unlisted_rows() {
        use super::super::container::SigElement;
        use super::half_inputs;
        use crate::native::{PsInput, StageIo, Usage};

        let sig = |row: u32, half: bool| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half,
        };
        let ps_in = |row: u8, attr: u8, pack2: bool, row2: u8| PsInput {
            usage: Usage::Generic,
            row,
            mask: 3,
            attr,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        // Row 0 has row2 == row (0) and pairs with unlisted row 1;
        // row 2 explicitly names row2 == 4 (not stolen by row 0);
        // row 5 has row2 == row (5) and no remaining unlisted row after it.
        let inputs = vec![
            sig(0, true),
            sig(1, true),
            sig(2, true),
            sig(3, false),
            sig(4, true),
            sig(5, true),
        ];
        let io = StageIo::Ps {
            inputs: vec![
                ps_in(0, 0, true, 0),
                ps_in(2, 1, true, 4),
                ps_in(5, 3, true, 5),
            ],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let map = half_inputs(&io, &inputs);
        assert_eq!(map.get(&0), Some(&(true, Some(1))));
        assert_eq!(map.get(&2), Some(&(true, Some(4))));
        assert_eq!(map.get(&5), Some(&(true, None)));

        // Duplicate entries for the same row keep the first entry and do not
        // consume an extra free row; an earlier unpaired row (0) must not steal
        // a free row (3) that comes after a later fp16 row (2).
        let inputs2 = vec![sig(0, true), sig(2, true), sig(3, true)];
        let io2 = StageIo::Ps {
            inputs: vec![
                ps_in(0, 0, true, 0),
                ps_in(0, 9, false, 0),
                ps_in(2, 1, true, 2),
            ],
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let map2 = half_inputs(&io2, &inputs2);
        assert_eq!(map2.get(&0), Some(&(true, None)));
        assert_eq!(map2.get(&2), Some(&(true, Some(3))));
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }
    pub(super) fn lower_compute(
        words: &[u32], layout: Vec<RootParam>, used: Vec<UsedResource>, wave64: bool, lds: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_compute_with_views(words, layout, used, wave64, lds, &[])
    }

    pub(super) fn lower_compute_with_views(
        words: &[u32], layout: Vec<RootParam>, used: Vec<UsedResource>, wave64: bool, lds: bool,
        uint_srvs: &'static [(u8, u32, u32)],
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), optionally 512 bytes of LDS.
                rsrc2: u32::from(lds) << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        super::lower_program(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            psv: None,
            target: crate::emit::Target::ANY,
            console_cf: false,
            per_lane_only: false,
        }, false, uint_srvs)
    }

    #[test]
    fn only_a_used_structured_srv_count_requests_a_host_word() {
        let words = [
            0xBE84_0300, 0xBE85_0380, // s[4:5] = descriptor table at s0
            0xF408_0202, 0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xBF06_800A,              // s_cmp_eq_u32 s10, 0
            0xBF85_0001,              // s_cbranch_scc1 to second end
            S_ENDPGM, S_ENDPGM,
        ];
        let compile = |code: &[u32], kind, stride| {
            lower_with(code, vec![root(RootType::Table, 0, 0)], vec![UsedResource {
                kind, root: RootType::Table, table_offset_dwords: 0, lower_bound: 0, stride,
                dwords: 4, user_data: 0, slot: 0,
            }])
        };
        let requested = |s: Lowered| s.user_data.iter()
            .any(|u| u.root == RootType::Other(super::super::USER_VIEW_RECORDS));
        assert!(requested(compile(&words, ResourceType::Srv, 16).unwrap()));
        assert!(compile(&words, ResourceType::Srv, 0).is_err());
        assert!(compile(&words, ResourceType::Uav, 16).is_err());
        let mut unused = words;
        unused[4] = 0xBF06_8008; // compare the view address instead
        assert!(!requested(compile(&unused, ResourceType::Srv, 16).unwrap()));
    }


    /// From #305: 64-bit image atomics declare 64-bit integers and typed
    /// 64-bit atomics.
    #[test]
    fn image_atomics_64_bit_lower_with_typed_int64_flag() {
        use crate::native::{RootParam, RootType};

        let run = |mimg_w0: u32| {
            let words = [
                // s_mov_b32 s1, 0
                0xBE81_0380u32,
                // s_load_dwordx8 s[4:11], s[0:1], null
                0xF40C_0100,
                0xFA00_0000,
                // v_mov_b32 v0..v5, 0/1/2
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0281,
                0x7E06_0280,
                0x7E08_0282,
                0x7E0A_0280,
                // image_atomic_* (2D, unorm=1, glc=1) v[2..], v0, s[4:11]
                mimg_w0,
                0x0001_0200,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 8,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 8).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            })
        };

        // image_atomic_add (op 0x11), dim=1 (2D), unorm=1, dmask=3, glc=1
        let add = run(0xF044_1388).unwrap();
        // image_atomic_cmpswap (op 0x10), dim=1 (2D), unorm=1, dmask=0xf, glc=1
        let cmp = run(0xF040_1F88).unwrap();
        for lowered in [add, cmp] {
            let sfi0 = crate::dxbc::Container::parse(&lowered.dxil)
                .unwrap()
                .part(crate::dxbc::FourCC::SFI0)
                .unwrap()
                .data;
            let features = u64::from_le_bytes(sfi0.try_into().unwrap());
            assert_ne!(features & 0x8000, 0, "INT64 feature bit missing");
            assert_ne!(
                features & 0x40_0000,
                0,
                "ATOMIC_INT64_ON_TYPED_RESOURCE feature bit missing"
            );
        }
    }

    #[test]
    fn cube_and_msaa_descriptor_data_lower() {
        use crate::native::{RootParam, RootType};

        for dim in [3u32, 6, 7] {
            // s_mov_b32 s1, 0
            // s_load_dwordx8 s[4:11], s[0:1], null
            // s_lshr_b32 s12, s7, 16        (reads descriptor word 3 as data)
            // s_cmp_lg_u32 s12, 0
            // s_cbranch_scc1 -> s_endpgm
            // s_mov_b32 s13, 0
            // v_mov_b32 v0..v3, 0
            // image_get_resinfo (dim) v0, v0, s[4:11]  (op 0x0e, dmask=1)
            // s_endpgm
            let mimg_w0 = 0xF000_0000 | (0x0E << 18) | (1 << 8) | (dim << 3);
            let words = [
                0xBE81_0380u32,
                0xF40C_0100,
                0xFA00_0000,
                0x900C_9007,
                0xBF07_800C,
                0xBF85_0001,
                0xBE8D_0380,
                0x7E00_0280,
                mimg_w0,
                0x0001_0000,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            let out = lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            });
            // Cube (3) and MSAA (6, 7) descriptor_data() all lower without failing
            // on "an image dimension (cube or MSAA)".
            out.unwrap();
        }
    }

    #[test]
    fn msaa_image_load_and_get_resinfo_lower() {
        use crate::native::{RootParam, RootType};

        for (dim, op) in [(6u32, 0x00u32), (6, 0x0E), (7, 0x00), (7, 0x0E)] {
            // s_mov_b32 s1, 0
            // s_load_dwordx8 s[4:11], s[0:1], null
            // v_mov_b32 v0..v3, 0
            // image_load (0x00) / image_get_resinfo (0x0e) v[0..3], v[0..3], s[4:11]
            // v_cmp_ne_u32 vcc, v0, 0
            // s_cbranch_vccz -> s_endpgm
            // s_mov_b32 s12, 0
            // s_endpgm
            let mimg_w0 = 0xF000_0000 | (op << 18) | (0xF << 8) | (dim << 3);
            let words = [
                0xBE81_0380u32,
                0xF40C_0100,
                0xFA00_0000,
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                mimg_w0,
                0x0001_0000,
                0x7D9A_0080, // v_cmpx_ne_u32_e32 v0, 0 -> sets EXEC, effectful?
                0xBF81_0000,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }

    #[test]
    fn partial_channel_image_store_lowers() {
        use crate::native::{RootParam, RootType};

        // image_store (op 0x08), dim=1 (2D), unorm=1
        // Test 32-bit partial masks (0x1, 0x3, 0x5) and d16 partial mask (0x7).
        for (dmask, d16) in [(0x1u32, false), (0x3, false), (0x5, false), (0x7, true)] {
            let mimg_w0 = 0xF000_0000
                | (0x08 << 18)
                | (u32::from(d16) << 15)
                | (dmask << 8)
                | (1 << 7)
                | (1 << 3);
            let words = [
                // s_mov_b32 s1, 0
                0xBE81_0380u32,
                // s_load_dwordx8 s[4:11], s[0:1], null
                0xF40C_0100,
                0xFA00_0000,
                // v_mov_b32 v0..v3, 0 (coords v0..v1, data v2..v3)
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                // image_store v[2..], v[0..1], s[4:11]
                mimg_w0,
                0x0001_0200,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }

    #[test]
    fn partial_channel_typed_buffer_store_lowers() {
        use crate::native::{RootParam, RootType};

        // MUBUF buffer_store_format_{x,xy,xyz,d16_xyz} (ops 0x04, 0x05, 0x06, 0x86)
        // with idxen=1, vaddr=v0, vdata=v1, srsrc=s[4:7].
        for op in [0x04u32, 0x05, 0x06, 0x86] {
            let mubuf_w0 = 0xE000_2000 | (op << 18);
            let words = [
                // s_mov_b32 s1, 0
                0xBE81_0380u32,
                // s_load_dwordx4 s[4:7], s[0:1], null
                0xF408_0100,
                0xFA00_0000,
                // v_mov_b32 v0..v3, 0
                0x7E00_0280,
                0x7E02_0280,
                0x7E04_0280,
                0x7E06_0280,
                // buffer_store_format_* v[1..], v0, s[4:7], 0 idxen
                mubuf_w0,
                0x8001_0100,
                S_ENDPGM,
            ];
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 4,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 0,
                    threads: [1, 1, 1],
                }),
                code,
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let st = structure::structure(&p).unwrap();
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: vec![],
                outputs: vec![],
                prims: vec![],
                mesh_stride: 0,
                target: crate::emit::Target::ANY,
                psv: None,
                console_cf: false,
                per_lane_only: false,
            })
            .unwrap();
        }
    }
