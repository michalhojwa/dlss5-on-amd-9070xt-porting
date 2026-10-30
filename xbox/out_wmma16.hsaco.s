	.amdgcn_target "amdgcn-amd-amdhsa--gfx1103"
	.amdhsa_code_object_version 6
	.text
	.protected	wmma_16x16x16_w32       ; -- Begin function wmma_16x16x16_w32
	.globl	wmma_16x16x16_w32
	.p2align	8
	.type	wmma_16x16x16_w32,@function
wmma_16x16x16_w32:                      ; @wmma_16x16x16_w32
; %bb.0:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[0:1], s[0:1], 0x10
	v_lshlrev_b32_e32 v1, 6, v0
	s_waitcnt lgkmcnt(0)
	s_clause 0x7
	global_load_b128 v[9:12], v1, s[6:7]
	global_load_b128 v[13:16], v1, s[0:1]
	global_load_b128 v[17:20], v1, s[6:7] offset:16
	global_load_b128 v[21:24], v1, s[0:1] offset:16
	global_load_b128 v[25:28], v1, s[6:7] offset:32
	global_load_b128 v[29:32], v1, s[0:1] offset:32
	global_load_b128 v[33:36], v1, s[6:7] offset:48
	global_load_b128 v[37:40], v1, s[0:1] offset:48
	v_dual_mov_b32 v1, 0 :: v_dual_lshlrev_b32 v0, 5, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_mov_b32_e32 v2, v1
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v7, v1
	v_mov_b32_e32 v8, v1
	s_waitcnt vmcnt(7)
	v_cvt_f16_f32_e32 v9, v9
	s_waitcnt vmcnt(6)
	v_cvt_f16_f32_e32 v41, v13
	v_cvt_f16_f32_e32 v42, v10
	v_cvt_f16_f32_e32 v43, v14
	v_cvt_f16_f32_e32 v10, v11
	v_cvt_f16_f32_e32 v44, v15
	v_cvt_f16_f32_e32 v45, v12
	v_cvt_f16_f32_e32 v46, v16
	s_waitcnt vmcnt(5)
	v_cvt_f16_f32_e32 v11, v17
	s_waitcnt vmcnt(4)
	v_cvt_f16_f32_e32 v17, v21
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v47, v22
	v_cvt_f16_f32_e32 v12, v19
	v_cvt_f16_f32_e32 v19, v23
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v48, v24
	s_waitcnt vmcnt(3)
	v_cvt_f16_f32_e32 v13, v25
	s_waitcnt vmcnt(2)
	v_cvt_f16_f32_e32 v21, v29
	v_cvt_f16_f32_e32 v22, v26
	v_cvt_f16_f32_e32 v25, v30
	v_cvt_f16_f32_e32 v14, v27
	v_cvt_f16_f32_e32 v26, v31
	v_cvt_f16_f32_e32 v23, v28
	v_cvt_f16_f32_e32 v27, v32
	s_waitcnt vmcnt(1)
	v_cvt_f16_f32_e32 v15, v33
	s_waitcnt vmcnt(0)
	v_cvt_f16_f32_e32 v28, v37
	v_cvt_f16_f32_e32 v16, v35
	v_cvt_f16_f32_e32 v24, v36
	v_cvt_f16_f32_e32 v29, v34
	v_cvt_f16_f32_e32 v30, v38
	v_cvt_f16_f32_e32 v31, v39
	v_cvt_f16_f32_e32 v32, v40
	v_pack_b32_f16 v16, v16, v24
	v_pack_b32_f16 v15, v15, v29
	v_pack_b32_f16 v14, v14, v23
	v_pack_b32_f16 v13, v13, v22
	v_pack_b32_f16 v12, v12, v20
	v_pack_b32_f16 v11, v11, v18
	v_pack_b32_f16 v10, v10, v45
	v_pack_b32_f16 v9, v9, v42
	v_pack_b32_f16 v24, v31, v32
	v_pack_b32_f16 v23, v28, v30
	v_pack_b32_f16 v22, v26, v27
	v_pack_b32_f16 v21, v21, v25
	v_pack_b32_f16 v20, v19, v48
	v_pack_b32_f16 v19, v17, v47
	v_pack_b32_f16 v18, v44, v46
	v_pack_b32_f16 v17, v41, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[9:16], v[17:24], v[1:8]
	global_store_b32 v0, v1, s[4:5]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_16x16x16_w32
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 49
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	wmma_16x16x16_w32, .Lfunc_end0-wmma_16x16x16_w32
                                        ; -- End function
	.set wmma_16x16x16_w32.num_vgpr, 49
	.set wmma_16x16x16_w32.num_agpr, 0
	.set wmma_16x16x16_w32.numbered_sgpr, 8
	.set wmma_16x16x16_w32.private_seg_size, 0
	.set wmma_16x16x16_w32.uses_vcc, 0
	.set wmma_16x16x16_w32.uses_flat_scratch, 0
	.set wmma_16x16x16_w32.has_dyn_sized_stack, 0
	.set wmma_16x16x16_w32.has_recursion, 0
	.set wmma_16x16x16_w32.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 448
; TotalNumSgprs: 8
; NumVgprs: 49
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 8
; NumVGPRsForWavesPerEU: 49
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_524d7cf39f0236f1,@object ; @__hip_cuid_524d7cf39f0236f1
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_524d7cf39f0236f1
__hip_cuid_524d7cf39f0236f1:
	.byte	0                               ; 0x0
	.size	__hip_cuid_524d7cf39f0236f1, 1

	.ident	"clang version 21.0.0git (git@github.com:AMD-Lightning-Internal/llvm-project 590b9320a5be90e40268759c6203c01fde121e68)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_524d7cf39f0236f1
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 32
    .name:           wmma_16x16x16_w32
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .sgpr_spill_count: 0
    .symbol:         wmma_16x16x16_w32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     49
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1103
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
