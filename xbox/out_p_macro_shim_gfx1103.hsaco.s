	.amdgcn_target "amdgcn-amd-amdhsa--gfx1103"
	.amdhsa_code_object_version 6
	.text
	.protected	shim                    ; -- Begin function shim
	.globl	shim
	.p2align	8
	.type	shim,@function
shim:                                   ; @shim
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_dual_mov_b32 v1, 2 :: v_dual_lshlrev_b32 v0, 2, v0
	s_waitcnt lgkmcnt(0)
	global_store_b32 v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel shim
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
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
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
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
		.amdhsa_inst_pref_size 1
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
	.size	shim, .Lfunc_end0-shim
                                        ; -- End function
	.set shim.num_vgpr, 2
	.set shim.num_agpr, 0
	.set shim.numbered_sgpr, 2
	.set shim.private_seg_size, 0
	.set shim.uses_vcc, 0
	.set shim.uses_flat_scratch, 0
	.set shim.has_dyn_sized_stack, 0
	.set shim.has_recursion, 0
	.set shim.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 32
; TotalNumSgprs: 2
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 2
; NumVGPRsForWavesPerEU: 2
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
	.type	__hip_cuid_fad860ea95e7e82b,@object ; @__hip_cuid_fad860ea95e7e82b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_fad860ea95e7e82b
__hip_cuid_fad860ea95e7e82b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_fad860ea95e7e82b, 1

	.ident	"clang version 21.0.0git (git@github.com:AMD-Lightning-Internal/llvm-project 590b9320a5be90e40268759c6203c01fde121e68)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_fad860ea95e7e82b
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           shim
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .sgpr_spill_count: 0
    .symbol:         shim.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1103
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
