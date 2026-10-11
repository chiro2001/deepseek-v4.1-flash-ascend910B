
/usr/local/Ascend/cann-9.1.0/lib64/libruntime_v200.so:	file format elf64-littleaarch64

Disassembly of section .text:

00000000004e1280 <_ZNSt8_Rb_treeI19rtKernelDfxInfoTypeSt4pairIKS0_PFvS0_jjPKhmEESt10_Select1stIS7_ESt4lessIS0_ESaIS7_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorIS7_ERS2_>:
  4f9b74: d1006000     	sub	x0, x0, #24
  4f9b78: eb17001f     	cmp	x0, x23
  4f9b7c: 54000040     	b.eq	0x4f9b84
  4f9b80: 97fff8bc     	bl	0x4f7e70
  4f9b84: d1002294     	sub	x20, x20, #8
  4f9b88: 17fffff8     	b	0x4f9b68
  4f9b8c: aa0003f3     	mov	x19, x0
  4f9b90: 17fffff3     	b	0x4f9b5c
  4f9b94: d0000560     	adrp	x0, 0x5a7000
  4f9b98: 913b8000     	add	x0, x0, #3808
  4f9b9c: 97ed3b39     	bl	0x48880
  4f9ba0: d0000560     	adrp	x0, 0x5a7000
  4f9ba4: 913b8000     	add	x0, x0, #3808
  4f9ba8: 97ed3b36     	bl	0x48880
  4f9bac: d0000560     	adrp	x0, 0x5a7000
  4f9bb0: 913b8000     	add	x0, x0, #3808
  4f9bb4: 97ed3b33     	bl	0x48880
  4f9bb8: 17ffffb1     	b	0x4f9a7c
  4f9bbc: d503201f     	nop
  4f9bc0: d108c3ff     	sub	sp, sp, #560
  4f9bc4: a9017bfd     	stp	x29, x30, [sp, #16]
  4f9bc8: 910043fd     	add	x29, sp, #16
  4f9bcc: a90253f3     	stp	x19, x20, [sp, #32]
  4f9bd0: d0001334     	adrp	x20, 0x75f000
  4f9bd4: f9002ffa     	str	x26, [sp, #88]
  4f9bd8: aa0003fa     	mov	x26, x0
  4f9bdc: f942a681     	ldr	x1, [x20, #1352]
  4f9be0: f9400800     	ldr	x0, [x0, #16]
  4f9be4: f9400022     	ldr	x2, [x1]
  4f9be8: f9010fa2     	str	x2, [x29, #536]
  4f9bec: d2800002     	mov	x2, #0
  4f9bf0: b4000dc0     	cbz	x0, 0x4f9da8
  4f9bf4: f9400001     	ldr	x1, [x0]
  4f9bf8: b4002ba1     	cbz	x1, 0x4fa16c
  4f9bfc: f9400400     	ldr	x0, [x0, #8]
  4f9c00: b4000820     	cbz	x0, 0x4f9d04
  4f9c04: a90363b7     	stp	x23, x24, [x29, #48]
  4f9c08: 90000bb7     	adrp	x23, 0x66d000
  4f9c0c: 52886878     	mov	w24, #17219
  4f9c10: 913c42f7     	add	x23, x23, #3856
  4f9c14: f90023b9     	str	x25, [x29, #64]
  4f9c18: 528848b9     	mov	w25, #16965
  4f9c1c: a9025bb5     	stp	x21, x22, [x29, #32]
  4f9c20: 910082f7     	add	x23, x23, #32
  4f9c24: d2800015     	mov	x21, #0
  4f9c28: d2800016     	mov	x22, #0
  4f9c2c: 72a82938     	movk	w24, #16713, lsl #16
  4f9c30: f9002bbb     	str	x27, [x29, #80]
  4f9c34: 72a86ab9     	movk	w25, #17237, lsl #16
  4f9c38: 14000012     	b	0x4f9c80
  4f9c3c: d503201f     	nop
  4f9c40: 71000a7f     	cmp	w19, #2
  4f9c44: 540007a0     	b.eq	0x4f9d38
  4f9c48: 71000e7f     	cmp	w19, #3
  4f9c4c: 54000d21     	b.ne	0x4f9df0
  4f9c50: 91001021     	add	x1, x1, #4
  4f9c54: aa1a03e0     	mov	x0, x26
  4f9c58: 97fffd8a     	bl	0x4f9280
  4f9c5c: 2a0003f3     	mov	w19, w0
  4f9c60: 35002680     	cbnz	w0, 0x4fa130
  4f9c64: f9400b40     	ldr	x0, [x26, #16]
  4f9c68: 910006d6     	add	x22, x22, #1
  4f9c6c: 910052b5     	add	x21, x21, #20
  4f9c70: f9400401     	ldr	x1, [x0, #8]
  4f9c74: eb16003f     	cmp	x1, x22
  4f9c78: 540003e9     	b.ls	0x4f9cf4
  4f9c7c: f9400001     	ldr	x1, [x0]
  4f9c80: b8756833     	ldr	w19, [x1, x21]
  4f9c84: 8b150021     	add	x1, x1, x21
  4f9c88: 7100067f     	cmp	w19, #1
  4f9c8c: 54fffda1     	b.ne	0x4f9c40
  4f9c90: b9400433     	ldr	w19, [x1, #4]
  4f9c94: 7100067f     	cmp	w19, #1
  4f9c98: 54000840     	b.eq	0x4f9da0
  4f9c9c: 35002ad3     	cbnz	w19, 0x4fa1f4
  4f9ca0: 3900d35f     	strb	wzr, [x26, #52]
  4f9ca4: 52800001     	mov	w1, #0
  4f9ca8: 528000e0     	mov	w0, #7
  4f9cac: 97ed3f15     	bl	0x49900
  4f9cb0: 7100041f     	cmp	w0, #1
  4f9cb4: 54fffd81     	b.ne	0x4f9c64
  4f9cb8: 52800000     	mov	w0, #0
  4f9cbc: b0000a04     	adrp	x4, 0x63a000
  4f9cc0: b0000a01     	adrp	x1, 0x63a000
  4f9cc4: 2a1303e5     	mov	w5, w19
  4f9cc8: 9120c021     	add	x1, x1, #2096
  4f9ccc: 912d8084     	add	x4, x4, #2912
  4f9cd0: aa1703e3     	mov	x3, x23
  4f9cd4: 52801002     	mov	w2, #128
  4f9cd8: 97ed3b8e     	bl	0x48b10
  4f9cdc: 910006d6     	add	x22, x22, #1
  4f9ce0: f9400b40     	ldr	x0, [x26, #16]
  4f9ce4: 910052b5     	add	x21, x21, #20
  4f9ce8: f9400401     	ldr	x1, [x0, #8]
  4f9cec: eb16003f     	cmp	x1, x22
  4f9cf0: 54fffc68     	b.hi	0x4f9c7c
  4f9cf4: a9425bb5     	ldp	x21, x22, [x29, #32]
  4f9cf8: a94363b7     	ldp	x23, x24, [x29, #48]
  4f9cfc: f94023b9     	ldr	x25, [x29, #64]
  4f9d00: f9402bbb     	ldr	x27, [x29, #80]
  4f9d04: 52800013     	mov	w19, #0
  4f9d08: f942a694     	ldr	x20, [x20, #1352]
  4f9d0c: 2a1303e0     	mov	w0, w19
  4f9d10: f9410fa2     	ldr	x2, [x29, #536]
  4f9d14: f9400281     	ldr	x1, [x20]
  4f9d18: ca010041     	eor	x1, x2, x1
  4f9d1c: b5004001     	cbnz	x1, 0x4fa51c
  4f9d20: a94253f3     	ldp	x19, x20, [sp, #32]
  4f9d24: a9417bfd     	ldp	x29, x30, [sp, #16]
  4f9d28: f9402ffa     	ldr	x26, [sp, #88]
  4f9d2c: 9108c3ff     	add	sp, sp, #560
  4f9d30: d65f03c0     	ret
  4f9d34: d503201f     	nop
  4f9d38: 39406340     	ldrb	w0, [x26, #24]
  4f9d3c: 35002a80     	cbnz	w0, 0x4fa28c
  4f9d40: b9400433     	ldr	w19, [x1, #4]
  4f9d44: 528a48db     	mov	w27, #21062
  4f9d48: 72a8283b     	movk	w27, #16705, lsl #16
  4f9d4c: 6b19027f     	cmp	w19, w25
  4f9d50: 7a581264     	ccmp	w19, w24, #4, ne
  4f9d54: 7a5b1264     	ccmp	w19, w27, #4, ne
  4f9d58: 54002d81     	b.ne	0x4fa308
  4f9d5c: b9003353     	str	w19, [x26, #48]
  4f9d60: 52800001     	mov	w1, #0
  4f9d64: 528000e0     	mov	w0, #7
  4f9d68: 97ed3ee6     	bl	0x49900
  4f9d6c: 7100041f     	cmp	w0, #1
  4f9d70: 54fff7a1     	b.ne	0x4f9c64
  4f9d74: b0000a04     	adrp	x4, 0x63a000
  4f9d78: b0000a01     	adrp	x1, 0x63a000
  4f9d7c: 2a1303e5     	mov	w5, w19
  4f9d80: 9132a084     	add	x4, x4, #3240
  4f9d84: aa1703e3     	mov	x3, x23
  4f9d88: 528011c2     	mov	w2, #142
  4f9d8c: 9120c021     	add	x1, x1, #2096
  4f9d90: 52800000     	mov	w0, #0
  4f9d94: 97ed3b5f     	bl	0x48b10
  4f9d98: 17ffffb3     	b	0x4f9c64
  4f9d9c: d503201f     	nop
  4f9da0: 3900d353     	strb	w19, [x26, #52]
  4f9da4: 17ffffc0     	b	0x4f9ca4
  4f9da8: 52800001     	mov	w1, #0
  4f9dac: 528000e0     	mov	w0, #7
  4f9db0: 97ed3ed4     	bl	0x49900
  4f9db4: 7100041f     	cmp	w0, #1
  4f9db8: 54fffa61     	b.ne	0x4f9d04
  4f9dbc: 90000ba3     	adrp	x3, 0x66d000
  4f9dc0: 913c4063     	add	x3, x3, #3856
  4f9dc4: b0000a04     	adrp	x4, 0x63a000
  4f9dc8: b0000a01     	adrp	x1, 0x63a000
  4f9dcc: 9128c084     	add	x4, x4, #2608
  4f9dd0: 91008063     	add	x3, x3, #32
  4f9dd4: 52800d82     	mov	w2, #108
  4f9dd8: 9120c021     	add	x1, x1, #2096
  4f9ddc: 52800000     	mov	w0, #0
  4f9de0: 52800013     	mov	w19, #0
  4f9de4: 97ed3b4b     	bl	0x48b10
  4f9de8: 17ffffc8     	b	0x4f9d08
  4f9dec: d503201f     	nop
  4f9df0: 90000ba0     	adrp	x0, 0x66d000
  4f9df4: 9101e3b9     	add	x25, x29, #120
  4f9df8: 913c4001     	add	x1, x0, #3856
  4f9dfc: aa1903e2     	mov	x2, x25
  4f9e00: 91008021     	add	x1, x1, #32
  4f9e04: 910243bb     	add	x27, x29, #144
  4f9e08: 9102c3b6     	add	x22, x29, #176
  4f9e0c: aa1b03e0     	mov	x0, x27
  4f9e10: f9002fbc     	str	x28, [x29, #88]
  4f9e14: 97ed4293     	bl	0x4a860
  4f9e18: 910162c0     	add	x0, x22, #88
  4f9e1c: 97ed38d9     	bl	0x48180
  4f9e20: d0001320     	adrp	x0, 0x75f000
  4f9e24: d0001338     	adrp	x24, 0x75f000
  4f9e28: 4f000400     	movi	v0.4s, #0
  4f9e2c: 3907a3bf     	strb	wzr, [x29, #488]
  4f9e30: f9429000     	ldr	x0, [x0, #1312]
  4f9e34: f9419b01     	ldr	x1, [x24, #816]
  4f9e38: 3907a7bf     	strb	wzr, [x29, #489]
  4f9e3c: f900f3bf     	str	xzr, [x29, #480]
  4f9e40: 91004021     	add	x1, x1, #16
  4f9e44: a940f01a     	ldp	x26, x28, [x0, #8]
  4f9e48: f9005bba     	str	x26, [x29, #176]
  4f9e4c: f90087a1     	str	x1, [x29, #264]
  4f9e50: d2800001     	mov	x1, #0
  4f9e54: 3d807fa0     	str	q0, [x29, #496]
  4f9e58: 3d8083a0     	str	q0, [x29, #512]
  4f9e5c: f85e8340     	ldur	x0, [x26, #-24]
  4f9e60: f8206adc     	str	x28, [x22, x0]
  4f9e64: f9405ba0     	ldr	x0, [x29, #176]
  4f9e68: f85e8000     	ldur	x0, [x0, #-24]
  4f9e6c: 8b0002c0     	add	x0, x22, x0
  4f9e70: 97ed3fc8     	bl	0x49d90
  4f9e74: d0001320     	adrp	x0, 0x75f000
  4f9e78: d0001337     	adrp	x23, 0x75f000
  4f9e7c: 4f000400     	movi	v0.4s, #0
  4f9e80: d0001335     	adrp	x21, 0x75f000
  4f9e84: f944e2e1     	ldr	x1, [x23, #2496]
  4f9e88: f9433802     	ldr	x2, [x0, #1648]
  4f9e8c: 910102c0     	add	x0, x22, #64
  4f9e90: 91006023     	add	x3, x1, #24
  4f9e94: 91010021     	add	x1, x1, #64
  4f9e98: 91004042     	add	x2, x2, #16
  4f9e9c: f90087a1     	str	x1, [x29, #264]
  4f9ea0: a90b0ba3     	stp	x3, x2, [x29, #176]
  4f9ea4: 3d8033a0     	str	q0, [x29, #192]
  4f9ea8: 3d8037a0     	str	q0, [x29, #208]
  4f9eac: 3d803ba0     	str	q0, [x29, #224]
  4f9eb0: 97ed43d8     	bl	0x4ae10
  4f9eb4: d0001320     	adrp	x0, 0x75f000
  4f9eb8: 52800203     	mov	w3, #16
  4f9ebc: 910022c1     	add	x1, x22, #8
