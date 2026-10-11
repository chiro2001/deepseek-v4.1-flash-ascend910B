
/usr/local/Ascend/cann-9.1.0/lib64/libruntime.so:	file format elf64-littleaarch64

Disassembly of section .text:

000000000004e5e8 <rtsKernelArgsAppend>:
   4e5e8: a9bb7bfd     	stp	x29, x30, [sp, #-80]!
   4e5ec: 910003fd     	mov	x29, sp
   4e5f0: a90153f3     	stp	x19, x20, [sp, #16]
   4e5f4: b0000513     	adrp	x19, 0xef000
   4e5f8: a9025bf5     	stp	x21, x22, [sp, #32]
   4e5fc: aa0103f5     	mov	x21, x1
   4e600: f9001bf7     	str	x23, [sp, #48]
   4e604: aa0003f4     	mov	x20, x0
   4e608: f947c665     	ldr	x5, [x19, #3976]
   4e60c: aa0203f6     	mov	x22, x2
   4e610: 52800000     	mov	w0, #0
   4e614: aa0303f7     	mov	x23, x3
   4e618: f94000a1     	ldr	x1, [x5]
   4e61c: f90027a1     	str	x1, [x29, #72]
   4e620: d2800001     	mov	x1, #0
   4e624: 940179e9     	bl	0xacdc8
   4e628: b4000640     	cbz	x0, 0x4e6f0
   4e62c: f9400005     	ldr	x5, [x0]
   4e630: aa1403e1     	mov	x1, x20
   4e634: aa1703e4     	mov	x4, x23
   4e638: aa1603e3     	mov	x3, x22
   4e63c: aa1503e2     	mov	x2, x21
   4e640: f94788a5     	ldr	x5, [x5, #3856]
   4e644: d63f00a0     	blr	x5
   4e648: 2a0003f4     	mov	w20, w0
   4e64c: 52800000     	mov	w0, #0
   4e650: 35000194     	cbnz	w20, 0x4e680
   4e654: f947c673     	ldr	x19, [x19, #3976]
   4e658: f94027a2     	ldr	x2, [x29, #72]
   4e65c: f9400261     	ldr	x1, [x19]
   4e660: ca010041     	eor	x1, x2, x1
   4e664: b5000861     	cbnz	x1, 0x4e770
   4e668: a94153f3     	ldp	x19, x20, [sp, #16]
   4e66c: a9425bf5     	ldp	x21, x22, [sp, #32]
   4e670: f9401bf7     	ldr	x23, [sp, #48]
   4e674: a8c57bfd     	ldp	x29, x30, [sp], #80
   4e678: d65f03c0     	ret
   4e67c: d503201f     	nop
   4e680: 97ff03f0     	bl	0xf640
   4e684: 90000355     	adrp	x21, 0xb6000
   4e688: 2a1403e1     	mov	w1, w20
   4e68c: 910103a8     	add	x8, x29, #64
   4e690: 97ff048c     	bl	0xf8c0
   4e694: f94023a4     	ldr	x4, [x29, #64]
   4e698: 910fe2a2     	add	x2, x21, #1016
   4e69c: 90000323     	adrp	x3, 0xb2000
   4e6a0: b0000320     	adrp	x0, 0xb3000
   4e6a4: 9114e063     	add	x3, x3, #1336
   4e6a8: 91018042     	add	x2, x2, #96
   4e6ac: 52802001     	mov	w1, #256
   4e6b0: 911b6000     	add	x0, x0, #1752
   4e6b4: 97ff0257     	bl	0xf010
   4e6b8: 910fe2a1     	add	x1, x21, #1016
   4e6bc: 2a1403e0     	mov	w0, w20
   4e6c0: 91018021     	add	x1, x1, #96
   4e6c4: 97ff0517     	bl	0xfb20
   4e6c8: 97ff04ce     	bl	0xfa00
   4e6cc: b0000501     	adrp	x1, 0xef000
   4e6d0: f94023a2     	ldr	x2, [x29, #64]
   4e6d4: f947f821     	ldr	x1, [x1, #4080]
   4e6d8: d1006040     	sub	x0, x2, #24
   4e6dc: eb01001f     	cmp	x0, x1
   4e6e0: 54000281     	b.ne	0x4e730
   4e6e4: 2a1403e0     	mov	w0, w20
   4e6e8: 9400f18a     	bl	0x8ad10
   4e6ec: 17ffffda     	b	0x4e654
   4e6f0: 90000342     	adrp	x2, 0xb6000
   4e6f4: 910fe042     	add	x2, x2, #1016
   4e6f8: 91018042     	add	x2, x2, #96
   4e6fc: 90000323     	adrp	x3, 0xb2000
   4e700: b0000320     	adrp	x0, 0xb3000
   4e704: 91140063     	add	x3, x3, #1280
   4e708: 911b6000     	add	x0, x0, #1752
   4e70c: 52978f04     	mov	w4, #48248
   4e710: 72a000e4     	movk	w4, #7, lsl #16
   4e714: 52801fa1     	mov	w1, #253
   4e718: 97ff023e     	bl	0xf010
   4e71c: 97ff04b9     	bl	0xfa00
   4e720: 52978f00     	mov	w0, #48248
   4e724: 72a000e0     	movk	w0, #7, lsl #16
   4e728: 17ffffcb     	b	0x4e654
   4e72c: d503201f     	nop
   4e730: b0000501     	adrp	x1, 0xef000
   4e734: f947ec21     	ldr	x1, [x1, #4056]
   4e738: b4000141     	cbz	x1, 0x4e760
   4e73c: d1002042     	sub	x2, x2, #8
   4e740: 885ffc41     	ldaxr	w1, [x2]
   4e744: 51000423     	sub	w3, w1, #1
   4e748: 8804fc43     	stlxr	w4, w3, [x2]
   4e74c: 35ffffa4     	cbnz	w4, 0x4e740
   4e750: 7100003f     	cmp	w1, #0
   4e754: 54fffc8c     	b.gt	0x4e6e4
   4e758: 97ff029a     	bl	0xf1c0
   4e75c: 17ffffe2     	b	0x4e6e4
   4e760: b85f8041     	ldur	w1, [x2, #-8]
   4e764: 51000423     	sub	w3, w1, #1
   4e768: b81f8043     	stur	w3, [x2, #-8]
   4e76c: 17fffff9     	b	0x4e750
   4e770: 97ff0418     	bl	0xf7d0
   4e774: b0000502     	adrp	x2, 0xef000
   4e778: f94023a1     	ldr	x1, [x29, #64]
   4e77c: aa0003f3     	mov	x19, x0
   4e780: f947f842     	ldr	x2, [x2, #4080]
   4e784: d1006020     	sub	x0, x1, #24
   4e788: eb02001f     	cmp	x0, x2
   4e78c: 54000040     	b.eq	0x4e794
   4e790: 97fffd26     	bl	0x4dc28
   4e794: aa1303e0     	mov	x0, x19
   4e798: 97ff05ba     	bl	0xfe80
   4e79c: d503201f     	nop
