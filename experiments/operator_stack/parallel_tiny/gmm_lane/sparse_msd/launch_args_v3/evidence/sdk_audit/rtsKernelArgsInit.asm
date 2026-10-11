
/usr/local/Ascend/cann-9.1.0/lib64/libruntime.so:	file format elf64-littleaarch64

Disassembly of section .text:

000000000004e448 <rtsKernelArgsInit>:
   4e448: a9bc7bfd     	stp	x29, x30, [sp, #-64]!
   4e44c: 910003fd     	mov	x29, sp
   4e450: a90153f3     	stp	x19, x20, [sp, #16]
   4e454: b0000513     	adrp	x19, 0xef000
   4e458: f90013f5     	str	x21, [sp, #32]
   4e45c: aa0103f5     	mov	x21, x1
   4e460: f947c663     	ldr	x3, [x19, #3976]
   4e464: aa0003f4     	mov	x20, x0
   4e468: 52800000     	mov	w0, #0
   4e46c: f9400061     	ldr	x1, [x3]
   4e470: f9001fa1     	str	x1, [x29, #56]
   4e474: d2800001     	mov	x1, #0
   4e478: 94017a54     	bl	0xacdc8
   4e47c: b40005e0     	cbz	x0, 0x4e538
   4e480: f9400003     	ldr	x3, [x0]
   4e484: aa1403e1     	mov	x1, x20
   4e488: aa1503e2     	mov	x2, x21
   4e48c: f9476c63     	ldr	x3, [x3, #3800]
   4e490: d63f0060     	blr	x3
   4e494: 2a0003f4     	mov	w20, w0
   4e498: 52800000     	mov	w0, #0
   4e49c: 35000174     	cbnz	w20, 0x4e4c8
   4e4a0: f947c673     	ldr	x19, [x19, #3976]
   4e4a4: f9401fa2     	ldr	x2, [x29, #56]
   4e4a8: f9400261     	ldr	x1, [x19]
   4e4ac: ca010041     	eor	x1, x2, x1
   4e4b0: b5000841     	cbnz	x1, 0x4e5b8
   4e4b4: a94153f3     	ldp	x19, x20, [sp, #16]
   4e4b8: f94013f5     	ldr	x21, [sp, #32]
   4e4bc: a8c47bfd     	ldp	x29, x30, [sp], #64
   4e4c0: d65f03c0     	ret
   4e4c4: d503201f     	nop
   4e4c8: 97ff045e     	bl	0xf640
   4e4cc: 90000355     	adrp	x21, 0xb6000
   4e4d0: 2a1403e1     	mov	w1, w20
   4e4d4: 9100c3a8     	add	x8, x29, #48
   4e4d8: 97ff04fa     	bl	0xf8c0
   4e4dc: f9401ba4     	ldr	x4, [x29, #48]
   4e4e0: 910fe2a2     	add	x2, x21, #1016
   4e4e4: 90000323     	adrp	x3, 0xb2000
   4e4e8: b0000320     	adrp	x0, 0xb3000
   4e4ec: 9114e063     	add	x3, x3, #1336
   4e4f0: 91012042     	add	x2, x2, #72
   4e4f4: 52801e81     	mov	w1, #244
   4e4f8: 911b6000     	add	x0, x0, #1752
   4e4fc: 97ff02c5     	bl	0xf010
   4e500: 910fe2a1     	add	x1, x21, #1016
   4e504: 2a1403e0     	mov	w0, w20
   4e508: 91012021     	add	x1, x1, #72
   4e50c: 97ff0585     	bl	0xfb20
   4e510: 97ff053c     	bl	0xfa00
   4e514: b0000501     	adrp	x1, 0xef000
   4e518: f9401ba2     	ldr	x2, [x29, #48]
   4e51c: f947f821     	ldr	x1, [x1, #4080]
   4e520: d1006040     	sub	x0, x2, #24
   4e524: eb01001f     	cmp	x0, x1
   4e528: 54000281     	b.ne	0x4e578
   4e52c: 2a1403e0     	mov	w0, w20
   4e530: 9400f1f8     	bl	0x8ad10
   4e534: 17ffffdb     	b	0x4e4a0
   4e538: 90000342     	adrp	x2, 0xb6000
   4e53c: 910fe042     	add	x2, x2, #1016
   4e540: 91012042     	add	x2, x2, #72
   4e544: 90000323     	adrp	x3, 0xb2000
   4e548: b0000320     	adrp	x0, 0xb3000
   4e54c: 91140063     	add	x3, x3, #1280
   4e550: 911b6000     	add	x0, x0, #1752
   4e554: 52978f04     	mov	w4, #48248
   4e558: 72a000e4     	movk	w4, #7, lsl #16
   4e55c: 52801e21     	mov	w1, #241
   4e560: 97ff02ac     	bl	0xf010
   4e564: 97ff0527     	bl	0xfa00
   4e568: 52978f00     	mov	w0, #48248
   4e56c: 72a000e0     	movk	w0, #7, lsl #16
   4e570: 17ffffcc     	b	0x4e4a0
   4e574: d503201f     	nop
   4e578: b0000501     	adrp	x1, 0xef000
   4e57c: f947ec21     	ldr	x1, [x1, #4056]
   4e580: b4000141     	cbz	x1, 0x4e5a8
   4e584: d1002042     	sub	x2, x2, #8
   4e588: 885ffc41     	ldaxr	w1, [x2]
   4e58c: 51000423     	sub	w3, w1, #1
   4e590: 8804fc43     	stlxr	w4, w3, [x2]
   4e594: 35ffffa4     	cbnz	w4, 0x4e588
   4e598: 7100003f     	cmp	w1, #0
   4e59c: 54fffc8c     	b.gt	0x4e52c
   4e5a0: 97ff0308     	bl	0xf1c0
   4e5a4: 17ffffe2     	b	0x4e52c
   4e5a8: b85f8041     	ldur	w1, [x2, #-8]
   4e5ac: 51000423     	sub	w3, w1, #1
   4e5b0: b81f8043     	stur	w3, [x2, #-8]
   4e5b4: 17fffff9     	b	0x4e598
   4e5b8: 97ff0486     	bl	0xf7d0
   4e5bc: b0000502     	adrp	x2, 0xef000
   4e5c0: f9401ba1     	ldr	x1, [x29, #48]
   4e5c4: aa0003f3     	mov	x19, x0
   4e5c8: f947f842     	ldr	x2, [x2, #4080]
   4e5cc: d1006020     	sub	x0, x1, #24
   4e5d0: eb02001f     	cmp	x0, x2
   4e5d4: 54000040     	b.eq	0x4e5dc
   4e5d8: 97fffd94     	bl	0x4dc28
   4e5dc: aa1303e0     	mov	x0, x19
   4e5e0: 97ff0628     	bl	0xfe80
   4e5e4: d503201f     	nop
