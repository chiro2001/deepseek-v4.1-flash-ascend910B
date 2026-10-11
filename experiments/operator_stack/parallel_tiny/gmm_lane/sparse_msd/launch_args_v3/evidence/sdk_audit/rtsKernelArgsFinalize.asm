
/usr/local/Ascend/cann-9.1.0/lib64/libruntime.so:	file format elf64-littleaarch64

Disassembly of section .text:

000000000004eff8 <rtsKernelArgsFinalize>:
   4eff8: a9bc7bfd     	stp	x29, x30, [sp, #-64]!
   4effc: 910003fd     	mov	x29, sp
   4f000: a90153f3     	stp	x19, x20, [sp, #16]
   4f004: 90000513     	adrp	x19, 0xef000
   4f008: aa0003f4     	mov	x20, x0
   4f00c: 52800000     	mov	w0, #0
   4f010: f947c662     	ldr	x2, [x19, #3976]
   4f014: f9400041     	ldr	x1, [x2]
   4f018: f9001fa1     	str	x1, [x29, #56]
   4f01c: d2800001     	mov	x1, #0
   4f020: 9401776a     	bl	0xacdc8
   4f024: b40005e0     	cbz	x0, 0x4f0e0
   4f028: f9400002     	ldr	x2, [x0]
   4f02c: aa1403e1     	mov	x1, x20
   4f030: f9478442     	ldr	x2, [x2, #3848]
   4f034: d63f0040     	blr	x2
   4f038: 2a0003f4     	mov	w20, w0
   4f03c: 52800000     	mov	w0, #0
   4f040: 35000154     	cbnz	w20, 0x4f068
   4f044: f947c673     	ldr	x19, [x19, #3976]
   4f048: f9401fa2     	ldr	x2, [x29, #56]
   4f04c: f9400261     	ldr	x1, [x19]
   4f050: ca010041     	eor	x1, x2, x1
   4f054: b5000861     	cbnz	x1, 0x4f160
   4f058: a94153f3     	ldp	x19, x20, [sp, #16]
   4f05c: a8c47bfd     	ldp	x29, x30, [sp], #64
   4f060: d65f03c0     	ret
   4f064: d503201f     	nop
   4f068: f90013b5     	str	x21, [x29, #32]
   4f06c: 97ff0175     	bl	0xf640
   4f070: 2a1403e1     	mov	w1, w20
   4f074: 9100c3a8     	add	x8, x29, #48
   4f078: 97ff0212     	bl	0xf8c0
   4f07c: f0000335     	adrp	x21, 0xb6000
   4f080: f9401ba4     	ldr	x4, [x29, #48]
   4f084: 910fe2a2     	add	x2, x21, #1016
   4f088: f0000303     	adrp	x3, 0xb2000
   4f08c: 90000320     	adrp	x0, 0xb3000
   4f090: 9114e063     	add	x3, x3, #1336
   4f094: 91046042     	add	x2, x2, #280
   4f098: 52802ca1     	mov	w1, #357
   4f09c: 911b6000     	add	x0, x0, #1752
   4f0a0: 97feffdc     	bl	0xf010
   4f0a4: 910fe2a1     	add	x1, x21, #1016
   4f0a8: 2a1403e0     	mov	w0, w20
   4f0ac: 91046021     	add	x1, x1, #280
   4f0b0: 97ff029c     	bl	0xfb20
   4f0b4: 97ff0253     	bl	0xfa00
   4f0b8: 90000501     	adrp	x1, 0xef000
   4f0bc: f9401ba2     	ldr	x2, [x29, #48]
   4f0c0: f947f821     	ldr	x1, [x1, #4080]
   4f0c4: d1006040     	sub	x0, x2, #24
   4f0c8: eb01001f     	cmp	x0, x1
   4f0cc: 540002a1     	b.ne	0x4f120
   4f0d0: 2a1403e0     	mov	w0, w20
   4f0d4: 9400ef0f     	bl	0x8ad10
   4f0d8: f94013b5     	ldr	x21, [x29, #32]
   4f0dc: 17ffffda     	b	0x4f044
   4f0e0: f0000322     	adrp	x2, 0xb6000
   4f0e4: 910fe042     	add	x2, x2, #1016
   4f0e8: 91046042     	add	x2, x2, #280
   4f0ec: f0000303     	adrp	x3, 0xb2000
   4f0f0: 90000320     	adrp	x0, 0xb3000
   4f0f4: 91140063     	add	x3, x3, #1280
   4f0f8: 911b6000     	add	x0, x0, #1752
   4f0fc: 52978f04     	mov	w4, #48248
   4f100: 72a000e4     	movk	w4, #7, lsl #16
   4f104: 52802c61     	mov	w1, #355
   4f108: 97feffc2     	bl	0xf010
   4f10c: 97ff023d     	bl	0xfa00
   4f110: 52978f00     	mov	w0, #48248
   4f114: 72a000e0     	movk	w0, #7, lsl #16
   4f118: 17ffffcb     	b	0x4f044
   4f11c: d503201f     	nop
   4f120: 90000501     	adrp	x1, 0xef000
   4f124: f947ec21     	ldr	x1, [x1, #4056]
   4f128: b4000141     	cbz	x1, 0x4f150
   4f12c: d1002042     	sub	x2, x2, #8
   4f130: 885ffc41     	ldaxr	w1, [x2]
   4f134: 51000423     	sub	w3, w1, #1
   4f138: 8804fc43     	stlxr	w4, w3, [x2]
   4f13c: 35ffffa4     	cbnz	w4, 0x4f130
   4f140: 7100003f     	cmp	w1, #0
   4f144: 54fffc6c     	b.gt	0x4f0d0
   4f148: 97ff001e     	bl	0xf1c0
   4f14c: 17ffffe1     	b	0x4f0d0
   4f150: b85f8041     	ldur	w1, [x2, #-8]
   4f154: 51000423     	sub	w3, w1, #1
   4f158: b81f8043     	stur	w3, [x2, #-8]
   4f15c: 17fffff9     	b	0x4f140
   4f160: f90013b5     	str	x21, [x29, #32]
   4f164: 97ff019b     	bl	0xf7d0
   4f168: 90000502     	adrp	x2, 0xef000
   4f16c: f9401ba1     	ldr	x1, [x29, #48]
   4f170: aa0003f3     	mov	x19, x0
   4f174: f947f842     	ldr	x2, [x2, #4080]
   4f178: d1006020     	sub	x0, x1, #24
   4f17c: eb02001f     	cmp	x0, x2
   4f180: 54000040     	b.eq	0x4f188
   4f184: 97fffaa9     	bl	0x4dc28
   4f188: aa1303e0     	mov	x0, x19
   4f18c: 97ff033d     	bl	0xfe80
