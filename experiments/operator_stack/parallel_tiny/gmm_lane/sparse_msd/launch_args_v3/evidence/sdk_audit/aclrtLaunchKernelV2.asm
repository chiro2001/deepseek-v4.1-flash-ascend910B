
/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so:	file format elf64-littleaarch64

Disassembly of section .text:

00000000000193c8 <aclrtLaunchKernelV2>:
   193c8: a9ba7bfd     	stp	x29, x30, [sp, #-96]!
   193cc: 910003fd     	mov	x29, sp
   193d0: a90153f3     	stp	x19, x20, [sp, #16]
   193d4: 900001d4     	adrp	x20, 0x51000
   193d8: 91044293     	add	x19, x20, #272
   193dc: a9025bf5     	stp	x21, x22, [sp, #32]
   193e0: a90363f7     	stp	x23, x24, [sp, #48]
   193e4: aa0003f5     	mov	x21, x0
   193e8: a9046bf9     	stp	x25, x26, [sp, #64]
   193ec: 2a0103f6     	mov	w22, w1
   193f0: aa0203f7     	mov	x23, x2
   193f4: aa0303f8     	mov	x24, x3
   193f8: aa0403f9     	mov	x25, x4
   193fc: aa0503fa     	mov	x26, x5
   19400: 910ac260     	add	x0, x19, #688
   19404: 08dffc01     	ldarb	w1, [x0]
   19408: 36000201     	tbz	w1, #0, 0x19448
   1940c: f9415e66     	ldr	x6, [x19, #696]
   19410: b4000386     	cbz	x6, 0x19480
   19414: a94153f3     	ldp	x19, x20, [sp, #16]
   19418: aa1a03e5     	mov	x5, x26
   1941c: aa1903e4     	mov	x4, x25
   19420: aa1803e3     	mov	x3, x24
   19424: a9446bf9     	ldp	x25, x26, [sp, #64]
   19428: aa1703e2     	mov	x2, x23
   1942c: 2a1603e1     	mov	w1, w22
   19430: aa1503e0     	mov	x0, x21
   19434: a94363f7     	ldp	x23, x24, [sp, #48]
   19438: a9425bf5     	ldp	x21, x22, [sp, #32]
   1943c: a8c67bfd     	ldp	x29, x30, [sp], #96
   19440: d61f00c0     	br	x6
   19444: d503201f     	nop
   19448: 97ffbfde     	bl	0x93c0
   1944c: 34fffe00     	cbz	w0, 0x1940c
   19450: 97ffc218     	bl	0x9cb0
   19454: 91044294     	add	x20, x20, #272
   19458: d00000a1     	adrp	x1, 0x2f000
   1945c: 91312021     	add	x1, x1, #3144
   19460: 97ffbffc     	bl	0x9450
   19464: aa0003e6     	mov	x6, x0
   19468: 910ac280     	add	x0, x20, #688
   1946c: f9015e86     	str	x6, [x20, #696]
   19470: f9002fa6     	str	x6, [x29, #88]
   19474: 97ffc00f     	bl	0x94b0
   19478: f9402fa6     	ldr	x6, [x29, #88]
   1947c: 17ffffe5     	b	0x19410
   19480: d0000081     	adrp	x1, 0x2b000
   19484: 91110021     	add	x1, x1, #1088
   19488: 913a8021     	add	x1, x1, #3744
   1948c: 900000a0     	adrp	x0, 0x2d000
   19490: 91328000     	add	x0, x0, #3232
   19494: 97ffc073     	bl	0x9660
   19498: 12800000     	mov	w0, #-1
   1949c: 97ffbff9     	bl	0x9480
   194a0: 91044294     	add	x20, x20, #272
   194a4: aa0003f3     	mov	x19, x0
   194a8: 910ac280     	add	x0, x20, #688
   194ac: 97ffc04d     	bl	0x95e0
   194b0: aa1303e0     	mov	x0, x19
   194b4: 97ffc077     	bl	0x9690
