
/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000013370 <aclrtLaunchKernelWithConfig>:
   13370: a9ba7bfd     	stp	x29, x30, [sp, #-96]!
   13374: 910003fd     	mov	x29, sp
   13378: a90153f3     	stp	x19, x20, [sp, #16]
   1337c: b00001f4     	adrp	x20, 0x50000
   13380: 91004293     	add	x19, x20, #16
   13384: a9025bf5     	stp	x21, x22, [sp, #32]
   13388: a90363f7     	stp	x23, x24, [sp, #48]
   1338c: aa0003f5     	mov	x21, x0
   13390: a9046bf9     	stp	x25, x26, [sp, #64]
   13394: 2a0103f6     	mov	w22, w1
   13398: aa0203f7     	mov	x23, x2
   1339c: aa0303f8     	mov	x24, x3
   133a0: aa0403f9     	mov	x25, x4
   133a4: aa0503fa     	mov	x26, x5
   133a8: 91304260     	add	x0, x19, #3088
   133ac: 08dffc01     	ldarb	w1, [x0]
   133b0: 36000201     	tbz	w1, #0, 0x133f0
   133b4: f9460e66     	ldr	x6, [x19, #3096]
   133b8: b4000386     	cbz	x6, 0x13428
   133bc: a94153f3     	ldp	x19, x20, [sp, #16]
   133c0: aa1a03e5     	mov	x5, x26
   133c4: aa1903e4     	mov	x4, x25
   133c8: aa1803e3     	mov	x3, x24
   133cc: a9446bf9     	ldp	x25, x26, [sp, #64]
   133d0: aa1703e2     	mov	x2, x23
   133d4: 2a1603e1     	mov	w1, w22
   133d8: aa1503e0     	mov	x0, x21
   133dc: a94363f7     	ldp	x23, x24, [sp, #48]
   133e0: a9425bf5     	ldp	x21, x22, [sp, #32]
   133e4: a8c67bfd     	ldp	x29, x30, [sp], #96
   133e8: d61f00c0     	br	x6
   133ec: d503201f     	nop
   133f0: 97ffd7f4     	bl	0x93c0
   133f4: 34fffe00     	cbz	w0, 0x133b4
   133f8: 97ffda2e     	bl	0x9cb0
   133fc: 91004294     	add	x20, x20, #16
   13400: 900000e1     	adrp	x1, 0x2f000
   13404: 9100a021     	add	x1, x1, #40
   13408: 97ffd812     	bl	0x9450
   1340c: aa0003e6     	mov	x6, x0
   13410: 91304280     	add	x0, x20, #3088
   13414: f9060e86     	str	x6, [x20, #3096]
   13418: f9002fa6     	str	x6, [x29, #88]
   1341c: 97ffd825     	bl	0x94b0
   13420: f9402fa6     	ldr	x6, [x29, #88]
   13424: 17ffffe5     	b	0x133b8
   13428: 900000c1     	adrp	x1, 0x2b000
   1342c: 91110021     	add	x1, x1, #1088
   13430: 910a0021     	add	x1, x1, #640
   13434: d00000c0     	adrp	x0, 0x2d000
   13438: 91328000     	add	x0, x0, #3232
   1343c: 97ffd889     	bl	0x9660
   13440: 12800000     	mov	w0, #-1
   13444: 97ffd80f     	bl	0x9480
   13448: 91004294     	add	x20, x20, #16
   1344c: aa0003f3     	mov	x19, x0
   13450: 91304280     	add	x0, x20, #3088
   13454: 97ffd863     	bl	0x95e0
   13458: aa1303e0     	mov	x0, x19
   1345c: 97ffd88d     	bl	0x9690
