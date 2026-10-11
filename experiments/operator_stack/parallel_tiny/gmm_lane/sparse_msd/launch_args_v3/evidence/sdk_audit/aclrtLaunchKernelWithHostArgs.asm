
/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so:	file format elf64-littleaarch64

Disassembly of section .text:

00000000000194b8 <aclrtLaunchKernelWithHostArgs>:
   194b8: a9b97bfd     	stp	x29, x30, [sp, #-112]!
   194bc: 910003fd     	mov	x29, sp
   194c0: a90153f3     	stp	x19, x20, [sp, #16]
   194c4: 900001d3     	adrp	x19, 0x51000
   194c8: a90573fb     	stp	x27, x28, [sp, #80]
   194cc: 9104427c     	add	x28, x19, #272
   194d0: a9025bf5     	stp	x21, x22, [sp, #32]
   194d4: aa0003f4     	mov	x20, x0
   194d8: a90363f7     	stp	x23, x24, [sp, #48]
   194dc: 2a0103f5     	mov	w21, w1
   194e0: a9046bf9     	stp	x25, x26, [sp, #64]
   194e4: aa0203f6     	mov	x22, x2
   194e8: aa0303f7     	mov	x23, x3
   194ec: aa0403f8     	mov	x24, x4
   194f0: aa0503f9     	mov	x25, x5
   194f4: aa0603fa     	mov	x26, x6
   194f8: aa0703fb     	mov	x27, x7
   194fc: 910b0380     	add	x0, x28, #704
   19500: 08dffc01     	ldarb	w1, [x0]
   19504: 36000261     	tbz	w1, #0, 0x19550
   19508: f9416788     	ldr	x8, [x28, #712]
   1950c: b40003e8     	cbz	x8, 0x19588
   19510: aa1b03e7     	mov	x7, x27
   19514: aa1a03e6     	mov	x6, x26
   19518: a94573fb     	ldp	x27, x28, [sp, #80]
   1951c: aa1903e5     	mov	x5, x25
   19520: aa1803e4     	mov	x4, x24
   19524: aa1703e3     	mov	x3, x23
   19528: a9446bf9     	ldp	x25, x26, [sp, #64]
   1952c: aa1603e2     	mov	x2, x22
   19530: a94363f7     	ldp	x23, x24, [sp, #48]
   19534: 2a1503e1     	mov	w1, w21
   19538: aa1403e0     	mov	x0, x20
   1953c: a9425bf5     	ldp	x21, x22, [sp, #32]
   19540: a94153f3     	ldp	x19, x20, [sp, #16]
   19544: a8c77bfd     	ldp	x29, x30, [sp], #112
   19548: d61f0100     	br	x8
   1954c: d503201f     	nop
   19550: 97ffbf9c     	bl	0x93c0
   19554: 34fffda0     	cbz	w0, 0x19508
   19558: 97ffc1d6     	bl	0x9cb0
   1955c: 91044273     	add	x19, x19, #272
   19560: d00000a1     	adrp	x1, 0x2f000
   19564: 91318021     	add	x1, x1, #3168
   19568: 97ffbfba     	bl	0x9450
   1956c: aa0003e8     	mov	x8, x0
   19570: 910b0260     	add	x0, x19, #704
   19574: f9016668     	str	x8, [x19, #712]
   19578: f90037a8     	str	x8, [x29, #104]
   1957c: 97ffbfcd     	bl	0x94b0
   19580: f94037a8     	ldr	x8, [x29, #104]
   19584: 17ffffe2     	b	0x1950c
   19588: d0000081     	adrp	x1, 0x2b000
   1958c: 91110021     	add	x1, x1, #1088
   19590: 913ae021     	add	x1, x1, #3768
   19594: 900000a0     	adrp	x0, 0x2d000
   19598: 91328000     	add	x0, x0, #3232
   1959c: 97ffc031     	bl	0x9660
   195a0: 12800000     	mov	w0, #-1
   195a4: 97ffbfb7     	bl	0x9480
   195a8: 91044273     	add	x19, x19, #272
   195ac: aa0003f4     	mov	x20, x0
   195b0: 910b0260     	add	x0, x19, #704
   195b4: 97ffc00b     	bl	0x95e0
   195b8: aa1403e0     	mov	x0, x20
   195bc: 97ffc035     	bl	0x9690
