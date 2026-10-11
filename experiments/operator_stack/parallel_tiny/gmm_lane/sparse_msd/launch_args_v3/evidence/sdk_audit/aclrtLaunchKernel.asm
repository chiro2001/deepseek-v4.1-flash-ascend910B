
/usr/local/Ascend/cann-9.1.0/lib64/libascendcl.so:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000011868 <aclrtLaunchKernel>:
   11868: a9ba7bfd     	stp	x29, x30, [sp, #-96]!
   1186c: 910003fd     	mov	x29, sp
   11870: a90153f3     	stp	x19, x20, [sp, #16]
   11874: f00001f4     	adrp	x20, 0x50000
   11878: 91004293     	add	x19, x20, #16
   1187c: a9025bf5     	stp	x21, x22, [sp, #32]
   11880: a90363f7     	stp	x23, x24, [sp, #48]
   11884: aa0003f5     	mov	x21, x0
   11888: f90023f9     	str	x25, [sp, #64]
   1188c: 2a0103f6     	mov	w22, w1
   11890: aa0203f7     	mov	x23, x2
   11894: aa0303f8     	mov	x24, x3
   11898: 91280260     	add	x0, x19, #2560
   1189c: aa0403f9     	mov	x25, x4
   118a0: 08dffc01     	ldarb	w1, [x0]
   118a4: 360001e1     	tbz	w1, #0, 0x118e0
   118a8: f9450665     	ldr	x5, [x19, #2568]
   118ac: b4000365     	cbz	x5, 0x11918
   118b0: a94153f3     	ldp	x19, x20, [sp, #16]
   118b4: aa1903e4     	mov	x4, x25
   118b8: aa1803e3     	mov	x3, x24
   118bc: f94023f9     	ldr	x25, [sp, #64]
   118c0: aa1703e2     	mov	x2, x23
   118c4: 2a1603e1     	mov	w1, w22
   118c8: a94363f7     	ldp	x23, x24, [sp, #48]
   118cc: aa1503e0     	mov	x0, x21
   118d0: a9425bf5     	ldp	x21, x22, [sp, #32]
   118d4: a8c67bfd     	ldp	x29, x30, [sp], #96
   118d8: d61f00a0     	br	x5
   118dc: d503201f     	nop
   118e0: 97ffdeb8     	bl	0x93c0
   118e4: 34fffe20     	cbz	w0, 0x118a8
   118e8: 97ffe0f2     	bl	0x9cb0
   118ec: 91004294     	add	x20, x20, #16
   118f0: b00000e1     	adrp	x1, 0x2e000
   118f4: 91316021     	add	x1, x1, #3160
   118f8: 97ffded6     	bl	0x9450
   118fc: aa0003e5     	mov	x5, x0
   11900: 91280280     	add	x0, x20, #2560
   11904: f9050685     	str	x5, [x20, #2568]
   11908: f9002fa5     	str	x5, [x29, #88]
   1190c: 97ffdee9     	bl	0x94b0
   11910: f9402fa5     	ldr	x5, [x29, #88]
   11914: 17ffffe6     	b	0x118ac
   11918: b00000c1     	adrp	x1, 0x2a000
   1191c: 910d0021     	add	x1, x1, #832
   11920: 913ec021     	add	x1, x1, #4016
   11924: 900000e0     	adrp	x0, 0x2d000
   11928: 91328000     	add	x0, x0, #3232
   1192c: 97ffdf4d     	bl	0x9660
   11930: 12800000     	mov	w0, #-1
   11934: 97ffded3     	bl	0x9480
   11938: 91004294     	add	x20, x20, #16
   1193c: aa0003f3     	mov	x19, x0
   11940: 91280280     	add	x0, x20, #2560
   11944: 97ffdf27     	bl	0x95e0
   11948: aa1303e0     	mov	x0, x19
   1194c: 97ffdf51     	bl	0x9690
