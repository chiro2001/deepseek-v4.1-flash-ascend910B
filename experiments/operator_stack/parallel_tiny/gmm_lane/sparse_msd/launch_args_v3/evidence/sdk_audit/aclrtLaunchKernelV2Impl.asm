
/usr/local/Ascend/cann-9.1.0/lib64/libacl_rt_impl.so:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000087750 <aclrtLaunchKernelV2Impl>:
   87750: a9b37bfd     	stp	x29, x30, [sp, #-208]!
   87754: 910003fd     	mov	x29, sp
   87758: a90153f3     	stp	x19, x20, [sp, #16]
   8775c: 900004d3     	adrp	x19, 0x11f000
   87760: a9025bf5     	stp	x21, x22, [sp, #32]
   87764: aa0203f6     	mov	x22, x2
   87768: f9469666     	ldr	x6, [x19, #3368]
   8776c: 9101c3b4     	add	x20, x29, #112
   87770: a90363f7     	stp	x23, x24, [sp, #48]
   87774: 2a0103f7     	mov	w23, w1
   87778: f94000c2     	ldr	x2, [x6]
   8777c: f90067a2     	str	x2, [x29, #200]
   87780: d2800002     	mov	x2, #0
   87784: 528017e1     	mov	w1, #191
   87788: a9046bf9     	stp	x25, x26, [sp, #64]
   8778c: 72a00061     	movk	w1, #3, lsl #16
   87790: aa0003f5     	mov	x21, x0
   87794: aa0303fa     	mov	x26, x3
   87798: aa1403e0     	mov	x0, x20
   8779c: aa0403f9     	mov	x25, x4
   877a0: aa0503f8     	mov	x24, x5
   877a4: f9002bfb     	str	x27, [sp, #80]
   877a8: 97fea614     	bl	0x30ff8
   877ac: 52800021     	mov	w1, #1
   877b0: 52800600     	mov	w0, #48
   877b4: 97fe8b23     	bl	0x2a440
   877b8: 7100041f     	cmp	w0, #1
   877bc: 54000420     	b.eq	0x87840
   877c0: b4000715     	cbz	x21, 0x878a0
   877c4: b40012f6     	cbz	x22, 0x87a20
   877c8: d2800006     	mov	x6, #0
   877cc: 2a1a03e5     	mov	w5, w26
   877d0: aa1603e4     	mov	x4, x22
   877d4: aa1903e3     	mov	x3, x25
   877d8: aa1803e2     	mov	x2, x24
   877dc: 2a1703e1     	mov	w1, w23
   877e0: aa1503e0     	mov	x0, x21
   877e4: 97fe8f73     	bl	0x2b5b0
   877e8: 2a0003f5     	mov	w21, w0
   877ec: 340000a0     	cbz	w0, 0x87800
   877f0: 52944120     	mov	w0, #41481
   877f4: 72a00020     	movk	w0, #1, lsl #16
   877f8: 6b0002bf     	cmp	w21, w0
   877fc: 54000d20     	b.eq	0x879a0
   87800: aa1403e0     	mov	x0, x20
   87804: 97fea5ad     	bl	0x30eb8
   87808: f9469673     	ldr	x19, [x19, #3368]
   8780c: 2a1503e0     	mov	w0, w21
   87810: f94067a2     	ldr	x2, [x29, #200]
   87814: f9400261     	ldr	x1, [x19]
   87818: ca010041     	eor	x1, x2, x1
   8781c: b5001b61     	cbnz	x1, 0x87b88
   87820: a94153f3     	ldp	x19, x20, [sp, #16]
   87824: a9425bf5     	ldp	x21, x22, [sp, #32]
   87828: a94363f7     	ldp	x23, x24, [sp, #48]
   8782c: a9446bf9     	ldp	x25, x26, [sp, #64]
   87830: f9402bfb     	ldr	x27, [sp, #80]
   87834: a8cd7bfd     	ldp	x29, x30, [sp], #208
   87838: d65f03c0     	ret
   8783c: d503201f     	nop
   87840: f0000280     	adrp	x0, 0xda000
   87844: f000029b     	adrp	x27, 0xda000
   87848: 91032800     	add	x0, x0, #202
   8784c: 9103237b     	add	x27, x27, #200
   87850: 14000005     	b	0x87864
   87854: d503201f     	nop
   87858: 7100bc3f     	cmp	w1, #47
   8785c: 9a80137b     	csel	x27, x27, x0, ne
   87860: 91000400     	add	x0, x0, #1
   87864: 385ff001     	ldurb	w1, [x0, #-1]
   87868: 35ffff81     	cbnz	w1, 0x87858
   8786c: 97fea181     	bl	0x2fe70
   87870: 90000306     	adrp	x6, 0xe7000
   87874: 912740c6     	add	x6, x6, #2512
   87878: f0000282     	adrp	x2, 0xda000
   8787c: 2a0003e5     	mov	w5, w0
   87880: 912180c6     	add	x6, x6, #2144
   87884: 528039a4     	mov	w4, #461
   87888: aa1b03e3     	mov	x3, x27
   8788c: 9135a042     	add	x2, x2, #3432
   87890: 52800021     	mov	w1, #1
   87894: 52800600     	mov	w0, #48
   87898: 97fe8bee     	bl	0x2a850
   8789c: b5fff955     	cbnz	x21, 0x877c4
   878a0: f0000280     	adrp	x0, 0xda000
   878a4: f0000295     	adrp	x21, 0xda000
   878a8: 91032800     	add	x0, x0, #202
   878ac: 910322b5     	add	x21, x21, #200
   878b0: 14000005     	b	0x878c4
   878b4: d503201f     	nop
   878b8: 7100bc3f     	cmp	w1, #47
   878bc: 9a8012b5     	csel	x21, x21, x0, ne
   878c0: 91000400     	add	x0, x0, #1
   878c4: 385ff001     	ldurb	w1, [x0, #-1]
   878c8: 35ffff81     	cbnz	w1, 0x878b8
   878cc: 97fea169     	bl	0x2fe70
   878d0: 90000316     	adrp	x22, 0xe7000
   878d4: 912742c6     	add	x6, x22, #2512
   878d8: f0000297     	adrp	x23, 0xda000
   878dc: f0000222     	adrp	x2, 0xce000
   878e0: 2a0003e5     	mov	w5, w0
   878e4: 910722e7     	add	x7, x23, #456
   878e8: 912180c6     	add	x6, x6, #2144
   878ec: 528039c4     	mov	w4, #462
   878f0: aa1503e3     	mov	x3, x21
   878f4: 913ee042     	add	x2, x2, #4024
   878f8: 52800061     	mov	w1, #3
   878fc: 52800600     	mov	w0, #48
   87900: 97fe8bd4     	bl	0x2a850
   87904: 912742c0     	add	x0, x22, #2512
   87908: 9101a3a8     	add	x8, x29, #104
   8790c: 9121e000     	add	x0, x0, #2168
   87910: 97fea320     	bl	0x30590
   87914: d00004a1     	adrp	x1, 0x11d000
   87918: 910223b5     	add	x21, x29, #136
   8791c: 910a2021     	add	x1, x1, #648
   87920: d2800042     	mov	x2, #2
   87924: 91140021     	add	x1, x1, #1280
   87928: aa1503e0     	mov	x0, x21
   8792c: 97ffdbe3     	bl	0x7e8b8
   87930: 910722e0     	add	x0, x23, #456
   87934: f94037a1     	ldr	x1, [x29, #104]
   87938: 910283b6     	add	x22, x29, #160
   8793c: a90b83a1     	stp	x1, x0, [x29, #184]
   87940: d2800042     	mov	x2, #2
   87944: 9102e3a1     	add	x1, x29, #184
   87948: aa1603e0     	mov	x0, x22
   8794c: 97ffdbdb     	bl	0x7e8b8
   87950: f0000220     	adrp	x0, 0xce000
   87954: aa1603e2     	mov	x2, x22
   87958: aa1503e1     	mov	x1, x21
   8795c: 913fa000     	add	x0, x0, #4072
   87960: 97fea172     	bl	0x2ff28
   87964: f94053a0     	ldr	x0, [x29, #160]
   87968: b4000040     	cbz	x0, 0x87970
   8796c: 97fe898d     	bl	0x29fa0
   87970: f94047a0     	ldr	x0, [x29, #136]
   87974: b4000040     	cbz	x0, 0x8797c
   87978: 97fe898a     	bl	0x29fa0
   8797c: 900004c0     	adrp	x0, 0x11f000
   87980: f94037b5     	ldr	x21, [x29, #104]
   87984: f947c400     	ldr	x0, [x0, #3976]
   87988: d10062b6     	sub	x22, x21, #24
   8798c: eb0002df     	cmp	x22, x0
   87990: 54000dc1     	b.ne	0x87b48
   87994: 5290d415     	mov	w21, #34464
   87998: 72a00035     	movk	w21, #1, lsl #16
   8799c: 17ffff99     	b	0x87800
   879a0: 52800041     	mov	w1, #2
   879a4: 52800600     	mov	w0, #48
   879a8: 97fe8aa6     	bl	0x2a440
   879ac: 7100041f     	cmp	w0, #1
   879b0: 54000321     	b.ne	0x87a14
   879b4: f0000280     	adrp	x0, 0xda000
   879b8: f0000295     	adrp	x21, 0xda000
   879bc: 91032800     	add	x0, x0, #202
   879c0: 910322b5     	add	x21, x21, #200
   879c4: 14000004     	b	0x879d4
   879c8: 7100bc3f     	cmp	w1, #47
   879cc: 9a8012b5     	csel	x21, x21, x0, ne
   879d0: 91000400     	add	x0, x0, #1
   879d4: 385ff001     	ldurb	w1, [x0, #-1]
   879d8: 35ffff81     	cbnz	w1, 0x879c8
   879dc: 97fea125     	bl	0x2fe70
   879e0: 90000306     	adrp	x6, 0xe7000
   879e4: 52944127     	mov	w7, #41481
   879e8: 912740c6     	add	x6, x6, #2512
   879ec: f0000282     	adrp	x2, 0xda000
   879f0: 2a0003e5     	mov	w5, w0
   879f4: 72a00027     	movk	w7, #1, lsl #16
   879f8: 912180c6     	add	x6, x6, #2144
   879fc: 52803b44     	mov	w4, #474
   87a00: aa1503e3     	mov	x3, x21
   87a04: 91368042     	add	x2, x2, #3488
   87a08: 52800041     	mov	w1, #2
   87a0c: 52800600     	mov	w0, #48
   87a10: 97fe8b90     	bl	0x2a850
   87a14: 52944135     	mov	w21, #41481
   87a18: 72a00035     	movk	w21, #1, lsl #16
   87a1c: 17ffff79     	b	0x87800
   87a20: f0000280     	adrp	x0, 0xda000
   87a24: f0000295     	adrp	x21, 0xda000
   87a28: 91032800     	add	x0, x0, #202
   87a2c: 910322b5     	add	x21, x21, #200
   87a30: 14000005     	b	0x87a44
   87a34: d503201f     	nop
   87a38: 7100bc3f     	cmp	w1, #47
   87a3c: 9a8012b5     	csel	x21, x21, x0, ne
   87a40: 91000400     	add	x0, x0, #1
   87a44: 385ff001     	ldurb	w1, [x0, #-1]
   87a48: 35ffff81     	cbnz	w1, 0x87a38
   87a4c: 97fea109     	bl	0x2fe70
   87a50: 90000316     	adrp	x22, 0xe7000
   87a54: 912742c6     	add	x6, x22, #2512
   87a58: f0000297     	adrp	x23, 0xda000
   87a5c: f0000222     	adrp	x2, 0xce000
   87a60: 2a0003e5     	mov	w5, w0
   87a64: 911cc2e7     	add	x7, x23, #1840
   87a68: 912180c6     	add	x6, x6, #2144
   87a6c: 528039e4     	mov	w4, #463
   87a70: aa1503e3     	mov	x3, x21
   87a74: 913ee042     	add	x2, x2, #4024
   87a78: 52800061     	mov	w1, #3
   87a7c: 52800600     	mov	w0, #48
   87a80: 97fe8b74     	bl	0x2a850
   87a84: 912742c0     	add	x0, x22, #2512
   87a88: 9101a3a8     	add	x8, x29, #104
   87a8c: 9121e000     	add	x0, x0, #2168
   87a90: 97fea2c0     	bl	0x30590
   87a94: d00004a1     	adrp	x1, 0x11d000
   87a98: 910223b5     	add	x21, x29, #136
   87a9c: 910a2021     	add	x1, x1, #648
   87aa0: d2800042     	mov	x2, #2
   87aa4: 91144021     	add	x1, x1, #1296
   87aa8: aa1503e0     	mov	x0, x21
   87aac: 97ffdb83     	bl	0x7e8b8
   87ab0: 911cc2e0     	add	x0, x23, #1840
   87ab4: f94037a1     	ldr	x1, [x29, #104]
   87ab8: 910283b6     	add	x22, x29, #160
   87abc: a90b83a1     	stp	x1, x0, [x29, #184]
   87ac0: d2800042     	mov	x2, #2
   87ac4: 9102e3a1     	add	x1, x29, #184
   87ac8: aa1603e0     	mov	x0, x22
   87acc: 97ffdb7b     	bl	0x7e8b8
   87ad0: f0000220     	adrp	x0, 0xce000
   87ad4: aa1603e2     	mov	x2, x22
   87ad8: aa1503e1     	mov	x1, x21
   87adc: 913fa000     	add	x0, x0, #4072
   87ae0: 97fea112     	bl	0x2ff28
   87ae4: f94053a0     	ldr	x0, [x29, #160]
   87ae8: b4000040     	cbz	x0, 0x87af0
   87aec: 97fe892d     	bl	0x29fa0
   87af0: f94047a0     	ldr	x0, [x29, #136]
   87af4: b4000040     	cbz	x0, 0x87afc
   87af8: 97fe892a     	bl	0x29fa0
   87afc: 900004c0     	adrp	x0, 0x11f000
   87b00: f94037b5     	ldr	x21, [x29, #104]
   87b04: f947c400     	ldr	x0, [x0, #3976]
   87b08: d10062b6     	sub	x22, x21, #24
   87b0c: eb0002df     	cmp	x22, x0
   87b10: 54fff420     	b.eq	0x87994
   87b14: 900004c0     	adrp	x0, 0x11f000
   87b18: f9479800     	ldr	x0, [x0, #3888]
   87b1c: b40002a0     	cbz	x0, 0x87b70
   87b20: d10022b5     	sub	x21, x21, #8
   87b24: 885ffeb7     	ldaxr	w23, [x21]
   87b28: 510006e0     	sub	w0, w23, #1
   87b2c: 8801fea0     	stlxr	w1, w0, [x21]
   87b30: 35ffffa1     	cbnz	w1, 0x87b24
   87b34: 710002ff     	cmp	w23, #0
   87b38: 54fff2ec     	b.gt	0x87994
   87b3c: aa1603e0     	mov	x0, x22
   87b40: 97fe8918     	bl	0x29fa0
   87b44: 17ffff94     	b	0x87994
   87b48: 900004c0     	adrp	x0, 0x11f000
   87b4c: f9479800     	ldr	x0, [x0, #3888]
   87b50: b4000100     	cbz	x0, 0x87b70
   87b54: d10022b5     	sub	x21, x21, #8
   87b58: 885ffeb7     	ldaxr	w23, [x21]
   87b5c: 510006e0     	sub	w0, w23, #1
   87b60: 8801fea0     	stlxr	w1, w0, [x21]
   87b64: 34fffe81     	cbz	w1, 0x87b34
   87b68: 17fffffc     	b	0x87b58
   87b6c: d503201f     	nop
   87b70: b85f82b7     	ldur	w23, [x21, #-8]
   87b74: 12800001     	mov	w1, #-1
   87b78: 2a1703e0     	mov	w0, w23
   87b7c: 97fe8d29     	bl	0x2b020
   87b80: b81f82a0     	stur	w0, [x21, #-8]
   87b84: 17ffffec     	b	0x87b34
   87b88: 97fe8bd6     	bl	0x2aae0
   87b8c: aa0003f3     	mov	x19, x0
   87b90: aa1403e0     	mov	x0, x20
   87b94: 97fea4c9     	bl	0x30eb8
   87b98: aa1303e0     	mov	x0, x19
   87b9c: 97fe8f35     	bl	0x2b870
   87ba0: aa0003f3     	mov	x19, x0
   87ba4: f94047a0     	ldr	x0, [x29, #136]
   87ba8: b4000040     	cbz	x0, 0x87bb0
   87bac: 97fe88fd     	bl	0x29fa0
   87bb0: f94037a0     	ldr	x0, [x29, #104]
   87bb4: d1006000     	sub	x0, x0, #24
   87bb8: 97ffdb0e     	bl	0x7e7f0
   87bbc: 17fffff5     	b	0x87b90
   87bc0: aa0003f3     	mov	x19, x0
   87bc4: 17fffffb     	b	0x87bb0
   87bc8: f94053a1     	ldr	x1, [x29, #160]
   87bcc: aa0003f3     	mov	x19, x0
   87bd0: b4fffea1     	cbz	x1, 0x87ba4
   87bd4: aa0103e0     	mov	x0, x1
   87bd8: 97fe88f2     	bl	0x29fa0
   87bdc: 17fffff2     	b	0x87ba4
   87be0: 17fffffa     	b	0x87bc8
   87be4: 17ffffef     	b	0x87ba0
   87be8: 17fffff6     	b	0x87bc0
   87bec: d503201f     	nop
