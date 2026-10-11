
/usr/local/Ascend/cann-9.1.0/lib64/libacl_rt_impl.so:	file format elf64-littleaarch64

Disassembly of section .text:

00000000000826d0 <aclrtLaunchKernelImpl>:
   826d0: a9b37bfd     	stp	x29, x30, [sp, #-208]!
   826d4: 910003fd     	mov	x29, sp
   826d8: a90153f3     	stp	x19, x20, [sp, #16]
   826dc: b00004f3     	adrp	x19, 0x11f000
   826e0: a9025bf5     	stp	x21, x22, [sp, #32]
   826e4: aa0003f4     	mov	x20, x0
   826e8: f9469665     	ldr	x5, [x19, #3368]
   826ec: 910163b5     	add	x21, x29, #88
   826f0: a90363f7     	stp	x23, x24, [sp, #48]
   826f4: 2a0103f7     	mov	w23, w1
   826f8: f94000a0     	ldr	x0, [x5]
   826fc: f90067a0     	str	x0, [x29, #200]
   82700: d2800000     	mov	x0, #0
   82704: 52800b01     	mov	w1, #88
   82708: f90023f9     	str	x25, [sp, #64]
   8270c: aa0203f6     	mov	x22, x2
   82710: aa0403f8     	mov	x24, x4
   82714: 72a00061     	movk	w1, #3, lsl #16
   82718: aa1503e0     	mov	x0, x21
   8271c: aa0303f9     	mov	x25, x3
   82720: 97feba36     	bl	0x30ff8
   82724: b4001174     	cbz	x20, 0x82950
   82728: b4000456     	cbz	x22, 0x827b0
   8272c: 52800020     	mov	w0, #1
   82730: a90affbf     	stp	xzr, xzr, [x29, #168]
   82734: a90bffbf     	stp	xzr, xzr, [x29, #184]
   82738: d2800004     	mov	x4, #0
   8273c: 3902ffa0     	strb	w0, [x29, #191]
   82740: aa1803e3     	mov	x3, x24
   82744: 910283a2     	add	x2, x29, #160
   82748: 2a1703e1     	mov	w1, w23
   8274c: aa1403e0     	mov	x0, x20
   82750: f90053b6     	str	x22, [x29, #160]
   82754: b900b3b9     	str	w25, [x29, #176]
   82758: 97fe9e02     	bl	0x29f60
   8275c: 2a0003f4     	mov	w20, w0
   82760: 340000a0     	cbz	w0, 0x82774
   82764: 52944120     	mov	w0, #41481
   82768: 72a00020     	movk	w0, #1, lsl #16
   8276c: 6b00029f     	cmp	w20, w0
   82770: 54000b00     	b.eq	0x828d0
   82774: aa1503e0     	mov	x0, x21
   82778: 97feb9d0     	bl	0x30eb8
   8277c: f9469673     	ldr	x19, [x19, #3368]
   82780: 2a1403e0     	mov	w0, w20
   82784: f94067a2     	ldr	x2, [x29, #200]
   82788: f9400261     	ldr	x1, [x19]
   8278c: ca010041     	eor	x1, x2, x1
   82790: b5001941     	cbnz	x1, 0x82ab8
   82794: a94153f3     	ldp	x19, x20, [sp, #16]
   82798: a9425bf5     	ldp	x21, x22, [sp, #32]
   8279c: a94363f7     	ldp	x23, x24, [sp, #48]
   827a0: f94023f9     	ldr	x25, [sp, #64]
   827a4: a8cd7bfd     	ldp	x29, x30, [sp], #208
   827a8: d65f03c0     	ret
   827ac: d503201f     	nop
   827b0: 900002c0     	adrp	x0, 0xda000
   827b4: 900002d4     	adrp	x20, 0xda000
   827b8: 91032800     	add	x0, x0, #202
   827bc: 91032294     	add	x20, x20, #200
   827c0: 14000005     	b	0x827d4
   827c4: d503201f     	nop
   827c8: 7100bc3f     	cmp	w1, #47
   827cc: 9a801294     	csel	x20, x20, x0, ne
   827d0: 91000400     	add	x0, x0, #1
   827d4: 385ff001     	ldurb	w1, [x0, #-1]
   827d8: 35ffff81     	cbnz	w1, 0x827c8
   827dc: 97feb5a5     	bl	0x2fe70
   827e0: b0000336     	adrp	x22, 0xe7000
   827e4: 912742c6     	add	x6, x22, #2512
   827e8: 900002d7     	adrp	x23, 0xda000
   827ec: 90000262     	adrp	x2, 0xce000
   827f0: 2a0003e5     	mov	w5, w0
   827f4: 911cc2e7     	add	x7, x23, #1840
   827f8: 910f00c6     	add	x6, x6, #960
   827fc: 52800f44     	mov	w4, #122
   82800: aa1403e3     	mov	x3, x20
   82804: 913ee042     	add	x2, x2, #4024
   82808: 52800061     	mov	w1, #3
   8280c: 52800600     	mov	w0, #48
   82810: 97fea010     	bl	0x2a850
   82814: 912742c0     	add	x0, x22, #2512
   82818: 910143a8     	add	x8, x29, #80
   8281c: 910f6000     	add	x0, x0, #984
   82820: 97feb75c     	bl	0x30590
   82824: d2800200     	mov	x0, #16
   82828: a9077fbf     	stp	xzr, xzr, [x29, #112]
   8282c: f90043bf     	str	xzr, [x29, #128]
   82830: 97fea40c     	bl	0x2b860
   82834: f00004c1     	adrp	x1, 0x11d000
   82838: 910a2021     	add	x1, x1, #648
   8283c: 91080021     	add	x1, x1, #512
   82840: aa0003e2     	mov	x2, x0
   82844: d2800200     	mov	x0, #16
   82848: 8b000043     	add	x3, x2, x0
   8284c: 911cc2f7     	add	x23, x23, #1840
   82850: f9402bb4     	ldr	x20, [x29, #80]
   82854: a9441424     	ldp	x4, x5, [x1, #64]
   82858: a9001444     	stp	x4, x5, [x2]
   8285c: a9070fa2     	stp	x2, x3, [x29, #112]
   82860: a9087fa3     	stp	x3, xzr, [x29, #128]
   82864: a9097fbf     	stp	xzr, xzr, [x29, #144]
   82868: 97fea3fe     	bl	0x2b860
   8286c: 90000263     	adrp	x3, 0xce000
   82870: a9005c14     	stp	x20, x23, [x0]
   82874: 91004001     	add	x1, x0, #16
   82878: 910223a2     	add	x2, x29, #136
   8287c: a90887a0     	stp	x0, x1, [x29, #136]
   82880: 913fa060     	add	x0, x3, #4072
   82884: f9004fa1     	str	x1, [x29, #152]
   82888: 9101c3a1     	add	x1, x29, #112
   8288c: 97feb5a7     	bl	0x2ff28
   82890: f94047a0     	ldr	x0, [x29, #136]
   82894: b4000040     	cbz	x0, 0x8289c
   82898: 97fe9dc2     	bl	0x29fa0
   8289c: f9403ba0     	ldr	x0, [x29, #112]
   828a0: b4000040     	cbz	x0, 0x828a8
   828a4: 97fe9dbf     	bl	0x29fa0
   828a8: b00004e0     	adrp	x0, 0x11f000
   828ac: f9402bb4     	ldr	x20, [x29, #80]
   828b0: f947c400     	ldr	x0, [x0, #3976]
   828b4: d1006296     	sub	x22, x20, #24
   828b8: eb0002df     	cmp	x22, x0
   828bc: 54000de1     	b.ne	0x82a78
   828c0: 5290d414     	mov	w20, #34464
   828c4: 72a00034     	movk	w20, #1, lsl #16
   828c8: 17ffffab     	b	0x82774
   828cc: d503201f     	nop
   828d0: 52800041     	mov	w1, #2
   828d4: 52800600     	mov	w0, #48
   828d8: 97fe9eda     	bl	0x2a440
   828dc: 7100041f     	cmp	w0, #1
   828e0: 54000321     	b.ne	0x82944
   828e4: 900002c0     	adrp	x0, 0xda000
   828e8: 900002d4     	adrp	x20, 0xda000
   828ec: 91032800     	add	x0, x0, #202
   828f0: 91032294     	add	x20, x20, #200
   828f4: 14000004     	b	0x82904
   828f8: 7100bc3f     	cmp	w1, #47
   828fc: 9a801294     	csel	x20, x20, x0, ne
   82900: 91000400     	add	x0, x0, #1
   82904: 385ff001     	ldurb	w1, [x0, #-1]
   82908: 35ffff81     	cbnz	w1, 0x828f8
   8290c: 97feb559     	bl	0x2fe70
   82910: b0000326     	adrp	x6, 0xe7000
   82914: 52944127     	mov	w7, #41481
   82918: 912740c6     	add	x6, x6, #2512
   8291c: 900002c2     	adrp	x2, 0xda000
   82920: 2a0003e5     	mov	w5, w0
   82924: 72a00027     	movk	w7, #1, lsl #16
   82928: 910f00c6     	add	x6, x6, #960
   8292c: 52801084     	mov	w4, #132
   82930: aa1403e3     	mov	x3, x20
   82934: 911d0042     	add	x2, x2, #1856
   82938: 52800041     	mov	w1, #2
   8293c: 52800600     	mov	w0, #48
   82940: 97fe9fc4     	bl	0x2a850
   82944: 52944134     	mov	w20, #41481
   82948: 72a00034     	movk	w20, #1, lsl #16
   8294c: 17ffff8a     	b	0x82774
   82950: 900002c0     	adrp	x0, 0xda000
   82954: 900002d4     	adrp	x20, 0xda000
   82958: 91032800     	add	x0, x0, #202
   8295c: 91032294     	add	x20, x20, #200
   82960: 14000005     	b	0x82974
   82964: d503201f     	nop
   82968: 7100bc3f     	cmp	w1, #47
   8296c: 9a801294     	csel	x20, x20, x0, ne
   82970: 91000400     	add	x0, x0, #1
   82974: 385ff001     	ldurb	w1, [x0, #-1]
   82978: 35ffff81     	cbnz	w1, 0x82968
   8297c: 97feb53d     	bl	0x2fe70
   82980: b0000336     	adrp	x22, 0xe7000
   82984: 912742c6     	add	x6, x22, #2512
   82988: 900002d7     	adrp	x23, 0xda000
   8298c: 90000262     	adrp	x2, 0xce000
   82990: 2a0003e5     	mov	w5, w0
   82994: 910722e7     	add	x7, x23, #456
   82998: 910f00c6     	add	x6, x6, #960
   8299c: 52800f24     	mov	w4, #121
   829a0: aa1403e3     	mov	x3, x20
   829a4: 913ee042     	add	x2, x2, #4024
   829a8: 52800061     	mov	w1, #3
   829ac: 52800600     	mov	w0, #48
   829b0: 97fe9fa8     	bl	0x2a850
   829b4: 912742c0     	add	x0, x22, #2512
   829b8: 910143a8     	add	x8, x29, #80
   829bc: 910f6000     	add	x0, x0, #984
   829c0: 97feb6f4     	bl	0x30590
   829c4: f00004c1     	adrp	x1, 0x11d000
   829c8: 9101c3b4     	add	x20, x29, #112
   829cc: 910a2021     	add	x1, x1, #648
   829d0: d2800042     	mov	x2, #2
   829d4: 9108c021     	add	x1, x1, #560
   829d8: aa1403e0     	mov	x0, x20
   829dc: 97ffefb7     	bl	0x7e8b8
   829e0: f9402ba0     	ldr	x0, [x29, #80]
   829e4: 910223b6     	add	x22, x29, #136
   829e8: 910722f7     	add	x23, x23, #456
   829ec: 910283a1     	add	x1, x29, #160
   829f0: a90a5fa0     	stp	x0, x23, [x29, #160]
   829f4: d2800042     	mov	x2, #2
   829f8: aa1603e0     	mov	x0, x22
   829fc: 97ffefaf     	bl	0x7e8b8
   82a00: 90000260     	adrp	x0, 0xce000
   82a04: aa1603e2     	mov	x2, x22
   82a08: aa1403e1     	mov	x1, x20
   82a0c: 913fa000     	add	x0, x0, #4072
   82a10: 97feb546     	bl	0x2ff28
   82a14: f94047a0     	ldr	x0, [x29, #136]
   82a18: b4000040     	cbz	x0, 0x82a20
   82a1c: 97fe9d61     	bl	0x29fa0
   82a20: f9403ba0     	ldr	x0, [x29, #112]
   82a24: b4000040     	cbz	x0, 0x82a2c
   82a28: 97fe9d5e     	bl	0x29fa0
   82a2c: b00004e0     	adrp	x0, 0x11f000
   82a30: f9402bb4     	ldr	x20, [x29, #80]
   82a34: f947c400     	ldr	x0, [x0, #3976]
   82a38: d1006296     	sub	x22, x20, #24
   82a3c: eb0002df     	cmp	x22, x0
   82a40: 54fff400     	b.eq	0x828c0
   82a44: b00004e0     	adrp	x0, 0x11f000
   82a48: f9479800     	ldr	x0, [x0, #3888]
   82a4c: b40002a0     	cbz	x0, 0x82aa0
   82a50: d1002294     	sub	x20, x20, #8
   82a54: 885ffe97     	ldaxr	w23, [x20]
   82a58: 510006e0     	sub	w0, w23, #1
   82a5c: 8801fe80     	stlxr	w1, w0, [x20]
   82a60: 35ffffa1     	cbnz	w1, 0x82a54
   82a64: 710002ff     	cmp	w23, #0
   82a68: 54fff2cc     	b.gt	0x828c0
   82a6c: aa1603e0     	mov	x0, x22
   82a70: 97fe9d4c     	bl	0x29fa0
   82a74: 17ffff93     	b	0x828c0
   82a78: b00004e0     	adrp	x0, 0x11f000
   82a7c: f9479800     	ldr	x0, [x0, #3888]
   82a80: b4000100     	cbz	x0, 0x82aa0
   82a84: d1002294     	sub	x20, x20, #8
   82a88: 885ffe97     	ldaxr	w23, [x20]
   82a8c: 510006e0     	sub	w0, w23, #1
   82a90: 8801fe80     	stlxr	w1, w0, [x20]
   82a94: 34fffe81     	cbz	w1, 0x82a64
   82a98: 17fffffc     	b	0x82a88
   82a9c: d503201f     	nop
   82aa0: b85f8297     	ldur	w23, [x20, #-8]
   82aa4: 12800001     	mov	w1, #-1
   82aa8: 2a1703e0     	mov	w0, w23
   82aac: 97fea15d     	bl	0x2b020
   82ab0: b81f8280     	stur	w0, [x20, #-8]
   82ab4: 17ffffec     	b	0x82a64
   82ab8: 97fea00a     	bl	0x2aae0
   82abc: f94047a1     	ldr	x1, [x29, #136]
   82ac0: aa0003f3     	mov	x19, x0
   82ac4: b4000061     	cbz	x1, 0x82ad0
   82ac8: aa0103e0     	mov	x0, x1
   82acc: 97fe9d35     	bl	0x29fa0
   82ad0: f9403ba0     	ldr	x0, [x29, #112]
   82ad4: b4000040     	cbz	x0, 0x82adc
   82ad8: 97fe9d32     	bl	0x29fa0
   82adc: f9402ba0     	ldr	x0, [x29, #80]
   82ae0: d1006000     	sub	x0, x0, #24
   82ae4: 97ffef43     	bl	0x7e7f0
   82ae8: aa1503e0     	mov	x0, x21
   82aec: 97feb8f3     	bl	0x30eb8
   82af0: aa1303e0     	mov	x0, x19
   82af4: 97fea35f     	bl	0x2b870
   82af8: aa0003f3     	mov	x19, x0
   82afc: 17fffffb     	b	0x82ae8
   82b00: f9403ba1     	ldr	x1, [x29, #112]
   82b04: aa0003f3     	mov	x19, x0
   82b08: b4fffea1     	cbz	x1, 0x82adc
   82b0c: aa0103e0     	mov	x0, x1
   82b10: 97fe9d24     	bl	0x29fa0
   82b14: 17fffff2     	b	0x82adc
   82b18: 17ffffe9     	b	0x82abc
   82b1c: 17ffffe8     	b	0x82abc
   82b20: aa0003f3     	mov	x19, x0
   82b24: 17ffffeb     	b	0x82ad0
   82b28: aa0003f3     	mov	x19, x0
   82b2c: 17ffffec     	b	0x82adc
