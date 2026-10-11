
/usr/local/Ascend/cann-9.1.0/lib64/libruntime_v100.so:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000409760 <_ZNSt8_Rb_treeI19rtKernelDfxInfoTypeSt4pairIKS0_PFvS0_jjPKhmEESt10_Select1stIS7_ESt4lessIS0_ESaIS7_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorIS7_ERS2_>:
  42f554: d1006000     	sub	x0, x0, #24
  42f558: eb17001f     	cmp	x0, x23
  42f55c: 54000040     	b.eq	0x42f564
  42f560: 97fff8bc     	bl	0x42d850
  42f564: d1002294     	sub	x20, x20, #8
  42f568: 17fffff8     	b	0x42f548
  42f56c: aa0003f3     	mov	x19, x0
  42f570: 17fffff3     	b	0x42f53c
  42f574: d00006a0     	adrp	x0, 0x505000
  42f578: 91114000     	add	x0, x0, #1104
  42f57c: 97f04b01     	bl	0x42180
  42f580: d00006a0     	adrp	x0, 0x505000
  42f584: 91114000     	add	x0, x0, #1104
  42f588: 97f04afe     	bl	0x42180
  42f58c: d00006a0     	adrp	x0, 0x505000
  42f590: 91114000     	add	x0, x0, #1104
  42f594: 97f04afb     	bl	0x42180
  42f598: 17ffffb1     	b	0x42f45c
  42f59c: d503201f     	nop
  42f5a0: d108c3ff     	sub	sp, sp, #560
  42f5a4: a9017bfd     	stp	x29, x30, [sp, #16]
  42f5a8: 910043fd     	add	x29, sp, #16
  42f5ac: a90253f3     	stp	x19, x20, [sp, #32]
  42f5b0: 90001294     	adrp	x20, 0x67f000
  42f5b4: f9002ffa     	str	x26, [sp, #88]
  42f5b8: aa0003fa     	mov	x26, x0
  42f5bc: f9434681     	ldr	x1, [x20, #1672]
  42f5c0: f9400800     	ldr	x0, [x0, #16]
  42f5c4: f9400022     	ldr	x2, [x1]
  42f5c8: f9010fa2     	str	x2, [x29, #536]
  42f5cc: d2800002     	mov	x2, #0
  42f5d0: b4000dc0     	cbz	x0, 0x42f788
  42f5d4: f9400001     	ldr	x1, [x0]
  42f5d8: b4002ba1     	cbz	x1, 0x42fb4c
  42f5dc: f9400400     	ldr	x0, [x0, #8]
  42f5e0: b4000820     	cbz	x0, 0x42f6e4
  42f5e4: a90363b7     	stp	x23, x24, [x29, #48]
  42f5e8: f0000c17     	adrp	x23, 0x5b2000
  42f5ec: 52886878     	mov	w24, #17219
  42f5f0: 912d22f7     	add	x23, x23, #2888
  42f5f4: f90023b9     	str	x25, [x29, #64]
  42f5f8: 528848b9     	mov	w25, #16965
  42f5fc: a9025bb5     	stp	x21, x22, [x29, #32]
  42f600: 910082f7     	add	x23, x23, #32
  42f604: d2800015     	mov	x21, #0
  42f608: d2800016     	mov	x22, #0
  42f60c: 72a82938     	movk	w24, #16713, lsl #16
  42f610: f9002bbb     	str	x27, [x29, #80]
  42f614: 72a86ab9     	movk	w25, #17237, lsl #16
  42f618: 14000012     	b	0x42f660
  42f61c: d503201f     	nop
  42f620: 71000a7f     	cmp	w19, #2
  42f624: 540007a0     	b.eq	0x42f718
  42f628: 71000e7f     	cmp	w19, #3
  42f62c: 54000d21     	b.ne	0x42f7d0
  42f630: 91001021     	add	x1, x1, #4
  42f634: aa1a03e0     	mov	x0, x26
  42f638: 97fffd8a     	bl	0x42ec60
  42f63c: 2a0003f3     	mov	w19, w0
  42f640: 35002680     	cbnz	w0, 0x42fb10
  42f644: f9400b40     	ldr	x0, [x26, #16]
  42f648: 910006d6     	add	x22, x22, #1
  42f64c: 910052b5     	add	x21, x21, #20
  42f650: f9400401     	ldr	x1, [x0, #8]
  42f654: eb16003f     	cmp	x1, x22
  42f658: 540003e9     	b.ls	0x42f6d4
  42f65c: f9400001     	ldr	x1, [x0]
  42f660: b8756833     	ldr	w19, [x1, x21]
  42f664: 8b150021     	add	x1, x1, x21
  42f668: 7100067f     	cmp	w19, #1
  42f66c: 54fffda1     	b.ne	0x42f620
  42f670: b9400433     	ldr	w19, [x1, #4]
  42f674: 7100067f     	cmp	w19, #1
  42f678: 54000840     	b.eq	0x42f780
  42f67c: 35002ad3     	cbnz	w19, 0x42fbd4
  42f680: 3900d35f     	strb	wzr, [x26, #52]
  42f684: 52800001     	mov	w1, #0
  42f688: 528000e0     	mov	w0, #7
  42f68c: 97f04db1     	bl	0x42d50
  42f690: 7100041f     	cmp	w0, #1
  42f694: 54fffd81     	b.ne	0x42f644
  42f698: 52800000     	mov	w0, #0
  42f69c: d0000a84     	adrp	x4, 0x581000
  42f6a0: d0000a81     	adrp	x1, 0x581000
  42f6a4: 2a1303e5     	mov	w5, w19
  42f6a8: 91122021     	add	x1, x1, #1160
  42f6ac: 911ee084     	add	x4, x4, #1976
  42f6b0: aa1703e3     	mov	x3, x23
  42f6b4: 52801002     	mov	w2, #128
  42f6b8: 97f04b1a     	bl	0x42320
  42f6bc: 910006d6     	add	x22, x22, #1
  42f6c0: f9400b40     	ldr	x0, [x26, #16]
  42f6c4: 910052b5     	add	x21, x21, #20
  42f6c8: f9400401     	ldr	x1, [x0, #8]
  42f6cc: eb16003f     	cmp	x1, x22
  42f6d0: 54fffc68     	b.hi	0x42f65c
  42f6d4: a9425bb5     	ldp	x21, x22, [x29, #32]
  42f6d8: a94363b7     	ldp	x23, x24, [x29, #48]
  42f6dc: f94023b9     	ldr	x25, [x29, #64]
  42f6e0: f9402bbb     	ldr	x27, [x29, #80]
  42f6e4: 52800013     	mov	w19, #0
  42f6e8: f9434694     	ldr	x20, [x20, #1672]
  42f6ec: 2a1303e0     	mov	w0, w19
  42f6f0: f9410fa2     	ldr	x2, [x29, #536]
  42f6f4: f9400281     	ldr	x1, [x20]
  42f6f8: ca010041     	eor	x1, x2, x1
  42f6fc: b5004001     	cbnz	x1, 0x42fefc
  42f700: a94253f3     	ldp	x19, x20, [sp, #32]
  42f704: a9417bfd     	ldp	x29, x30, [sp, #16]
  42f708: f9402ffa     	ldr	x26, [sp, #88]
  42f70c: 9108c3ff     	add	sp, sp, #560
  42f710: d65f03c0     	ret
  42f714: d503201f     	nop
  42f718: 39406340     	ldrb	w0, [x26, #24]
  42f71c: 35002a80     	cbnz	w0, 0x42fc6c
  42f720: b9400433     	ldr	w19, [x1, #4]
  42f724: 528a48db     	mov	w27, #21062
  42f728: 72a8283b     	movk	w27, #16705, lsl #16
  42f72c: 6b19027f     	cmp	w19, w25
  42f730: 7a581264     	ccmp	w19, w24, #4, ne
  42f734: 7a5b1264     	ccmp	w19, w27, #4, ne
  42f738: 54002d81     	b.ne	0x42fce8
  42f73c: b9003353     	str	w19, [x26, #48]
  42f740: 52800001     	mov	w1, #0
  42f744: 528000e0     	mov	w0, #7
  42f748: 97f04d82     	bl	0x42d50
  42f74c: 7100041f     	cmp	w0, #1
  42f750: 54fff7a1     	b.ne	0x42f644
  42f754: d0000a84     	adrp	x4, 0x581000
  42f758: d0000a81     	adrp	x1, 0x581000
  42f75c: 2a1303e5     	mov	w5, w19
  42f760: 91240084     	add	x4, x4, #2304
  42f764: aa1703e3     	mov	x3, x23
  42f768: 528011c2     	mov	w2, #142
  42f76c: 91122021     	add	x1, x1, #1160
  42f770: 52800000     	mov	w0, #0
  42f774: 97f04aeb     	bl	0x42320
  42f778: 17ffffb3     	b	0x42f644
  42f77c: d503201f     	nop
  42f780: 3900d353     	strb	w19, [x26, #52]
  42f784: 17ffffc0     	b	0x42f684
  42f788: 52800001     	mov	w1, #0
  42f78c: 528000e0     	mov	w0, #7
  42f790: 97f04d70     	bl	0x42d50
  42f794: 7100041f     	cmp	w0, #1
  42f798: 54fffa61     	b.ne	0x42f6e4
  42f79c: f0000c03     	adrp	x3, 0x5b2000
  42f7a0: 912d2063     	add	x3, x3, #2888
  42f7a4: d0000a84     	adrp	x4, 0x581000
  42f7a8: d0000a81     	adrp	x1, 0x581000
  42f7ac: 911a2084     	add	x4, x4, #1672
  42f7b0: 91008063     	add	x3, x3, #32
  42f7b4: 52800d82     	mov	w2, #108
  42f7b8: 91122021     	add	x1, x1, #1160
  42f7bc: 52800000     	mov	w0, #0
  42f7c0: 52800013     	mov	w19, #0
  42f7c4: 97f04ad7     	bl	0x42320
  42f7c8: 17ffffc8     	b	0x42f6e8
  42f7cc: d503201f     	nop
  42f7d0: f0000c00     	adrp	x0, 0x5b2000
  42f7d4: 9101e3b9     	add	x25, x29, #120
  42f7d8: 912d2001     	add	x1, x0, #2888
  42f7dc: aa1903e2     	mov	x2, x25
  42f7e0: 91008021     	add	x1, x1, #32
  42f7e4: 910243bb     	add	x27, x29, #144
  42f7e8: 9102c3b6     	add	x22, x29, #176
  42f7ec: aa1b03e0     	mov	x0, x27
  42f7f0: f9002fbc     	str	x28, [x29, #88]
  42f7f4: 97f0b2eb     	bl	0x5c3a0
  42f7f8: 910162c0     	add	x0, x22, #88
  42f7fc: 97f04939     	bl	0x41ce0
  42f800: 90001280     	adrp	x0, 0x67f000
  42f804: 90001298     	adrp	x24, 0x67f000
  42f808: 4f000400     	movi	v0.4s, #0
  42f80c: 3907a3bf     	strb	wzr, [x29, #488]
  42f810: f9433000     	ldr	x0, [x0, #1632]
  42f814: f9424f01     	ldr	x1, [x24, #1176]
  42f818: 3907a7bf     	strb	wzr, [x29, #489]
  42f81c: f900f3bf     	str	xzr, [x29, #480]
  42f820: 91004021     	add	x1, x1, #16
  42f824: a940f01a     	ldp	x26, x28, [x0, #8]
  42f828: f9005bba     	str	x26, [x29, #176]
  42f82c: f90087a1     	str	x1, [x29, #264]
  42f830: d2800001     	mov	x1, #0
  42f834: 3d807fa0     	str	q0, [x29, #496]
  42f838: 3d8083a0     	str	q0, [x29, #512]
  42f83c: f85e8340     	ldur	x0, [x26, #-24]
  42f840: f8206adc     	str	x28, [x22, x0]
  42f844: f9405ba0     	ldr	x0, [x29, #176]
  42f848: f85e8000     	ldur	x0, [x0, #-24]
  42f84c: 8b0002c0     	add	x0, x22, x0
  42f850: 97f04e18     	bl	0x430b0
  42f854: 90001280     	adrp	x0, 0x67f000
  42f858: 90001297     	adrp	x23, 0x67f000
  42f85c: 4f000400     	movi	v0.4s, #0
  42f860: 90001295     	adrp	x21, 0x67f000
  42f864: f9452ee1     	ldr	x1, [x23, #2648]
  42f868: f943b802     	ldr	x2, [x0, #1904]
  42f86c: 910102c0     	add	x0, x22, #64
  42f870: 91006023     	add	x3, x1, #24
  42f874: 91010021     	add	x1, x1, #64
  42f878: 91004042     	add	x2, x2, #16
  42f87c: f90087a1     	str	x1, [x29, #264]
  42f880: a90b0ba3     	stp	x3, x2, [x29, #176]
  42f884: 3d8033a0     	str	q0, [x29, #192]
  42f888: 3d8037a0     	str	q0, [x29, #208]
  42f88c: 3d803ba0     	str	q0, [x29, #224]
  42f890: 97f05120     	bl	0x43d10
  42f894: 90001280     	adrp	x0, 0x67f000
  42f898: 52800203     	mov	w3, #16
  42f89c: 910022c1     	add	x1, x22, #8
