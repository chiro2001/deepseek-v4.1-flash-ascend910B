
/usr/local/Ascend/cann-9.1.0/lib64/libacl_rt_impl.so:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000083708 <aclrtLaunchKernelWithConfigImpl>:
   83708: a9b07bfd     	stp	x29, x30, [sp, #-256]!
   8370c: 910003fd     	mov	x29, sp
   83710: a90153f3     	stp	x19, x20, [sp, #16]
   83714: 900004f3     	adrp	x19, 0x11f000
   83718: a9046bf9     	stp	x25, x26, [sp, #64]
   8371c: aa0203f9     	mov	x25, x2
   83720: f9469666     	ldr	x6, [x19, #3368]
   83724: 9101e3b4     	add	x20, x29, #120
   83728: a9025bf5     	stp	x21, x22, [sp, #32]
   8372c: 2a0103f6     	mov	w22, w1
   83730: f94000c2     	ldr	x2, [x6]
   83734: f9007fa2     	str	x2, [x29, #248]
   83738: d2800002     	mov	x2, #0
   8373c: 52800ca1     	mov	w1, #101
   83740: a90363f7     	stp	x23, x24, [sp, #48]
   83744: 72a00061     	movk	w1, #3, lsl #16
   83748: aa0003f5     	mov	x21, x0
   8374c: aa0303fa     	mov	x26, x3
   83750: aa1403e0     	mov	x0, x20
   83754: aa0403f7     	mov	x23, x4
   83758: aa0503f8     	mov	x24, x5
   8375c: f9002bfb     	str	x27, [sp, #80]
   83760: 97feb626     	bl	0x30ff8
   83764: 52800021     	mov	w1, #1
   83768: 52800600     	mov	w0, #48
   8376c: 97fe9b35     	bl	0x2a440
   83770: 7100041f     	cmp	w0, #1
   83774: 54000420     	b.eq	0x837f8
   83778: b4000715     	cbz	x21, 0x83858
   8377c: 34000d36     	cbz	w22, 0x83920
   83780: b4001717     	cbz	x23, 0x83a60
   83784: b5001eb8     	cbnz	x24, 0x83b58
   83788: d2800005     	mov	x5, #0
   8378c: aa1703e4     	mov	x4, x23
   83790: aa1a03e3     	mov	x3, x26
   83794: aa1903e2     	mov	x2, x25
   83798: 2a1603e1     	mov	w1, w22
   8379c: aa1503e0     	mov	x0, x21
   837a0: 97fe9ac8     	bl	0x2a2c0
   837a4: 2a0003f5     	mov	w21, w0
   837a8: 340000a0     	cbz	w0, 0x837bc
   837ac: 52944120     	mov	w0, #41481
   837b0: 72a00020     	movk	w0, #1, lsl #16
   837b4: 6b0002bf     	cmp	w21, w0
   837b8: 540020c0     	b.eq	0x83bd0
   837bc: aa1403e0     	mov	x0, x20
   837c0: 97feb5be     	bl	0x30eb8
   837c4: f9469673     	ldr	x19, [x19, #3368]
   837c8: 2a1503e0     	mov	w0, w21
   837cc: f9407fa2     	ldr	x2, [x29, #248]
   837d0: f9400261     	ldr	x1, [x19]
   837d4: ca010041     	eor	x1, x2, x1
   837d8: b50028a1     	cbnz	x1, 0x83cec
   837dc: a94153f3     	ldp	x19, x20, [sp, #16]
   837e0: a9425bf5     	ldp	x21, x22, [sp, #32]
   837e4: a94363f7     	ldp	x23, x24, [sp, #48]
   837e8: a9446bf9     	ldp	x25, x26, [sp, #64]
   837ec: f9402bfb     	ldr	x27, [sp, #80]
   837f0: a8d07bfd     	ldp	x29, x30, [sp], #256
   837f4: d65f03c0     	ret
   837f8: f00002a0     	adrp	x0, 0xda000
   837fc: f00002bb     	adrp	x27, 0xda000
   83800: 91032800     	add	x0, x0, #202
   83804: 9103237b     	add	x27, x27, #200
   83808: 14000005     	b	0x8381c
   8380c: d503201f     	nop
   83810: 7100bc3f     	cmp	w1, #47
   83814: 9a80137b     	csel	x27, x27, x0, ne
   83818: 91000400     	add	x0, x0, #1
   8381c: 385ff001     	ldurb	w1, [x0, #-1]
   83820: 35ffff81     	cbnz	w1, 0x83810
   83824: 97feb193     	bl	0x2fe70
   83828: 90000326     	adrp	x6, 0xe7000
   8382c: 912740c6     	add	x6, x6, #2512
   83830: f00002a2     	adrp	x2, 0xda000
   83834: 2a0003e5     	mov	w5, w0
   83838: 9112c0c6     	add	x6, x6, #1200
   8383c: 52801ce4     	mov	w4, #231
   83840: aa1b03e3     	mov	x3, x27
   83844: 91236042     	add	x2, x2, #2264
   83848: 52800021     	mov	w1, #1
   8384c: 52800600     	mov	w0, #48
   83850: 97fe9c00     	bl	0x2a850
   83854: b5fff955     	cbnz	x21, 0x8377c
   83858: f00002a0     	adrp	x0, 0xda000
   8385c: f00002b5     	adrp	x21, 0xda000
   83860: 91032800     	add	x0, x0, #202
   83864: 910322b5     	add	x21, x21, #200
   83868: 14000005     	b	0x8387c
   8386c: d503201f     	nop
   83870: 7100bc3f     	cmp	w1, #47
   83874: 9a8012b5     	csel	x21, x21, x0, ne
   83878: 91000400     	add	x0, x0, #1
   8387c: 385ff001     	ldurb	w1, [x0, #-1]
   83880: 35ffff81     	cbnz	w1, 0x83870
   83884: 97feb17b     	bl	0x2fe70
   83888: 90000336     	adrp	x22, 0xe7000
   8388c: 912742c6     	add	x6, x22, #2512
   83890: f00002b7     	adrp	x23, 0xda000
   83894: f0000242     	adrp	x2, 0xce000
   83898: 2a0003e5     	mov	w5, w0
   8389c: 910722e7     	add	x7, x23, #456
   838a0: 9112c0c6     	add	x6, x6, #1200
   838a4: 52801d04     	mov	w4, #232
   838a8: aa1503e3     	mov	x3, x21
   838ac: 913ee042     	add	x2, x2, #4024
   838b0: 52800061     	mov	w1, #3
   838b4: 52800600     	mov	w0, #48
   838b8: 97fe9be6     	bl	0x2a850
   838bc: 912742c0     	add	x0, x22, #2512
   838c0: 9101c3a8     	add	x8, x29, #112
   838c4: 91134000     	add	x0, x0, #1232
   838c8: 97feb332     	bl	0x30590
   838cc: d00004c1     	adrp	x1, 0x11d000
   838d0: 910243b5     	add	x21, x29, #144
   838d4: 910a2021     	add	x1, x1, #648
   838d8: d2800042     	mov	x2, #2
   838dc: 910a8021     	add	x1, x1, #672
   838e0: aa1503e0     	mov	x0, x21
   838e4: 97ffebf5     	bl	0x7e8b8
   838e8: f9403ba0     	ldr	x0, [x29, #112]
   838ec: 9102a3b6     	add	x22, x29, #168
   838f0: 910722f7     	add	x23, x23, #456
   838f4: 910363a1     	add	x1, x29, #216
   838f8: a90ddfa0     	stp	x0, x23, [x29, #216]
   838fc: d2800042     	mov	x2, #2
   83900: aa1603e0     	mov	x0, x22
   83904: 97ffebed     	bl	0x7e8b8
   83908: f0000240     	adrp	x0, 0xce000
   8390c: aa1603e2     	mov	x2, x22
   83910: aa1503e1     	mov	x1, x21
   83914: 913fa000     	add	x0, x0, #4072
   83918: 97feb184     	bl	0x2ff28
   8391c: 14000082     	b	0x83b24
   83920: f00002a0     	adrp	x0, 0xda000
   83924: f00002b5     	adrp	x21, 0xda000
   83928: 91032800     	add	x0, x0, #202
   8392c: 910322b5     	add	x21, x21, #200
   83930: 14000005     	b	0x83944
   83934: d503201f     	nop
   83938: 7100bc3f     	cmp	w1, #47
   8393c: 9a8012b5     	csel	x21, x21, x0, ne
   83940: 91000400     	add	x0, x0, #1
   83944: 385ff001     	ldurb	w1, [x0, #-1]
   83948: 35ffff81     	cbnz	w1, 0x83938
   8394c: 97feb149     	bl	0x2fe70
   83950: 90000336     	adrp	x22, 0xe7000
   83954: 912742c6     	add	x6, x22, #2512
   83958: f00002b7     	adrp	x23, 0xda000
   8395c: b0000282     	adrp	x2, 0xd4000
   83960: 2a0003e5     	mov	w5, w0
   83964: 912462e7     	add	x7, x23, #2328
   83968: 9112c0c6     	add	x6, x6, #1200
   8396c: 52801d24     	mov	w4, #233
   83970: aa1503e3     	mov	x3, x21
   83974: 91302042     	add	x2, x2, #3080
   83978: 52800061     	mov	w1, #3
   8397c: 52800600     	mov	w0, #48
   83980: 97fe9bb4     	bl	0x2a850
   83984: 900004e0     	adrp	x0, 0x11f000
   83988: d00002a2     	adrp	x2, 0xd9000
   8398c: 9101a3a8     	add	x8, x29, #104
   83990: 52800003     	mov	w3, #0
   83994: f947a400     	ldr	x0, [x0, #3912]
   83998: 913a8042     	add	x2, x2, #3744
   8399c: d2800201     	mov	x1, #16
   839a0: 97feeb92     	bl	0x3e7e8
   839a4: 912742c0     	add	x0, x22, #2512
   839a8: 9101c3a8     	add	x8, x29, #112
   839ac: 91134000     	add	x0, x0, #1232
   839b0: 97feb2f8     	bl	0x30590
   839b4: d00004c1     	adrp	x1, 0x11d000
   839b8: 910243b5     	add	x21, x29, #144
   839bc: 910a2021     	add	x1, x1, #648
   839c0: d2800082     	mov	x2, #4
   839c4: 910ac021     	add	x1, x1, #688
   839c8: aa1503e0     	mov	x0, x21
   839cc: 97ffebbb     	bl	0x7e8b8
   839d0: a9468ba1     	ldp	x1, x2, [x29, #104]
   839d4: 9102a3b6     	add	x22, x29, #168
   839d8: b0000280     	adrp	x0, 0xd4000
   839dc: 9130e000     	add	x0, x0, #3128
   839e0: a90d87a2     	stp	x2, x1, [x29, #216]
   839e4: 912462f7     	add	x23, x23, #2328
   839e8: 910363a1     	add	x1, x29, #216
   839ec: a90e83b7     	stp	x23, x0, [x29, #232]
   839f0: d2800082     	mov	x2, #4
   839f4: aa1603e0     	mov	x0, x22
   839f8: 97ffebb0     	bl	0x7e8b8
   839fc: f0000260     	adrp	x0, 0xd2000
   83a00: aa1603e2     	mov	x2, x22
   83a04: aa1503e1     	mov	x1, x21
   83a08: 9100e000     	add	x0, x0, #56
   83a0c: 97feb147     	bl	0x2ff28
   83a10: f94057a0     	ldr	x0, [x29, #168]
   83a14: b4000040     	cbz	x0, 0x83a1c
   83a18: 97fe9962     	bl	0x29fa0
   83a1c: f9404ba0     	ldr	x0, [x29, #144]
   83a20: b4000040     	cbz	x0, 0x83a28
   83a24: 97fe995f     	bl	0x29fa0
   83a28: 900004f5     	adrp	x21, 0x11f000
   83a2c: f9403bb6     	ldr	x22, [x29, #112]
   83a30: f947c6a0     	ldr	x0, [x21, #3976]
   83a34: d10062d7     	sub	x23, x22, #24
   83a38: eb0002ff     	cmp	x23, x0
   83a3c: 540010a1     	b.ne	0x83c50
   83a40: f94037b6     	ldr	x22, [x29, #104]
   83a44: f947c6b5     	ldr	x21, [x21, #3976]
   83a48: d10062d7     	sub	x23, x22, #24
   83a4c: eb1502ff     	cmp	x23, x21
   83a50: 540011c1     	b.ne	0x83c88
   83a54: 5290d415     	mov	w21, #34464
   83a58: 72a00035     	movk	w21, #1, lsl #16
   83a5c: 17ffff58     	b	0x837bc
   83a60: f00002a0     	adrp	x0, 0xda000
   83a64: f00002b5     	adrp	x21, 0xda000
   83a68: 91032800     	add	x0, x0, #202
   83a6c: 910322b5     	add	x21, x21, #200
   83a70: 14000005     	b	0x83a84
   83a74: d503201f     	nop
   83a78: 7100bc3f     	cmp	w1, #47
   83a7c: 9a8012b5     	csel	x21, x21, x0, ne
   83a80: 91000400     	add	x0, x0, #1
   83a84: 385ff001     	ldurb	w1, [x0, #-1]
   83a88: 35ffff81     	cbnz	w1, 0x83a78
   83a8c: 97feb0f9     	bl	0x2fe70
   83a90: 90000336     	adrp	x22, 0xe7000
   83a94: 912742c6     	add	x6, x22, #2512
   83a98: f00002b7     	adrp	x23, 0xda000
   83a9c: f0000242     	adrp	x2, 0xce000
   83aa0: 2a0003e5     	mov	w5, w0
   83aa4: 910e02e7     	add	x7, x23, #896
   83aa8: 9112c0c6     	add	x6, x6, #1200
   83aac: 52801d44     	mov	w4, #234
   83ab0: aa1503e3     	mov	x3, x21
   83ab4: 913ee042     	add	x2, x2, #4024
   83ab8: 52800061     	mov	w1, #3
   83abc: 52800600     	mov	w0, #48
   83ac0: 97fe9b64     	bl	0x2a850
   83ac4: 912742c0     	add	x0, x22, #2512
   83ac8: 9101c3a8     	add	x8, x29, #112
   83acc: 91134000     	add	x0, x0, #1232
   83ad0: 97feb2b0     	bl	0x30590
   83ad4: d00004c1     	adrp	x1, 0x11d000
   83ad8: 910243b5     	add	x21, x29, #144
   83adc: 910a2021     	add	x1, x1, #648
   83ae0: d2800042     	mov	x2, #2
   83ae4: 910b4021     	add	x1, x1, #720
   83ae8: aa1503e0     	mov	x0, x21
   83aec: 97ffeb73     	bl	0x7e8b8
   83af0: f9403ba0     	ldr	x0, [x29, #112]
   83af4: 9102a3b6     	add	x22, x29, #168
   83af8: 910e02f7     	add	x23, x23, #896
   83afc: 910363a1     	add	x1, x29, #216
   83b00: a90ddfa0     	stp	x0, x23, [x29, #216]
   83b04: d2800042     	mov	x2, #2
   83b08: aa1603e0     	mov	x0, x22
   83b0c: 97ffeb6b     	bl	0x7e8b8
   83b10: f0000240     	adrp	x0, 0xce000
   83b14: aa1603e2     	mov	x2, x22
   83b18: aa1503e1     	mov	x1, x21
   83b1c: 913fa000     	add	x0, x0, #4072
   83b20: 97feb102     	bl	0x2ff28
   83b24: f94057a0     	ldr	x0, [x29, #168]
   83b28: b4000040     	cbz	x0, 0x83b30
   83b2c: 97fe991d     	bl	0x29fa0
   83b30: f9404ba0     	ldr	x0, [x29, #144]
   83b34: b4000040     	cbz	x0, 0x83b3c
   83b38: 97fe991a     	bl	0x29fa0
   83b3c: f9403ba0     	ldr	x0, [x29, #112]
   83b40: 5290d415     	mov	w21, #34464
   83b44: 72a00035     	movk	w21, #1, lsl #16
   83b48: d1006000     	sub	x0, x0, #24
   83b4c: 97ffeb29     	bl	0x7e7f0
   83b50: 17ffff1b     	b	0x837bc
   83b54: d503201f     	nop
   83b58: b00004e1     	adrp	x1, 0x120000
   83b5c: 91022021     	add	x1, x1, #136
   83b60: 90000320     	adrp	x0, 0xe7000
   83b64: 91274000     	add	x0, x0, #2512
   83b68: 9102a3a8     	add	x8, x29, #168
   83b6c: 91134000     	add	x0, x0, #1232
   83b70: a9400c22     	ldp	x2, x3, [x1]
   83b74: a90c0fa2     	stp	x2, x3, [x29, #192]
   83b78: f9400821     	ldr	x1, [x1, #16]
   83b7c: f9006ba1     	str	x1, [x29, #208]
   83b80: 97feb284     	bl	0x30590
   83b84: 910403a2     	add	x2, x29, #256
   83b88: f94057a0     	ldr	x0, [x29, #168]
   83b8c: f00002c3     	adrp	x3, 0xde000
   83b90: 900002a1     	adrp	x1, 0xd7000
   83b94: 91320063     	add	x3, x3, #3200
   83b98: 91010021     	add	x1, x1, #64
   83b9c: f81d8c40     	str	x0, [x2, #-40]!
   83ba0: d0000260     	adrp	x0, 0xd1000
   83ba4: a90e07a3     	stp	x3, x1, [x29, #224]
   83ba8: 91256000     	add	x0, x0, #2392
   83bac: d2800063     	mov	x3, #3
   83bb0: 910303a1     	add	x1, x29, #192
   83bb4: 97feb21d     	bl	0x30428
   83bb8: f94057a0     	ldr	x0, [x29, #168]
   83bbc: 5290d415     	mov	w21, #34464
   83bc0: 72a00035     	movk	w21, #1, lsl #16
   83bc4: d1006000     	sub	x0, x0, #24
   83bc8: 97ffeb0a     	bl	0x7e7f0
   83bcc: 17fffefc     	b	0x837bc
   83bd0: 52800041     	mov	w1, #2
   83bd4: 52800600     	mov	w0, #48
   83bd8: 97fe9a1a     	bl	0x2a440
   83bdc: 7100041f     	cmp	w0, #1
   83be0: 54000321     	b.ne	0x83c44
   83be4: f00002a0     	adrp	x0, 0xda000
   83be8: f00002b5     	adrp	x21, 0xda000
   83bec: 91032800     	add	x0, x0, #202
   83bf0: 910322b5     	add	x21, x21, #200
   83bf4: 14000004     	b	0x83c04
   83bf8: 7100bc3f     	cmp	w1, #47
   83bfc: 9a8012b5     	csel	x21, x21, x0, ne
   83c00: 91000400     	add	x0, x0, #1
   83c04: 385ff001     	ldurb	w1, [x0, #-1]
   83c08: 35ffff81     	cbnz	w1, 0x83bf8
   83c0c: 97feb099     	bl	0x2fe70
   83c10: 90000326     	adrp	x6, 0xe7000
   83c14: 52944127     	mov	w7, #41481
   83c18: 912740c6     	add	x6, x6, #2512
   83c1c: f00002a2     	adrp	x2, 0xda000
   83c20: 2a0003e5     	mov	w5, w0
   83c24: 72a00027     	movk	w7, #1, lsl #16
   83c28: 9112c0c6     	add	x6, x6, #1200
   83c2c: 52801ea4     	mov	w4, #245
   83c30: aa1503e3     	mov	x3, x21
   83c34: 9124a042     	add	x2, x2, #2344
   83c38: 52800041     	mov	w1, #2
   83c3c: 52800600     	mov	w0, #48
   83c40: 97fe9b04     	bl	0x2a850
   83c44: 52944135     	mov	w21, #41481
   83c48: 72a00035     	movk	w21, #1, lsl #16
   83c4c: 17fffedc     	b	0x837bc
   83c50: 900004e0     	adrp	x0, 0x11f000
   83c54: f9479800     	ldr	x0, [x0, #3888]
   83c58: b4000320     	cbz	x0, 0x83cbc
   83c5c: d10022d6     	sub	x22, x22, #8
   83c60: 885ffed8     	ldaxr	w24, [x22]
   83c64: 51000700     	sub	w0, w24, #1
   83c68: 8801fec0     	stlxr	w1, w0, [x22]
   83c6c: 35ffffa1     	cbnz	w1, 0x83c60
   83c70: 7100031f     	cmp	w24, #0
   83c74: 54ffee6c     	b.gt	0x83a40
   83c78: aa1703e0     	mov	x0, x23
   83c7c: 97fe98c9     	bl	0x29fa0
   83c80: 17ffff70     	b	0x83a40
   83c84: d503201f     	nop
   83c88: 900004e0     	adrp	x0, 0x11f000
   83c8c: f9479800     	ldr	x0, [x0, #3888]
   83c90: b4000220     	cbz	x0, 0x83cd4
   83c94: d10022d6     	sub	x22, x22, #8
   83c98: 885ffed5     	ldaxr	w21, [x22]
   83c9c: 510006a0     	sub	w0, w21, #1
   83ca0: 8801fec0     	stlxr	w1, w0, [x22]
   83ca4: 35ffffa1     	cbnz	w1, 0x83c98
   83ca8: 710002bf     	cmp	w21, #0
   83cac: 54ffed4c     	b.gt	0x83a54
   83cb0: aa1703e0     	mov	x0, x23
   83cb4: 97fe98bb     	bl	0x29fa0
   83cb8: 17ffff67     	b	0x83a54
   83cbc: b85f82d8     	ldur	w24, [x22, #-8]
   83cc0: 12800001     	mov	w1, #-1
   83cc4: 2a1803e0     	mov	w0, w24
   83cc8: 97fe9cd6     	bl	0x2b020
   83ccc: b81f82c0     	stur	w0, [x22, #-8]
   83cd0: 17ffffe8     	b	0x83c70
   83cd4: b85f82d5     	ldur	w21, [x22, #-8]
   83cd8: 12800001     	mov	w1, #-1
   83cdc: 2a1503e0     	mov	w0, w21
   83ce0: 97fe9cd0     	bl	0x2b020
   83ce4: b81f82c0     	stur	w0, [x22, #-8]
   83ce8: 17fffff0     	b	0x83ca8
   83cec: 97fe9b7d     	bl	0x2aae0
   83cf0: f94057a1     	ldr	x1, [x29, #168]
   83cf4: aa0003f3     	mov	x19, x0
   83cf8: b4000061     	cbz	x1, 0x83d04
   83cfc: aa0103e0     	mov	x0, x1
   83d00: 97fe98a8     	bl	0x29fa0
   83d04: f9404ba0     	ldr	x0, [x29, #144]
   83d08: b4000040     	cbz	x0, 0x83d10
   83d0c: 97fe98a5     	bl	0x29fa0
   83d10: f9403ba0     	ldr	x0, [x29, #112]
   83d14: d1006000     	sub	x0, x0, #24
   83d18: 97ffeab6     	bl	0x7e7f0
   83d1c: aa1403e0     	mov	x0, x20
   83d20: 97feb466     	bl	0x30eb8
   83d24: aa1303e0     	mov	x0, x19
   83d28: 97fe9ed2     	bl	0x2b870
   83d2c: aa0003f3     	mov	x19, x0
   83d30: f94037a0     	ldr	x0, [x29, #104]
   83d34: d1006000     	sub	x0, x0, #24
   83d38: 97ffeaae     	bl	0x7e7f0
   83d3c: 17fffff8     	b	0x83d1c
   83d40: f94057a1     	ldr	x1, [x29, #168]
   83d44: aa0003f3     	mov	x19, x0
   83d48: b4000061     	cbz	x1, 0x83d54
   83d4c: aa0103e0     	mov	x0, x1
   83d50: 97fe9894     	bl	0x29fa0
   83d54: f9404ba0     	ldr	x0, [x29, #144]
   83d58: b4000040     	cbz	x0, 0x83d60
   83d5c: 97fe9891     	bl	0x29fa0
   83d60: f9403ba0     	ldr	x0, [x29, #112]
   83d64: d1006000     	sub	x0, x0, #24
   83d68: 97ffeaa2     	bl	0x7e7f0
   83d6c: 17fffff1     	b	0x83d30
   83d70: aa0003f3     	mov	x19, x0
   83d74: 17ffffea     	b	0x83d1c
   83d78: aa0003f3     	mov	x19, x0
   83d7c: 17fffff9     	b	0x83d60
   83d80: aa0003f3     	mov	x19, x0
   83d84: 17fffff4     	b	0x83d54
   83d88: aa0003f3     	mov	x19, x0
   83d8c: 17ffffde     	b	0x83d04
   83d90: aa0003f3     	mov	x19, x0
   83d94: 17ffffdf     	b	0x83d10
   83d98: f94057a1     	ldr	x1, [x29, #168]
   83d9c: aa0003f3     	mov	x19, x0
   83da0: d1006020     	sub	x0, x1, #24
   83da4: 97ffea93     	bl	0x7e7f0
   83da8: 17ffffdd     	b	0x83d1c
   83dac: 17ffffd1     	b	0x83cf0
   83db0: 17fffff6     	b	0x83d88
   83db4: 17fffff7     	b	0x83d90
