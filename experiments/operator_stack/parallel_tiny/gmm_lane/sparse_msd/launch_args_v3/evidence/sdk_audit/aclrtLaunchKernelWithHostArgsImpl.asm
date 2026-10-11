
/usr/local/Ascend/cann-9.1.0/lib64/libacl_rt_impl.so:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000087bf0 <aclrtLaunchKernelWithHostArgsImpl>:
   87bf0: a9b37bfd     	stp	x29, x30, [sp, #-208]!
   87bf4: 910003fd     	mov	x29, sp
   87bf8: a90153f3     	stp	x19, x20, [sp, #16]
   87bfc: 900004d3     	adrp	x19, 0x11f000
   87c00: a9025bf5     	stp	x21, x22, [sp, #32]
   87c04: aa0003f4     	mov	x20, x0
   87c08: f9469668     	ldr	x8, [x19, #3368]
   87c0c: 9101c3b5     	add	x21, x29, #112
   87c10: a90363f7     	stp	x23, x24, [sp, #48]
   87c14: 2a0103f7     	mov	w23, w1
   87c18: f9400100     	ldr	x0, [x8]
   87c1c: f90067a0     	str	x0, [x29, #200]
   87c20: d2800000     	mov	x0, #0
   87c24: 52801801     	mov	w1, #192
   87c28: a9046bf9     	stp	x25, x26, [sp, #64]
   87c2c: aa0203f8     	mov	x24, x2
   87c30: a90573fb     	stp	x27, x28, [sp, #80]
   87c34: aa0303f9     	mov	x25, x3
   87c38: aa0403f6     	mov	x22, x4
   87c3c: aa0503fa     	mov	x26, x5
   87c40: aa0603fb     	mov	x27, x6
   87c44: aa0703fc     	mov	x28, x7
   87c48: 72a00061     	movk	w1, #3, lsl #16
   87c4c: aa1503e0     	mov	x0, x21
   87c50: 97fea4ea     	bl	0x30ff8
   87c54: b4000834     	cbz	x20, 0x87d58
   87c58: b4001016     	cbz	x22, 0x87e58
   87c5c: 2a1c03e7     	mov	w7, w28
   87c60: aa1b03e6     	mov	x6, x27
   87c64: 2a1a03e5     	mov	w5, w26
   87c68: aa1603e4     	mov	x4, x22
   87c6c: aa1903e3     	mov	x3, x25
   87c70: aa1803e2     	mov	x2, x24
   87c74: 2a1703e1     	mov	w1, w23
   87c78: aa1403e0     	mov	x0, x20
   87c7c: 97fe8a4d     	bl	0x2a5b0
   87c80: 2a0003f4     	mov	w20, w0
   87c84: 340000a0     	cbz	w0, 0x87c98
   87c88: 52944120     	mov	w0, #41481
   87c8c: 72a00020     	movk	w0, #1, lsl #16
   87c90: 6b00029f     	cmp	w20, w0
   87c94: 54000220     	b.eq	0x87cd8
   87c98: aa1503e0     	mov	x0, x21
   87c9c: 97fea487     	bl	0x30eb8
   87ca0: f9469673     	ldr	x19, [x19, #3368]
   87ca4: 2a1403e0     	mov	w0, w20
   87ca8: f94067a2     	ldr	x2, [x29, #200]
   87cac: f9400261     	ldr	x1, [x19]
   87cb0: ca010041     	eor	x1, x2, x1
   87cb4: b5001861     	cbnz	x1, 0x87fc0
   87cb8: a94153f3     	ldp	x19, x20, [sp, #16]
   87cbc: a9425bf5     	ldp	x21, x22, [sp, #32]
   87cc0: a94363f7     	ldp	x23, x24, [sp, #48]
   87cc4: a9446bf9     	ldp	x25, x26, [sp, #64]
   87cc8: a94573fb     	ldp	x27, x28, [sp, #80]
   87ccc: a8cd7bfd     	ldp	x29, x30, [sp], #208
   87cd0: d65f03c0     	ret
   87cd4: d503201f     	nop
   87cd8: 52800041     	mov	w1, #2
   87cdc: 52800600     	mov	w0, #48
   87ce0: 97fe89d8     	bl	0x2a440
   87ce4: 7100041f     	cmp	w0, #1
   87ce8: 54000321     	b.ne	0x87d4c
   87cec: f0000280     	adrp	x0, 0xda000
   87cf0: f0000294     	adrp	x20, 0xda000
   87cf4: 91032800     	add	x0, x0, #202
   87cf8: 91032294     	add	x20, x20, #200
   87cfc: 14000004     	b	0x87d0c
   87d00: 7100bc3f     	cmp	w1, #47
   87d04: 9a801294     	csel	x20, x20, x0, ne
   87d08: 91000400     	add	x0, x0, #1
   87d0c: 385ff001     	ldurb	w1, [x0, #-1]
   87d10: 35ffff81     	cbnz	w1, 0x87d00
   87d14: 97fea057     	bl	0x2fe70
   87d18: 90000306     	adrp	x6, 0xe7000
   87d1c: 52944127     	mov	w7, #41481
   87d20: 912740c6     	add	x6, x6, #2512
   87d24: f0000282     	adrp	x2, 0xda000
   87d28: 2a0003e5     	mov	w5, w0
   87d2c: 72a00027     	movk	w7, #1, lsl #16
   87d30: 912240c6     	add	x6, x6, #2192
   87d34: 52803f44     	mov	w4, #506
   87d38: aa1403e3     	mov	x3, x20
   87d3c: 91382042     	add	x2, x2, #3592
   87d40: 52800041     	mov	w1, #2
   87d44: 52800600     	mov	w0, #48
   87d48: 97fe8ac2     	bl	0x2a850
   87d4c: 52944134     	mov	w20, #41481
   87d50: 72a00034     	movk	w20, #1, lsl #16
   87d54: 17ffffd1     	b	0x87c98
   87d58: f0000280     	adrp	x0, 0xda000
   87d5c: f0000294     	adrp	x20, 0xda000
   87d60: 91032800     	add	x0, x0, #202
   87d64: 91032294     	add	x20, x20, #200
   87d68: 14000005     	b	0x87d7c
   87d6c: d503201f     	nop
   87d70: 7100bc3f     	cmp	w1, #47
   87d74: 9a801294     	csel	x20, x20, x0, ne
   87d78: 91000400     	add	x0, x0, #1
   87d7c: 385ff001     	ldurb	w1, [x0, #-1]
   87d80: 35ffff81     	cbnz	w1, 0x87d70
   87d84: 97fea03b     	bl	0x2fe70
   87d88: 90000316     	adrp	x22, 0xe7000
   87d8c: 912742c6     	add	x6, x22, #2512
   87d90: f0000297     	adrp	x23, 0xda000
   87d94: f0000222     	adrp	x2, 0xce000
   87d98: 2a0003e5     	mov	w5, w0
   87d9c: 910722e7     	add	x7, x23, #456
   87da0: 912240c6     	add	x6, x6, #2192
   87da4: 52803d24     	mov	w4, #489
   87da8: aa1403e3     	mov	x3, x20
   87dac: 913ee042     	add	x2, x2, #4024
   87db0: 52800061     	mov	w1, #3
   87db4: 52800600     	mov	w0, #48
   87db8: 97fe8aa6     	bl	0x2a850
   87dbc: 912742c0     	add	x0, x22, #2512
   87dc0: 9101a3a8     	add	x8, x29, #104
   87dc4: 9122e000     	add	x0, x0, #2232
   87dc8: 97fea1f2     	bl	0x30590
   87dcc: d00004a1     	adrp	x1, 0x11d000
   87dd0: 910223b4     	add	x20, x29, #136
   87dd4: 910a2021     	add	x1, x1, #648
   87dd8: d2800042     	mov	x2, #2
   87ddc: 91148021     	add	x1, x1, #1312
   87de0: aa1403e0     	mov	x0, x20
   87de4: 97ffdab5     	bl	0x7e8b8
   87de8: 910722e0     	add	x0, x23, #456
   87dec: f94037a1     	ldr	x1, [x29, #104]
   87df0: 910283b6     	add	x22, x29, #160
   87df4: a90b83a1     	stp	x1, x0, [x29, #184]
   87df8: d2800042     	mov	x2, #2
   87dfc: 9102e3a1     	add	x1, x29, #184
   87e00: aa1603e0     	mov	x0, x22
   87e04: 97ffdaad     	bl	0x7e8b8
   87e08: f0000220     	adrp	x0, 0xce000
   87e0c: aa1603e2     	mov	x2, x22
   87e10: aa1403e1     	mov	x1, x20
   87e14: 913fa000     	add	x0, x0, #4072
   87e18: 97fea044     	bl	0x2ff28
   87e1c: f94053a0     	ldr	x0, [x29, #160]
   87e20: b4000040     	cbz	x0, 0x87e28
   87e24: 97fe885f     	bl	0x29fa0
   87e28: f94047a0     	ldr	x0, [x29, #136]
   87e2c: b4000040     	cbz	x0, 0x87e34
   87e30: 97fe885c     	bl	0x29fa0
   87e34: 900004c0     	adrp	x0, 0x11f000
   87e38: f94037b4     	ldr	x20, [x29, #104]
   87e3c: f947c400     	ldr	x0, [x0, #3976]
   87e40: d1006296     	sub	x22, x20, #24
   87e44: eb0002df     	cmp	x22, x0
   87e48: 540009c1     	b.ne	0x87f80
   87e4c: 5290d414     	mov	w20, #34464
   87e50: 72a00034     	movk	w20, #1, lsl #16
   87e54: 17ffff91     	b	0x87c98
   87e58: f0000280     	adrp	x0, 0xda000
   87e5c: f0000294     	adrp	x20, 0xda000
   87e60: 91032800     	add	x0, x0, #202
   87e64: 91032294     	add	x20, x20, #200
   87e68: 14000005     	b	0x87e7c
   87e6c: d503201f     	nop
   87e70: 7100bc3f     	cmp	w1, #47
   87e74: 9a801294     	csel	x20, x20, x0, ne
   87e78: 91000400     	add	x0, x0, #1
   87e7c: 385ff001     	ldurb	w1, [x0, #-1]
   87e80: 35ffff81     	cbnz	w1, 0x87e70
   87e84: 97fe9ffb     	bl	0x2fe70
   87e88: 90000316     	adrp	x22, 0xe7000
   87e8c: 912742c6     	add	x6, x22, #2512
   87e90: f0000297     	adrp	x23, 0xda000
   87e94: f0000222     	adrp	x2, 0xce000
   87e98: 2a0003e5     	mov	w5, w0
   87e9c: 9137e2e7     	add	x7, x23, #3576
   87ea0: 912240c6     	add	x6, x6, #2192
   87ea4: 52803d44     	mov	w4, #490
   87ea8: aa1403e3     	mov	x3, x20
   87eac: 913ee042     	add	x2, x2, #4024
   87eb0: 52800061     	mov	w1, #3
   87eb4: 52800600     	mov	w0, #48
   87eb8: 97fe8a66     	bl	0x2a850
   87ebc: 912742c0     	add	x0, x22, #2512
   87ec0: 9101a3a8     	add	x8, x29, #104
   87ec4: 9122e000     	add	x0, x0, #2232
   87ec8: 97fea1b2     	bl	0x30590
   87ecc: d00004a1     	adrp	x1, 0x11d000
   87ed0: 910223b4     	add	x20, x29, #136
   87ed4: 910a2021     	add	x1, x1, #648
   87ed8: d2800042     	mov	x2, #2
   87edc: 9114c021     	add	x1, x1, #1328
   87ee0: aa1403e0     	mov	x0, x20
   87ee4: 97ffda75     	bl	0x7e8b8
   87ee8: 9137e2e0     	add	x0, x23, #3576
   87eec: f94037a1     	ldr	x1, [x29, #104]
   87ef0: 910283b6     	add	x22, x29, #160
   87ef4: a90b83a1     	stp	x1, x0, [x29, #184]
   87ef8: d2800042     	mov	x2, #2
   87efc: 9102e3a1     	add	x1, x29, #184
   87f00: aa1603e0     	mov	x0, x22
   87f04: 97ffda6d     	bl	0x7e8b8
   87f08: f0000220     	adrp	x0, 0xce000
   87f0c: aa1603e2     	mov	x2, x22
   87f10: aa1403e1     	mov	x1, x20
   87f14: 913fa000     	add	x0, x0, #4072
   87f18: 97fea004     	bl	0x2ff28
   87f1c: f94053a0     	ldr	x0, [x29, #160]
   87f20: b4000040     	cbz	x0, 0x87f28
   87f24: 97fe881f     	bl	0x29fa0
   87f28: f94047a0     	ldr	x0, [x29, #136]
   87f2c: b4000040     	cbz	x0, 0x87f34
   87f30: 97fe881c     	bl	0x29fa0
   87f34: 900004c0     	adrp	x0, 0x11f000
   87f38: f94037b4     	ldr	x20, [x29, #104]
   87f3c: f947c400     	ldr	x0, [x0, #3976]
   87f40: d1006296     	sub	x22, x20, #24
   87f44: eb0002df     	cmp	x22, x0
   87f48: 54fff820     	b.eq	0x87e4c
   87f4c: 900004c0     	adrp	x0, 0x11f000
   87f50: f9479800     	ldr	x0, [x0, #3888]
   87f54: b40002a0     	cbz	x0, 0x87fa8
   87f58: d1002294     	sub	x20, x20, #8
   87f5c: 885ffe97     	ldaxr	w23, [x20]
   87f60: 510006e0     	sub	w0, w23, #1
   87f64: 8801fe80     	stlxr	w1, w0, [x20]
   87f68: 35ffffa1     	cbnz	w1, 0x87f5c
   87f6c: 710002ff     	cmp	w23, #0
   87f70: 54fff6ec     	b.gt	0x87e4c
   87f74: aa1603e0     	mov	x0, x22
   87f78: 97fe880a     	bl	0x29fa0
   87f7c: 17ffffb4     	b	0x87e4c
   87f80: 900004c0     	adrp	x0, 0x11f000
   87f84: f9479800     	ldr	x0, [x0, #3888]
   87f88: b4000100     	cbz	x0, 0x87fa8
   87f8c: d1002294     	sub	x20, x20, #8
   87f90: 885ffe97     	ldaxr	w23, [x20]
   87f94: 510006e0     	sub	w0, w23, #1
   87f98: 8801fe80     	stlxr	w1, w0, [x20]
   87f9c: 34fffe81     	cbz	w1, 0x87f6c
   87fa0: 17fffffc     	b	0x87f90
   87fa4: d503201f     	nop
   87fa8: b85f8297     	ldur	w23, [x20, #-8]
   87fac: 12800001     	mov	w1, #-1
   87fb0: 2a1703e0     	mov	w0, w23
   87fb4: 97fe8c1b     	bl	0x2b020
   87fb8: b81f8280     	stur	w0, [x20, #-8]
   87fbc: 17ffffec     	b	0x87f6c
   87fc0: 97fe8ac8     	bl	0x2aae0
   87fc4: aa0003f3     	mov	x19, x0
   87fc8: aa1503e0     	mov	x0, x21
   87fcc: 97fea3bb     	bl	0x30eb8
   87fd0: aa1303e0     	mov	x0, x19
   87fd4: 97fe8e27     	bl	0x2b870
   87fd8: aa0003f3     	mov	x19, x0
   87fdc: f94047a0     	ldr	x0, [x29, #136]
   87fe0: b4000040     	cbz	x0, 0x87fe8
   87fe4: 97fe87ef     	bl	0x29fa0
   87fe8: f94037a0     	ldr	x0, [x29, #104]
   87fec: d1006000     	sub	x0, x0, #24
   87ff0: 97ffda00     	bl	0x7e7f0
   87ff4: 17fffff5     	b	0x87fc8
   87ff8: aa0003f3     	mov	x19, x0
   87ffc: 17fffffb     	b	0x87fe8
   88000: f94053a1     	ldr	x1, [x29, #160]
   88004: aa0003f3     	mov	x19, x0
   88008: b4fffea1     	cbz	x1, 0x87fdc
   8800c: aa0103e0     	mov	x0, x1
   88010: 97fe87e4     	bl	0x29fa0
   88014: 17fffff2     	b	0x87fdc
   88018: 17fffffa     	b	0x88000
   8801c: 17ffffef     	b	0x87fd8
   88020: 17fffff6     	b	0x87ff8
   88024: d503201f     	nop
