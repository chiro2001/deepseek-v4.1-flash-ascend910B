
/usr/local/Ascend/cann-9.1.0/lib64/libruntime.so:	file format elf64-littleaarch64

Disassembly of section .text:

0000000000055f88 <rtsLaunchKernelWithDevArgs>:
   55f88: d10983ff     	sub	sp, sp, #608
   55f8c: a9007bfd     	stp	x29, x30, [sp]
   55f90: 910003fd     	mov	x29, sp
   55f94: a90153f3     	stp	x19, x20, [sp, #16]
   55f98: aa0003f3     	mov	x19, x0
   55f9c: d00004c0     	adrp	x0, 0xef000
   55fa0: a9025bf5     	stp	x21, x22, [sp, #32]
   55fa4: a90363f7     	stp	x23, x24, [sp, #48]
   55fa8: aa0603f6     	mov	x22, x6
   55fac: a9046bf9     	stp	x25, x26, [sp, #64]
   55fb0: 2a0103f9     	mov	w25, w1
   55fb4: f9002bfb     	str	x27, [sp, #80]
   55fb8: b000031b     	adrp	x27, 0xb6000
   55fbc: f947c400     	ldr	x0, [x0, #3976]
   55fc0: 910fe374     	add	x20, x27, #1016
   55fc4: aa0203f5     	mov	x21, x2
   55fc8: aa0303fa     	mov	x26, x3
   55fcc: f9400001     	ldr	x1, [x0]
   55fd0: f9012fa1     	str	x1, [x29, #600]
   55fd4: d2800001     	mov	x1, #0
   55fd8: aa0403f8     	mov	x24, x4
   55fdc: 2a0503f7     	mov	w23, w5
   55fe0: 910a6294     	add	x20, x20, #664
   55fe4: 97fee7df     	bl	0xff60
   55fe8: aa1403e1     	mov	x1, x20
   55fec: 97fee49d     	bl	0xf260
   55ff0: b4003116     	cbz	x22, 0x56610
   55ff4: 9102c3ba     	add	x26, x29, #176
   55ff8: aa1403e1     	mov	x1, x20
   55ffc: aa1a03e2     	mov	x2, x26
   56000: 910323b9     	add	x25, x29, #200
   56004: 9103c3b3     	add	x19, x29, #240
   56008: aa1903e0     	mov	x0, x25
   5600c: 97fee4f1     	bl	0xf3d0
   56010: b00004d4     	adrp	x20, 0xef000
   56014: 91016260     	add	x0, x19, #88
   56018: 97fee436     	bl	0xf0f0
   5601c: b00004c0     	adrp	x0, 0xef000
   56020: f947aa81     	ldr	x1, [x20, #3920]
   56024: 4f000400     	movi	v0.4s, #0
   56028: 3908a3bf     	strb	wzr, [x29, #552]
   5602c: f947c800     	ldr	x0, [x0, #3984]
   56030: 91004021     	add	x1, x1, #16
   56034: 3908a7bf     	strb	wzr, [x29, #553]
   56038: f900a7a1     	str	x1, [x29, #328]
   5603c: d2800001     	mov	x1, #0
   56040: f90113bf     	str	xzr, [x29, #544]
   56044: a940e015     	ldp	x21, x24, [x0, #8]
   56048: f9007bb5     	str	x21, [x29, #240]
   5604c: 3d808fa0     	str	q0, [x29, #560]
   56050: 3d8093a0     	str	q0, [x29, #576]
   56054: f85e82a0     	ldur	x0, [x21, #-24]
   56058: f8206a78     	str	x24, [x19, x0]
   5605c: f9407ba0     	ldr	x0, [x29, #240]
   56060: f85e8000     	ldur	x0, [x0, #-24]
   56064: 8b000260     	add	x0, x19, x0
   56068: 97fee636     	bl	0xf940
   5606c: b00004c0     	adrp	x0, 0xef000
   56070: f9002fbc     	str	x28, [x29, #88]
   56074: b00004dc     	adrp	x28, 0xef000
   56078: b00004d7     	adrp	x23, 0xef000
   5607c: 4f000400     	movi	v0.4s, #0
   56080: f947ac01     	ldr	x1, [x0, #3928]
   56084: f947f782     	ldr	x2, [x28, #4072]
   56088: 91010260     	add	x0, x19, #64
   5608c: 91006023     	add	x3, x1, #24
   56090: 91010021     	add	x1, x1, #64
   56094: 91004042     	add	x2, x2, #16
   56098: f9007ba3     	str	x3, [x29, #240]
   5609c: f900a7a1     	str	x1, [x29, #328]
   560a0: f9007fa2     	str	x2, [x29, #248]
   560a4: 3d8043a0     	str	q0, [x29, #256]
   560a8: 3d8047a0     	str	q0, [x29, #272]
   560ac: 3d804ba0     	str	q0, [x29, #288]
   560b0: 97fee5d8     	bl	0xf810
   560b4: b00004c0     	adrp	x0, 0xef000
   560b8: 52800203     	mov	w3, #16
   560bc: 91002261     	add	x1, x19, #8
   560c0: f947a402     	ldr	x2, [x0, #3912]
   560c4: f947fae0     	ldr	x0, [x23, #4080]
   560c8: 91004042     	add	x2, x2, #16
   560cc: b9013ba3     	str	w3, [x29, #312]
   560d0: 91006000     	add	x0, x0, #24
   560d4: f9007fa2     	str	x2, [x29, #248]
   560d8: f900a3a0     	str	x0, [x29, #320]
   560dc: 91016260     	add	x0, x19, #88
   560e0: 97fee618     	bl	0xf940
   560e4: aa1603e1     	mov	x1, x22
   560e8: aa1303e0     	mov	x0, x19
   560ec: 97fee441     	bl	0xf1f0
   560f0: f947fae0     	ldr	x0, [x23, #4080]
   560f4: f94093a1     	ldr	x1, [x29, #288]
   560f8: 91006000     	add	x0, x0, #24
   560fc: f9006ba0     	str	x0, [x29, #208]
   56100: b4003a01     	cbz	x1, 0x56840
   56104: a95103a4     	ldp	x4, x0, [x29, #272]
   56108: 52800003     	mov	w3, #0
   5610c: aa1a03e2     	mov	x2, x26
   56110: eb04003f     	cmp	x1, x4
   56114: 54003469     	b.ls	0x567a0
   56118: 97fee47a     	bl	0xf300
   5611c: f9406ba1     	ldr	x1, [x29, #208]
   56120: d1002022     	sub	x2, x1, #8
   56124: b9400042     	ldr	w2, [x2]
   56128: 37f83b42     	tbnz	w2, #31, 0x56890
   5612c: d1002001     	sub	x1, x0, #8
   56130: b9400021     	ldr	w1, [x1]
   56134: 37f839a1     	tbnz	w1, #31, 0x56868
   56138: f9406ba2     	ldr	x2, [x29, #208]
   5613c: f947fae1     	ldr	x1, [x23, #4080]
   56140: f9006ba0     	str	x0, [x29, #208]
   56144: d1006040     	sub	x0, x2, #24
   56148: eb01001f     	cmp	x0, x1
   5614c: 540040a1     	b.ne	0x56960
   56150: b00004c0     	adrp	x0, 0xef000
   56154: b00004c1     	adrp	x1, 0xef000
   56158: f940a3a3     	ldr	x3, [x29, #320]
   5615c: f947ac00     	ldr	x0, [x0, #3928]
   56160: f947a421     	ldr	x1, [x1, #3912]
   56164: 91006004     	add	x4, x0, #24
   56168: f947fae2     	ldr	x2, [x23, #4080]
   5616c: 91010000     	add	x0, x0, #64
   56170: 91004021     	add	x1, x1, #16
   56174: f900a7a0     	str	x0, [x29, #328]
   56178: d1006060     	sub	x0, x3, #24
   5617c: a90f07a4     	stp	x4, x1, [x29, #240]
   56180: eb02001f     	cmp	x0, x2
   56184: 54003be1     	b.ne	0x56900
   56188: f947f79c     	ldr	x28, [x28, #4072]
   5618c: 91010260     	add	x0, x19, #64
   56190: 9100439c     	add	x28, x28, #16
   56194: f9007fbc     	str	x28, [x29, #248]
   56198: 97fee752     	bl	0xfee0
   5619c: f9007bb5     	str	x21, [x29, #240]
   561a0: f947aa94     	ldr	x20, [x20, #3920]
   561a4: 91016260     	add	x0, x19, #88
   561a8: f85e82a1     	ldur	x1, [x21, #-24]
   561ac: 91004294     	add	x20, x20, #16
   561b0: f8216a78     	str	x24, [x19, x1]
   561b4: f900a7b4     	str	x20, [x29, #328]
   561b8: 97fee606     	bl	0xf9d0
   561bc: aa1a03e2     	mov	x2, x26
   561c0: 91004320     	add	x0, x25, #16
   561c4: d00002e1     	adrp	x1, 0xb4000
   561c8: 911fc021     	add	x1, x1, #2032
   561cc: 97fee481     	bl	0xf3d0
   561d0: 91006320     	add	x0, x25, #24
   561d4: b00002e1     	adrp	x1, 0xb3000
   561d8: aa1a03e2     	mov	x2, x26
   561dc: 91204021     	add	x1, x1, #2064
   561e0: 97fee47c     	bl	0xf3d0
   561e4: f90053bf     	str	xzr, [x29, #160]
   561e8: 4f000400     	movi	v0.4s, #0
   561ec: d2800400     	mov	x0, #32
   561f0: 3d8027a0     	str	q0, [x29, #144]
   561f4: 97fee71f     	bl	0xfe70
   561f8: aa0003f8     	mov	x24, x0
   561fc: b00004c0     	adrp	x0, 0xef000
   56200: f947fafc     	ldr	x28, [x23, #4080]
   56204: 91008301     	add	x1, x24, #32
   56208: d2800013     	mov	x19, #0
   5620c: f947ec00     	ldr	x0, [x0, #4056]
   56210: a90683b8     	stp	x24, x0, [x29, #104]
   56214: f9003fbc     	str	x28, [x29, #120]
   56218: f9004bb8     	str	x24, [x29, #144]
   5621c: f90053a1     	str	x1, [x29, #160]
   56220: 14000008     	b	0x56240
   56224: d503201f     	nop
   56228: eb1c02bf     	cmp	x21, x28
   5622c: 54001e21     	b.ne	0x565f0
   56230: f8008714     	str	x20, [x24], #8
   56234: 91002273     	add	x19, x19, #8
   56238: f100827f     	cmp	x19, #32
   5623c: 54000600     	b.eq	0x562fc
   56240: f8736b34     	ldr	x20, [x25, x19]
   56244: d1006295     	sub	x21, x20, #24
   56248: 910042a0     	add	x0, x21, #16
   5624c: b9400001     	ldr	w1, [x0]
   56250: 36fffec1     	tbz	w1, #31, 0x56228
   56254: a97e8696     	ldp	x22, x1, [x20, #-24]
   56258: 928000c0     	mov	x0, #-7
   5625c: f2e7ffe0     	movk	x0, #16383, lsl #48
   56260: eb0002df     	cmp	x22, x0
   56264: 540069c8     	b.hi	0x56f9c
   56268: eb0102df     	cmp	x22, x1
   5626c: 54000089     	b.ls	0x5627c
   56270: d37ff820     	lsl	x0, x1, #1
   56274: eb0002df     	cmp	x22, x0
   56278: 9a8022d6     	csel	x22, x22, x0, hs
   5627c: 9100e6c0     	add	x0, x22, #57
   56280: f140041f     	cmp	x0, #1, lsl #12         // =4096
   56284: fa568022     	ccmp	x1, x22, #2, hi
   56288: 54000102     	b.hs	0x562a8
   5628c: 914006d6     	add	x22, x22, #1, lsl #12   // =4096
   56290: 92402c00     	and	x0, x0, #0xfff
   56294: cb0002c0     	sub	x0, x22, x0
   56298: 928000c1     	mov	x1, #-7
   5629c: f2e7ffe1     	movk	x1, #16383, lsl #48
   562a0: eb01001f     	cmp	x0, x1
   562a4: 9a819016     	csel	x22, x0, x1, ls
   562a8: 910066c0     	add	x0, x22, #25
   562ac: 97fee6f1     	bl	0xfe70
   562b0: f9000416     	str	x22, [x0, #8]
   562b4: aa0003e3     	mov	x3, x0
   562b8: b900101f     	str	wzr, [x0, #16]
   562bc: 91006004     	add	x4, x0, #24
   562c0: f94002a2     	ldr	x2, [x21]
   562c4: b40000c2     	cbz	x2, 0x562dc
   562c8: f100045f     	cmp	x2, #1
   562cc: 54002a21     	b.ne	0x56810
   562d0: 39400280     	ldrb	w0, [x20]
   562d4: 39006060     	strb	w0, [x3, #24]
   562d8: f94002a2     	ldr	x2, [x21]
   562dc: f9403fa0     	ldr	x0, [x29, #120]
   562e0: eb00007f     	cmp	x3, x0
   562e4: 54002b61     	b.ne	0x56850
   562e8: aa0403f4     	mov	x20, x4
   562ec: f8008714     	str	x20, [x24], #8
   562f0: 91002273     	add	x19, x19, #8
   562f4: f100827f     	cmp	x19, #32
   562f8: 54fffa41     	b.ne	0x56240
   562fc: 910243b3     	add	x19, x29, #144
   56300: 910fe363     	add	x3, x27, #1016
   56304: b00002f4     	adrp	x20, 0xb3000
   56308: aa1303e4     	mov	x4, x19
   5630c: 910a6063     	add	x3, x3, #664
   56310: 52801122     	mov	w2, #137
   56314: 911b6281     	add	x1, x20, #1752
   56318: 52800060     	mov	w0, #3
   5631c: f9004fb8     	str	x24, [x29, #152]
   56320: 97fee5c4     	bl	0xfa30
   56324: 97fee62b     	bl	0xfbd0
   56328: 910223b5     	add	x21, x29, #136
   5632c: aa0003f6     	mov	x22, x0
   56330: 900002e1     	adrp	x1, 0xb2000
   56334: 910203a2     	add	x2, x29, #128
   56338: 9137a021     	add	x1, x1, #3560
   5633c: aa1503e0     	mov	x0, x21
   56340: 97fee424     	bl	0xf3d0
   56344: aa1a03e8     	mov	x8, x26
   56348: 52800060     	mov	w0, #3
   5634c: 97fee3b5     	bl	0xf220
   56350: aa1303e3     	mov	x3, x19
   56354: aa1a03e2     	mov	x2, x26
   56358: aa1503e1     	mov	x1, x21
   5635c: aa1603e0     	mov	x0, x22
   56360: 97fee53c     	bl	0xf850
   56364: a94b57b3     	ldp	x19, x21, [x29, #176]
   56368: b00004d6     	adrp	x22, 0xef000
   5636c: eb15027f     	cmp	x19, x21
   56370: 54000180     	b.eq	0x563a0
   56374: f947eec0     	ldr	x0, [x22, #4056]
   56378: f947faf8     	ldr	x24, [x23, #4080]
   5637c: b4000be0     	cbz	x0, 0x564f8
   56380: f9400261     	ldr	x1, [x19]
   56384: d1006020     	sub	x0, x1, #24
   56388: eb18001f     	cmp	x0, x24
   5638c: 54001f61     	b.ne	0x56778
   56390: 91002273     	add	x19, x19, #8
   56394: eb1302bf     	cmp	x21, x19
   56398: 54ffff41     	b.ne	0x56380
   5639c: f9405bb5     	ldr	x21, [x29, #176]
   563a0: b4000075     	cbz	x21, 0x563ac
   563a4: aa1503e0     	mov	x0, x21
   563a8: 97fee386     	bl	0xf1c0
   563ac: f94047a2     	ldr	x2, [x29, #136]
   563b0: f947fae1     	ldr	x1, [x23, #4080]
   563b4: d1006040     	sub	x0, x2, #24
   563b8: eb01001f     	cmp	x0, x1
   563bc: 54002721     	b.ne	0x568a0
   563c0: a94957b3     	ldp	x19, x21, [x29, #144]
   563c4: eb15027f     	cmp	x19, x21
   563c8: 54000180     	b.eq	0x563f8
   563cc: f947eec0     	ldr	x0, [x22, #4056]
   563d0: f947faf8     	ldr	x24, [x23, #4080]
   563d4: b4000b60     	cbz	x0, 0x56540
   563d8: f9400261     	ldr	x1, [x19]
   563dc: d1006020     	sub	x0, x1, #24
   563e0: eb18001f     	cmp	x0, x24
   563e4: 54001b61     	b.ne	0x56750
   563e8: 91002273     	add	x19, x19, #8
   563ec: eb1302bf     	cmp	x21, x19
   563f0: 54ffff41     	b.ne	0x563d8
   563f4: f9404bb5     	ldr	x21, [x29, #144]
   563f8: b4000075     	cbz	x21, 0x56404
   563fc: aa1503e0     	mov	x0, x21
   56400: 97fee370     	bl	0xf1c0
   56404: f947eec0     	ldr	x0, [x22, #4056]
   56408: f947fae1     	ldr	x1, [x23, #4080]
   5640c: f94073a2     	ldr	x2, [x29, #224]
   56410: b4000bc0     	cbz	x0, 0x56588
   56414: d1006040     	sub	x0, x2, #24
   56418: eb01001f     	cmp	x0, x1
   5641c: 54004401     	b.ne	0x56c9c
   56420: f9406fa2     	ldr	x2, [x29, #216]
   56424: f947fae1     	ldr	x1, [x23, #4080]
   56428: d1006040     	sub	x0, x2, #24
   5642c: eb01001f     	cmp	x0, x1
   56430: 54004561     	b.ne	0x56cdc
   56434: f9406ba2     	ldr	x2, [x29, #208]
   56438: f947fae1     	ldr	x1, [x23, #4080]
   5643c: d1006040     	sub	x0, x2, #24
   56440: eb01001f     	cmp	x0, x1
   56444: 54004a21     	b.ne	0x56d88
   56448: f94067a2     	ldr	x2, [x29, #200]
   5644c: f947fae1     	ldr	x1, [x23, #4080]
   56450: d1006040     	sub	x0, x2, #24
   56454: eb01001f     	cmp	x0, x1
   56458: 54002981     	b.ne	0x56988
   5645c: 97fee479     	bl	0xf640
   56460: 52800021     	mov	w1, #1
   56464: aa1a03e8     	mov	x8, x26
   56468: 72a0e221     	movk	w1, #1809, lsl #16
   5646c: 97fee515     	bl	0xf8c0
   56470: f9405ba4     	ldr	x4, [x29, #176]
   56474: 910fe362     	add	x2, x27, #1016
   56478: 900002e3     	adrp	x3, 0xb2000
   5647c: 910a6042     	add	x2, x2, #664
   56480: 9114e063     	add	x3, x3, #1336
   56484: 52801121     	mov	w1, #137
   56488: 911b6280     	add	x0, x20, #1752
   5648c: 97fee2e1     	bl	0xf010
   56490: 97fee55c     	bl	0xfa00
   56494: 52800020     	mov	w0, #1
   56498: 72a0e220     	movk	w0, #1809, lsl #16
   5649c: 9400d21d     	bl	0x8ad10
   564a0: f9405ba1     	ldr	x1, [x29, #176]
   564a4: 2a0003f4     	mov	w20, w0
   564a8: f947faf7     	ldr	x23, [x23, #4080]
   564ac: d1006020     	sub	x0, x1, #24
   564b0: eb17001f     	cmp	x0, x23
   564b4: 540020e1     	b.ne	0x568d0
   564b8: b00004c0     	adrp	x0, 0xef000
   564bc: f947c401     	ldr	x1, [x0, #3976]
   564c0: 2a1403e0     	mov	w0, w20
   564c4: f9412fa2     	ldr	x2, [x29, #600]
   564c8: f9400021     	ldr	x1, [x1]
   564cc: ca010041     	eor	x1, x2, x1
   564d0: b5005881     	cbnz	x1, 0x56fe0
   564d4: a94153f3     	ldp	x19, x20, [sp, #16]
   564d8: a9425bf5     	ldp	x21, x22, [sp, #32]
   564dc: a94363f7     	ldp	x23, x24, [sp, #48]
   564e0: a9446bf9     	ldp	x25, x26, [sp, #64]
   564e4: f9402fbc     	ldr	x28, [x29, #88]
   564e8: f9402bfb     	ldr	x27, [sp, #80]
   564ec: a9407bfd     	ldp	x29, x30, [sp]
   564f0: 910983ff     	add	sp, sp, #608
   564f4: d65f03c0     	ret
   564f8: f9400261     	ldr	x1, [x19]
   564fc: d1006020     	sub	x0, x1, #24
   56500: eb18001f     	cmp	x0, x24
   56504: 54000101     	b.ne	0x56524
   56508: 91002273     	add	x19, x19, #8
   5650c: eb1302bf     	cmp	x21, x19
   56510: 54fff460     	b.eq	0x5639c
   56514: f9400261     	ldr	x1, [x19]
   56518: d1006020     	sub	x0, x1, #24
   5651c: eb18001f     	cmp	x0, x24
   56520: 54ffff40     	b.eq	0x56508
   56524: b85f8022     	ldur	w2, [x1, #-8]
   56528: 51000443     	sub	w3, w2, #1
   5652c: b81f8023     	stur	w3, [x1, #-8]
   56530: 7100005f     	cmp	w2, #0
   56534: 54fffeac     	b.gt	0x56508
   56538: 97fee322     	bl	0xf1c0
   5653c: 17fffff3     	b	0x56508
   56540: f9400261     	ldr	x1, [x19]
   56544: d1006020     	sub	x0, x1, #24
   56548: eb18001f     	cmp	x0, x24
   5654c: 54000101     	b.ne	0x5656c
   56550: 91002273     	add	x19, x19, #8
   56554: eb1302bf     	cmp	x21, x19
   56558: 54fff4e0     	b.eq	0x563f4
   5655c: f9400261     	ldr	x1, [x19]
   56560: d1006020     	sub	x0, x1, #24
   56564: eb18001f     	cmp	x0, x24
   56568: 54ffff40     	b.eq	0x56550
   5656c: b85f8022     	ldur	w2, [x1, #-8]
   56570: 51000443     	sub	w3, w2, #1
   56574: b81f8023     	stur	w3, [x1, #-8]
   56578: 7100005f     	cmp	w2, #0
   5657c: 54fffeac     	b.gt	0x56550
   56580: 97fee310     	bl	0xf1c0
   56584: 17fffff3     	b	0x56550
   56588: d1006040     	sub	x0, x2, #24
   5658c: eb01001f     	cmp	x0, x1
   56590: 54003981     	b.ne	0x56cc0
   56594: f9406fa2     	ldr	x2, [x29, #216]
   56598: f947fae1     	ldr	x1, [x23, #4080]
   5659c: d1006040     	sub	x0, x2, #24
   565a0: eb01001f     	cmp	x0, x1
   565a4: 54003c81     	b.ne	0x56d34
   565a8: f9406ba2     	ldr	x2, [x29, #208]
   565ac: f947fae1     	ldr	x1, [x23, #4080]
   565b0: d1006040     	sub	x0, x2, #24
   565b4: eb01001f     	cmp	x0, x1
   565b8: 54004161     	b.ne	0x56de4
   565bc: f94067a2     	ldr	x2, [x29, #200]
   565c0: f947fae1     	ldr	x1, [x23, #4080]
   565c4: d1006040     	sub	x0, x2, #24
   565c8: eb01001f     	cmp	x0, x1
   565cc: 54fff480     	b.eq	0x5645c
   565d0: b85f8041     	ldur	w1, [x2, #-8]
   565d4: 51000423     	sub	w3, w1, #1
   565d8: b81f8043     	stur	w3, [x2, #-8]
   565dc: 7100003f     	cmp	w1, #0
   565e0: 54fff3ec     	b.gt	0x5645c
   565e4: 97fee2f7     	bl	0xf1c0
   565e8: 17ffff9d     	b	0x5645c
   565ec: d503201f     	nop
   565f0: f9403ba1     	ldr	x1, [x29, #112]
   565f4: b40011e1     	cbz	x1, 0x56830
   565f8: 885ffc01     	ldaxr	w1, [x0]
   565fc: 11000421     	add	w1, w1, #1
   56600: 8802fc01     	stlxr	w2, w1, [x0]
   56604: 34ffe162     	cbz	w2, 0x56230
   56608: 17fffffc     	b	0x565f8
   5660c: d503201f     	nop
   56610: f9002fbc     	str	x28, [x29, #88]
   56614: 52800000     	mov	w0, #0
   56618: 940159ec     	bl	0xacdc8
   5661c: aa0003fc     	mov	x28, x0
   56620: b4003700     	cbz	x0, 0x56d00
   56624: b4001c73     	cbz	x19, 0x569b0
   56628: b940ce60     	ldr	w0, [x19, #204]
   5662c: a90b7fbf     	stp	xzr, xzr, [x29, #176]
   56630: a90cffbf     	stp	xzr, xzr, [x29, #200]
   56634: 7100041f     	cmp	w0, #1
   56638: a90dffbf     	stp	xzr, xzr, [x29, #216]
   5663c: a90f7fbf     	stp	xzr, xzr, [x29, #240]
   56640: a9107fbf     	stp	xzr, xzr, [x29, #256]
   56644: a9117fbf     	stp	xzr, xzr, [x29, #272]
   56648: a9127fbf     	stp	xzr, xzr, [x29, #288]
   5664c: f90077bf     	str	xzr, [x29, #232]
   56650: f9009bbf     	str	xzr, [x29, #304]
   56654: 54001120     	b.eq	0x56878
   56658: 9103c3a0     	add	x0, x29, #240
   5665c: 52800021     	mov	w1, #1
   56660: f9005fa0     	str	x0, [x29, #184]
   56664: 52800040     	mov	w0, #2
   56668: 390463a1     	strb	w1, [x29, #280]
   5666c: f9007bb8     	str	x24, [x29, #240]
   56670: b9010bb7     	str	w23, [x29, #264]
   56674: 910fe376     	add	x22, x27, #1016
   56678: b900b3a0     	str	w0, [x29, #176]
   5667c: 910a62d6     	add	x22, x22, #664
   56680: 910223a1     	add	x1, x29, #136
   56684: aa1603e2     	mov	x2, x22
   56688: f90047bf     	str	xzr, [x29, #136]
   5668c: aa1503e0     	mov	x0, x21
   56690: 9400ce9e     	bl	0x8a108
   56694: 2a0003f4     	mov	w20, w0
   56698: 35fff100     	cbnz	w0, 0x564b8
   5669c: f9400386     	ldr	x6, [x28]
   566a0: aa1303e1     	mov	x1, x19
   566a4: f94047a4     	ldr	x4, [x29, #136]
   566a8: aa1a03e5     	mov	x5, x26
   566ac: 9102c3a3     	add	x3, x29, #176
   566b0: 2a1903e2     	mov	w2, w25
   566b4: aa1c03e0     	mov	x0, x28
   566b8: f94050c6     	ldr	x6, [x6, #160]
   566bc: d63f00c0     	blr	x6
   566c0: 2a0003f3     	mov	w19, w0
   566c4: 52800120     	mov	w0, #9
   566c8: 72a0e100     	movk	w0, #1800, lsl #16
   566cc: 6b00027f     	cmp	w19, w0
   566d0: 54003840     	b.eq	0x56dd8
   566d4: 528001a0     	mov	w0, #13
   566d8: 72a0e220     	movk	w0, #1809, lsl #16
   566dc: 6b00027f     	cmp	w19, w0
   566e0: 54003900     	b.eq	0x56e00
   566e4: 34ffeeb3     	cbz	w19, 0x564b8
   566e8: 97fee3d6     	bl	0xf640
   566ec: 2a1303e1     	mov	w1, w19
   566f0: 910243a8     	add	x8, x29, #144
   566f4: 97fee473     	bl	0xf8c0
   566f8: f9404ba4     	ldr	x4, [x29, #144]
   566fc: 900002e3     	adrp	x3, 0xb2000
   56700: b00002e0     	adrp	x0, 0xb3000
   56704: 9114e063     	add	x3, x3, #1336
   56708: aa1603e2     	mov	x2, x22
   5670c: 528014e1     	mov	w1, #167
   56710: 911b6000     	add	x0, x0, #1752
   56714: 97fee23f     	bl	0xf010
   56718: 2a1303e0     	mov	w0, w19
   5671c: aa1603e1     	mov	x1, x22
   56720: 97fee500     	bl	0xfb20
   56724: 97fee4b7     	bl	0xfa00
   56728: b00004c1     	adrp	x1, 0xef000
   5672c: f9404ba2     	ldr	x2, [x29, #144]
   56730: f947f821     	ldr	x1, [x1, #4080]
   56734: d1006040     	sub	x0, x2, #24
   56738: eb01001f     	cmp	x0, x1
   5673c: 54003e81     	b.ne	0x56f0c
   56740: 2a1303e0     	mov	w0, w19
   56744: 9400d173     	bl	0x8ad10
   56748: 2a0003f4     	mov	w20, w0
   5674c: 17ffff5b     	b	0x564b8
   56750: d1002021     	sub	x1, x1, #8
   56754: 885ffc22     	ldaxr	w2, [x1]
   56758: 51000443     	sub	w3, w2, #1
   5675c: 8804fc23     	stlxr	w4, w3, [x1]
   56760: 35ffffa4     	cbnz	w4, 0x56754
   56764: 7100005f     	cmp	w2, #0
   56768: 54ffe40c     	b.gt	0x563e8
   5676c: 97fee295     	bl	0xf1c0
   56770: 17ffff1e     	b	0x563e8
   56774: d503201f     	nop
   56778: d1002021     	sub	x1, x1, #8
   5677c: 885ffc22     	ldaxr	w2, [x1]
   56780: 51000443     	sub	w3, w2, #1
   56784: 8804fc23     	stlxr	w4, w3, [x1]
   56788: 35ffffa4     	cbnz	w4, 0x5677c
   5678c: 7100005f     	cmp	w2, #0
   56790: 54ffe00c     	b.gt	0x56390
   56794: 97fee28b     	bl	0xf1c0
   56798: 17fffefe     	b	0x56390
   5679c: d503201f     	nop
   567a0: aa0403e1     	mov	x1, x4
   567a4: 97fee2d7     	bl	0xf300
   567a8: f9406ba1     	ldr	x1, [x29, #208]
   567ac: d1002022     	sub	x2, x1, #8
   567b0: b9400042     	ldr	w2, [x2]
   567b4: 37f80722     	tbnz	w2, #31, 0x56898
   567b8: d1002001     	sub	x1, x0, #8
   567bc: b9400021     	ldr	w1, [x1]
   567c0: 37f80581     	tbnz	w1, #31, 0x56870
   567c4: f9406ba2     	ldr	x2, [x29, #208]
   567c8: f947fae1     	ldr	x1, [x23, #4080]
   567cc: f9006ba0     	str	x0, [x29, #208]
   567d0: d1006040     	sub	x0, x2, #24
   567d4: eb01001f     	cmp	x0, x1
   567d8: 54ffcbc0     	b.eq	0x56150
   567dc: b00004d6     	adrp	x22, 0xef000
   567e0: f947eed6     	ldr	x22, [x22, #4056]
   567e4: b4002f36     	cbz	x22, 0x56dc8
   567e8: d1002042     	sub	x2, x2, #8
   567ec: 885ffc41     	ldaxr	w1, [x2]
   567f0: 51000423     	sub	w3, w1, #1
   567f4: 8804fc43     	stlxr	w4, w3, [x2]
   567f8: 35ffffa4     	cbnz	w4, 0x567ec
   567fc: 7100003f     	cmp	w1, #0
   56800: 54ffca8c     	b.gt	0x56150
   56804: 97fee26f     	bl	0xf1c0
   56808: 17fffe52     	b	0x56150
   5680c: d503201f     	nop
   56810: aa0403e0     	mov	x0, x4
   56814: f90033a3     	str	x3, [x29, #96]
   56818: aa1403e1     	mov	x1, x20
   5681c: 97fee2c5     	bl	0xf330
   56820: f94002a2     	ldr	x2, [x21]
   56824: aa0003e4     	mov	x4, x0
   56828: f94033a3     	ldr	x3, [x29, #96]
   5682c: 17fffeac     	b	0x562dc
   56830: b85f8280     	ldur	w0, [x20, #-8]
   56834: 11000400     	add	w0, w0, #1
   56838: b81f8280     	stur	w0, [x20, #-8]
   5683c: 17fffe7d     	b	0x56230
   56840: 91014261     	add	x1, x19, #80
   56844: 91002320     	add	x0, x25, #8
   56848: 97fee4da     	bl	0xfbb0
   5684c: 17fffe41     	b	0x56150
   56850: 8b020060     	add	x0, x3, x2
   56854: f9000062     	str	x2, [x3]
   56858: aa0403f4     	mov	x20, x4
   5685c: 3900601f     	strb	wzr, [x0, #24]
   56860: 17fffea3     	b	0x562ec
   56864: d503201f     	nop
   56868: b81f801f     	stur	wzr, [x0, #-8]
   5686c: 17fffe33     	b	0x56138
   56870: b81f801f     	stur	wzr, [x0, #-8]
   56874: 17ffffd4     	b	0x567c4
   56878: 910323a1     	add	x1, x29, #200
   5687c: 39039fa0     	strb	w0, [x29, #231]
   56880: f9005fa1     	str	x1, [x29, #184]
   56884: f90067b8     	str	x24, [x29, #200]
   56888: b900dbb7     	str	w23, [x29, #216]
   5688c: 17ffff7a     	b	0x56674
   56890: b81f803f     	stur	wzr, [x1, #-8]
   56894: 17fffe26     	b	0x5612c
   56898: b81f803f     	stur	wzr, [x1, #-8]
   5689c: 17ffffc7     	b	0x567b8
   568a0: f947eec1     	ldr	x1, [x22, #4056]
   568a4: b4000461     	cbz	x1, 0x56930
   568a8: d1002042     	sub	x2, x2, #8
   568ac: 885ffc41     	ldaxr	w1, [x2]
   568b0: 51000423     	sub	w3, w1, #1
   568b4: 8804fc43     	stlxr	w4, w3, [x2]
   568b8: 35ffffa4     	cbnz	w4, 0x568ac
   568bc: 7100003f     	cmp	w1, #0
   568c0: 54ffd80c     	b.gt	0x563c0
   568c4: 97fee23f     	bl	0xf1c0
   568c8: 17fffebe     	b	0x563c0
   568cc: d503201f     	nop
   568d0: f947eed6     	ldr	x22, [x22, #4056]
   568d4: b4000376     	cbz	x22, 0x56940
   568d8: d1002021     	sub	x1, x1, #8
   568dc: 885ffc22     	ldaxr	w2, [x1]
   568e0: 51000443     	sub	w3, w2, #1
   568e4: 8804fc23     	stlxr	w4, w3, [x1]
   568e8: 35ffffa4     	cbnz	w4, 0x568dc
   568ec: 7100005f     	cmp	w2, #0
   568f0: 54ffde4c     	b.gt	0x564b8
   568f4: 97fee233     	bl	0xf1c0
   568f8: 17fffef0     	b	0x564b8
   568fc: d503201f     	nop
   56900: b00004d6     	adrp	x22, 0xef000
   56904: f947eed6     	ldr	x22, [x22, #4056]
   56908: b4000256     	cbz	x22, 0x56950
   5690c: d1002063     	sub	x3, x3, #8
   56910: 885ffc61     	ldaxr	w1, [x3]
   56914: 51000422     	sub	w2, w1, #1
   56918: 8804fc62     	stlxr	w4, w2, [x3]
   5691c: 35ffffa4     	cbnz	w4, 0x56910
   56920: 7100003f     	cmp	w1, #0
   56924: 54ffc32c     	b.gt	0x56188
   56928: 97fee226     	bl	0xf1c0
   5692c: 17fffe17     	b	0x56188
   56930: b85f8041     	ldur	w1, [x2, #-8]
   56934: 51000423     	sub	w3, w1, #1
   56938: b81f8043     	stur	w3, [x2, #-8]
   5693c: 17ffffe0     	b	0x568bc
   56940: b85f8022     	ldur	w2, [x1, #-8]
   56944: 51000443     	sub	w3, w2, #1
   56948: b81f8023     	stur	w3, [x1, #-8]
   5694c: 17ffffe8     	b	0x568ec
   56950: b85f8061     	ldur	w1, [x3, #-8]
   56954: 51000422     	sub	w2, w1, #1
   56958: b81f8062     	stur	w2, [x3, #-8]
   5695c: 17fffff1     	b	0x56920
   56960: b00004d6     	adrp	x22, 0xef000
   56964: f947eed6     	ldr	x22, [x22, #4056]
   56968: b4002316     	cbz	x22, 0x56dc8
   5696c: d1002042     	sub	x2, x2, #8
   56970: 885ffc41     	ldaxr	w1, [x2]
   56974: 51000423     	sub	w3, w1, #1
   56978: 8804fc43     	stlxr	w4, w3, [x2]
   5697c: 34fff404     	cbz	w4, 0x567fc
   56980: 17fffffc     	b	0x56970
   56984: d503201f     	nop
   56988: d1002042     	sub	x2, x2, #8
   5698c: 885ffc41     	ldaxr	w1, [x2]
   56990: 51000423     	sub	w3, w1, #1
   56994: 8804fc43     	stlxr	w4, w3, [x2]
   56998: 35ffffa4     	cbnz	w4, 0x5698c
   5699c: 7100003f     	cmp	w1, #0
   569a0: 54ffd5ec     	b.gt	0x5645c
   569a4: 97fee207     	bl	0xf1c0
   569a8: 17fffead     	b	0x5645c
   569ac: d503201f     	nop
   569b0: 9102c3ba     	add	x26, x29, #176
   569b4: 9103c3b3     	add	x19, x29, #240
   569b8: aa1a03e2     	mov	x2, x26
   569bc: aa1403e1     	mov	x1, x20
   569c0: aa1303e0     	mov	x0, x19
   569c4: 97fee283     	bl	0xf3d0
   569c8: 91002260     	add	x0, x19, #8
   569cc: b00002e1     	adrp	x1, 0xb3000
   569d0: aa1a03e2     	mov	x2, x26
   569d4: 911ba021     	add	x1, x1, #1768
   569d8: 97fee27e     	bl	0xf3d0
   569dc: f90053bf     	str	xzr, [x29, #160]
   569e0: 4f000400     	movi	v0.4s, #0
   569e4: d2800200     	mov	x0, #16
   569e8: 3d8027a0     	str	q0, [x29, #144]
   569ec: 97fee521     	bl	0xfe70
   569f0: b00004d7     	adrp	x23, 0xef000
   569f4: b00004c1     	adrp	x1, 0xef000
   569f8: aa0003f9     	mov	x25, x0
   569fc: f947fafc     	ldr	x28, [x23, #4080]
   56a00: d2800014     	mov	x20, #0
   56a04: f947ec21     	ldr	x1, [x1, #4056]
   56a08: a90687a0     	stp	x0, x1, [x29, #104]
   56a0c: f9004ba0     	str	x0, [x29, #144]
   56a10: 91004000     	add	x0, x0, #16
   56a14: f9003fbc     	str	x28, [x29, #120]
   56a18: f90053a0     	str	x0, [x29, #160]
   56a1c: 14000007     	b	0x56a38
   56a20: eb1c02df     	cmp	x22, x28
   56a24: 54001c41     	b.ne	0x56dac
   56a28: f8008735     	str	x21, [x25], #8
   56a2c: 91002294     	add	x20, x20, #8
   56a30: f100429f     	cmp	x20, #16
   56a34: 54000660     	b.eq	0x56b00
   56a38: f8746a75     	ldr	x21, [x19, x20]
   56a3c: d10062b6     	sub	x22, x21, #24
   56a40: 910042c0     	add	x0, x22, #16
   56a44: b9400001     	ldr	w1, [x0]
   56a48: 36fffec1     	tbz	w1, #31, 0x56a20
   56a4c: a97e86b8     	ldp	x24, x1, [x21, #-24]
   56a50: 928000c0     	mov	x0, #-7
   56a54: f2e7ffe0     	movk	x0, #16383, lsl #48
   56a58: eb00031f     	cmp	x24, x0
   56a5c: 54003608     	b.hi	0x5711c
   56a60: eb01031f     	cmp	x24, x1
   56a64: 54000089     	b.ls	0x56a74
   56a68: d37ff820     	lsl	x0, x1, #1
   56a6c: eb00031f     	cmp	x24, x0
   56a70: 9a802318     	csel	x24, x24, x0, hs
   56a74: 9100e700     	add	x0, x24, #57
   56a78: f140041f     	cmp	x0, #1, lsl #12         // =4096
   56a7c: fa588022     	ccmp	x1, x24, #2, hi
   56a80: 54000102     	b.hs	0x56aa0
   56a84: 91400718     	add	x24, x24, #1, lsl #12   // =4096
   56a88: 92402c00     	and	x0, x0, #0xfff
   56a8c: cb000300     	sub	x0, x24, x0
   56a90: 928000c1     	mov	x1, #-7
   56a94: f2e7ffe1     	movk	x1, #16383, lsl #48
   56a98: eb01001f     	cmp	x0, x1
   56a9c: 9a819018     	csel	x24, x0, x1, ls
   56aa0: 91006700     	add	x0, x24, #25
   56aa4: 97fee4f3     	bl	0xfe70
   56aa8: f9000418     	str	x24, [x0, #8]
   56aac: aa0003e3     	mov	x3, x0
   56ab0: b900101f     	str	wzr, [x0, #16]
   56ab4: 91006004     	add	x4, x0, #24
   56ab8: f94002c2     	ldr	x2, [x22]
   56abc: b40000c2     	cbz	x2, 0x56ad4
   56ac0: f100045f     	cmp	x2, #1
   56ac4: 54001b61     	b.ne	0x56e30
   56ac8: 394002a0     	ldrb	w0, [x21]
   56acc: 39006060     	strb	w0, [x3, #24]
   56ad0: f94002c2     	ldr	x2, [x22]
   56ad4: f9403fa0     	ldr	x0, [x29, #120]
   56ad8: eb00007f     	cmp	x3, x0
   56adc: 54000061     	b.ne	0x56ae8
   56ae0: aa0403f5     	mov	x21, x4
   56ae4: 17ffffd1     	b	0x56a28
   56ae8: 8b020060     	add	x0, x3, x2
   56aec: f9000062     	str	x2, [x3]
   56af0: aa0403f5     	mov	x21, x4
   56af4: 3900601f     	strb	wzr, [x0, #24]
   56af8: 17ffffcc     	b	0x56a28
   56afc: d503201f     	nop
   56b00: 910243b3     	add	x19, x29, #144
   56b04: 910fe363     	add	x3, x27, #1016
   56b08: b00002e1     	adrp	x1, 0xb3000
   56b0c: aa1303e4     	mov	x4, x19
   56b10: 910a6063     	add	x3, x3, #664
   56b14: 52801182     	mov	w2, #140
   56b18: 911b6021     	add	x1, x1, #1752
   56b1c: 52800080     	mov	w0, #4
   56b20: f9004fb9     	str	x25, [x29, #152]
   56b24: 97fee3c3     	bl	0xfa30
   56b28: 97fee42a     	bl	0xfbd0
   56b2c: 910223b5     	add	x21, x29, #136
   56b30: aa0003f4     	mov	x20, x0
   56b34: 900002e1     	adrp	x1, 0xb2000
   56b38: 910203a2     	add	x2, x29, #128
   56b3c: 91340021     	add	x1, x1, #3328
   56b40: aa1503e0     	mov	x0, x21
   56b44: 97fee223     	bl	0xf3d0
   56b48: aa1a03e8     	mov	x8, x26
   56b4c: 52800080     	mov	w0, #4
   56b50: 97fee1b4     	bl	0xf220
   56b54: aa1303e3     	mov	x3, x19
   56b58: aa1a03e2     	mov	x2, x26
   56b5c: aa1503e1     	mov	x1, x21
   56b60: aa1403e0     	mov	x0, x20
   56b64: 97fee33b     	bl	0xf850
   56b68: a94b53b3     	ldp	x19, x20, [x29, #176]
   56b6c: b00004d6     	adrp	x22, 0xef000
   56b70: eb14027f     	cmp	x19, x20
   56b74: 54000180     	b.eq	0x56ba4
   56b78: f947eec0     	ldr	x0, [x22, #4056]
   56b7c: f947faf5     	ldr	x21, [x23, #4080]
   56b80: b4000680     	cbz	x0, 0x56c50
   56b84: f9400261     	ldr	x1, [x19]
   56b88: d1006020     	sub	x0, x1, #24
   56b8c: eb15001f     	cmp	x0, x21
   56b90: 540013e1     	b.ne	0x56e0c
   56b94: 91002273     	add	x19, x19, #8
   56b98: eb13029f     	cmp	x20, x19
   56b9c: 54ffff41     	b.ne	0x56b84
   56ba0: f9405bb4     	ldr	x20, [x29, #176]
   56ba4: b4000074     	cbz	x20, 0x56bb0
   56ba8: aa1403e0     	mov	x0, x20
   56bac: 97fee185     	bl	0xf1c0
   56bb0: f94047a2     	ldr	x2, [x29, #136]
   56bb4: f947fae1     	ldr	x1, [x23, #4080]
   56bb8: d1006040     	sub	x0, x2, #24
   56bbc: eb01001f     	cmp	x0, x1
   56bc0: 540017e1     	b.ne	0x56ebc
   56bc4: a94953b3     	ldp	x19, x20, [x29, #144]
   56bc8: eb14027f     	cmp	x19, x20
   56bcc: 54000180     	b.eq	0x56bfc
   56bd0: f947eec0     	ldr	x0, [x22, #4056]
   56bd4: f947faf5     	ldr	x21, [x23, #4080]
   56bd8: b4000500     	cbz	x0, 0x56c78
   56bdc: f9400261     	ldr	x1, [x19]
   56be0: d1006020     	sub	x0, x1, #24
   56be4: eb15001f     	cmp	x0, x21
   56be8: 54001341     	b.ne	0x56e50
   56bec: 91002273     	add	x19, x19, #8
   56bf0: eb13029f     	cmp	x20, x19
   56bf4: 54ffff41     	b.ne	0x56bdc
   56bf8: f9404bb4     	ldr	x20, [x29, #144]
   56bfc: b4000074     	cbz	x20, 0x56c08
   56c00: aa1403e0     	mov	x0, x20
   56c04: 97fee16f     	bl	0xf1c0
   56c08: f947eed6     	ldr	x22, [x22, #4056]
   56c0c: f9407fa2     	ldr	x2, [x29, #248]
   56c10: f947fae1     	ldr	x1, [x23, #4080]
   56c14: d1006040     	sub	x0, x2, #24
   56c18: eb01001f     	cmp	x0, x1
   56c1c: b40009b6     	cbz	x22, 0x56d50
   56c20: 54001641     	b.ne	0x56ee8
   56c24: f9407ba1     	ldr	x1, [x29, #240]
   56c28: f947faf7     	ldr	x23, [x23, #4080]
   56c2c: d1006020     	sub	x0, x1, #24
   56c30: eb17001f     	cmp	x0, x23
   56c34: 540019a1     	b.ne	0x56f68
   56c38: 97fee372     	bl	0xfa00
   56c3c: 52800020     	mov	w0, #1
   56c40: 72a0e220     	movk	w0, #1809, lsl #16
   56c44: 9400d033     	bl	0x8ad10
   56c48: 2a0003f4     	mov	w20, w0
   56c4c: 17fffe1b     	b	0x564b8
   56c50: f9400261     	ldr	x1, [x19]
   56c54: d1006020     	sub	x0, x1, #24
   56c58: eb15001f     	cmp	x0, x21
   56c5c: 540010c1     	b.ne	0x56e74
   56c60: 91002273     	add	x19, x19, #8
   56c64: eb14027f     	cmp	x19, x20
   56c68: 54ffff41     	b.ne	0x56c50
   56c6c: f9405bb4     	ldr	x20, [x29, #176]
   56c70: 17ffffcd     	b	0x56ba4
   56c74: d503201f     	nop
   56c78: f9400261     	ldr	x1, [x19]
   56c7c: d1006020     	sub	x0, x1, #24
   56c80: eb15001f     	cmp	x0, x21
   56c84: 54001061     	b.ne	0x56e90
   56c88: 91002273     	add	x19, x19, #8
   56c8c: eb14027f     	cmp	x19, x20
   56c90: 54ffff41     	b.ne	0x56c78
   56c94: f9404bb4     	ldr	x20, [x29, #144]
   56c98: 17ffffd9     	b	0x56bfc
   56c9c: d1002042     	sub	x2, x2, #8
   56ca0: 885ffc41     	ldaxr	w1, [x2]
   56ca4: 51000423     	sub	w3, w1, #1
   56ca8: 8804fc43     	stlxr	w4, w3, [x2]
   56cac: 35ffffa4     	cbnz	w4, 0x56ca0
   56cb0: 7100003f     	cmp	w1, #0
   56cb4: 54ffbb6c     	b.gt	0x56420
   56cb8: 97fee142     	bl	0xf1c0
   56cbc: 17fffdd9     	b	0x56420
   56cc0: b85f8041     	ldur	w1, [x2, #-8]
   56cc4: 51000423     	sub	w3, w1, #1
   56cc8: b81f8043     	stur	w3, [x2, #-8]
   56ccc: 7100003f     	cmp	w1, #0
   56cd0: 54ffc62c     	b.gt	0x56594
   56cd4: 97fee13b     	bl	0xf1c0
   56cd8: 17fffe2f     	b	0x56594
   56cdc: d1002042     	sub	x2, x2, #8
   56ce0: 885ffc41     	ldaxr	w1, [x2]
   56ce4: 51000423     	sub	w3, w1, #1
   56ce8: 8804fc43     	stlxr	w4, w3, [x2]
   56cec: 35ffffa4     	cbnz	w4, 0x56ce0
   56cf0: 7100003f     	cmp	w1, #0
   56cf4: 54ffba0c     	b.gt	0x56434
   56cf8: 97fee132     	bl	0xf1c0
   56cfc: 17fffdce     	b	0x56434
   56d00: aa1403e2     	mov	x2, x20
   56d04: 900002e3     	adrp	x3, 0xb2000
   56d08: b00002e0     	adrp	x0, 0xb3000
   56d0c: 91140063     	add	x3, x3, #1280
   56d10: 911b6000     	add	x0, x0, #1752
   56d14: 52978f04     	mov	w4, #48248
   56d18: 72a000e4     	movk	w4, #7, lsl #16
   56d1c: 52801161     	mov	w1, #139
   56d20: 52978f14     	mov	w20, #48248
   56d24: 72a000f4     	movk	w20, #7, lsl #16
   56d28: 97fee0ba     	bl	0xf010
   56d2c: 97fee335     	bl	0xfa00
   56d30: 17fffde2     	b	0x564b8
   56d34: b85f8041     	ldur	w1, [x2, #-8]
   56d38: 51000423     	sub	w3, w1, #1
   56d3c: b81f8043     	stur	w3, [x2, #-8]
   56d40: 7100003f     	cmp	w1, #0
   56d44: 54ffc32c     	b.gt	0x565a8
   56d48: 97fee11e     	bl	0xf1c0
   56d4c: 17fffe17     	b	0x565a8
   56d50: 54000fe1     	b.ne	0x56f4c
   56d54: f9407ba1     	ldr	x1, [x29, #240]
   56d58: f947faf7     	ldr	x23, [x23, #4080]
   56d5c: d1006020     	sub	x0, x1, #24
   56d60: eb17001f     	cmp	x0, x23
   56d64: 54fff6a0     	b.eq	0x56c38
   56d68: b85f8022     	ldur	w2, [x1, #-8]
   56d6c: 51000443     	sub	w3, w2, #1
   56d70: b81f8023     	stur	w3, [x1, #-8]
   56d74: 7100005f     	cmp	w2, #0
   56d78: 54fff60c     	b.gt	0x56c38
   56d7c: 97fee111     	bl	0xf1c0
   56d80: 17ffffae     	b	0x56c38
   56d84: d503201f     	nop
   56d88: d1002042     	sub	x2, x2, #8
   56d8c: 885ffc41     	ldaxr	w1, [x2]
   56d90: 51000423     	sub	w3, w1, #1
   56d94: 8804fc43     	stlxr	w4, w3, [x2]
   56d98: 35ffffa4     	cbnz	w4, 0x56d8c
   56d9c: 7100003f     	cmp	w1, #0
   56da0: 54ffb54c     	b.gt	0x56448
   56da4: 97fee107     	bl	0xf1c0
   56da8: 17fffda8     	b	0x56448
   56dac: f9403ba1     	ldr	x1, [x29, #112]
   56db0: b40007e1     	cbz	x1, 0x56eac
   56db4: 885ffc01     	ldaxr	w1, [x0]
   56db8: 11000421     	add	w1, w1, #1
   56dbc: 8802fc01     	stlxr	w2, w1, [x0]
   56dc0: 34ffe342     	cbz	w2, 0x56a28
   56dc4: 17fffffc     	b	0x56db4
   56dc8: b85f8041     	ldur	w1, [x2, #-8]
   56dcc: 51000423     	sub	w3, w1, #1
   56dd0: b81f8043     	stur	w3, [x2, #-8]
   56dd4: 17fffe8a     	b	0x567fc
   56dd8: 52944134     	mov	w20, #41481
   56ddc: 72a00034     	movk	w20, #1, lsl #16
   56de0: 17fffdb6     	b	0x564b8
   56de4: b85f8041     	ldur	w1, [x2, #-8]
   56de8: 51000423     	sub	w3, w1, #1
   56dec: b81f8043     	stur	w3, [x2, #-8]
   56df0: 7100003f     	cmp	w1, #0
   56df4: 54ffbe4c     	b.gt	0x565bc
   56df8: 97fee0f2     	bl	0xf1c0
   56dfc: 17fffdf0     	b	0x565bc
   56e00: 52851314     	mov	w20, #10392
   56e04: 72a00074     	movk	w20, #3, lsl #16
   56e08: 17fffdac     	b	0x564b8
   56e0c: d1002021     	sub	x1, x1, #8
   56e10: 885ffc22     	ldaxr	w2, [x1]
   56e14: 51000443     	sub	w3, w2, #1
   56e18: 8804fc23     	stlxr	w4, w3, [x1]
   56e1c: 35ffffa4     	cbnz	w4, 0x56e10
   56e20: 7100005f     	cmp	w2, #0
   56e24: 54ffeb8c     	b.gt	0x56b94
   56e28: 97fee0e6     	bl	0xf1c0
   56e2c: 17ffff5a     	b	0x56b94
   56e30: aa0403e0     	mov	x0, x4
   56e34: f90033a3     	str	x3, [x29, #96]
   56e38: aa1503e1     	mov	x1, x21
   56e3c: 97fee13d     	bl	0xf330
   56e40: f94002c2     	ldr	x2, [x22]
   56e44: aa0003e4     	mov	x4, x0
   56e48: f94033a3     	ldr	x3, [x29, #96]
   56e4c: 17ffff22     	b	0x56ad4
   56e50: d1002021     	sub	x1, x1, #8
   56e54: 885ffc22     	ldaxr	w2, [x1]
   56e58: 51000443     	sub	w3, w2, #1
   56e5c: 8804fc23     	stlxr	w4, w3, [x1]
   56e60: 35ffffa4     	cbnz	w4, 0x56e54
   56e64: 7100005f     	cmp	w2, #0
   56e68: 54ffec2c     	b.gt	0x56bec
   56e6c: 97fee0d5     	bl	0xf1c0
   56e70: 17ffff5f     	b	0x56bec
   56e74: b85f8022     	ldur	w2, [x1, #-8]
   56e78: 51000443     	sub	w3, w2, #1
   56e7c: b81f8023     	stur	w3, [x1, #-8]
   56e80: 7100005f     	cmp	w2, #0
   56e84: 54ffeeec     	b.gt	0x56c60
   56e88: 97fee0ce     	bl	0xf1c0
   56e8c: 17ffff75     	b	0x56c60
   56e90: b85f8022     	ldur	w2, [x1, #-8]
   56e94: 51000443     	sub	w3, w2, #1
   56e98: b81f8023     	stur	w3, [x1, #-8]
   56e9c: 7100005f     	cmp	w2, #0
   56ea0: 54ffef4c     	b.gt	0x56c88
   56ea4: 97fee0c7     	bl	0xf1c0
   56ea8: 17ffff78     	b	0x56c88
   56eac: b85f82a0     	ldur	w0, [x21, #-8]
   56eb0: 11000400     	add	w0, w0, #1
   56eb4: b81f82a0     	stur	w0, [x21, #-8]
   56eb8: 17fffedc     	b	0x56a28
   56ebc: f947eec1     	ldr	x1, [x22, #4056]
   56ec0: b40003e1     	cbz	x1, 0x56f3c
   56ec4: d1002042     	sub	x2, x2, #8
   56ec8: 885ffc41     	ldaxr	w1, [x2]
   56ecc: 51000423     	sub	w3, w1, #1
   56ed0: 8804fc43     	stlxr	w4, w3, [x2]
   56ed4: 35ffffa4     	cbnz	w4, 0x56ec8
   56ed8: 7100003f     	cmp	w1, #0
   56edc: 54ffe74c     	b.gt	0x56bc4
   56ee0: 97fee0b8     	bl	0xf1c0
   56ee4: 17ffff38     	b	0x56bc4
   56ee8: d1002042     	sub	x2, x2, #8
   56eec: 885ffc41     	ldaxr	w1, [x2]
   56ef0: 51000423     	sub	w3, w1, #1
   56ef4: 8804fc43     	stlxr	w4, w3, [x2]
   56ef8: 35ffffa4     	cbnz	w4, 0x56eec
   56efc: 7100003f     	cmp	w1, #0
   56f00: 54ffe92c     	b.gt	0x56c24
   56f04: 97fee0af     	bl	0xf1c0
   56f08: 17ffff47     	b	0x56c24
   56f0c: b00004c1     	adrp	x1, 0xef000
   56f10: f947ec21     	ldr	x1, [x1, #4056]
   56f14: b40003c1     	cbz	x1, 0x56f8c
   56f18: d1002042     	sub	x2, x2, #8
   56f1c: 885ffc41     	ldaxr	w1, [x2]
   56f20: 51000423     	sub	w3, w1, #1
   56f24: 8804fc43     	stlxr	w4, w3, [x2]
   56f28: 35ffffa4     	cbnz	w4, 0x56f1c
   56f2c: 7100003f     	cmp	w1, #0
   56f30: 54ffc08c     	b.gt	0x56740
   56f34: 97fee0a3     	bl	0xf1c0
   56f38: 17fffe02     	b	0x56740
   56f3c: b85f8041     	ldur	w1, [x2, #-8]
   56f40: 51000423     	sub	w3, w1, #1
   56f44: b81f8043     	stur	w3, [x2, #-8]
   56f48: 17ffffe4     	b	0x56ed8
   56f4c: b85f8041     	ldur	w1, [x2, #-8]
   56f50: 51000423     	sub	w3, w1, #1
   56f54: b81f8043     	stur	w3, [x2, #-8]
   56f58: 7100003f     	cmp	w1, #0
   56f5c: 54ffefcc     	b.gt	0x56d54
   56f60: 97fee098     	bl	0xf1c0
   56f64: 17ffff7c     	b	0x56d54
   56f68: d1002021     	sub	x1, x1, #8
   56f6c: 885ffc22     	ldaxr	w2, [x1]
   56f70: 51000443     	sub	w3, w2, #1
   56f74: 8804fc23     	stlxr	w4, w3, [x1]
   56f78: 35ffffa4     	cbnz	w4, 0x56f6c
   56f7c: 7100005f     	cmp	w2, #0
   56f80: 54ffe5cc     	b.gt	0x56c38
   56f84: 97fee08f     	bl	0xf1c0
   56f88: 17ffff2c     	b	0x56c38
   56f8c: b85f8041     	ldur	w1, [x2, #-8]
   56f90: 51000423     	sub	w3, w1, #1
   56f94: b81f8043     	stur	w3, [x2, #-8]
   56f98: 17ffffe5     	b	0x56f2c
   56f9c: 900002e0     	adrp	x0, 0xb2000
   56fa0: 9113a000     	add	x0, x0, #1256
   56fa4: 97fedffb     	bl	0xef90
   56fa8: 97fee36a     	bl	0xfd50
   56fac: f947faf3     	ldr	x19, [x23, #4080]
   56fb0: f94037a0     	ldr	x0, [x29, #104]
   56fb4: eb18001f     	cmp	x0, x24
   56fb8: 54000280     	b.eq	0x57008
   56fbc: f9400000     	ldr	x0, [x0]
   56fc0: d1006000     	sub	x0, x0, #24
   56fc4: eb13001f     	cmp	x0, x19
   56fc8: 54000040     	b.eq	0x56fd0
   56fcc: 97ffdb17     	bl	0x4dc28
   56fd0: f94037a0     	ldr	x0, [x29, #104]
   56fd4: 91002000     	add	x0, x0, #8
   56fd8: f90037a0     	str	x0, [x29, #104]
   56fdc: 17fffff5     	b	0x56fb0
   56fe0: 97fee1fc     	bl	0xf7d0
   56fe4: b00004c2     	adrp	x2, 0xef000
   56fe8: f9404ba1     	ldr	x1, [x29, #144]
   56fec: aa0003f3     	mov	x19, x0
   56ff0: f947f842     	ldr	x2, [x2, #4080]
   56ff4: d1006020     	sub	x0, x1, #24
   56ff8: eb02001f     	cmp	x0, x2
   56ffc: 54000621     	b.ne	0x570c0
   57000: aa1303e0     	mov	x0, x19
   57004: 97fee39f     	bl	0xfe80
   57008: 97fee08a     	bl	0xf230
   5700c: aa0003f4     	mov	x20, x0
   57010: f9404ba0     	ldr	x0, [x29, #144]
   57014: b4000040     	cbz	x0, 0x5701c
   57018: 97fee06a     	bl	0xf1c0
   5701c: f947faf7     	ldr	x23, [x23, #4080]
   57020: 91006339     	add	x25, x25, #24
   57024: 910303b3     	add	x19, x29, #192
   57028: eb13033f     	cmp	x25, x19
   5702c: 54000320     	b.eq	0x57090
   57030: f9400320     	ldr	x0, [x25]
   57034: d1006000     	sub	x0, x0, #24
   57038: eb17001f     	cmp	x0, x23
   5703c: 54000040     	b.eq	0x57044
   57040: 97ffdafa     	bl	0x4dc28
   57044: d1002339     	sub	x25, x25, #8
   57048: 17fffff8     	b	0x57028
   5704c: aa0003f4     	mov	x20, x0
   57050: 900004d7     	adrp	x23, 0xef000
   57054: f9404ba0     	ldr	x0, [x29, #144]
   57058: b4000040     	cbz	x0, 0x57060
   5705c: 97fee059     	bl	0xf1c0
   57060: f9407fa0     	ldr	x0, [x29, #248]
   57064: f947fae1     	ldr	x1, [x23, #4080]
   57068: d1006000     	sub	x0, x0, #24
   5706c: eb01001f     	cmp	x0, x1
   57070: 54000040     	b.eq	0x57078
   57074: 97ffdaed     	bl	0x4dc28
   57078: f9407ba0     	ldr	x0, [x29, #240]
   5707c: f947faf7     	ldr	x23, [x23, #4080]
   57080: d1006000     	sub	x0, x0, #24
   57084: eb17001f     	cmp	x0, x23
   57088: 54000040     	b.eq	0x57090
   5708c: 97ffdae7     	bl	0x4dc28
   57090: aa1403e0     	mov	x0, x20
   57094: 97fee37b     	bl	0xfe80
   57098: aa0003f4     	mov	x20, x0
   5709c: aa1303e0     	mov	x0, x19
   570a0: 97fee344     	bl	0xfdb0
   570a4: 17ffffde     	b	0x5701c
   570a8: f9405ba1     	ldr	x1, [x29, #176]
   570ac: aa0003f3     	mov	x19, x0
   570b0: f947faf7     	ldr	x23, [x23, #4080]
   570b4: d1006020     	sub	x0, x1, #24
   570b8: eb17001f     	cmp	x0, x23
   570bc: 54fffa20     	b.eq	0x57000
   570c0: 97ffdada     	bl	0x4dc28
   570c4: 17ffffcf     	b	0x57000
   570c8: aa0003f4     	mov	x20, x0
   570cc: 97fee2dd     	bl	0xfc40
   570d0: 17ffffd0     	b	0x57010
   570d4: aa0003f4     	mov	x20, x0
   570d8: aa1303e0     	mov	x0, x19
   570dc: 97fee241     	bl	0xf9e0
   570e0: aa1403e0     	mov	x0, x20
   570e4: 97fee367     	bl	0xfe80
   570e8: aa0003f6     	mov	x22, x0
   570ec: 91002260     	add	x0, x19, #8
   570f0: 97fee0c0     	bl	0xf3f0
   570f4: f9007bb5     	str	x21, [x29, #240]
   570f8: f85e82a0     	ldur	x0, [x21, #-24]
   570fc: f8206a78     	str	x24, [x19, x0]
   57100: f947aa81     	ldr	x1, [x20, #3920]
   57104: 91016260     	add	x0, x19, #88
   57108: 91004021     	add	x1, x1, #16
   5710c: f900a7a1     	str	x1, [x29, #328]
   57110: 97fee230     	bl	0xf9d0
   57114: aa1603e0     	mov	x0, x22
   57118: 97fee35a     	bl	0xfe80
   5711c: f00002c0     	adrp	x0, 0xb2000
   57120: 9113a000     	add	x0, x0, #1256
   57124: 97fedf9b     	bl	0xef90
   57128: aa0003f4     	mov	x20, x0
   5712c: f94047a0     	ldr	x0, [x29, #136]
   57130: f947fae1     	ldr	x1, [x23, #4080]
   57134: d1006000     	sub	x0, x0, #24
   57138: eb01001f     	cmp	x0, x1
   5713c: 54000040     	b.eq	0x57144
   57140: 97ffdaba     	bl	0x4dc28
   57144: aa1403e0     	mov	x0, x20
   57148: 17ffffd4     	b	0x57098
   5714c: aa0003f4     	mov	x20, x0
   57150: aa1a03e0     	mov	x0, x26
   57154: 97fee317     	bl	0xfdb0
   57158: 17fffff5     	b	0x5712c
   5715c: aa0003f6     	mov	x22, x0
   57160: f9002fbc     	str	x28, [x29, #88]
   57164: 17ffffe7     	b	0x57100
   57168: f9406ba1     	ldr	x1, [x29, #208]
   5716c: aa0003f4     	mov	x20, x0
   57170: f947faf7     	ldr	x23, [x23, #4080]
   57174: d1006020     	sub	x0, x1, #24
   57178: eb17001f     	cmp	x0, x23
   5717c: 54fffae0     	b.eq	0x570d8
   57180: 97ffdaaa     	bl	0x4dc28
   57184: 17ffffd5     	b	0x570d8
   57188: aa0003f4     	mov	x20, x0
   5718c: aa1303e0     	mov	x0, x19
   57190: 97fee308     	bl	0xfdb0
   57194: 17ffffb3     	b	0x57060
   57198: 97fee2ee     	bl	0xfd50
   5719c: f947faf3     	ldr	x19, [x23, #4080]
   571a0: f94037a0     	ldr	x0, [x29, #104]
   571a4: eb19001f     	cmp	x0, x25
   571a8: 54000140     	b.eq	0x571d0
   571ac: f9400000     	ldr	x0, [x0]
   571b0: d1006000     	sub	x0, x0, #24
   571b4: eb13001f     	cmp	x0, x19
   571b8: 54000040     	b.eq	0x571c0
   571bc: 97ffda9b     	bl	0x4dc28
   571c0: f94037a0     	ldr	x0, [x29, #104]
   571c4: 91002000     	add	x0, x0, #8
   571c8: f90037a0     	str	x0, [x29, #104]
   571cc: 17fffff5     	b	0x571a0
   571d0: 97fee018     	bl	0xf230
   571d4: aa0003f4     	mov	x20, x0
   571d8: aa1a03e0     	mov	x0, x26
   571dc: 97fee2f5     	bl	0xfdb0
   571e0: f94047a0     	ldr	x0, [x29, #136]
   571e4: f947fae1     	ldr	x1, [x23, #4080]
   571e8: d1006000     	sub	x0, x0, #24
   571ec: eb01001f     	cmp	x0, x1
   571f0: 54000040     	b.eq	0x571f8
   571f4: 97ffda8d     	bl	0x4dc28
   571f8: aa1403e0     	mov	x0, x20
   571fc: 17ffffe3     	b	0x57188
   57200: aa0003f4     	mov	x20, x0
   57204: 17fffff7     	b	0x571e0
   57208: aa0003f4     	mov	x20, x0
   5720c: 97fee28d     	bl	0xfc40
   57210: 17ffff91     	b	0x57054
   57214: d503201f     	nop
