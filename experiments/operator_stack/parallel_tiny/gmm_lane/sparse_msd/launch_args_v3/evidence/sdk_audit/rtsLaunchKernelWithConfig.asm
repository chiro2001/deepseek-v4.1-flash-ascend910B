
/usr/local/Ascend/cann-9.1.0/lib64/libruntime.so:	file format elf64-littleaarch64

Disassembly of section .text:

00000000000552e8 <rtsLaunchKernelWithConfig>:
   552e8: d10943ff     	sub	sp, sp, #592
   552ec: a9007bfd     	stp	x29, x30, [sp]
   552f0: 910003fd     	mov	x29, sp
   552f4: a9025bf5     	stp	x21, x22, [sp, #32]
   552f8: aa0003f5     	mov	x21, x0
   552fc: d00004c0     	adrp	x0, 0xef000
   55300: a90153f3     	stp	x19, x20, [sp, #16]
   55304: a90363f7     	stp	x23, x24, [sp, #48]
   55308: 2a0103f7     	mov	w23, w1
   5530c: a9046bf9     	stp	x25, x26, [sp, #64]
   55310: aa0503f6     	mov	x22, x5
   55314: a90573fb     	stp	x27, x28, [sp, #80]
   55318: b000031b     	adrp	x27, 0xb6000
   5531c: f947c400     	ldr	x0, [x0, #3976]
   55320: 910fe373     	add	x19, x27, #1016
   55324: aa0203f4     	mov	x20, x2
   55328: aa0303f8     	mov	x24, x3
   5532c: f9400001     	ldr	x1, [x0]
   55330: f90127a1     	str	x1, [x29, #584]
   55334: d2800001     	mov	x1, #0
   55338: aa0403f9     	mov	x25, x4
   5533c: 9109e273     	add	x19, x19, #632
   55340: 97feeb08     	bl	0xff60
   55344: aa1303e1     	mov	x1, x19
   55348: 97fee7c6     	bl	0xf260
   5534c: b40030b6     	cbz	x22, 0x55960
   55350: 9102a3ba     	add	x26, x29, #168
   55354: aa1303e1     	mov	x1, x19
   55358: aa1a03e2     	mov	x2, x26
   5535c: 910303b9     	add	x25, x29, #192
   55360: 910383b3     	add	x19, x29, #224
   55364: aa1903e0     	mov	x0, x25
   55368: 97fee81a     	bl	0xf3d0
   5536c: d00004d4     	adrp	x20, 0xef000
   55370: 91016260     	add	x0, x19, #88
   55374: 97fee75f     	bl	0xf0f0
   55378: d00004c0     	adrp	x0, 0xef000
   5537c: f947aa81     	ldr	x1, [x20, #3920]
   55380: 4f000400     	movi	v0.4s, #0
   55384: 390863bf     	strb	wzr, [x29, #536]
   55388: f947c800     	ldr	x0, [x0, #3984]
   5538c: 91004021     	add	x1, x1, #16
   55390: 390867bf     	strb	wzr, [x29, #537]
   55394: f9009fa1     	str	x1, [x29, #312]
   55398: d2800001     	mov	x1, #0
   5539c: f9010bbf     	str	xzr, [x29, #528]
   553a0: a940e015     	ldp	x21, x24, [x0, #8]
   553a4: f90073b5     	str	x21, [x29, #224]
   553a8: 3d808ba0     	str	q0, [x29, #544]
   553ac: 3d808fa0     	str	q0, [x29, #560]
   553b0: f85e82a0     	ldur	x0, [x21, #-24]
   553b4: f8206a78     	str	x24, [x19, x0]
   553b8: f94073a0     	ldr	x0, [x29, #224]
   553bc: f85e8000     	ldur	x0, [x0, #-24]
   553c0: 8b000260     	add	x0, x19, x0
   553c4: 97fee95f     	bl	0xf940
   553c8: d00004c0     	adrp	x0, 0xef000
   553cc: d00004dc     	adrp	x28, 0xef000
   553d0: 4f000400     	movi	v0.4s, #0
   553d4: d00004d7     	adrp	x23, 0xef000
   553d8: f947ac01     	ldr	x1, [x0, #3928]
   553dc: 91010260     	add	x0, x19, #64
   553e0: f947f782     	ldr	x2, [x28, #4072]
   553e4: 91006023     	add	x3, x1, #24
   553e8: 91010021     	add	x1, x1, #64
   553ec: 91004042     	add	x2, x2, #16
   553f0: f9009fa1     	str	x1, [x29, #312]
   553f4: a90e0ba3     	stp	x3, x2, [x29, #224]
   553f8: 3d803fa0     	str	q0, [x29, #240]
   553fc: 3d8043a0     	str	q0, [x29, #256]
   55400: 3d8047a0     	str	q0, [x29, #272]
   55404: 97fee903     	bl	0xf810
   55408: d00004c0     	adrp	x0, 0xef000
   5540c: 52800203     	mov	w3, #16
   55410: 91002261     	add	x1, x19, #8
   55414: f947a402     	ldr	x2, [x0, #3912]
   55418: f947fae0     	ldr	x0, [x23, #4080]
   5541c: 91004042     	add	x2, x2, #16
   55420: b9012ba3     	str	w3, [x29, #296]
   55424: 91006000     	add	x0, x0, #24
   55428: f90077a2     	str	x2, [x29, #232]
   5542c: f9009ba0     	str	x0, [x29, #304]
   55430: 91016260     	add	x0, x19, #88
   55434: 97fee943     	bl	0xf940
   55438: aa1603e1     	mov	x1, x22
   5543c: aa1303e0     	mov	x0, x19
   55440: 97fee76c     	bl	0xf1f0
   55444: f947fae0     	ldr	x0, [x23, #4080]
   55448: f9408ba1     	ldr	x1, [x29, #272]
   5544c: 91006000     	add	x0, x0, #24
   55450: f90067a0     	str	x0, [x29, #200]
   55454: b40036e1     	cbz	x1, 0x55b30
   55458: a95003a4     	ldp	x4, x0, [x29, #256]
   5545c: 52800003     	mov	w3, #0
   55460: aa1a03e2     	mov	x2, x26
   55464: eb04003f     	cmp	x1, x4
   55468: 540031c9     	b.ls	0x55aa0
   5546c: 97fee7a5     	bl	0xf300
   55470: f94067a1     	ldr	x1, [x29, #200]
   55474: d1002022     	sub	x2, x1, #8
   55478: b9400042     	ldr	w2, [x2]
   5547c: 37f83762     	tbnz	w2, #31, 0x55b68
   55480: d1002001     	sub	x1, x0, #8
   55484: b9400021     	ldr	w1, [x1]
   55488: 37f836c1     	tbnz	w1, #31, 0x55b60
   5548c: f94067a2     	ldr	x2, [x29, #200]
   55490: f947fae1     	ldr	x1, [x23, #4080]
   55494: f90067a0     	str	x0, [x29, #200]
   55498: d1006040     	sub	x0, x2, #24
   5549c: eb01001f     	cmp	x0, x1
   554a0: 54003cc1     	b.ne	0x55c38
   554a4: d00004c0     	adrp	x0, 0xef000
   554a8: d00004c1     	adrp	x1, 0xef000
   554ac: f9409ba3     	ldr	x3, [x29, #304]
   554b0: f947ac00     	ldr	x0, [x0, #3928]
   554b4: f947a421     	ldr	x1, [x1, #3912]
   554b8: 91006004     	add	x4, x0, #24
   554bc: f947fae2     	ldr	x2, [x23, #4080]
   554c0: 91010000     	add	x0, x0, #64
   554c4: 91004021     	add	x1, x1, #16
   554c8: f9009fa0     	str	x0, [x29, #312]
   554cc: d1006060     	sub	x0, x3, #24
   554d0: a90e07a4     	stp	x4, x1, [x29, #224]
   554d4: eb02001f     	cmp	x0, x2
   554d8: 54003801     	b.ne	0x55bd8
   554dc: f947f79c     	ldr	x28, [x28, #4072]
   554e0: 91010260     	add	x0, x19, #64
   554e4: 9100439c     	add	x28, x28, #16
   554e8: f90077bc     	str	x28, [x29, #232]
   554ec: 97feea7d     	bl	0xfee0
   554f0: f90073b5     	str	x21, [x29, #224]
   554f4: f947aa94     	ldr	x20, [x20, #3920]
   554f8: 91016260     	add	x0, x19, #88
   554fc: f85e82a1     	ldur	x1, [x21, #-24]
   55500: 91004294     	add	x20, x20, #16
   55504: f8216a78     	str	x24, [x19, x1]
   55508: f9009fb4     	str	x20, [x29, #312]
   5550c: 97fee931     	bl	0xf9d0
   55510: aa1a03e2     	mov	x2, x26
   55514: 91004320     	add	x0, x25, #16
   55518: f00002e1     	adrp	x1, 0xb4000
   5551c: 911fc021     	add	x1, x1, #2032
   55520: 97fee7ac     	bl	0xf3d0
   55524: 91006320     	add	x0, x25, #24
   55528: d00002e1     	adrp	x1, 0xb3000
   5552c: aa1a03e2     	mov	x2, x26
   55530: 91204021     	add	x1, x1, #2064
   55534: 97fee7a7     	bl	0xf3d0
   55538: f90053bf     	str	xzr, [x29, #160]
   5553c: 4f000400     	movi	v0.4s, #0
   55540: d2800400     	mov	x0, #32
   55544: 3d8027a0     	str	q0, [x29, #144]
   55548: 97feea4a     	bl	0xfe70
   5554c: aa0003f8     	mov	x24, x0
   55550: d00004c0     	adrp	x0, 0xef000
   55554: f947fafc     	ldr	x28, [x23, #4080]
   55558: 91008301     	add	x1, x24, #32
   5555c: d2800013     	mov	x19, #0
   55560: f947ec00     	ldr	x0, [x0, #4056]
   55564: a90683b8     	stp	x24, x0, [x29, #104]
   55568: f9003fbc     	str	x28, [x29, #120]
   5556c: f9004bb8     	str	x24, [x29, #144]
   55570: f90053a1     	str	x1, [x29, #160]
   55574: 14000007     	b	0x55590
   55578: eb1c02bf     	cmp	x21, x28
   5557c: 54001e21     	b.ne	0x55940
   55580: f8008714     	str	x20, [x24], #8
   55584: 91002273     	add	x19, x19, #8
   55588: f100827f     	cmp	x19, #32
   5558c: 54000600     	b.eq	0x5564c
   55590: f8736b34     	ldr	x20, [x25, x19]
   55594: d1006295     	sub	x21, x20, #24
   55598: 910042a0     	add	x0, x21, #16
   5559c: b9400001     	ldr	w1, [x0]
   555a0: 36fffec1     	tbz	w1, #31, 0x55578
   555a4: a97e8696     	ldp	x22, x1, [x20, #-24]
   555a8: 928000c0     	mov	x0, #-7
   555ac: f2e7ffe0     	movk	x0, #16383, lsl #48
   555b0: eb0002df     	cmp	x22, x0
   555b4: 540041c8     	b.hi	0x55dec
   555b8: eb0102df     	cmp	x22, x1
   555bc: 54000089     	b.ls	0x555cc
   555c0: d37ff820     	lsl	x0, x1, #1
   555c4: eb0002df     	cmp	x22, x0
   555c8: 9a8022d6     	csel	x22, x22, x0, hs
   555cc: 9100e6c0     	add	x0, x22, #57
   555d0: f140041f     	cmp	x0, #1, lsl #12         // =4096
   555d4: fa568022     	ccmp	x1, x22, #2, hi
   555d8: 54000102     	b.hs	0x555f8
   555dc: 914006d6     	add	x22, x22, #1, lsl #12   // =4096
   555e0: 92402c00     	and	x0, x0, #0xfff
   555e4: cb0002c0     	sub	x0, x22, x0
   555e8: 928000c1     	mov	x1, #-7
   555ec: f2e7ffe1     	movk	x1, #16383, lsl #48
   555f0: eb01001f     	cmp	x0, x1
   555f4: 9a819016     	csel	x22, x0, x1, ls
   555f8: 910066c0     	add	x0, x22, #25
   555fc: 97feea1d     	bl	0xfe70
   55600: f9000416     	str	x22, [x0, #8]
   55604: aa0003e3     	mov	x3, x0
   55608: b900101f     	str	wzr, [x0, #16]
   5560c: 91006004     	add	x4, x0, #24
   55610: f94002a2     	ldr	x2, [x21]
   55614: b40000c2     	cbz	x2, 0x5562c
   55618: f100045f     	cmp	x2, #1
   5561c: 54002721     	b.ne	0x55b00
   55620: 39400280     	ldrb	w0, [x20]
   55624: 39006060     	strb	w0, [x3, #24]
   55628: f94002a2     	ldr	x2, [x21]
   5562c: f9403fa0     	ldr	x0, [x29, #120]
   55630: eb00007f     	cmp	x3, x0
   55634: 54002861     	b.ne	0x55b40
   55638: aa0403f4     	mov	x20, x4
   5563c: f8008714     	str	x20, [x24], #8
   55640: 91002273     	add	x19, x19, #8
   55644: f100827f     	cmp	x19, #32
   55648: 54fffa41     	b.ne	0x55590
   5564c: 910243b3     	add	x19, x29, #144
   55650: 910fe363     	add	x3, x27, #1016
   55654: d00002f4     	adrp	x20, 0xb3000
   55658: aa1303e4     	mov	x4, x19
   5565c: 9109e063     	add	x3, x3, #632
   55660: 52800e82     	mov	w2, #116
   55664: 911b6281     	add	x1, x20, #1752
   55668: 52800060     	mov	w0, #3
   5566c: f9004fb8     	str	x24, [x29, #152]
   55670: 97fee8f0     	bl	0xfa30
   55674: 97fee957     	bl	0xfbd0
   55678: 910223b5     	add	x21, x29, #136
   5567c: aa0003f6     	mov	x22, x0
   55680: b00002e1     	adrp	x1, 0xb2000
   55684: 910203a2     	add	x2, x29, #128
   55688: 9137a021     	add	x1, x1, #3560
   5568c: aa1503e0     	mov	x0, x21
   55690: 97fee750     	bl	0xf3d0
   55694: aa1a03e8     	mov	x8, x26
   55698: 52800060     	mov	w0, #3
   5569c: 97fee6e1     	bl	0xf220
   556a0: aa1303e3     	mov	x3, x19
   556a4: aa1a03e2     	mov	x2, x26
   556a8: aa1503e1     	mov	x1, x21
   556ac: aa1603e0     	mov	x0, x22
   556b0: 97fee868     	bl	0xf850
   556b4: a94ad7b3     	ldp	x19, x21, [x29, #168]
   556b8: d00004d6     	adrp	x22, 0xef000
   556bc: eb15027f     	cmp	x19, x21
   556c0: 54000180     	b.eq	0x556f0
   556c4: f947eec0     	ldr	x0, [x22, #4056]
   556c8: f947faf8     	ldr	x24, [x23, #4080]
   556cc: b4000be0     	cbz	x0, 0x55848
   556d0: f9400261     	ldr	x1, [x19]
   556d4: d1006020     	sub	x0, x1, #24
   556d8: eb18001f     	cmp	x0, x24
   556dc: 54001ce1     	b.ne	0x55a78
   556e0: 91002273     	add	x19, x19, #8
   556e4: eb1302bf     	cmp	x21, x19
   556e8: 54ffff41     	b.ne	0x556d0
   556ec: f94057b5     	ldr	x21, [x29, #168]
   556f0: b4000075     	cbz	x21, 0x556fc
   556f4: aa1503e0     	mov	x0, x21
   556f8: 97fee6b2     	bl	0xf1c0
   556fc: f94047a2     	ldr	x2, [x29, #136]
   55700: f947fae1     	ldr	x1, [x23, #4080]
   55704: d1006040     	sub	x0, x2, #24
   55708: eb01001f     	cmp	x0, x1
   5570c: 54002361     	b.ne	0x55b78
   55710: a94957b3     	ldp	x19, x21, [x29, #144]
   55714: eb15027f     	cmp	x19, x21
   55718: 54000180     	b.eq	0x55748
   5571c: f947eec0     	ldr	x0, [x22, #4056]
   55720: f947faf8     	ldr	x24, [x23, #4080]
   55724: b4000b60     	cbz	x0, 0x55890
   55728: f9400261     	ldr	x1, [x19]
   5572c: d1006020     	sub	x0, x1, #24
   55730: eb18001f     	cmp	x0, x24
   55734: 540018e1     	b.ne	0x55a50
   55738: 91002273     	add	x19, x19, #8
   5573c: eb1302bf     	cmp	x21, x19
   55740: 54ffff41     	b.ne	0x55728
   55744: f9404bb5     	ldr	x21, [x29, #144]
   55748: b4000075     	cbz	x21, 0x55754
   5574c: aa1503e0     	mov	x0, x21
   55750: 97fee69c     	bl	0xf1c0
   55754: f947eec0     	ldr	x0, [x22, #4056]
   55758: f947fae1     	ldr	x1, [x23, #4080]
   5575c: f9406fa2     	ldr	x2, [x29, #216]
   55760: b4000bc0     	cbz	x0, 0x558d8
   55764: d1006040     	sub	x0, x2, #24
   55768: eb01001f     	cmp	x0, x1
   5576c: 54002ac1     	b.ne	0x55cc4
   55770: f9406ba2     	ldr	x2, [x29, #208]
   55774: f947fae1     	ldr	x1, [x23, #4080]
   55778: d1006040     	sub	x0, x2, #24
   5577c: eb01001f     	cmp	x0, x1
   55780: 54002c21     	b.ne	0x55d04
   55784: f94067a2     	ldr	x2, [x29, #200]
   55788: f947fae1     	ldr	x1, [x23, #4080]
   5578c: d1006040     	sub	x0, x2, #24
   55790: eb01001f     	cmp	x0, x1
   55794: 54002d81     	b.ne	0x55d44
   55798: f94063a2     	ldr	x2, [x29, #192]
   5579c: f947fae1     	ldr	x1, [x23, #4080]
   557a0: d1006040     	sub	x0, x2, #24
   557a4: eb01001f     	cmp	x0, x1
   557a8: 54002601     	b.ne	0x55c68
   557ac: 97fee7a5     	bl	0xf640
   557b0: 52800021     	mov	w1, #1
   557b4: aa1a03e8     	mov	x8, x26
   557b8: 72a0e221     	movk	w1, #1809, lsl #16
   557bc: 97fee841     	bl	0xf8c0
   557c0: f94057a4     	ldr	x4, [x29, #168]
   557c4: 910fe362     	add	x2, x27, #1016
   557c8: b00002e3     	adrp	x3, 0xb2000
   557cc: 9109e042     	add	x2, x2, #632
   557d0: 9114e063     	add	x3, x3, #1336
   557d4: 52800e81     	mov	w1, #116
   557d8: 911b6280     	add	x0, x20, #1752
   557dc: 97fee60d     	bl	0xf010
   557e0: 97fee888     	bl	0xfa00
   557e4: 52800020     	mov	w0, #1
   557e8: 72a0e220     	movk	w0, #1809, lsl #16
   557ec: 9400d549     	bl	0x8ad10
   557f0: f94057a1     	ldr	x1, [x29, #168]
   557f4: 2a0003f4     	mov	w20, w0
   557f8: f947faf7     	ldr	x23, [x23, #4080]
   557fc: d1006020     	sub	x0, x1, #24
   55800: eb17001f     	cmp	x0, x23
   55804: 54001d21     	b.ne	0x55ba8
   55808: d00004c0     	adrp	x0, 0xef000
   5580c: f947c401     	ldr	x1, [x0, #3976]
   55810: 2a1403e0     	mov	w0, w20
   55814: f94127a2     	ldr	x2, [x29, #584]
   55818: f9400021     	ldr	x1, [x1]
   5581c: ca010041     	eor	x1, x2, x1
   55820: b5003081     	cbnz	x1, 0x55e30
   55824: a94153f3     	ldp	x19, x20, [sp, #16]
   55828: a9425bf5     	ldp	x21, x22, [sp, #32]
   5582c: a94363f7     	ldp	x23, x24, [sp, #48]
   55830: a9446bf9     	ldp	x25, x26, [sp, #64]
   55834: a94573fb     	ldp	x27, x28, [sp, #80]
   55838: a9407bfd     	ldp	x29, x30, [sp]
   5583c: 910943ff     	add	sp, sp, #592
   55840: d65f03c0     	ret
   55844: d503201f     	nop
   55848: f9400261     	ldr	x1, [x19]
   5584c: d1006020     	sub	x0, x1, #24
   55850: eb18001f     	cmp	x0, x24
   55854: 54000101     	b.ne	0x55874
   55858: 91002273     	add	x19, x19, #8
   5585c: eb15027f     	cmp	x19, x21
   55860: 54fff460     	b.eq	0x556ec
   55864: f9400261     	ldr	x1, [x19]
   55868: d1006020     	sub	x0, x1, #24
   5586c: eb18001f     	cmp	x0, x24
   55870: 54ffff40     	b.eq	0x55858
   55874: b85f8022     	ldur	w2, [x1, #-8]
   55878: 51000443     	sub	w3, w2, #1
   5587c: b81f8023     	stur	w3, [x1, #-8]
   55880: 7100005f     	cmp	w2, #0
   55884: 54fffeac     	b.gt	0x55858
   55888: 97fee64e     	bl	0xf1c0
   5588c: 17fffff3     	b	0x55858
   55890: f9400261     	ldr	x1, [x19]
   55894: d1006020     	sub	x0, x1, #24
   55898: eb18001f     	cmp	x0, x24
   5589c: 54000101     	b.ne	0x558bc
   558a0: 91002273     	add	x19, x19, #8
   558a4: eb15027f     	cmp	x19, x21
   558a8: 54fff4e0     	b.eq	0x55744
   558ac: f9400261     	ldr	x1, [x19]
   558b0: d1006020     	sub	x0, x1, #24
   558b4: eb18001f     	cmp	x0, x24
   558b8: 54ffff40     	b.eq	0x558a0
   558bc: b85f8022     	ldur	w2, [x1, #-8]
   558c0: 51000443     	sub	w3, w2, #1
   558c4: b81f8023     	stur	w3, [x1, #-8]
   558c8: 7100005f     	cmp	w2, #0
   558cc: 54fffeac     	b.gt	0x558a0
   558d0: 97fee63c     	bl	0xf1c0
   558d4: 17fffff3     	b	0x558a0
   558d8: d1006040     	sub	x0, x2, #24
   558dc: eb01001f     	cmp	x0, x1
   558e0: 54002041     	b.ne	0x55ce8
   558e4: f9406ba2     	ldr	x2, [x29, #208]
   558e8: f947fae1     	ldr	x1, [x23, #4080]
   558ec: d1006040     	sub	x0, x2, #24
   558f0: eb01001f     	cmp	x0, x1
   558f4: 540021a1     	b.ne	0x55d28
   558f8: f94067a2     	ldr	x2, [x29, #200]
   558fc: f947fae1     	ldr	x1, [x23, #4080]
   55900: d1006040     	sub	x0, x2, #24
   55904: eb01001f     	cmp	x0, x1
   55908: 540023e1     	b.ne	0x55d84
   5590c: f94063a2     	ldr	x2, [x29, #192]
   55910: f947fae1     	ldr	x1, [x23, #4080]
   55914: d1006040     	sub	x0, x2, #24
   55918: eb01001f     	cmp	x0, x1
   5591c: 54fff480     	b.eq	0x557ac
   55920: b85f8041     	ldur	w1, [x2, #-8]
   55924: 51000423     	sub	w3, w1, #1
   55928: b81f8043     	stur	w3, [x2, #-8]
   5592c: 7100003f     	cmp	w1, #0
   55930: 54fff3ec     	b.gt	0x557ac
   55934: 97fee623     	bl	0xf1c0
   55938: 17ffff9d     	b	0x557ac
   5593c: d503201f     	nop
   55940: f9403ba1     	ldr	x1, [x29, #112]
   55944: b4000ee1     	cbz	x1, 0x55b20
   55948: 885ffc01     	ldaxr	w1, [x0]
   5594c: 11000421     	add	w1, w1, #1
   55950: 8802fc01     	stlxr	w2, w1, [x0]
   55954: 34ffe162     	cbz	w2, 0x55580
   55958: 17fffffc     	b	0x55948
   5595c: d503201f     	nop
   55960: 52800000     	mov	w0, #0
   55964: 94015d19     	bl	0xacdc8
   55968: aa0003f6     	mov	x22, x0
   5596c: b4001920     	cbz	x0, 0x55c90
   55970: 52800060     	mov	w0, #3
   55974: aa1303e2     	mov	x2, x19
   55978: 29157fa0     	stp	w0, wzr, [x29, #168]
   5597c: 910223a1     	add	x1, x29, #136
   55980: aa1403e0     	mov	x0, x20
   55984: f90047bf     	str	xzr, [x29, #136]
   55988: f9005bb9     	str	x25, [x29, #176]
   5598c: 9400d1df     	bl	0x8a108
   55990: 2a0003f4     	mov	w20, w0
   55994: 35fff3a0     	cbnz	w0, 0x55808
   55998: f94002c6     	ldr	x6, [x22]
   5599c: aa1503e1     	mov	x1, x21
   559a0: f94047a4     	ldr	x4, [x29, #136]
   559a4: aa1803e5     	mov	x5, x24
   559a8: 9102a3a3     	add	x3, x29, #168
   559ac: 2a1703e2     	mov	w2, w23
   559b0: aa1603e0     	mov	x0, x22
   559b4: f94050c6     	ldr	x6, [x6, #160]
   559b8: d63f00c0     	blr	x6
   559bc: 2a0003f5     	mov	w21, w0
   559c0: 52800120     	mov	w0, #9
   559c4: 72a0e100     	movk	w0, #1800, lsl #16
   559c8: 6b0002bf     	cmp	w21, w0
   559cc: 54001ce0     	b.eq	0x55d68
   559d0: 528001a0     	mov	w0, #13
   559d4: 72a0e220     	movk	w0, #1809, lsl #16
   559d8: 6b0002bf     	cmp	w21, w0
   559dc: 54001e20     	b.eq	0x55da0
   559e0: 34fff155     	cbz	w21, 0x55808
   559e4: 97fee717     	bl	0xf640
   559e8: 2a1503e1     	mov	w1, w21
   559ec: 910243a8     	add	x8, x29, #144
   559f0: 97fee7b4     	bl	0xf8c0
   559f4: f9404ba4     	ldr	x4, [x29, #144]
   559f8: b00002e3     	adrp	x3, 0xb2000
   559fc: d00002e0     	adrp	x0, 0xb3000
   55a00: 9114e063     	add	x3, x3, #1336
   55a04: aa1303e2     	mov	x2, x19
   55a08: 52800fe1     	mov	w1, #127
   55a0c: 911b6000     	add	x0, x0, #1752
   55a10: 97fee580     	bl	0xf010
   55a14: 910fe361     	add	x1, x27, #1016
   55a18: 2a1503e0     	mov	w0, w21
   55a1c: 9109e021     	add	x1, x1, #632
   55a20: 97fee840     	bl	0xfb20
   55a24: 97fee7f7     	bl	0xfa00
   55a28: d00004c1     	adrp	x1, 0xef000
   55a2c: f9404ba2     	ldr	x2, [x29, #144]
   55a30: f947f821     	ldr	x1, [x1, #4080]
   55a34: d1006040     	sub	x0, x2, #24
   55a38: eb01001f     	cmp	x0, x1
   55a3c: 54001b81     	b.ne	0x55dac
   55a40: 2a1503e0     	mov	w0, w21
   55a44: 9400d4b3     	bl	0x8ad10
   55a48: 2a0003f4     	mov	w20, w0
   55a4c: 17ffff6f     	b	0x55808
   55a50: d1002021     	sub	x1, x1, #8
   55a54: 885ffc22     	ldaxr	w2, [x1]
   55a58: 51000443     	sub	w3, w2, #1
   55a5c: 8804fc23     	stlxr	w4, w3, [x1]
   55a60: 35ffffa4     	cbnz	w4, 0x55a54
   55a64: 7100005f     	cmp	w2, #0
   55a68: 54ffe68c     	b.gt	0x55738
   55a6c: 97fee5d5     	bl	0xf1c0
   55a70: 17ffff32     	b	0x55738
   55a74: d503201f     	nop
   55a78: d1002021     	sub	x1, x1, #8
   55a7c: 885ffc22     	ldaxr	w2, [x1]
   55a80: 51000443     	sub	w3, w2, #1
   55a84: 8804fc23     	stlxr	w4, w3, [x1]
   55a88: 35ffffa4     	cbnz	w4, 0x55a7c
   55a8c: 7100005f     	cmp	w2, #0
   55a90: 54ffe28c     	b.gt	0x556e0
   55a94: 97fee5cb     	bl	0xf1c0
   55a98: 17ffff12     	b	0x556e0
   55a9c: d503201f     	nop
   55aa0: aa0403e1     	mov	x1, x4
   55aa4: 97fee617     	bl	0xf300
   55aa8: f94067a1     	ldr	x1, [x29, #200]
   55aac: d1002022     	sub	x2, x1, #8
   55ab0: b9400042     	ldr	w2, [x2]
   55ab4: 37f805e2     	tbnz	w2, #31, 0x55b70
   55ab8: d1002001     	sub	x1, x0, #8
   55abc: b9400021     	ldr	w1, [x1]
   55ac0: 37f804c1     	tbnz	w1, #31, 0x55b58
   55ac4: f94067a2     	ldr	x2, [x29, #200]
   55ac8: f947fae1     	ldr	x1, [x23, #4080]
   55acc: f90067a0     	str	x0, [x29, #200]
   55ad0: d1006040     	sub	x0, x2, #24
   55ad4: eb01001f     	cmp	x0, x1
   55ad8: 54ffce60     	b.eq	0x554a4
   55adc: d00004d6     	adrp	x22, 0xef000
   55ae0: f947eed6     	ldr	x22, [x22, #4056]
   55ae4: b4001496     	cbz	x22, 0x55d74
   55ae8: d1002042     	sub	x2, x2, #8
   55aec: 885ffc41     	ldaxr	w1, [x2]
   55af0: 51000423     	sub	w3, w1, #1
   55af4: 8804fc43     	stlxr	w4, w3, [x2]
   55af8: 34000b04     	cbz	w4, 0x55c58
   55afc: 17fffffc     	b	0x55aec
   55b00: aa0403e0     	mov	x0, x4
   55b04: f90033a3     	str	x3, [x29, #96]
   55b08: aa1403e1     	mov	x1, x20
   55b0c: 97fee609     	bl	0xf330
   55b10: f94002a2     	ldr	x2, [x21]
   55b14: aa0003e4     	mov	x4, x0
   55b18: f94033a3     	ldr	x3, [x29, #96]
   55b1c: 17fffec4     	b	0x5562c
   55b20: b85f8280     	ldur	w0, [x20, #-8]
   55b24: 11000400     	add	w0, w0, #1
   55b28: b81f8280     	stur	w0, [x20, #-8]
   55b2c: 17fffe95     	b	0x55580
   55b30: 91014261     	add	x1, x19, #80
   55b34: 91002320     	add	x0, x25, #8
   55b38: 97fee81e     	bl	0xfbb0
   55b3c: 17fffe5a     	b	0x554a4
   55b40: 8b020060     	add	x0, x3, x2
   55b44: f9000062     	str	x2, [x3]
   55b48: aa0403f4     	mov	x20, x4
   55b4c: 3900601f     	strb	wzr, [x0, #24]
   55b50: 17fffebb     	b	0x5563c
   55b54: d503201f     	nop
   55b58: b81f801f     	stur	wzr, [x0, #-8]
   55b5c: 17ffffda     	b	0x55ac4
   55b60: b81f801f     	stur	wzr, [x0, #-8]
   55b64: 17fffe4a     	b	0x5548c
   55b68: b81f803f     	stur	wzr, [x1, #-8]
   55b6c: 17fffe45     	b	0x55480
   55b70: b81f803f     	stur	wzr, [x1, #-8]
   55b74: 17ffffd1     	b	0x55ab8
   55b78: f947eec1     	ldr	x1, [x22, #4056]
   55b7c: b40004e1     	cbz	x1, 0x55c18
   55b80: d1002042     	sub	x2, x2, #8
   55b84: 885ffc41     	ldaxr	w1, [x2]
   55b88: 51000423     	sub	w3, w1, #1
   55b8c: 8804fc43     	stlxr	w4, w3, [x2]
   55b90: 35ffffa4     	cbnz	w4, 0x55b84
   55b94: 7100003f     	cmp	w1, #0
   55b98: 54ffdbcc     	b.gt	0x55710
   55b9c: 97fee589     	bl	0xf1c0
   55ba0: 17fffedc     	b	0x55710
   55ba4: d503201f     	nop
   55ba8: f947eed6     	ldr	x22, [x22, #4056]
   55bac: b40002f6     	cbz	x22, 0x55c08
   55bb0: d1002021     	sub	x1, x1, #8
   55bb4: 885ffc22     	ldaxr	w2, [x1]
   55bb8: 51000443     	sub	w3, w2, #1
   55bbc: 8804fc23     	stlxr	w4, w3, [x1]
   55bc0: 35ffffa4     	cbnz	w4, 0x55bb4
   55bc4: 7100005f     	cmp	w2, #0
   55bc8: 54ffe20c     	b.gt	0x55808
   55bcc: 97fee57d     	bl	0xf1c0
   55bd0: 17ffff0e     	b	0x55808
   55bd4: d503201f     	nop
   55bd8: d00004d6     	adrp	x22, 0xef000
   55bdc: f947eed6     	ldr	x22, [x22, #4056]
   55be0: b4000256     	cbz	x22, 0x55c28
   55be4: d1002063     	sub	x3, x3, #8
   55be8: 885ffc61     	ldaxr	w1, [x3]
   55bec: 51000422     	sub	w2, w1, #1
   55bf0: 8804fc62     	stlxr	w4, w2, [x3]
   55bf4: 35ffffa4     	cbnz	w4, 0x55be8
   55bf8: 7100003f     	cmp	w1, #0
   55bfc: 54ffc70c     	b.gt	0x554dc
   55c00: 97fee570     	bl	0xf1c0
   55c04: 17fffe36     	b	0x554dc
   55c08: b85f8022     	ldur	w2, [x1, #-8]
   55c0c: 51000443     	sub	w3, w2, #1
   55c10: b81f8023     	stur	w3, [x1, #-8]
   55c14: 17ffffec     	b	0x55bc4
   55c18: b85f8041     	ldur	w1, [x2, #-8]
   55c1c: 51000423     	sub	w3, w1, #1
   55c20: b81f8043     	stur	w3, [x2, #-8]
   55c24: 17ffffdc     	b	0x55b94
   55c28: b85f8061     	ldur	w1, [x3, #-8]
   55c2c: 51000422     	sub	w2, w1, #1
   55c30: b81f8062     	stur	w2, [x3, #-8]
   55c34: 17fffff1     	b	0x55bf8
   55c38: d00004d6     	adrp	x22, 0xef000
   55c3c: f947eed6     	ldr	x22, [x22, #4056]
   55c40: b40009b6     	cbz	x22, 0x55d74
   55c44: d1002042     	sub	x2, x2, #8
   55c48: 885ffc41     	ldaxr	w1, [x2]
   55c4c: 51000423     	sub	w3, w1, #1
   55c50: 8804fc43     	stlxr	w4, w3, [x2]
   55c54: 35ffffa4     	cbnz	w4, 0x55c48
   55c58: 7100003f     	cmp	w1, #0
   55c5c: 54ffc24c     	b.gt	0x554a4
   55c60: 97fee558     	bl	0xf1c0
   55c64: 17fffe10     	b	0x554a4
   55c68: d1002042     	sub	x2, x2, #8
   55c6c: 885ffc41     	ldaxr	w1, [x2]
   55c70: 51000423     	sub	w3, w1, #1
   55c74: 8804fc43     	stlxr	w4, w3, [x2]
   55c78: 35ffffa4     	cbnz	w4, 0x55c6c
   55c7c: 7100003f     	cmp	w1, #0
   55c80: 54ffd96c     	b.gt	0x557ac
   55c84: 97fee54f     	bl	0xf1c0
   55c88: 17fffec9     	b	0x557ac
   55c8c: d503201f     	nop
   55c90: aa1303e2     	mov	x2, x19
   55c94: b00002e3     	adrp	x3, 0xb2000
   55c98: d00002e0     	adrp	x0, 0xb3000
   55c9c: 91140063     	add	x3, x3, #1280
   55ca0: 911b6000     	add	x0, x0, #1752
   55ca4: 52978f04     	mov	w4, #48248
   55ca8: 72a000e4     	movk	w4, #7, lsl #16
   55cac: 52800ec1     	mov	w1, #118
   55cb0: 52978f14     	mov	w20, #48248
   55cb4: 72a000f4     	movk	w20, #7, lsl #16
   55cb8: 97fee4d6     	bl	0xf010
   55cbc: 97fee751     	bl	0xfa00
   55cc0: 17fffed2     	b	0x55808
   55cc4: d1002042     	sub	x2, x2, #8
   55cc8: 885ffc41     	ldaxr	w1, [x2]
   55ccc: 51000423     	sub	w3, w1, #1
   55cd0: 8804fc43     	stlxr	w4, w3, [x2]
   55cd4: 35ffffa4     	cbnz	w4, 0x55cc8
   55cd8: 7100003f     	cmp	w1, #0
   55cdc: 54ffd4ac     	b.gt	0x55770
   55ce0: 97fee538     	bl	0xf1c0
   55ce4: 17fffea3     	b	0x55770
   55ce8: b85f8041     	ldur	w1, [x2, #-8]
   55cec: 51000423     	sub	w3, w1, #1
   55cf0: b81f8043     	stur	w3, [x2, #-8]
   55cf4: 7100003f     	cmp	w1, #0
   55cf8: 54ffdf6c     	b.gt	0x558e4
   55cfc: 97fee531     	bl	0xf1c0
   55d00: 17fffef9     	b	0x558e4
   55d04: d1002042     	sub	x2, x2, #8
   55d08: 885ffc41     	ldaxr	w1, [x2]
   55d0c: 51000423     	sub	w3, w1, #1
   55d10: 8804fc43     	stlxr	w4, w3, [x2]
   55d14: 35ffffa4     	cbnz	w4, 0x55d08
   55d18: 7100003f     	cmp	w1, #0
   55d1c: 54ffd34c     	b.gt	0x55784
   55d20: 97fee528     	bl	0xf1c0
   55d24: 17fffe98     	b	0x55784
   55d28: b85f8041     	ldur	w1, [x2, #-8]
   55d2c: 51000423     	sub	w3, w1, #1
   55d30: b81f8043     	stur	w3, [x2, #-8]
   55d34: 7100003f     	cmp	w1, #0
   55d38: 54ffde0c     	b.gt	0x558f8
   55d3c: 97fee521     	bl	0xf1c0
   55d40: 17fffeee     	b	0x558f8
   55d44: d1002042     	sub	x2, x2, #8
   55d48: 885ffc41     	ldaxr	w1, [x2]
   55d4c: 51000423     	sub	w3, w1, #1
   55d50: 8804fc43     	stlxr	w4, w3, [x2]
   55d54: 35ffffa4     	cbnz	w4, 0x55d48
   55d58: 7100003f     	cmp	w1, #0
   55d5c: 54ffd1ec     	b.gt	0x55798
   55d60: 97fee518     	bl	0xf1c0
   55d64: 17fffe8d     	b	0x55798
   55d68: 52944134     	mov	w20, #41481
   55d6c: 72a00034     	movk	w20, #1, lsl #16
   55d70: 17fffea6     	b	0x55808
   55d74: b85f8041     	ldur	w1, [x2, #-8]
   55d78: 51000423     	sub	w3, w1, #1
   55d7c: b81f8043     	stur	w3, [x2, #-8]
   55d80: 17ffffb6     	b	0x55c58
   55d84: b85f8041     	ldur	w1, [x2, #-8]
   55d88: 51000423     	sub	w3, w1, #1
   55d8c: b81f8043     	stur	w3, [x2, #-8]
   55d90: 7100003f     	cmp	w1, #0
   55d94: 54ffdbcc     	b.gt	0x5590c
   55d98: 97fee50a     	bl	0xf1c0
   55d9c: 17fffedc     	b	0x5590c
   55da0: 52851314     	mov	w20, #10392
   55da4: 72a00074     	movk	w20, #3, lsl #16
   55da8: 17fffe98     	b	0x55808
   55dac: d00004c1     	adrp	x1, 0xef000
   55db0: f947ec21     	ldr	x1, [x1, #4056]
   55db4: b4000141     	cbz	x1, 0x55ddc
   55db8: d1002042     	sub	x2, x2, #8
   55dbc: 885ffc41     	ldaxr	w1, [x2]
   55dc0: 51000423     	sub	w3, w1, #1
   55dc4: 8804fc43     	stlxr	w4, w3, [x2]
   55dc8: 35ffffa4     	cbnz	w4, 0x55dbc
   55dcc: 7100003f     	cmp	w1, #0
   55dd0: 54ffe38c     	b.gt	0x55a40
   55dd4: 97fee4fb     	bl	0xf1c0
   55dd8: 17ffff1a     	b	0x55a40
   55ddc: b85f8041     	ldur	w1, [x2, #-8]
   55de0: 51000423     	sub	w3, w1, #1
   55de4: b81f8043     	stur	w3, [x2, #-8]
   55de8: 17fffff9     	b	0x55dcc
   55dec: b00002e0     	adrp	x0, 0xb2000
   55df0: 9113a000     	add	x0, x0, #1256
   55df4: 97fee467     	bl	0xef90
   55df8: 97fee7d6     	bl	0xfd50
   55dfc: f947faf3     	ldr	x19, [x23, #4080]
   55e00: f94037a0     	ldr	x0, [x29, #104]
   55e04: eb18001f     	cmp	x0, x24
   55e08: 54000160     	b.eq	0x55e34
   55e0c: f9400000     	ldr	x0, [x0]
   55e10: d1006000     	sub	x0, x0, #24
   55e14: eb13001f     	cmp	x0, x19
   55e18: 54000040     	b.eq	0x55e20
   55e1c: 97ffdf83     	bl	0x4dc28
   55e20: f94037a0     	ldr	x0, [x29, #104]
   55e24: 91002000     	add	x0, x0, #8
   55e28: f90037a0     	str	x0, [x29, #104]
   55e2c: 17fffff5     	b	0x55e00
   55e30: 97fee668     	bl	0xf7d0
   55e34: 97fee4ff     	bl	0xf230
   55e38: aa0003f4     	mov	x20, x0
   55e3c: aa1303e0     	mov	x0, x19
   55e40: 97fee6e8     	bl	0xf9e0
   55e44: aa1403e0     	mov	x0, x20
   55e48: 97fee80e     	bl	0xfe80
   55e4c: aa0003f4     	mov	x20, x0
   55e50: 97fee77c     	bl	0xfc40
   55e54: f9404ba0     	ldr	x0, [x29, #144]
   55e58: b4000040     	cbz	x0, 0x55e60
   55e5c: 97fee4d9     	bl	0xf1c0
   55e60: f947faf7     	ldr	x23, [x23, #4080]
   55e64: 91006339     	add	x25, x25, #24
   55e68: 9102e3b3     	add	x19, x29, #184
   55e6c: eb19027f     	cmp	x19, x25
   55e70: 54000100     	b.eq	0x55e90
   55e74: f9400320     	ldr	x0, [x25]
   55e78: d1006000     	sub	x0, x0, #24
   55e7c: eb17001f     	cmp	x0, x23
   55e80: 54000040     	b.eq	0x55e88
   55e84: 97ffdf69     	bl	0x4dc28
   55e88: d1002339     	sub	x25, x25, #8
   55e8c: 17fffff8     	b	0x55e6c
   55e90: aa1403e0     	mov	x0, x20
   55e94: 97fee7fb     	bl	0xfe80
   55e98: d00004c2     	adrp	x2, 0xef000
   55e9c: f9404ba1     	ldr	x1, [x29, #144]
   55ea0: aa0003f3     	mov	x19, x0
   55ea4: f947f842     	ldr	x2, [x2, #4080]
   55ea8: d1006020     	sub	x0, x1, #24
   55eac: eb02001f     	cmp	x0, x2
   55eb0: 540002e1     	b.ne	0x55f0c
   55eb4: aa1303e0     	mov	x0, x19
   55eb8: 97fee7f2     	bl	0xfe80
   55ebc: aa0003f4     	mov	x20, x0
   55ec0: aa1a03e0     	mov	x0, x26
   55ec4: 97fee7bb     	bl	0xfdb0
   55ec8: f94047a0     	ldr	x0, [x29, #136]
   55ecc: f947fae1     	ldr	x1, [x23, #4080]
   55ed0: d1006000     	sub	x0, x0, #24
   55ed4: eb01001f     	cmp	x0, x1
   55ed8: 54000040     	b.eq	0x55ee0
   55edc: 97ffdf53     	bl	0x4dc28
   55ee0: aa1403e0     	mov	x0, x20
   55ee4: aa0003f4     	mov	x20, x0
   55ee8: aa1303e0     	mov	x0, x19
   55eec: 97fee7b1     	bl	0xfdb0
   55ef0: 17ffffdc     	b	0x55e60
   55ef4: f94057a1     	ldr	x1, [x29, #168]
   55ef8: aa0003f3     	mov	x19, x0
   55efc: f947faf7     	ldr	x23, [x23, #4080]
   55f00: d1006020     	sub	x0, x1, #24
   55f04: eb17001f     	cmp	x0, x23
   55f08: 54fffd60     	b.eq	0x55eb4
   55f0c: 97ffdf47     	bl	0x4dc28
   55f10: 17ffffe9     	b	0x55eb4
   55f14: aa0003f6     	mov	x22, x0
   55f18: 91002260     	add	x0, x19, #8
   55f1c: 97fee535     	bl	0xf3f0
   55f20: f90073b5     	str	x21, [x29, #224]
   55f24: f85e82a0     	ldur	x0, [x21, #-24]
   55f28: f8206a78     	str	x24, [x19, x0]
   55f2c: f947aa81     	ldr	x1, [x20, #3920]
   55f30: 91016260     	add	x0, x19, #88
   55f34: 91004021     	add	x1, x1, #16
   55f38: f9009fa1     	str	x1, [x29, #312]
   55f3c: 97fee6a5     	bl	0xf9d0
   55f40: aa1603e0     	mov	x0, x22
   55f44: 97fee7cf     	bl	0xfe80
   55f48: f94067a1     	ldr	x1, [x29, #200]
   55f4c: aa0003f4     	mov	x20, x0
   55f50: f947faf7     	ldr	x23, [x23, #4080]
   55f54: d1006020     	sub	x0, x1, #24
   55f58: eb17001f     	cmp	x0, x23
   55f5c: 54fff700     	b.eq	0x55e3c
   55f60: 97ffdf32     	bl	0x4dc28
   55f64: 17ffffb6     	b	0x55e3c
   55f68: aa0003f6     	mov	x22, x0
   55f6c: 17fffff0     	b	0x55f2c
   55f70: 17ffffdd     	b	0x55ee4
   55f74: aa0003f4     	mov	x20, x0
   55f78: 17ffffb7     	b	0x55e54
   55f7c: aa0003f4     	mov	x20, x0
   55f80: 17ffffd2     	b	0x55ec8
   55f84: d503201f     	nop
