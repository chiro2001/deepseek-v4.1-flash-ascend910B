
/usr/local/Ascend/cann-9.1.0/lib64/libruntime.so:	file format elf64-littleaarch64

Disassembly of section .text:

00000000000500b0 <rtsLaunchKernelWithHostArgs>:
   500b0: a9ab7bfd     	stp	x29, x30, [sp, #-336]!
   500b4: 910003fd     	mov	x29, sp
   500b8: a9025bf5     	stp	x21, x22, [sp, #32]
   500bc: f00004f5     	adrp	x21, 0xef000
   500c0: a90153f3     	stp	x19, x20, [sp, #16]
   500c4: aa0203f6     	mov	x22, x2
   500c8: a90363f7     	stp	x23, x24, [sp, #48]
   500cc: aa0003f3     	mov	x19, x0
   500d0: a9046bf9     	stp	x25, x26, [sp, #64]
   500d4: aa0303f8     	mov	x24, x3
   500d8: a90573fb     	stp	x27, x28, [sp, #80]
   500dc: 2a0703f4     	mov	w20, w7
   500e0: f947c6a2     	ldr	x2, [x21, #3976]
   500e4: aa0403fc     	mov	x28, x4
   500e8: 2a0503fb     	mov	w27, w5
   500ec: 2a0103f7     	mov	w23, w1
   500f0: f9400040     	ldr	x0, [x2]
   500f4: f900a7a0     	str	x0, [x29, #328]
   500f8: d2800000     	mov	x0, #0
   500fc: f90047a6     	str	x6, [x29, #136]
   50100: d0000339     	adrp	x25, 0xb6000
   50104: 97feff97     	bl	0xff60
   50108: 910fe323     	add	x3, x25, #1016
   5010c: 9108407a     	add	x26, x3, #528
   50110: aa1a03e1     	mov	x1, x26
   50114: 97fefc53     	bl	0xf260
   50118: 52800000     	mov	w0, #0
   5011c: 9401732b     	bl	0xacdc8
   50120: b4002780     	cbz	x0, 0x50610
   50124: b4000873     	cbz	x19, 0x50230
   50128: 910543a1     	add	x1, x29, #336
   5012c: aa0003e6     	mov	x6, x0
   50130: b940ce60     	ldr	w0, [x19, #204]
   50134: aa1a03e2     	mov	x2, x26
   50138: b90083a0     	str	w0, [x29, #128]
   5013c: aa1603e0     	mov	x0, x22
   50140: f8148c3f     	str	xzr, [x1, #-184]!
   50144: a90c7fbf     	stp	xzr, xzr, [x29, #192]
   50148: a90dffbf     	stp	xzr, xzr, [x29, #216]
   5014c: a90effbf     	stp	xzr, xzr, [x29, #232]
   50150: a9107fbf     	stp	xzr, xzr, [x29, #256]
   50154: a9117fbf     	stp	xzr, xzr, [x29, #272]
   50158: a9127fbf     	stp	xzr, xzr, [x29, #288]
   5015c: a9137fbf     	stp	xzr, xzr, [x29, #304]
   50160: f9003fa6     	str	x6, [x29, #120]
   50164: f9007fbf     	str	xzr, [x29, #248]
   50168: f900a3bf     	str	xzr, [x29, #320]
   5016c: 9400e7e7     	bl	0x8a108
   50170: 35000460     	cbnz	w0, 0x501fc
   50174: b94083a0     	ldr	w0, [x29, #128]
   50178: 12003e94     	and	w20, w20, #0xffff
   5017c: f9403fa6     	ldr	x6, [x29, #120]
   50180: 7100041f     	cmp	w0, #1
   50184: 54002320     	b.eq	0x505e8
   50188: 910403a0     	add	x0, x29, #256
   5018c: f94047a1     	ldr	x1, [x29, #136]
   50190: f90067a0     	str	x0, [x29, #200]
   50194: 52800040     	mov	w0, #2
   50198: 79023bb4     	strh	w20, [x29, #284]
   5019c: 3904a3bf     	strb	wzr, [x29, #296]
   501a0: a91007bc     	stp	x28, x1, [x29, #256]
   501a4: b9011bbb     	str	w27, [x29, #280]
   501a8: b900c3a0     	str	w0, [x29, #192]
   501ac: aa0603e0     	mov	x0, x6
   501b0: f9404fa4     	ldr	x4, [x29, #152]
   501b4: aa1303e1     	mov	x1, x19
   501b8: f94000c7     	ldr	x7, [x6]
   501bc: aa1803e5     	mov	x5, x24
   501c0: 910303a3     	add	x3, x29, #192
   501c4: 2a1703e2     	mov	w2, w23
   501c8: f94050e6     	ldr	x6, [x7, #160]
   501cc: d63f00c0     	blr	x6
   501d0: 2a0003f3     	mov	w19, w0
   501d4: 52800120     	mov	w0, #9
   501d8: 72a0e100     	movk	w0, #1800, lsl #16
   501dc: 6b00027f     	cmp	w19, w0
   501e0: 540028c0     	b.eq	0x506f8
   501e4: 528001a0     	mov	w0, #13
   501e8: 72a0e220     	movk	w0, #1809, lsl #16
   501ec: 6b00027f     	cmp	w19, w0
   501f0: 54002940     	b.eq	0x50718
   501f4: 52800000     	mov	w0, #0
   501f8: 35001b13     	cbnz	w19, 0x50558
   501fc: f947c6b5     	ldr	x21, [x21, #3976]
   50200: f940a7a2     	ldr	x2, [x29, #328]
   50204: f94002a1     	ldr	x1, [x21]
   50208: ca010041     	eor	x1, x2, x1
   5020c: b50030a1     	cbnz	x1, 0x50820
   50210: a94153f3     	ldp	x19, x20, [sp, #16]
   50214: a9425bf5     	ldp	x21, x22, [sp, #32]
   50218: a94363f7     	ldp	x23, x24, [sp, #48]
   5021c: a9446bf9     	ldp	x25, x26, [sp, #64]
   50220: a94573fb     	ldp	x27, x28, [sp, #80]
   50224: a8d57bfd     	ldp	x29, x30, [sp], #336
   50228: d65f03c0     	ret
   5022c: d503201f     	nop
   50230: 910303b3     	add	x19, x29, #192
   50234: 910403bb     	add	x27, x29, #256
   50238: aa1303e2     	mov	x2, x19
   5023c: aa1a03e1     	mov	x1, x26
   50240: aa1b03e0     	mov	x0, x27
   50244: f90043b3     	str	x19, [x29, #128]
   50248: 97fefc62     	bl	0xf3d0
   5024c: 91002360     	add	x0, x27, #8
   50250: f0000301     	adrp	x1, 0xb3000
   50254: aa1303e2     	mov	x2, x19
   50258: 911ba021     	add	x1, x1, #1768
   5025c: 97fefc5d     	bl	0xf3d0
   50260: f9005bbf     	str	xzr, [x29, #176]
   50264: 4f000400     	movi	v0.4s, #0
   50268: d2800200     	mov	x0, #16
   5026c: 3d802ba0     	str	q0, [x29, #160]
   50270: 97feff00     	bl	0xfe70
   50274: f00004fa     	adrp	x26, 0xef000
   50278: f00004e1     	adrp	x1, 0xef000
   5027c: aa0003f8     	mov	x24, x0
   50280: f947fb5c     	ldr	x28, [x26, #4080]
   50284: d2800013     	mov	x19, #0
   50288: f947ec21     	ldr	x1, [x1, #4056]
   5028c: a90707a0     	stp	x0, x1, [x29, #112]
   50290: f90053a0     	str	x0, [x29, #160]
   50294: 91004000     	add	x0, x0, #16
   50298: f90047bc     	str	x28, [x29, #136]
   5029c: f9005ba0     	str	x0, [x29, #176]
   502a0: 14000008     	b	0x502c0
   502a4: d503201f     	nop
   502a8: eb1c02df     	cmp	x22, x28
   502ac: 540018e1     	b.ne	0x505c8
   502b0: f8008714     	str	x20, [x24], #8
   502b4: 91002273     	add	x19, x19, #8
   502b8: f100427f     	cmp	x19, #16
   502bc: 54000600     	b.eq	0x5037c
   502c0: f8736b74     	ldr	x20, [x27, x19]
   502c4: d1006296     	sub	x22, x20, #24
   502c8: 910042c0     	add	x0, x22, #16
   502cc: b9400001     	ldr	w1, [x0]
   502d0: 36fffec1     	tbz	w1, #31, 0x502a8
   502d4: a97e8697     	ldp	x23, x1, [x20, #-24]
   502d8: 928000c0     	mov	x0, #-7
   502dc: f2e7ffe0     	movk	x0, #16383, lsl #48
   502e0: eb0002ff     	cmp	x23, x0
   502e4: 54002a08     	b.hi	0x50824
   502e8: eb0102ff     	cmp	x23, x1
   502ec: 54000089     	b.ls	0x502fc
   502f0: d37ff820     	lsl	x0, x1, #1
   502f4: eb0002ff     	cmp	x23, x0
   502f8: 9a8022f7     	csel	x23, x23, x0, hs
   502fc: 9100e6e0     	add	x0, x23, #57
   50300: f140041f     	cmp	x0, #1, lsl #12         // =4096
   50304: fa578022     	ccmp	x1, x23, #2, hi
   50308: 54000102     	b.hs	0x50328
   5030c: 914006f7     	add	x23, x23, #1, lsl #12   // =4096
   50310: 92402c00     	and	x0, x0, #0xfff
   50314: cb0002e0     	sub	x0, x23, x0
   50318: 928000c1     	mov	x1, #-7
   5031c: f2e7ffe1     	movk	x1, #16383, lsl #48
   50320: eb01001f     	cmp	x0, x1
   50324: 9a819017     	csel	x23, x0, x1, ls
   50328: 910066e0     	add	x0, x23, #25
   5032c: 97fefed1     	bl	0xfe70
   50330: f9000417     	str	x23, [x0, #8]
   50334: aa0003e3     	mov	x3, x0
   50338: b900101f     	str	wzr, [x0, #16]
   5033c: 91006004     	add	x4, x0, #24
   50340: f94002c2     	ldr	x2, [x22]
   50344: b40000c2     	cbz	x2, 0x5035c
   50348: f100045f     	cmp	x2, #1
   5034c: 54001a61     	b.ne	0x50698
   50350: 39400280     	ldrb	w0, [x20]
   50354: 39006060     	strb	w0, [x3, #24]
   50358: f94002c2     	ldr	x2, [x22]
   5035c: f94047a0     	ldr	x0, [x29, #136]
   50360: eb00007f     	cmp	x3, x0
   50364: 54001e21     	b.ne	0x50728
   50368: aa0403f4     	mov	x20, x4
   5036c: f8008714     	str	x20, [x24], #8
   50370: 91002273     	add	x19, x19, #8
   50374: f100427f     	cmp	x19, #16
   50378: 54fffa41     	b.ne	0x502c0
   5037c: 910283b3     	add	x19, x29, #160
   50380: 910fe323     	add	x3, x25, #1016
   50384: f0000301     	adrp	x1, 0xb3000
   50388: aa1303e4     	mov	x4, x19
   5038c: 91084063     	add	x3, x3, #528
   50390: 52800602     	mov	w2, #48
   50394: 911b6021     	add	x1, x1, #1752
   50398: 52800080     	mov	w0, #4
   5039c: f90057b8     	str	x24, [x29, #168]
   503a0: 97fefda4     	bl	0xfa30
   503a4: 97fefe0b     	bl	0xfbd0
   503a8: 910263b4     	add	x20, x29, #152
   503ac: aa0003f6     	mov	x22, x0
   503b0: d0000301     	adrp	x1, 0xb2000
   503b4: 910243a2     	add	x2, x29, #144
   503b8: 91340021     	add	x1, x1, #3328
   503bc: aa1403e0     	mov	x0, x20
   503c0: 97fefc04     	bl	0xf3d0
   503c4: f94043b7     	ldr	x23, [x29, #128]
   503c8: 52800080     	mov	w0, #4
   503cc: aa1703e8     	mov	x8, x23
   503d0: 97fefb94     	bl	0xf220
   503d4: aa1303e3     	mov	x3, x19
   503d8: aa1703e2     	mov	x2, x23
   503dc: aa1403e1     	mov	x1, x20
   503e0: aa1603e0     	mov	x0, x22
   503e4: 97fefd1b     	bl	0xf850
   503e8: a94c53b3     	ldp	x19, x20, [x29, #192]
   503ec: f00004f6     	adrp	x22, 0xef000
   503f0: eb14027f     	cmp	x19, x20
   503f4: 54000180     	b.eq	0x50424
   503f8: f947eec0     	ldr	x0, [x22, #4056]
   503fc: f947fb57     	ldr	x23, [x26, #4080]
   50400: b4000680     	cbz	x0, 0x504d0
   50404: f9400261     	ldr	x1, [x19]
   50408: d1006020     	sub	x0, x1, #24
   5040c: eb17001f     	cmp	x0, x23
   50410: 54001301     	b.ne	0x50670
   50414: 91002273     	add	x19, x19, #8
   50418: eb13029f     	cmp	x20, x19
   5041c: 54ffff41     	b.ne	0x50404
   50420: f94063b4     	ldr	x20, [x29, #192]
   50424: b4000074     	cbz	x20, 0x50430
   50428: aa1403e0     	mov	x0, x20
   5042c: 97fefb65     	bl	0xf1c0
   50430: f9404fa2     	ldr	x2, [x29, #152]
   50434: f947fb41     	ldr	x1, [x26, #4080]
   50438: d1006040     	sub	x0, x2, #24
   5043c: eb01001f     	cmp	x0, x1
   50440: 54001801     	b.ne	0x50740
   50444: a94a53b3     	ldp	x19, x20, [x29, #160]
   50448: eb14027f     	cmp	x19, x20
   5044c: 54000180     	b.eq	0x5047c
   50450: f947eec0     	ldr	x0, [x22, #4056]
   50454: f947fb57     	ldr	x23, [x26, #4080]
   50458: b4000500     	cbz	x0, 0x504f8
   5045c: f9400261     	ldr	x1, [x19]
   50460: d1006020     	sub	x0, x1, #24
   50464: eb17001f     	cmp	x0, x23
   50468: 54000f01     	b.ne	0x50648
   5046c: 91002273     	add	x19, x19, #8
   50470: eb13029f     	cmp	x20, x19
   50474: 54ffff41     	b.ne	0x5045c
   50478: f94053b4     	ldr	x20, [x29, #160]
   5047c: b4000074     	cbz	x20, 0x50488
   50480: aa1403e0     	mov	x0, x20
   50484: 97fefb4f     	bl	0xf1c0
   50488: f947eed6     	ldr	x22, [x22, #4056]
   5048c: f94087a2     	ldr	x2, [x29, #264]
   50490: f947fb41     	ldr	x1, [x26, #4080]
   50494: d1006040     	sub	x0, x2, #24
   50498: eb01001f     	cmp	x0, x1
   5049c: b4000436     	cbz	x22, 0x50520
   504a0: 54001661     	b.ne	0x5076c
   504a4: f94083a1     	ldr	x1, [x29, #256]
   504a8: f947fb5a     	ldr	x26, [x26, #4080]
   504ac: d1006020     	sub	x0, x1, #24
   504b0: eb1a001f     	cmp	x0, x26
   504b4: 540019c1     	b.ne	0x507ec
   504b8: 97fefd52     	bl	0xfa00
   504bc: 52800020     	mov	w0, #1
   504c0: 72a0e220     	movk	w0, #1809, lsl #16
   504c4: 9400ea13     	bl	0x8ad10
   504c8: 17ffff4d     	b	0x501fc
   504cc: d503201f     	nop
   504d0: f9400261     	ldr	x1, [x19]
   504d4: d1006020     	sub	x0, x1, #24
   504d8: eb17001f     	cmp	x0, x23
   504dc: 54000fe1     	b.ne	0x506d8
   504e0: 91002273     	add	x19, x19, #8
   504e4: eb13029f     	cmp	x20, x19
   504e8: 54ffff41     	b.ne	0x504d0
   504ec: f94063b4     	ldr	x20, [x29, #192]
   504f0: 17ffffcd     	b	0x50424
   504f4: d503201f     	nop
   504f8: f9400261     	ldr	x1, [x19]
   504fc: d1006020     	sub	x0, x1, #24
   50500: eb17001f     	cmp	x0, x23
   50504: 54000da1     	b.ne	0x506b8
   50508: 91002273     	add	x19, x19, #8
   5050c: eb14027f     	cmp	x19, x20
   50510: 54ffff41     	b.ne	0x504f8
   50514: f94053b4     	ldr	x20, [x29, #160]
   50518: 17ffffd9     	b	0x5047c
   5051c: d503201f     	nop
   50520: 54001381     	b.ne	0x50790
   50524: f94083a1     	ldr	x1, [x29, #256]
   50528: f947fb5a     	ldr	x26, [x26, #4080]
   5052c: d1006020     	sub	x0, x1, #24
   50530: eb1a001f     	cmp	x0, x26
   50534: 54fffc20     	b.eq	0x504b8
   50538: b85f8022     	ldur	w2, [x1, #-8]
   5053c: 51000443     	sub	w3, w2, #1
   50540: b81f8023     	stur	w3, [x1, #-8]
   50544: 7100005f     	cmp	w2, #0
   50548: 54fffb8c     	b.gt	0x504b8
   5054c: 97fefb1d     	bl	0xf1c0
   50550: 17ffffda     	b	0x504b8
   50554: d503201f     	nop
   50558: 97fefc3a     	bl	0xf640
   5055c: 2a1303e1     	mov	w1, w19
   50560: 910283a8     	add	x8, x29, #160
   50564: 97fefcd7     	bl	0xf8c0
   50568: f94053a4     	ldr	x4, [x29, #160]
   5056c: 910fe322     	add	x2, x25, #1016
   50570: d0000303     	adrp	x3, 0xb2000
   50574: f0000300     	adrp	x0, 0xb3000
   50578: 9114e063     	add	x3, x3, #1336
   5057c: 91084042     	add	x2, x2, #528
   50580: 52800a01     	mov	w1, #80
   50584: 911b6000     	add	x0, x0, #1752
   50588: 97fefaa2     	bl	0xf010
   5058c: 910fe321     	add	x1, x25, #1016
   50590: 2a1303e0     	mov	w0, w19
   50594: 91084021     	add	x1, x1, #528
   50598: 97fefd62     	bl	0xfb20
   5059c: 97fefd19     	bl	0xfa00
   505a0: f00004e1     	adrp	x1, 0xef000
   505a4: f94053a2     	ldr	x2, [x29, #160]
   505a8: f947f821     	ldr	x1, [x1, #4080]
   505ac: d1006040     	sub	x0, x2, #24
   505b0: eb01001f     	cmp	x0, x1
   505b4: 54001041     	b.ne	0x507bc
   505b8: 2a1303e0     	mov	w0, w19
   505bc: 9400e9d5     	bl	0x8ad10
   505c0: 17ffff0f     	b	0x501fc
   505c4: d503201f     	nop
   505c8: f9403fa1     	ldr	x1, [x29, #120]
   505cc: b40009e1     	cbz	x1, 0x50708
   505d0: 885ffc01     	ldaxr	w1, [x0]
   505d4: 11000421     	add	w1, w1, #1
   505d8: 8802fc01     	stlxr	w2, w1, [x0]
   505dc: 34ffe6a2     	cbz	w2, 0x502b0
   505e0: 17fffffc     	b	0x505d0
   505e4: d503201f     	nop
   505e8: 910363a0     	add	x0, x29, #216
   505ec: f94047a1     	ldr	x1, [x29, #136]
   505f0: f90067a0     	str	x0, [x29, #200]
   505f4: 7901ebb4     	strh	w20, [x29, #244]
   505f8: 3903dfbf     	strb	wzr, [x29, #247]
   505fc: a90d87bc     	stp	x28, x1, [x29, #216]
   50600: b900ebbb     	str	w27, [x29, #232]
   50604: b94083a0     	ldr	w0, [x29, #128]
   50608: 17fffee8     	b	0x501a8
   5060c: d503201f     	nop
   50610: aa1a03e2     	mov	x2, x26
   50614: d0000303     	adrp	x3, 0xb2000
   50618: f0000300     	adrp	x0, 0xb3000
   5061c: 91140063     	add	x3, x3, #1280
   50620: 911b6000     	add	x0, x0, #1752
   50624: 52978f04     	mov	w4, #48248
   50628: 72a000e4     	movk	w4, #7, lsl #16
   5062c: 528005e1     	mov	w1, #47
   50630: 97fefa78     	bl	0xf010
   50634: 97fefcf3     	bl	0xfa00
   50638: 52978f00     	mov	w0, #48248
   5063c: 72a000e0     	movk	w0, #7, lsl #16
   50640: 17fffeef     	b	0x501fc
   50644: d503201f     	nop
   50648: d1002021     	sub	x1, x1, #8
   5064c: 885ffc22     	ldaxr	w2, [x1]
   50650: 51000443     	sub	w3, w2, #1
   50654: 8804fc23     	stlxr	w4, w3, [x1]
   50658: 35ffffa4     	cbnz	w4, 0x5064c
   5065c: 7100005f     	cmp	w2, #0
   50660: 54fff06c     	b.gt	0x5046c
   50664: 97fefad7     	bl	0xf1c0
   50668: 17ffff81     	b	0x5046c
   5066c: d503201f     	nop
   50670: d1002021     	sub	x1, x1, #8
   50674: 885ffc22     	ldaxr	w2, [x1]
   50678: 51000443     	sub	w3, w2, #1
   5067c: 8804fc23     	stlxr	w4, w3, [x1]
   50680: 35ffffa4     	cbnz	w4, 0x50674
   50684: 7100005f     	cmp	w2, #0
   50688: 54ffec6c     	b.gt	0x50414
   5068c: 97fefacd     	bl	0xf1c0
   50690: 17ffff61     	b	0x50414
   50694: d503201f     	nop
   50698: aa0403e0     	mov	x0, x4
   5069c: f90037a3     	str	x3, [x29, #104]
   506a0: aa1403e1     	mov	x1, x20
   506a4: 97fefb23     	bl	0xf330
   506a8: f94002c2     	ldr	x2, [x22]
   506ac: aa0003e4     	mov	x4, x0
   506b0: f94037a3     	ldr	x3, [x29, #104]
   506b4: 17ffff2a     	b	0x5035c
   506b8: b85f8022     	ldur	w2, [x1, #-8]
   506bc: 51000443     	sub	w3, w2, #1
   506c0: b81f8023     	stur	w3, [x1, #-8]
   506c4: 7100005f     	cmp	w2, #0
   506c8: 54fff20c     	b.gt	0x50508
   506cc: 97fefabd     	bl	0xf1c0
   506d0: 17ffff8e     	b	0x50508
   506d4: d503201f     	nop
   506d8: b85f8022     	ldur	w2, [x1, #-8]
   506dc: 51000443     	sub	w3, w2, #1
   506e0: b81f8023     	stur	w3, [x1, #-8]
   506e4: 7100005f     	cmp	w2, #0
   506e8: 54ffefcc     	b.gt	0x504e0
   506ec: 97fefab5     	bl	0xf1c0
   506f0: 17ffff7c     	b	0x504e0
   506f4: d503201f     	nop
   506f8: 52944120     	mov	w0, #41481
   506fc: 72a00020     	movk	w0, #1, lsl #16
   50700: 17fffebf     	b	0x501fc
   50704: d503201f     	nop
   50708: b85f8280     	ldur	w0, [x20, #-8]
   5070c: 11000400     	add	w0, w0, #1
   50710: b81f8280     	stur	w0, [x20, #-8]
   50714: 17fffee7     	b	0x502b0
   50718: 52851300     	mov	w0, #10392
   5071c: 72a00060     	movk	w0, #3, lsl #16
   50720: 17fffeb7     	b	0x501fc
   50724: d503201f     	nop
   50728: 8b020060     	add	x0, x3, x2
   5072c: f9000062     	str	x2, [x3]
   50730: aa0403f4     	mov	x20, x4
   50734: 3900601f     	strb	wzr, [x0, #24]
   50738: 17ffff0d     	b	0x5036c
   5073c: d503201f     	nop
   50740: f947eec1     	ldr	x1, [x22, #4056]
   50744: b4000341     	cbz	x1, 0x507ac
   50748: d1002042     	sub	x2, x2, #8
   5074c: 885ffc41     	ldaxr	w1, [x2]
   50750: 51000423     	sub	w3, w1, #1
   50754: 8804fc43     	stlxr	w4, w3, [x2]
   50758: 35ffffa4     	cbnz	w4, 0x5074c
   5075c: 7100003f     	cmp	w1, #0
   50760: 54ffe72c     	b.gt	0x50444
   50764: 97fefa97     	bl	0xf1c0
   50768: 17ffff37     	b	0x50444
   5076c: d1002042     	sub	x2, x2, #8
   50770: 885ffc41     	ldaxr	w1, [x2]
   50774: 51000423     	sub	w3, w1, #1
   50778: 8804fc43     	stlxr	w4, w3, [x2]
   5077c: 35ffffa4     	cbnz	w4, 0x50770
   50780: 7100003f     	cmp	w1, #0
   50784: 54ffe90c     	b.gt	0x504a4
   50788: 97fefa8e     	bl	0xf1c0
   5078c: 17ffff46     	b	0x504a4
   50790: b85f8041     	ldur	w1, [x2, #-8]
   50794: 51000423     	sub	w3, w1, #1
   50798: b81f8043     	stur	w3, [x2, #-8]
   5079c: 7100003f     	cmp	w1, #0
   507a0: 54ffec2c     	b.gt	0x50524
   507a4: 97fefa87     	bl	0xf1c0
   507a8: 17ffff5f     	b	0x50524
   507ac: b85f8041     	ldur	w1, [x2, #-8]
   507b0: 51000423     	sub	w3, w1, #1
   507b4: b81f8043     	stur	w3, [x2, #-8]
   507b8: 17ffffe9     	b	0x5075c
   507bc: f00004e1     	adrp	x1, 0xef000
   507c0: f947ec21     	ldr	x1, [x1, #4056]
   507c4: b4000261     	cbz	x1, 0x50810
   507c8: d1002042     	sub	x2, x2, #8
   507cc: 885ffc41     	ldaxr	w1, [x2]
   507d0: 51000423     	sub	w3, w1, #1
   507d4: 8804fc43     	stlxr	w4, w3, [x2]
   507d8: 35ffffa4     	cbnz	w4, 0x507cc
   507dc: 7100003f     	cmp	w1, #0
   507e0: 54ffeecc     	b.gt	0x505b8
   507e4: 97fefa77     	bl	0xf1c0
   507e8: 17ffff74     	b	0x505b8
   507ec: d1002021     	sub	x1, x1, #8
   507f0: 885ffc22     	ldaxr	w2, [x1]
   507f4: 51000443     	sub	w3, w2, #1
   507f8: 8804fc23     	stlxr	w4, w3, [x1]
   507fc: 35ffffa4     	cbnz	w4, 0x507f0
   50800: 7100005f     	cmp	w2, #0
   50804: 54ffe5ac     	b.gt	0x504b8
   50808: 97fefa6e     	bl	0xf1c0
   5080c: 17ffff2b     	b	0x504b8
   50810: b85f8041     	ldur	w1, [x2, #-8]
   50814: 51000423     	sub	w3, w1, #1
   50818: b81f8043     	stur	w3, [x2, #-8]
   5081c: 17fffff0     	b	0x507dc
   50820: 97fefbec     	bl	0xf7d0
   50824: d0000300     	adrp	x0, 0xb2000
   50828: 9113a000     	add	x0, x0, #1256
   5082c: 97fef9d9     	bl	0xef90
   50830: 97fefd48     	bl	0xfd50
   50834: f947fb53     	ldr	x19, [x26, #4080]
   50838: f9403ba0     	ldr	x0, [x29, #112]
   5083c: eb18001f     	cmp	x0, x24
   50840: 54000140     	b.eq	0x50868
   50844: f9400000     	ldr	x0, [x0]
   50848: d1006000     	sub	x0, x0, #24
   5084c: eb13001f     	cmp	x0, x19
   50850: 54000040     	b.eq	0x50858
   50854: 97fff4f5     	bl	0x4dc28
   50858: f9403ba0     	ldr	x0, [x29, #112]
   5085c: 91002000     	add	x0, x0, #8
   50860: f9003ba0     	str	x0, [x29, #112]
   50864: 17fffff5     	b	0x50838
   50868: 97fefa72     	bl	0xf230
   5086c: f00004e2     	adrp	x2, 0xef000
   50870: f94053a1     	ldr	x1, [x29, #160]
   50874: aa0003f3     	mov	x19, x0
   50878: f947f842     	ldr	x2, [x2, #4080]
   5087c: d1006020     	sub	x0, x1, #24
   50880: eb02001f     	cmp	x0, x2
   50884: 54000040     	b.eq	0x5088c
   50888: 97fff4e8     	bl	0x4dc28
   5088c: aa1303e0     	mov	x0, x19
   50890: 97fefd7c     	bl	0xfe80
   50894: aa0003f4     	mov	x20, x0
   50898: 97fefcea     	bl	0xfc40
   5089c: f94053a0     	ldr	x0, [x29, #160]
   508a0: b4000040     	cbz	x0, 0x508a8
   508a4: 97fefa47     	bl	0xf1c0
   508a8: f94087a0     	ldr	x0, [x29, #264]
   508ac: f947fb41     	ldr	x1, [x26, #4080]
   508b0: d1006000     	sub	x0, x0, #24
   508b4: eb01001f     	cmp	x0, x1
   508b8: 54000040     	b.eq	0x508c0
   508bc: 97fff4db     	bl	0x4dc28
   508c0: f94083a0     	ldr	x0, [x29, #256]
   508c4: f947fb5a     	ldr	x26, [x26, #4080]
   508c8: d1006000     	sub	x0, x0, #24
   508cc: eb1a001f     	cmp	x0, x26
   508d0: 54000040     	b.eq	0x508d8
   508d4: 97fff4d5     	bl	0x4dc28
   508d8: aa1403e0     	mov	x0, x20
   508dc: 97fefd69     	bl	0xfe80
   508e0: aa0003f4     	mov	x20, x0
   508e4: f9404fa0     	ldr	x0, [x29, #152]
   508e8: f947fb41     	ldr	x1, [x26, #4080]
   508ec: d1006000     	sub	x0, x0, #24
   508f0: eb01001f     	cmp	x0, x1
   508f4: 54000040     	b.eq	0x508fc
   508f8: 97fff4cc     	bl	0x4dc28
   508fc: aa1403e0     	mov	x0, x20
   50900: aa0003f4     	mov	x20, x0
   50904: aa1303e0     	mov	x0, x19
   50908: 97fefd2a     	bl	0xfdb0
   5090c: 17ffffe7     	b	0x508a8
   50910: aa0003f4     	mov	x20, x0
   50914: f94043a0     	ldr	x0, [x29, #128]
   50918: 97fefd26     	bl	0xfdb0
   5091c: 17fffff2     	b	0x508e4
   50920: 17fffff8     	b	0x50900
   50924: aa0003f4     	mov	x20, x0
   50928: f00004fa     	adrp	x26, 0xef000
   5092c: 17ffffdc     	b	0x5089c
