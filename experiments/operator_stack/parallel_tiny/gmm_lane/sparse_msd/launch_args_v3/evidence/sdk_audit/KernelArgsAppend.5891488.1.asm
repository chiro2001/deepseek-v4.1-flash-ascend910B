
/usr/local/Ascend/cann-9.1.0/lib64/libruntime_v100.so:	file format elf64-littleaarch64

Disassembly of section .text:

000000000005c3a0 <_ZNSsC2EPKcRKSaIcE>:
   c1318: b5001601     	cbnz	x1, 0xc15d8
   c131c: a94153f3     	ldp	x19, x20, [sp, #16]
   c1320: a9425bf5     	ldp	x21, x22, [sp, #32]
   c1324: a94363f7     	ldp	x23, x24, [sp, #48]
   c1328: f94023f9     	ldr	x25, [sp, #64]
   c132c: a8ca7bfd     	ldp	x29, x30, [sp], #160
   c1330: d65f03c0     	ret
   c1334: d503201f     	nop
   c1338: f942a321     	ldr	x1, [x25, #1344]
   c133c: d2807800     	mov	x0, #960
   c1340: 97fe0a60     	bl	0x43cc0
   c1344: aa0003f8     	mov	x24, x0
   c1348: b4000d00     	cbz	x0, 0xc14e8
   c134c: aa1403e3     	mov	x3, x20
   c1350: aa1703e2     	mov	x2, x23
   c1354: aa1503e1     	mov	x1, x21
   c1358: 940da55a     	bl	0x42a8c0
   c135c: f90002d8     	str	x24, [x22]
   c1360: aa1803e0     	mov	x0, x24
   c1364: f9400301     	ldr	x1, [x24]
   c1368: f9403021     	ldr	x1, [x1, #96]
   c136c: d63f0020     	blr	x1
   c1370: 2a0003f4     	mov	w20, w0
   c1374: 35001020     	cbnz	w0, 0xc1578
   c1378: f94002d8     	ldr	x24, [x22]
   c137c: 17ffffd5     	b	0xc12d0
   c1380: f90002df     	str	xzr, [x22]
   c1384: 910223b4     	add	x20, x29, #136
   c1388: aa1403e8     	mov	x8, x20
   c138c: d2806400     	mov	x0, #800
   c1390: 9400313e     	bl	0xcd888
   c1394: 9101c3b5     	add	x21, x29, #112
   c1398: 90002281     	adrp	x1, 0x511000
   c139c: aa1503e2     	mov	x2, x21
   c13a0: 91040021     	add	x1, x1, #256
   c13a4: 91002280     	add	x0, x20, #8
   c13a8: 97fe6bfe     	bl	0x5c3a0
   c13ac: aa1403e1     	mov	x1, x20
   c13b0: d2800042     	mov	x2, #2
   c13b4: aa1503e0     	mov	x0, x21
   c13b8: 97fe9ad2     	bl	0x67f00
   c13bc: b00026e2     	adrp	x2, 0x59e000
   c13c0: 9134e042     	add	x2, x2, #3384
   c13c4: aa1503e3     	mov	x3, x21
   c13c8: 91168042     	add	x2, x2, #1440
   c13cc: 5280f821     	mov	w1, #1985
   c13d0: 52800180     	mov	w0, #12
   c13d4: 97fefeb5     	bl	0x80ea8
   c13d8: aa1503e0     	mov	x0, x21
   c13dc: d0002df4     	adrp	x20, 0x67f000
   c13e0: 940026c4     	bl	0xcaef0
   c13e4: f9404ba0     	ldr	x0, [x29, #144]
   c13e8: f9429e81     	ldr	x1, [x20, #1336]
   c13ec: d1006000     	sub	x0, x0, #24
   c13f0: eb01001f     	cmp	x0, x1
   c13f4: 54000e61     	b.ne	0xc15c0
   c13f8: f94047a0     	ldr	x0, [x29, #136]
   c13fc: f9429e94     	ldr	x20, [x20, #1336]
   c1400: d1006000     	sub	x0, x0, #24
   c1404: eb14001f     	cmp	x0, x20
   c1408: 54000e41     	b.ne	0xc15d0
   c140c: 52800054     	mov	w20, #2
   c1410: 72a0e0b4     	movk	w20, #1797, lsl #16
   c1414: 17ffffba     	b	0xc12fc
   c1418: b00026e3     	adrp	x3, 0x59e000
   c141c: 9134e063     	add	x3, x3, #3384
   c1420: aa1703e5     	mov	x5, x23
   c1424: 91168063     	add	x3, x3, #1440
   c1428: 90002284     	adrp	x4, 0x511000
   c142c: 91108084     	add	x4, x4, #1056
   c1430: 5280f642     	mov	w2, #1970
   c1434: 52800000     	mov	w0, #0
   c1438: 90002221     	adrp	x1, 0x505000
   c143c: 91136021     	add	x1, x1, #1240
   c1440: 97fe03b8     	bl	0x42320
   c1444: 12800001     	mov	w1, #-1
   c1448: aa1403e0     	mov	x0, x20
   c144c: 97fe97ed     	bl	0x67400
   c1450: aa0003f4     	mov	x20, x0
   c1454: 52800021     	mov	w1, #1
   c1458: 940aa5d8     	bl	0x36abb8
   c145c: 72001c1f     	tst	w0, #0xff
   c1460: 54fff0e1     	b.ne	0xc127c
   c1464: d0002223     	adrp	x3, 0x507000
   c1468: 91014063     	add	x3, x3, #80
   c146c: b00026f5     	adrp	x21, 0x59e000
   c1470: 90002234     	adrp	x20, 0x505000
   c1474: 9134e2b5     	add	x21, x21, #3384
   c1478: 91136294     	add	x20, x20, #1240
   c147c: 911682a2     	add	x2, x21, #1440
   c1480: 5280f681     	mov	w1, #1972
   c1484: aa1403e0     	mov	x0, x20
   c1488: 9101c3b6     	add	x22, x29, #112
   c148c: 97fe0469     	bl	0x42630
   c1490: 910183a2     	add	x2, x29, #96
   c1494: aa1603e0     	mov	x0, x22
   c1498: d0002221     	adrp	x1, 0x507000
   c149c: 91018021     	add	x1, x1, #96
   c14a0: 97fe6bc0     	bl	0x5c3a0
   c14a4: d0002224     	adrp	x4, 0x507000
   c14a8: aa1603e3     	mov	x3, x22
   c14ac: 5280f682     	mov	w2, #1972
   c14b0: 9101a084     	add	x4, x4, #104
   c14b4: 9116c2a1     	add	x1, x21, #1456
   c14b8: aa1403e0     	mov	x0, x20
   c14bc: 97fe0525     	bl	0x42950
   c14c0: d0002de1     	adrp	x1, 0x67f000
   c14c4: f9403ba0     	ldr	x0, [x29, #112]
   c14c8: f9429c21     	ldr	x1, [x1, #1336]
   c14cc: d1006000     	sub	x0, x0, #24
   c14d0: eb01001f     	cmp	x0, x1
   c14d4: 540007a1     	b.ne	0xc15c8
   c14d8: 52800034     	mov	w20, #1
   c14dc: 72a0e0f4     	movk	w20, #1799, lsl #16
   c14e0: 17ffff89     	b	0xc1304
   c14e4: d503201f     	nop
   c14e8: f90002df     	str	xzr, [x22]
   c14ec: 910223b4     	add	x20, x29, #136
   c14f0: aa1403e8     	mov	x8, x20
   c14f4: d2807800     	mov	x0, #960
   c14f8: 940030e4     	bl	0xcd888
   c14fc: 90002281     	adrp	x1, 0x511000
   c1500: 910163a2     	add	x2, x29, #88
   c1504: 91040021     	add	x1, x1, #256
   c1508: 91002280     	add	x0, x20, #8
   c150c: 97fe6ba5     	bl	0x5c3a0
   c1510: 9101c3b5     	add	x21, x29, #112
   c1514: aa1403e1     	mov	x1, x20
   c1518: d2800042     	mov	x2, #2
   c151c: aa1503e0     	mov	x0, x21
   c1520: 97fe9a78     	bl	0x67f00
   c1524: b00026e2     	adrp	x2, 0x59e000
   c1528: 9134e042     	add	x2, x2, #3384
   c152c: aa1503e3     	mov	x3, x21
   c1530: 91168042     	add	x2, x2, #1440
   c1534: 5280f741     	mov	w1, #1978
   c1538: 52800180     	mov	w0, #12
   c153c: 97fefe5b     	bl	0x80ea8
   c1540: 17ffffa6     	b	0xc13d8
   c1544: d503201f     	nop
   c1548: b00026e2     	adrp	x2, 0x59e000
   c154c: 9134e042     	add	x2, x2, #3384
   c1550: d0002223     	adrp	x3, 0x507000
   c1554: 90002220     	adrp	x0, 0x505000
   c1558: 913da063     	add	x3, x3, #3944
   c155c: 91168042     	add	x2, x2, #1440
   c1560: 5280f6c1     	mov	w1, #1974
   c1564: 91136000     	add	x0, x0, #1240
   c1568: 97fe0432     	bl	0x42630
   c156c: 52800034     	mov	w20, #1
   c1570: 72a0e234     	movk	w20, #1809, lsl #16
   c1574: 17ffff62     	b	0xc12fc
   c1578: b00026e2     	adrp	x2, 0x59e000
   c157c: 9134e042     	add	x2, x2, #3384
   c1580: 90002283     	adrp	x3, 0x511000
   c1584: 90002220     	adrp	x0, 0x505000
   c1588: 2a1403e4     	mov	w4, w20
   c158c: 91132063     	add	x3, x3, #1224
   c1590: 91168042     	add	x2, x2, #1440
   c1594: 5280f7a1     	mov	w1, #1981
   c1598: 91136000     	add	x0, x0, #1240
   c159c: 97fe0425     	bl	0x42630
   c15a0: f94002c0     	ldr	x0, [x22]
