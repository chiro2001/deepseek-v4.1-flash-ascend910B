
/usr/local/Ascend/cann-9.1.0/lib64/libruntime_v100.so:	file format elf64-littleaarch64

Disassembly of section .text:

000000000005c3a0 <_ZNSsC2EPKcRKSaIcE>:
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
   c15a4: b4ffeac0     	cbz	x0, 0xc12fc
   c15a8: f9400001     	ldr	x1, [x0]
   c15ac: f9400421     	ldr	x1, [x1, #8]
   c15b0: d63f0020     	blr	x1
   c15b4: f90002df     	str	xzr, [x22]
   c15b8: 17ffff51     	b	0xc12fc
   c15bc: d503201f     	nop
   c15c0: 97fe9982     	bl	0x67bc8
   c15c4: 17ffff8d     	b	0xc13f8
   c15c8: 97fe9980     	bl	0x67bc8
   c15cc: 17ffffc3     	b	0xc14d8
   c15d0: 97fe997e     	bl	0x67bc8
   c15d4: 17ffff8e     	b	0xc140c
   c15d8: 97fe0222     	bl	0x41e60
   c15dc: aa0003f3     	mov	x19, x0
   c15e0: d0002df4     	adrp	x20, 0x67f000
   c15e4: f9404ba0     	ldr	x0, [x29, #144]
   c15e8: f9429e81     	ldr	x1, [x20, #1336]
   c15ec: d1006000     	sub	x0, x0, #24
   c15f0: eb01001f     	cmp	x0, x1
   c15f4: 54000040     	b.eq	0xc15fc
   c15f8: 97fe9974     	bl	0x67bc8
   c15fc: f94047a0     	ldr	x0, [x29, #136]
   c1600: f9429e94     	ldr	x20, [x20, #1336]
   c1604: d1006000     	sub	x0, x0, #24
   c1608: eb14001f     	cmp	x0, x20
   c160c: 54000040     	b.eq	0xc1614
   c1610: 97fe996e     	bl	0x67bc8
   c1614: 910183a0     	add	x0, x29, #96
   c1618: 940aafaa     	bl	0x36d4c0
   c161c: aa1303e0     	mov	x0, x19
   c1620: 97fe08a4     	bl	0x438b0
   c1624: f942a321     	ldr	x1, [x25, #1344]
   c1628: aa0003f3     	mov	x19, x0
   c162c: aa1803e0     	mov	x0, x24
   c1630: 97fe031c     	bl	0x422a0
   c1634: 17fffff8     	b	0xc1614
   c1638: 17ffffe9     	b	0xc15dc
   c163c: aa0003f3     	mov	x19, x0
   c1640: 17fffff5     	b	0xc1614
   c1644: aa0003f3     	mov	x19, x0
   c1648: aa1503e0     	mov	x0, x21
   c164c: 94002629     	bl	0xcaef0
   c1650: 17ffffe4     	b	0xc15e0
   c1654: d0002de2     	adrp	x2, 0x67f000
   c1658: f9403ba1     	ldr	x1, [x29, #112]
   c165c: aa0003f3     	mov	x19, x0
   c1660: f9429c42     	ldr	x2, [x2, #1336]
   c1664: d1006020     	sub	x0, x1, #24
   c1668: eb02001f     	cmp	x0, x2
   c166c: 54fffd80     	b.eq	0xc161c
   c1670: 97fe9956     	bl	0x67bc8
   c1674: 17ffffea     	b	0xc161c
   c1678: 17ffffeb     	b	0xc1624
   c167c: 17fffff2     	b	0xc1644
   c1680: d10303ff     	sub	sp, sp, #192
   c1684: a9017bfd     	stp	x29, x30, [sp, #16]
   c1688: 910043fd     	add	x29, sp, #16
   c168c: a90253f3     	stp	x19, x20, [sp, #32]
   c1690: d0002df4     	adrp	x20, 0x67f000
   c1694: a9035bf5     	stp	x21, x22, [sp, #48]
   c1698: aa0203f5     	mov	x21, x2
   c169c: a90463f7     	stp	x23, x24, [sp, #64]
   c16a0: aa0003f8     	mov	x24, x0
   c16a4: f9002bf9     	str	x25, [sp, #80]
   c16a8: aa0103f6     	mov	x22, x1
   c16ac: f9434680     	ldr	x0, [x20, #1672]
   c16b0: 2a0303f7     	mov	w23, w3
   c16b4: 52800021     	mov	w1, #1
   c16b8: 12003c99     	and	w25, w4, #0xffff
   c16bc: f9400002     	ldr	x2, [x0]
   c16c0: f90057a2     	str	x2, [x29, #168]
   c16c4: d2800002     	mov	x2, #0
   c16c8: 528000e0     	mov	w0, #7
   c16cc: 97fe05a1     	bl	0x42d50
   c16d0: 7100041f     	cmp	w0, #1
   c16d4: 540006a0     	b.eq	0xc17a8
   c16d8: b4001d56     	cbz	x22, 0xc1a80
