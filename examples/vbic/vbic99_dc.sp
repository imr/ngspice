***** VBIC99 level9 DC test *****
.OPTION gmin=1.0e-15
vbe bx 0 0
vcb cx bx 0
vib bx b 0
vic cx c 0
ve ex 0 0
vie ex e 0
vs sx 0 0
vis sx s 0
q1 c b e s dt vbic99 area=1 m=1
.model vbic99 npn level=9
+tref = 27.0 rcx = 10.0 rci = 60.0 vo = 2.0
+gamm = 2e-11 hrcf = 2.0 rbx = 10.0 rbi = 40.0
+re = 2.0 rs = 20.0 rbp = 40.0 is = 1.0e-16
+nf = 1.0 nr = 1.0 fc = 0.9 cbeo = 0.0
+cje = 1.0e-13 pe = 0.75 me = 0.33 aje = -0.5
+cbco = 0.0 cjc = 2e-14 qco = 1e-12 cjep = 1e-13
+pc = 0.75 mc = 0.33 ajc = -0.5 cjcp = 4e-13
+ps = 0.75 ms = 0.33 ajs = -0.5 ibei = 1.0e-18
+wbe = 1.0 nei = 1.0 iben = 5.0e-15 nen = 2.0
+ibci = 2.0e-17 nci = 1.0 ibcn = 5.0e-15 ncn = 2.0
+avc1 = 2.0 avc2 = 15.0 isp = 1.0e-15 wsp = 1.0
+nfp = 1.0 ibeip = 0.0 ibenp = 0.0 ibcip = 0.0
+ncip = 1.0 ibcnp = 0.0 ncnp = 2.0 vef = 10.0
+ver = 4.0 ikf = 2e-3 ikr = 2e-4 ikp = 2e-4
+tf = 10e-12 qtf = 0.0 xtf = 20.0 vtf = 0.0
+itf = 8e-2 tr = 100e-12 td = 1e-20 kfn = 0.0
+afn = 1.0 bfn = 1.0 xre = 0 xrbi = 0
+xrci = 0 xrs = 0 xvo = 0 ea = 1.12
+eaie = 1.12 eaic = 1.12 eais = 1.12 eane = 1.12
+eanc = 1.12 eans = 1.12 xis = 3.0 xii = 3.0
+xin = 3.0 tnf = 0.0 tavc = 0.0 rth = 300.0
+cth = 0.0 vrt = 0.0 art = 0.1 ccso = 0.0
+qbm = 0.0 nkf = 0.5 xikf = 0 xrcx = 0
+xrbx = 0 xrbp = 0 isrr = 1.0 xisr = 0.0
+dear = 0.0 eap = 1.12 vbbe = 0.0 nbbe = 1.0
+ibbe = 1.0e-6 tvbbe1 = 0.0 tvbbe2 = 0.0 tnbbe = 0.0
+ebbe = 0.0
.temp 27
.control
dc vbe 0.1 1.1 0.02
plot i(vib) i(vic) abs(i(vis)) ylog
plot v(dt)
.endc
.end
