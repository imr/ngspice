***** VBIC99 level 9 Transient test *****
*
q1 c b 0 0 t vbic99
vcc vp 0 dc 5.0
vin in 0 dc 2.5 pulse (0 5 0 1n 1n 10n 25n)
r1 in b 100
r2 c vp 1k
*
.control
op
tran 50p 100n
set xbrushwidth=2
plot v(in) v(b) v(c) v(vp)
settype temperature v(t) 
plot v(t)
.endc
*
.model vbic99 npn
+ LEVEL = 9 TREF = 27.0 RCX = 10.26
+ RCI = 0.001 VO = 0 GAMM = 0
+ HRCF = 0 RBX = 122.23 RBI = 0.001
+ RE = 17.61 RS = 1 RBP = 1
+ IS = 4.70047e-25 NF = 1.09575 NR = 1.02
+ FC = 0.9 CBEO = 0 CJE = 7e-15
+ PE = 0.75 ME = 0.33 AJE = -0.5
+ CBCO = 0 CJC = 1.1e-14 QCO = 0
+ CJEP = 0 PC = 0.75 MC = 0.33
+ AJC = -0.5 CJCP = 3e-15 PS = 0.75
+ MS = 0.33 AJS = -0.5 IBEI = 1.484e-23
+ WBE = 1 NEI = 1.302 IBEN = 6.096e-18
+ NEN = 2.081 IBCI = 5.618e-24 NCI = 1.11
+ IBCN = 3.297e-14 NCN = 2 AVC1 = 0
+ AVC2 = 0 ISP = 0 WSP = 1
+ NFP = 1 IBEIP = 0 IBENP = 0
+ IBCIP = 0 NCIP = 1 IBCNP = 0
+ NCNP = 2 VEF = 800 VER = 700
+ IKF = 0 IKR = 0 IKP = 0
+ TF = 2.3e-12 QTF = 0 XTF = 0
+ VTF = 0 ITF = 0 TR = 0
+ TD = 1e-15 KFN = 0 AFN = 1
+ BFN = 1 XRE = 2 XRBI = 2
+ XRCI = 2 XRS = 2 XVO = 0
+ EA = 1.1095 EAIE = 1.489271 EAIC = 1.489271
+ EAIS = 1.12 EANE = 1.489271 EANC = 1.489271
+ EANS = 1.12 XIS = 3 XII = 3
+ XIN = 3 TNF = 0 TAVC = 0
+ RTH = 159.177 CTH = 100p VRT = 0
+ ART = 0.1 CCSO = 0 QBM = 0
+ NKF = 0.5 XIKF = 0 XRCX = 2
+ XRBX = 2 XRBP = 0 ISRR = 1
+ XISR = 0 DEAR = 0 EAP = 1.12
+ VBBE = 0 NBBE = 1 IBBE = 1e-06
+ TVBBE1 = 0 TVBBE2 = 0 TNBBE = 0
*
.end

