* converted by sgp2vbic and Q-S tweaked
.model mjl21194 npn (level=4 rcx=0.05 rbx=1.0e-01
+ rci=0.08 gamm=2.0e-06 vo=20 Hrcf=1.0 qco=1e-12
+ rbi=1.09204e+01 re=6.75706e-04 is=1e-10 nf=8.58602e-01 nr=9.25054e-01 fc=8.0e-01
+ cje=1.70807e-08 pe=4.0e-01 me=5.20397e-01 cjc=4.003635e-10 cjep=9.96365e-11 pc=9.5e-01
+ mc=2.38884e-01 cjcp=0 ps=7.5e-01 ms=5.0e-01 ibei=1e-12 nei=8.58602e-01
+ iben=7.00007e-12 nen=3.43749 ibci=1.926442e-11 nci=9.25054e-01 ibcn=3.25e-13 ncn=4
+ vef=4e+01 ver=4.18283393 ikf=7.6 nkf=5.0e-01 ikr=4.0 tf=1.0e-08
+ xtf=4.73046e+01 vtf=1.88154 itf=5.60261e-01 tr=1.0e-07 td=0 ea=1.11955
+ eaie=1.11955 eaic=1.11955 eane=1.11955 eanc=1.11955 xis=1.00001 xii=8.49187e-01
+ xin=8.49187e-01 kfn=0 afn=1
+ )

* greped from Infineon_VBIC.lib
.MODEL M_BFP780 NPN (Level=4
+ Tnom=25 Cbeo=2.47E-012 Cje=561.3E-015 Pe=0.7 Me=0.333 Aje=-1 Wbe=1
+ Cbco=10E-015 Cjc=668.6E-015 Pc=0.54 Mc=0.333 Ajc=-1 Cjep=2.616E-015
+ Cjcp=900E-015 Ps=0.6 Ms=0.3 Ajs=-0.5 Fc=0.94 Vef=545.4 Ver=3.291 Is=2.3E-015
+ Nf=0.9855 Ibei=1.893E-018 Nei=0.9399 Iben=4.77E-015 Nen=1.361 Ikf=1
+ Nr=0.9912 Ibci=157.5E-018 Nci=1.1 Ibcn=4.929E-015 Ncn=1.463 Ikr=0.01178
+ Wsp=1 Isp=1E-015 Nfp=1 Ibcip=1E-015 Ncip=1.029 Ibcnp=1E-015 Ncnp=1 Ikp=1E-3
+ Ibeip=1E-015 Ibenp=1E-015 Re=0.15 Rcx=0.01 Rci=2.665 Qco=1E-015
+ Vo=0.0005022 Gamm=5.659E-012 Hrcf=0.21 Rbx=5 Rbi=1.964 Rbp=265.5 Rs=26.56
+ Avc1=3.97 Avc2=29.52 Tf=1.6E-012 Qtf=50E-3 Xtf=30 Vtf=0.7 Itf=1 Tr=1E-015
+ Td=500E-015 Cth=0 Rth=80 Ea=1.12 Eaie=1.12 Eaic=1.12 Eais=1 Eane=1.12
+ Eanc=1.12 Eans=1 Xre=0 Xrb=0 Xrc=0 Xrs=0 Xvo=0 Xis=-1.631 Xii=0 Xin=0
+ Tnf=0 Tavc=0.002613 Kfn=0 Afn=1 Bfn=1)

* hspice vbic99_ac and vbic99_tran example
.model vbic99 npn level=9
+ LEVEL = 9 TREF=27 RCX = 10.26
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
+ RTH = 159.177 CTH = 10f VRT = 0
+ ART = 0.1 CCSO = 0 QBM = 0
+ NKF = 0.5 XIKF = 0 XRCX = 2
+ XRBX = 2 XRBP = 0 ISRR = 1
+ XISR = 0 DEAR = 0 EAP = 1.12
+ VBBE = 0 NBBE = 1 IBBE = 1e-06
+ TVBBE1 = 0 TVBBE2= 0 TNBBE = 0
+ EBBE = 0

* hspice vbic99_dc example
.model vbic99_dc npn level=9
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

* hspice vbic95 example
.model vbic95 npn Level=4
+ afn=1 ajc=-0.5 aje=0.5 ajs=0.5
+ avc1=0 avc2=0 bfn=1 cbco=0 cbeo=0 cjc=2e-14
+ cjcp=4e-13 cje=1e-13 cjep=1e-13 cth=0
+ ea=1.12 eaic=1.12 eaie=1.12 eais=1.12 eanc=1.12
+ eane=1.12 eans=1.12 fc=0.9 gamm=2e-11 hrcf=2
+ ibci=2e-17 ibcip=0 ibcn=5e-15 ibcnp=0
+ ibei=1e-18 ibeip=0 iben=5e-15 ibenp=0
+ ikf=2e-3 ikp=2e-4 ikr=2e-4 is=1e-16 isp=1e-15 itf=8e-2
+ kfn=0 mc=0.33 me=0.33 ms=0.33
+ nci=1 ncip=1 ncn=2 ncnp=2 nei=1 nen=2
+ nf=1 nfp=1 nr=1 pc=0.75 pe=0.75 ps=0.75 qco=1e-12 qtf=0
+ rbi=4 rbp=4 rbx=1 rci=6 rcx=1 re=0.2 rs=2
+ rth=300 tavc=0 td=2e-11 tf=10e-12 tnf=0 tr=100e-12
+ tnom=25 tref=25 vef=10 ver=4 vo=2
+ vtf=0 wbe=1 wsp=1
+ xii=3 xin=3 xis=3 xrb=0 xrc=0 xre=0 xrs=0 xtf=20 xvo=0

* unknown
.MODEL NX4 NPN LEVEL=4 
+ IS=1e-16 IBEI=1e-18 IBEN=5e-15 IBCI=2e-17 IBCN=5e-15 ISP=1e-15 RCX=10
+ RCI=60 RBX=10 RBI=40 RE=2 RS=20 RBP=40 VEF=10 VER=4 IKF=2e-3 ITF=8e-2
+ XTF=20 IKR=2e-4 IKP=2e-4 CJE=1e-15 CJC=10e-12 CJEP=1e-15 CJCP=1e-15 VO=2
+ GAMM=2e-11 HRCF=2 QCO=1e-12 AVC1=2 AVC2=15 TF=10e-12 TR=100e-12 TD=2e-11 RTH=300
