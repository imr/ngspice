.param
+qnva_is=1.000e+00
+qnva_re=1.000e+00
+qnva_cje=1.000e+00
+qnva_cjc=1.000e+00
+qnva_vef=1.000e+00
+qnva_rcx=1.000e+00
+qnva_rbx=1.000e+00
+qnva_rci=1.000e+00
+qnva_ibei=1.000e+00
+qnva_iben=0.000e+00
+qnva_cjep=1.000e+00
+qnva_cjcp=1.000e+00
+qnva_ais=0.000e+00
+qnva_aibei=0.000e+00
+qnva_aiben=0.000e+00

.subckt qnva c b e s le=2.02 par1=1
q1 c b e s qnvamod area=1
.model qnvamod npn level=9
+ tref=2.700e+01
+ ebbe=2.111e-05
+ vo=0.000e+00
+ gamm=1.493e-13               hrcf=0.000e+00
+ nf=1.000e+00                 nr=1.008e+00
+ fc=8.300e-01                 pe=6.500e-01
+ me=3.500e-01                 aje=-5.000e-01
+ qco=0.000e+00                pc=6.500e-01
+ mc=3.200e-01                 ajc=-5.000e-01
+ ps=6.000e-01                 ms=3.700e-01
+ ajs=-5.000e-01               wbe=4.800e-01
+ nei=1.000e+00                nen=2.000e+00
+ nci=1.000e+00                ncn=2.100e+00
+ avc1=2.258e+00
+ wsp=1.000e+00                nfp=1.000e+00
+ ncip=1.000e+00               ncnp=2.000e+00
+ ver=8.807e+00                qtf=0.000e+00
+ xtf=3.770e+00                vtf=1.667e+00
+ tr=3.500e-10                 td=0.000e+00
+ afn=1.000e+00                kfn=5.500e-14
+ bfn=1.000e+00
+ xrbi=0.000e+00               xrci=0.000e+00
+ xre=0.000e+00                xrs=0.000e+00
+ xvo=0.000e+00                ea=1.130e+00
+ eaie=1.130e+00               eaic=1.110e+00
+ eais=1.110e+00               eane=1.110e+00
+ eanc=1.110e+00               eans=1.110e+00
+ xis=4.864e+00                xii=2.952e+00
+ xin=2.952e+00
+ tnf=0.000e+00                tavc=6.580e-04
+ rth='(0.000e+00/le+0.000e+00)'
+ cth='(0.000e+00*le+0.000e+00)'
+ vrt=0.000e+00                art=1.000e-01
+ qbm=1.000e+00                nkf=5.000e-01
+ xikf=1.000e+00               xrcx=0.000e+00
+ xrbx=0.000e+00               xrbp=0.000e+00
+ isrr=1.230e+00               xisr=0.000e+00
+ dear=0.000e+00               eap=1.110e+00
+ vbbe=2.266e+00               nbbe=6.584e+00
+ tvbbe1=2.000e-04             tvbbe2=0.000e+00
+ tnbbe=-1.900e-03
+ vef='qnva_vef*1.500e+01'
+ tf='1.000e+00*4.300e-12'
+ rcx='qnva_rcx*(1/(le/3.800e+02+1/3.000e+02))'
+ rci='qnva_rci*(1/(le/7.200e+02+1/4.800e+02))'
+ rbx='qnva_rbx*(1/(le/2.000e+02+1/1.000e+04))'
+ rbi='1.000e+00*(1/(le/3.000e+01+1/1.000e+04))'
+ rbp='1/(le/3.000e+01+1/1.000e+04)'
+ re='qnva_re*(1.800e+01/le+(2.778e+00))'
+ rs='1/(le/1.500e+01+1/1.000e+04)'
+ is='(1+qnva_ais/sqrt(par1*le))*qnva_is*(le*1.181e-18+(3.202e-19))'
+ cbeo='(le*0.000e+00+(5.000e-15))'
+ cbco='(le*0.000e+00+(3.000e-15))'
+ cje='qnva_cje*(le*2.326e-15+(-4.733e-30))'
+ cjc='qnva_cjc*(le*1.196e-15+(2.381e-15))'
+ cjep='qnva_cjep*(le*1.196e-15+(2.381e-15))'
+ cjcp='qnva_cjcp*(le*1.824e-15+(1.814e-14))'
+ ibei='(1+qnva_aibei/sqrt(par1*le))*qnva_ibei*(le*4.032e-20+(3.890e-21))'
+ iben='exp((qnva_aiben/sqrt(par1*le))+(qnva_iben))*(le*4.125e-16+(-6.420e-20))'
+ ibci='(le*1.220e-20+(1.086e-23))'
+ ibcn='(le*1.636e-15+(1.557e-18))'
+ isp='(le*1.137e-18+(2.280e-18))'
+ ibeip='(le*6.558e-20+(2.802e-19))'
+ ibenp='(le*3.311e-20+(-5.172e-24))'
+ ibcip='(le*4.290e-17+(7.958e-16))'
+ ibcnp='(le*6.212e-16+(-1.632e-19))'
+ ikf='1.000e+00*(le*1.253e-03+(2.165e-07))'
+ ikr='(le*1.540e-04+(-1.077e-07))'
+ ikp='(le*4.007e-05+(4.615e-05))'
+ itf='(le*5.037e-03+(-2.423e-06))'
+ ccso='(le*0.000e+00+(1.000e-18))'
+ ibbe='(le*7.769e-09+(-1.684e-12))'
+ avc2='1/(1.590e-03/le+1/1.892e+01)'
.ends qnva
