HICUM2 Output Test Ic=f(Vc,Ib)

IB 0 B 200n
VC C 0 3.0
VS S 0 0.0
Q1 C B 0 S MOD

.control
dc vc 0.0 2.0 0.05 ib 5u 25u 5u
run
plot abs(i(vc))
.endc

.model MOD NPN LEVEL=8 tnom=27.00
+     c10=3.680e-30     qp0=1.480e-14     ich=2.470e-02     hfc=1.000e+00
+     hfe=1.000e+00    hjci=4.000e-02    hjei=1.000e+00    alit=4.500e-01
+     mcf=1.015e+00
+   cjei0=2.830e-14    vdei=7.500e-01     zei=3.660e-01
+   cjci0=1.690e-14    vdci=9.090e-01     zci=4.340e-01   vptci=1.370e+01
+      t0=2.310e-12    dt0h=5.000e-14    tbvl=0.000e+00    tef0=2.500e-13
+    gtfe=1.000e+00    thcs=9.000e-11    fthc=6.000e-01
+    alqf=2.250e-01
+    rci0=3.470e+01    vlim=8.000e-01     vpt=1.000e+01    vces=8.000e-02
+      tr=0.000e+00
+   ibeis=5.750e-19    mbei=1.065e+00   ireis=2.870e-15    mrei=2.045e+00
+   ibcis=0.000e+00    mbci=1.500e+00
+    favl=8.200e+01    qavl=3.220e-13
+    rbi0=3.559e+01   fdqr0=1.320e-01    fgeo=7.397e-01     fqi=8.957e-01
+   fcrbi=2.000e-01
+    latb=8.035e-01    latl=8.300e-02
+   cjep0=3.660e-15    vdep=7.500e-01     zep=8.530e-01
+   ibeps=1.020e-18    mbep=1.070e+00   ireps=5.090e-15    mrep=2.245e+00
+   ibets=0.000e+00    abet=4.000e+01
+   cjcx0=1.590e-14    vdcx=6.170e-01     zcx=2.570e-01   vptcx=2.550e+00
*obsolete in va2.2     ccox=4.670e-15     fbc=9.990e-01
+   ibcxs=2.200e-16    mbcx=1.172e+00
*obsolete in va2.2     ceox=1.420e-14
+     rbx=2.136e+01      re=3.459e+00     rcx=1.686e+01
+    itss=5.820e-19     msf=1.080e+00     tsf=1.000e-09    iscs=4.920e-14
+     msc=1.050e+00
*obsolete in va2.2      msr=1.000e+00
+    cjs0=4.720e-14     vds=5.480e-01      zs=2.430e-01    vpts=1.000e+10
+     rsu=0.000e+00     csu=0.000e+00
+      kf=4.870e-13      af=1.000e+00
*obsolete in va2.2    aljei=1.880e+00
*obsolete in va2.2    aljep=2.200e+00
*obsolete in va2.2     alhc=4.000e-02
*obsolete in va2.2     krbi=1.000e+00
+     vgb=1.014e+00     alb=-7.000e-03    alt0=1.000e-04     kt0=1.000e-07
+  zetaci=9.300e-01    alvs=1.000e-03   alces=4.000e-04 zetarbi=3.300e-01
+ zetarbx=1.000e-02 zetarcx=3.100e-01  zetare=0.000e+00   alfav=0.000e+00
+   alqav=0.000e+00     rth=0.000e+00     cth=0.000e+00

.end
