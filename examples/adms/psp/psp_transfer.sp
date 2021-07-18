psp102 nch transfer
*
vd  d 0 dc 0.1
vg  g 0 dc 0.0
vs  s 0 dc 0.0
vb  b 0 dc 0.0
m1  d g s b nch
+l=1.0e-06
+w=10.0e-06
+sa=0.0e+00
+sb=0.0e+00
+absource=1.0e-12
+lssource=1.0e-06
+lgsource=1.0e-06
+abdrain=1.0e-12
+lsdrain=1.0e-06
+lgdrain=1.0e-06
+mult=1.0e+00
*
.option temp=21
.control
dc vg 0 1.5 0.02 vb -3 0 0.5
plot abs(i(vd))
dc vg 0 1.5 0.01 vb -3 0 0.5
plot abs(i(vd)) ylog ylimit 1e-12 1e-03
.endc
*
.include psp102_nmos.mod

.end
