psp102 pch output
*
vd  d 0 dc -0.1
vg  g 0 dc 0.0
vs  s 0 dc 0.0
vb  b 0 dc 0.0
m1  d g s b pch
+l=0.1u
+w=1u
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
dc vd 0 -3.0 -0.05 vg 0 -1 -0.25 ; saturation
plot i(vs)
dc vg 0 -1.5 -0.05 vb 0 3.0 1
plot i(vs)
.endc
*

.include psp102_pmos.mod
*.include psp103_pmos.mod

.end
