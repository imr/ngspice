psp102 nch output
*
vd  d 0 dc 0.05
vg  g 0 dc 0.0
vs  s 0 dc 0.0
vb  b 0 dc 0.0
am1 d g s b nch
+L=0.1u
+W=1u
+sa=0.0e+00
*+sb=0.0e+00
*+absource=1.0e-12
*+lssource=1.0e-06
*+lgsource=1.0e-06
*+abdrain=1.0e-12
*+lsdrain=1.0e-06
*+lgdrain=1.0e-06
+mult=1.0e+00
*
.option temp=21

*.include Modelcards/psp103_nmos-2.mod
.include Modelcards/psp102_nmos.mod

.control
pre_osdi test_osdi_win/psp102.osdi
dc vd 0 2.0 0.05 vg 0 1.5 0.25
plot i(vs)
dc vg 0 1.5 0.05 vb 0 -3.0 -1
plot i(vs)
.endc

.end
