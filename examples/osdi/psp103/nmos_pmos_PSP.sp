***  NMOS and PMOS transistors PSP 103.8 (Id-Vgs, Vbs) (Id-Vds, Vgs) (Id-Vgs, T)  ***

nmn1  2 1 3 4 nch
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

vgsn 1 0 3.5
vdsn 2 0 0.1
vssn 3 0 0
vbsn 4 0 0

nmp1 22 11 33 44 pch
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

vgsp 11 0 -3.5
vdsp 22 0 -0.1
vssp 33 0 0
vbsp 44 0 0

* PSP modelparameters for PSP 103.3
.include Modelcards/psp103_nmos-2.mod
.include Modelcards/psp103_pmos-2.mod

.control
* Load the models dynamically
* pre_osdi ../osdi_libs/psp103.osdi
set xgridwidth=2
set xbrushwidth=3

* NMOS
dc vgsn 0 1.5 0.05 vbsn 0 -1.5 -0.3
plot vssn#branch ylabel 'Id over Vgs, Vbs 0 ... -1.5'
plot abs(vssn#branch) ylog ylabel 'Id over Vgs, Vbs 0 ... -1.5'
dc vdsn 0 1.6 0.01 vgsn 0 1.6 0.2
plot vssn#branch ylabel 'Id over Vds, Vgs 0 ... 1.6'
dc vgsn 0 1.5 0.05 temp -40 160 40
plot vssn#branch ylabel 'Id over Vds, Temp. -40 ... 160'
plot abs(vssn#branch) ylog ylabel 'Id over Vds, Temp. -40 ... 160'

* PMOS
dc vgsp 0 -1.5 -0.05 vbsp 0 1.5 0.3
plot vssp#branch ylabel 'Id over Vgs, Vbs 0 ... 1.5'
plot abs(vssp#branch) ylog ylabel 'Id over Vgs, Vbs 0 ... 1.5'
dc vdsp 0 -1.6 -0.01 vgsp 0 -1.6 -0.2
plot vssp#branch ylabel 'Id over Vds, Vgs 0 ... -1.6'
dc vgsp 0 -1.5 -0.05 temp -40 160 40
plot vssp#branch ylabel 'Id over Vds, Temp. -40 ... 160'
plot abs(vssp#branch) ylog ylabel 'Id over Vds, Temp. -40 ... 160'
.endc

.end
