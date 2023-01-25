* PSP models
* simple inverter

.param Vcc = 1.2
.csparam vcc='Vcc'

* Path to the models
.include Modelcards/psp103_nmos-2.mod
.include Modelcards/psp103_pmos-2.mod

* the voltage sources: 
Vdd vdd gnd DC 'Vcc'
V1 in gnd pulse(0 'Vcc' 0p 200p 100p 1n 2n)
Vmeas vss 0 0

Xnot1 in vdd vss out not1
*Rout out 0 1k

.subckt not1 a vdd vss z
*m01   z a     vdd     vdd pch  l=0.1u  w=1u  as=0.26235  ad=0.26235  ps=2.51   pd=2.51
nmp1  z a     vdd     vdd pch
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

*m02   z a     vss     vss nch  l=0.1u  w=0.5u as=0.131175 ad=0.131175 ps=1.52   pd=1.52
nmn1  z a     vss     vss nch
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
c3  a     vss   0.384f
c2  z     vss   0.576f
.ends

* simulation command: 
.tran 10ps 10ns
.dc V1 0 'vcc' 'vcc/100'

.control
* pre_osdi ../osdi_libs/psp103.osdi
run
*set nolegend
set xbrushwidth=2
plot in out
plot dc1.out
plot dc1.i(Vmeas)
rusage
.endc

.end
