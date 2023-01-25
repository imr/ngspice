* Mix up two models
* BSIMBULK model vers. 107
* PSP model vers. 103
* simple 5-stage ring oscillator

* Power supply
.param Vcc = 1.2

* Path to the models
.include Modelcards/model.l
.include Modelcards/psp103_nmos-2.mod
.include Modelcards/psp103_pmos-2.mod

* The voltage sources: 
Vdd vdd gnd DC 'Vcc'
V1 in gnd pulse(0 'Vcc' 0p 200p 100p 1n 2n)
Vmeas vss 0 0

* The circuit: five stages
Xnot1 in vdd vss in2 notbb
Xnot2 in2 vdd vss in3 notpsp
Xnot3 in3 vdd vss in4 notbb
Xnot4 in4 vdd vss in5 notpsp
Xnot5 in5 vdd vss in notbb

* Inverter BSIMBULK
.subckt notbb a vdd vss z
Np1 z a vdd vdd BSIMBULK_osdi_P  l=0.1u  w=1u  as=0.26235p  ad=0.26235p  ps=2.51u   pd=2.51u
Nn1 z a vss vss BSIMBULK_osdi_N l=0.1u  w=0.5u as=0.131175p ad=0.131175p ps=1.52u   pd=1.52u
c3  a     vss   0.384f
c2  z     vss   0.576f
.ends

* Inverter PSP
.subckt notpsp a vdd vss z
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

* Simulation command: 
.tran 10p 10n uic

.control
* Load the models dynamically
* pre_osdi ../osdi_libs/bsimbulk107.osdi  osdi_libs/psp103.osdi
* Run the simulation
run
* Plotting
set xbrushwidth=3
plot in
* Resource usage
rusage
.endc

.end
