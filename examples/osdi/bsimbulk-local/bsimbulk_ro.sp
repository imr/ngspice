* BSIMBULK model vers. 107
* simple 5-stage ring oscillator

.param Vcc = 1.2
.csparam vcc='Vcc'

* Path to the models
.include Modelcards/model.l

* the voltage sources: 
Vdd vdd gnd DC 'Vcc'
V1 in gnd pulse(0 'Vcc' 0p 200p 100p 1n 2n)
Vmeas vss 0 0

Xnot1 in vdd vss in2 not1
Xnot2 in2 vdd vss in3 not1
Xnot3 in3 vdd vss in4 not1
Xnot4 in4 vdd vss in5 not1
Xnot5 in5 vdd vss in not1

*Rout out 0 1k

.subckt not1 a vdd vss z
Np1 z a vdd vdd BSIMBULK_osdi_P  l=0.1u  w=1u  as=0.26235p  ad=0.26235p  ps=2.51u   pd=2.51u
Nn1 z a vss vss BSIMBULK_osdi_N l=0.1u  w=0.5u as=0.131175p ad=0.131175p ps=1.52u   pd=1.52u
c3  a     vss   0.384f
c2  z     vss   0.576f
.ends

* simulation command: 
.tran 10p 10n uic

.control
pre_osdi osdi_libs/bsimbulk107.osdi
run
set xbrushwidth=3
plot in 
rusage
.endc

.end
