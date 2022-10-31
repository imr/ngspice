* PSP models
* simple 5-stage ring oscillator

.param Vcc = 1.2
.csparam vcc='Vcc'

* Path to the models
.include psp102_nmos.mod
.include psp102_pmos.mod

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
m01   z a     vdd     vdd pch  l=0.1u  w=1u  as=0.26235  ad=0.26235  ps=2.51   pd=2.51
m02   z a     vss     vss nch  l=0.1u  w=0.5u as=0.131175 ad=0.131175 ps=1.52   pd=1.52
c3  a     vss   0.384f
c2  z     vss   0.576f
.ends

* simulation command: 
.tran 10p 10n uic

.control
run
set xbrushwidth=3
plot in 
rusage
.endc

.end
