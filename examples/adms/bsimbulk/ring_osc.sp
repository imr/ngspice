* Sample netlist: 17-stage ring oscillator *

.options abstol=1e-6 reltol=1e-6

*.hdl "bsimbulk.va"
.include "model.l"

v1 vdd 0 dc=1.0

.subckt inv vin vout vdd vss
    mn vout vin vss vss nch W=10e-6 L=10e-6
    mp vout vin vdd vdd pch W=10e-6 L=10e-6
.ends

x1 1 2 vdd 0 inv
x2 2 3 vdd 0 inv
x3 3 4 vdd 0 inv
x4 4 5 vdd 0 inv
x5 5 6 vdd 0 inv
x6 6 7 vdd 0 inv
x7 7 8 vdd 0 inv
x8 8 9 vdd 0 inv
x9 9 10 vdd 0 inv
x10 10 11 vdd 0 inv
x11 11 12 vdd 0 inv
x12 12 13 vdd 0 inv
x13 13 14 vdd 0 inv
x14 14 15 vdd 0 inv
x15 15 16 vdd 0 inv
x16 16 17 vdd 0 inv
x17 17 1 vdd 0 inv

*.ic 1=1

.tran 1n 10u
.print tran v(1)
.measure tran t1 when v(1)=0.5 cross=1
.measure tran t2 when v(1)=0.5 cross=7
*.measure tran period param'(t2-t1)/3'
*.measure tran delay_per_stage param'period/34'

.control
run
plot v(1)
.endc

.end
