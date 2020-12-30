* Sample netlist: Inverter transient *

.option post ingold numdgt=10
.temp 27

*.hdl "bsimbulk.va"
.include "model.l"

v1 vdd 0 dc=1.0
v2 in 0 dc=0.5 sin(0.5 0.5 1meg)

.subckt inv vin vout vdd vss
    mn vout vin vss vss nch W=10u L=10u
    mp vout vin vdd vdd pch W=10u L=10u
.ends

x1 in 1 vdd 0 inv
x2 1 2 vdd 0 inv
x3 2 3 vdd 0 inv
x4 3 4 vdd 0 inv
x5 4 out vdd 0 inv

.tran 10n 5u

.control
run
plot v(in) v(out)
.endc

.end
