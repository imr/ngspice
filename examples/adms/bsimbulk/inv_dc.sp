* Sample netlist: Inverter DC *

.option post ingold numdgt=10
.temp 27

*.hdl "bsimbulk.va"
.include "model.l"

v1 vdd 0 dc=1.0
v2 in 0 dc=0.5

.subckt inv vin vout vdd vss
    mn vout vin vss vss nch W=10u L=10u
    mp vout vin vdd vdd pch W=10u L=10u
.ends

x1 in out vdd 0 inv

.dc v2 0 1 0.01

.control
run
plot v(in) v(out)
.endc

.end
