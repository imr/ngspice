*Sample netlist for BSIM6.0
*Inverter DC Analysis
* (exec-spice "ngspice %s" t)

.option abstol=1e-6 reltol=1e-6 post ingold
.include "modelcard.nmos"
.include "modelcard.pmos"
* --- Voltage Sources ---
vdd   supply  0 dc=1.0
vin   vi 0 dc=0.5

* --- Inverter Subcircuit ---
.subckt inverter vin vout vdd gnd
    mXp1 vout vin vdd gnd  mp W=10u L=10u
    mXn1 vout vin gnd gnd  mn W=10u L=10u
.ends

* --- Inverter ---
Xinv1  vi vo supply 0 inverter

* --- Transient Analysis ---
.dc vin 0 1 0.01

.print dc v(vi) v(vo)

.control
run
plot v(vi) v(vo)
.endc
.end
