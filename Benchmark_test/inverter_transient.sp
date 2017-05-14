*Sample netlist for BSIM6.0
*Inverter Transient

.option abstol=1e-6 reltol=1e-6 post ingold
.hdl "bsim6.va"
.include "modelcard.nmos"
.include "modelcard.pmos"

* --- Voltage Sources ---
vdd   supply  0 dc=1.0
vin  vi  0 dc=0.5 sin (0.5 0.5 1MEG)

* --- Inverter Subcircuit ---
.subckt inverter vin vout vdd gnd
    Xp1 vout vin vdd gnd pmos W=10u L=10u
    Xn1 vout vin gnd gnd nmos W=10u L=10u
.ends

* --- Inverter ---
Xinv1  vi 1 supply 0 inverter
Xinv2  1 2 supply 0 inverter
Xinv3  2 3 supply 0 inverter
Xinv4  3 4 supply 0 inverter
Xinv5  4 vo supply 0 inverter

* --- Transient Analysis ---
.tran 10n 5u

.print tran v(vi) v(vo)

.end
