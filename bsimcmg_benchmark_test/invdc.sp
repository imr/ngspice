*Sample netlist for BSIM-MG
*Inverter Transient

.option abstol=1e-6 reltol=1e-6 post ingold

.hdl "bsimcmg.va"
.include "modelcard.nmos"
.include "modelcard.pmos"

* --- Voltage Sources ---
vdd   supply  0 dc=1.0
vin   vi 0 dc=0.5

* --- Inverter Subcircuit ---
.subckt mg_inv vin vout vdd gnd
    Xp1 vout vin vdd gnd pmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1
    Xn1 vout vin gnd gnd nmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1
.ends

* --- Inverter ---
Xinv1  vi vo supply 0 mg_inv

* --- Transient Analysis ---
.dc vin 0 1 0.01

.print dc v(vi) v(vo)

.end
