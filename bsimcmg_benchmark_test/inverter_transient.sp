*Sample netlist for BSIM-MG 
*Inverter Transient

.option abstol=1e-6 reltol=1e-6 post ingold

.hdl "bsimcmg.va"
.include "modelcard.nmos"
.include "modelcard.pmos"

* --- Voltage Sources ---
vdd   supply  0 dc=1.0
vsig  vi  0 dc=0.5 sin (0.5 0.5 1MEG)

* --- Inverter Subcircuit ---
.subckt mg_inv vin vout vdd gnd
    Xp1 vout vin vdd gnd pmos1 TFIN=15n L=30n NFIN=10 ASEO=1.5e-14 ADEO=1.5e-14 NRS=1 NRD=1
    Xn1 vout vin gnd gnd nmos1 TFIN=15n L=30n NFIN=10 ASEO=1.5e-14 ADEO=1.5e-14 NRS=1 NRD=1
.ends

* --- Inverter ---
Xinv1  vi 1 supply 0 mg_inv
Xinv2  1 2 supply 0 mg_inv
Xinv3  2 3 supply 0 mg_inv
Xinv4  3 4 supply 0 mg_inv
Xinv5  4 vo supply 0 mg_inv

* --- Transient Analysis ---
.tran 10n 5u

.print tran v(vi) v(vo)

.end
