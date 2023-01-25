*Sample netlist for BSIM-CMG 
* (exec-spice "ngspice %s" t)
*Inverter Transient

.include Modelcards/modelcard.nmos
.include Modelcards/modelcard.pmos
* --- Voltage Sources ---
vdd   supply  0 dc=1.0
vsig  vi  0 dc=0.5 sin (0.5 0.5 1MEG)

* --- Inverter Subcircuit ---
.subckt mg_inv vin vout vdd gnd
NP1 vout vin vdd vdd BSIMCMG_osdi_P
NN1 vout vin gnd gnd BSIMCMG_osdi_N
.ends

* --- Inverter ---
Xinv1  vi 1 supply 0 mg_inv
Xinv2  1 2 supply 0 mg_inv
Xinv3  2 3 supply 0 mg_inv
Xinv4  3 4 supply 0 mg_inv
Xinv5  4 vo supply 0 mg_inv

* --- Transient Analysis ---
.tran 20n 5u

.print tran v(vi) v(vo)

.control
* pre_osdi ../osdi_libs/bsimcmg.osdi
set xbrushwidth=3
run
plot v(vi) v(vo)
.endc

.end
