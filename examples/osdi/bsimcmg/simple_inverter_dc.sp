*Sample netlist for BSIM-CMG 
* (exec-spice "ngspice %s" t)
*Inverter DC

.include Modelcards/modelcard.nmos
.include Modelcards/modelcard.pmos
* --- Voltage Sources ---
vdd   supply  0 dc=1.0
vsig  vin  0 dc=0.5 sin (0.5 0.5 1MEG)

NP1 vout vin supply supply BSIMCMG_osdi_P
NN1 vout vin 0 0 BSIMCMG_osdi_N

* --- DC Analysis ---
.dc vsig 0 1 0.01

* --- Transient Analysis ---
*.tran 10n 2u

.control
* pre_osdi ../osdi_libs/bsimcmg.osdi
set xbrushwidth=3
run
plot v(vout) v(vin)
.endc

.end
