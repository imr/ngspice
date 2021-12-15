*Sample netlist for BSIM-MG
* (exec-spice "ngspice %s" t)
*Id-Vg Characteristics for NMOS (T = 27 C)

.option abstol=1e-6 reltol=1e-6 post ingold

*.hdl "bsimcmg.va"
.include "modelcard.nmos.1"

* --- Voltage Sources ---
vds supply  0 dc=0.05
vgs gate  0 dc=1
vbs bulk  0 dc=0
vt   t    0 dc= 0

* --- Transistor ---
m1 supply gate 0 bulk  t nmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1 D=40n

* --- DC Analysis ---
.dc vgs -0.5 1.0 0.01 vds 0.05 1 0.95
*.probe dc par'-i(vds)'
*.probe dc par'-i(vbs)'
*.print dc i(X1.d)

.save i(vds) i(vbs)

.control
set temp = 27
run
plot -i(vds)
plot -i(vbs)

set temp = -55
run
plot -i(vds)
plot -i(vbs)

set temp = 100
run
plot -i(vds)
plot -i(vbs)

.endc

.end
