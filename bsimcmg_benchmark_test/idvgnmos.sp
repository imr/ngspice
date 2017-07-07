*Sample netlist for BSIM-MG
*Id-Vg Characteristics for NMOS (T = 27 C)

.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27

.hdl "bsimcmg.va"
.include "modelcard.nmos.1"

* --- Voltage Sources ---
vds supply  0 dc=0.05
vgs gate  0 dc=1
vbs bulk  0 dc=0
vt   t    0 dc= 0

* --- Transistor ---
X1 supply gate 0 bulk  t nmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1

* --- DC Analysis ---
.dc vgs -0.5 1.0 0.01 vds 0.05 1 0.95
.probe dc par'-i(vds)'
.probe dc par'-i(vbs)'
.print dc i(X1.d)

.alter 
.temp -55

.alter
.temp 100

.end
