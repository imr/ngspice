*Sample netlist for BSIM-MG
*AC Response 

.option abstol=1e-6 reltol=1e-6 post ingold

.hdl "bsimcmg.va"
.include "modelcard.nmos"

.param myvdd=1.0

* --- Voltage Sources ---
vdd supply  0 dc=myvdd
vsig  gate  0 dc=0.5 ac=1
vbs bulk 0 dc=0

* --- Transistor ---
X1 vout gate 0 bulk nmos1 TFIN=15n L=30n NFIN=10 NRS=1 NRD=1

* --- Load ---
rl supply vout r=2k
cl supply vout c=10f

* --- AC Analysis ---
.ac dec 10 1k 1T

* For Bias Point Testing 
* .dc vsig -1 1.5 0.01

.print ac vm(vout) vp(vout)

*.alter
*.param myvdd=2.0

.end

