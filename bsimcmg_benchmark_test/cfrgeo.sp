*Sample netlist for BSIM-MG 
* Geometry-dependent Cfr
*
.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27

.hdl "bsimcmg.va"

.param hfin=30n

.model nmos2 bsimcmg
+ DEVTYPE=1
+ CGEOMOD=2
+ HEPI=10n
+ LSP=5n
+ EPSRSP=7.5
+ TGATE=40n 
+ TMASK=10n 
+ TSILI=0n 
+ CRATIO=1.0
+ EOT=1.0n 
+ TOXP=1.2n 
+ HFIN=hfin 

* --- Voltage Sources ---
vds supply  0 dc=0
vgs gate  0 dc=0
vbs bulk  0 dc=0

* --- Transistor ---
X1 supply gate 0 bulk nmos2 TFIN=10n L=30n NFIN=1 FPITCH=20n LRSD=40n
X2 supply gate 0 bulk nmos2 TFIN=10n L=30n NFIN=1 FPITCH=40n LRSD=40n
X3 supply gate 0 bulk nmos2 TFIN=10n L=30n NFIN=1 FPITCH=60n LRSD=40n
X4 supply gate 0 bulk nmos2 TFIN=10n L=30n NFIN=1 FPITCH=80n LRSD=40n

* --- DC Analysis ---
.dc vgs 0.0 1.0 1.5
.print dc par'hfin' X1:CFGEO X2:CFGEO X3:CFGEO X4:CFGEO

.alter
.param hfin=40n

.alter
.param hfin=50n

.alter
.param hfin=60n

.end
