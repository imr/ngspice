*Sample netlist for BSIM-MG
* Geometry-dependent Rds

.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27

.hdl "bsimcmg.va"

.model nmos2 bsimcmg
+ DEVTYPE=1
+ RGEOMOD=1
+ HEPI=15n
+ CRATIO=0.5
+ DELTAPRSD=12.42n
+ RHOC=1.0p
+ LSP=15n
+ HFIN=30n
+ NSD=2.0e+26
+ LINT = 0

.model pmos2 bsimcmg
+ DEVTYPE=0
+ RGEOMOD=1
+ HEPI=15n
+ CRATIO=0.5
+ DELTAPRSD=12.42n
+ RHOC=1.0p
+ LSP=15n
+ HFIN=30n
+ NSD=2.0e+26
+ LINT = 0

.param fp = 45n

* --- Voltage Sources ---
vds supply  0 dc=0
vgs gate  0 dc=0
vbs bulk  0 dc=0

* --- Transistor ---
Xn1 supply gate 0 bulk nmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp SDTERM=0 LRSD=20n
Xn2 supply gate 0 bulk nmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp SDTERM=0 LRSD=40n
Xn3 supply gate 0 bulk nmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp SDTERM=0 LRSD=60n
Xn4 supply gate 0 bulk nmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp SDTERM=0 LRSD=80n
Xp1 supply gate 0 bulk pmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp SDTERM=0 LRSD=20n
Xp2 supply gate 0 bulk pmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp SDTERM=0 LRSD=40n
Xp3 supply gate 0 bulk pmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp SDTERM=0 LRSD=60n
Xp4 supply gate 0 bulk pmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp SDTERM=0 LRSD=80n

* --- DC Analysis ---
.dc vgs 0.0 1.0 2.0
.print dc Xn1:RSGEO Xn2:RSGEO Xn3:RSGEO Xn4:RSGEO
.print dc Xp1:RSGEO Xp2:RSGEO Xp3:RSGEO Xp4:RSGEO

.alter 
.param fp=90n

.end
