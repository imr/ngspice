*Sample netlist for BSIM-MG 
* (exec-spice "ngspice %s" t)
* Geometry-dependent Cfr
*
.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27

*.hdl "bsimcmg.va"

.param hfin=30n

.model nmos2 NMOS level=17
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
M1 supply gate 0 bulk 0 nmos2 TFIN=10n L=30n NFIN=1 FPITCH=20n LRSD=40n
M2 supply gate 0 bulk 0 nmos2 TFIN=10n L=30n NFIN=1 FPITCH=40n LRSD=40n
M3 supply gate 0 bulk 0 nmos2 TFIN=10n L=30n NFIN=1 FPITCH=60n LRSD=40n
M4 supply gate 0 bulk 0 nmos2 TFIN=10n L=30n NFIN=1 FPITCH=80n LRSD=40n

* --- DC Analysis ---
.dc vgs 0.0 1.0 0.1
*.print dc par'hfin' M1:CFGEO M2:CFGEO M3:CFGEO M4:CFGEO

.control
save @m1[CFGEO] @m2[CFGEO] @m3[CFGEO] @m4[CFGEO]

showmod #nmos2 : HFIN
run
plot @m1[CFGEO] @m2[CFGEO] @m3[CFGEO] @m4[CFGEO]

altermod nmos2 hfin = 40n
showmod #nmos2 : HFIN
run
plot @m1[CFGEO] @m2[CFGEO] @m3[CFGEO] @m4[CFGEO]

altermod nmos2 hfin = 50n
showmod #nmos2 : HFIN
run
plot @m1[CFGEO] @m2[CFGEO] @m3[CFGEO] @m4[CFGEO]

altermod nmos2 hfin = 60n
showmod #nmos2 : HFIN
run
plot @m1[CFGEO] @m2[CFGEO] @m3[CFGEO] @m4[CFGEO]

.endc

.end
