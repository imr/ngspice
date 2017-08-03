*Sample netlist for BSIM-MG
* (exec-spice "ngspice %s" t)
* Geometry-dependent Rds

.option abstol=1e-6 reltol=1e-6 post ingold
.temp 27

*.hdl "bsimcmg.va"

.model nmos2 NMOS level=17
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

.model pmos2 PMOS level=17
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
Mn1 supply gate 0 bulk 0 nmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp LRSD=20n
Mn2 supply gate 0 bulk 0 nmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp LRSD=40n
Mn3 supply gate 0 bulk 0 nmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp LRSD=60n
Mn4 supply gate 0 bulk 0 nmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp LRSD=80n
Mp1 supply gate 0 bulk 0 pmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp LRSD=20n
Mp2 supply gate 0 bulk 0 pmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp LRSD=40n
Mp3 supply gate 0 bulk 0 pmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp LRSD=60n
Mp4 supply gate 0 bulk 0 pmos2 TFIN=15n L=30n NFIN=10 FPITCH=fp LRSD=80n

* --- DC Analysis ---
.dc vgs 0.0 1.0 0.1
.print dc Xn1:RSGEO Xn2:RSGEO Xn3:RSGEO Xn4:RSGEO
.print dc Xp1:RSGEO Xp2:RSGEO Xp3:RSGEO Xp4:RSGEO

.control
save @Mn1[RSGEO] @Mn2[RSGEO] @Mn3[RSGEO] @Mn4[RSGEO]
save @Mp1[RSGEO] @Mp2[RSGEO] @Mp3[RSGEO] @Mp4[RSGEO]
run
plot @Mn1[RSGEO] @Mn2[RSGEO] @Mn3[RSGEO] @Mn4[RSGEO]
plot @Mp1[RSGEO] @Mp2[RSGEO] @Mp3[RSGEO] @Mp4[RSGEO]

alter @mn1[FPITCH] = 90n
alter @mn2[FPITCH] = 90n
alter @mn3[FPITCH] = 90n
alter @mn4[FPITCH] = 90n
alter @mp1[FPITCH] = 90n
alter @mp2[FPITCH] = 90n
alter @mp3[FPITCH] = 90n
alter @mp4[FPITCH] = 90n
run
plot @Mn1[RSGEO] @Mn2[RSGEO] @Mn3[RSGEO] @Mn4[RSGEO]
plot @Mp1[RSGEO] @Mp2[RSGEO] @Mp3[RSGEO] @Mp4[RSGEO]

.endc

.end
