Code Model Test - 3d Table Model
* Ring oscillator made of NAND gates
*
*** analysis type ***
.control
option trtol=1
*dc V1 0.0 1.7 0.1 V2 0.3 1.7 0.3
*op
tran 100p 20n
*plot i(Vs) i(Vs2)
plot v(in1)
rusage
.endc
*
*** input sources ***
*
v1 d 0 DC 1.5
v2 g 0 DC 1.5
Vs s 0 0
Vs2 s2 0 0

vsinv vss 0 0
vdinv vdd 0 1.5
*
*********************

*xmosnt d g s tbmosn
*mn2 d g s2 s2 n1 l=0.13u w=10u ad=5p pd=6u as=5p ps=6u  rgeoMod=1

.SUBCKT NAND VDD VSS in1 in2 out
*   NODES:   VCC, Ground, INPUT(2), OUTPUT
xmospt1 out in2 vdd vdd tbmosp
xmosnt2 net.1 in2 vss vss tbmosn
xmospt3 out in1 vdd vdd tbmosp
xmosnt4 out in1 net.1 vss tbmosn
.ENDS NAND


xmosinv1 vdd vss in1 in1 out1 nand
xmosinv2 vdd vss out1 out1 out2 nand
xmosinv3 vdd vss out2 out2 out3 nand
xmosinv4 vdd vss out3 out3 out4 nand
xmosinv5 vdd vss out4 out4 in1 nand

.subckt tbmosn d g s b
*** table model of nmos transistor ***
cdg d g 0.01p
csg s g 0.014p
amos1 %vd(d s) %vd(g s) %vd(b s) %id(d s) mostable1
.model mostable1 table3d (offset=0.0 gain=0.5 order=3 file="bsim4n-3d-1.table")
* NMOS L=0.13u W=10.0u rgeoMod=1
* BSIM 4.7
* change width of transistor by modifying parameter "gain"
.ends

.subckt tbmosp d g s b
*** table model of pmos transistor ***
cdg d g 0.01p
csg s g 0.014p
amos2 %vd(d s) %vd(g s) %vd(b s) %id(d s) mostable2
.model mostable2 table3d (offset=0.0 gain=1 order=3 file="bsim4p-3d-1.table")
* PMOS L=0.13u W=10.0u rgeoMod=1
* BSIM 4.7
* change width of transistor by modifying parameter "gain"
.ends

.include ./modelcards/modelcard.nmos
.include ./modelcards/modelcard.pmos

.end
