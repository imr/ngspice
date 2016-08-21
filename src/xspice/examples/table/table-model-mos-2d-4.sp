Code Model Test - 2d Table Model
* Inverter
*
*** analysis type ***
.control
*option reltol=0.1
dc V1 0 1.5 0.01
*op
*ac lin 11 100 200
*tran 100p 20n
*plot i(Vs) i(Vs2)
plot i(vsinv)
plot v(in1) v(out1)
plot deriv(v(out1))
.endc
*
*** input sources ***
*
v1 in1 0 DC 0.75 ac 1
Vs2 s2 0 0

vsinv vss 0 0
vdinv vdd 0 1.5
*
*********************

*xmosnt d g s tbmosn
*mn2 d g s2 s2 n1 l=0.13u w=10u ad=5p pd=6u as=5p ps=6u  rgeoMod=1

.subckt inv vd vs in out
*mp2 out in vd vd p1 l=0.13u w=10u ad=5p pd=6u as=5p ps=6u
xmospt out in vd tbmosp
*mn2 out in vs vs n1 l=0.13u w=5u ad=5p pd=6u as=5p ps=6u
xmosnt out in vs tbmosn
.ends

xmosinv1 vdd vss in1 out1 inv

.subckt tbmosn d g s
*** table model of mos transistor ***
cdg d g 0.01p
csg s g 0.014p
amos1 %vd(d s) %vd(g s) %id(d s) mostable1
.model mostable1 table2d (offset=0.0 gain=0.5 order=2 file="bsim4n-2d-3.table")
* NMOS L=0.13u W=10.0u rgeoMod=1
* BSIM 4.7
* change width of transistor by modifying parameter "gain"
* source is always tied to bulk (we not yet have a 3D table model!)
.ends

.subckt tbmosp d g s
*** table model of pmos transistor ***
cdg d g 0.01p
csg s g 0.014p
amos2 %vd(d s) %vd(g s) %id(d s) mostable2
.model mostable2 table2d (offset=0.0 gain=1 order=3 file="bsim4p-2d-3.table")
* PMOS L=0.13u W=10.0u rgeoMod=1
* BSIM 4.7
* change width of transistor by modifying parameter "gain"
* source is always tied to bulk (we not yet have a 3D table model!)
.ends

.include ./modelcards/modelcard.nmos
.include ./modelcards/modelcard.pmos

.end
