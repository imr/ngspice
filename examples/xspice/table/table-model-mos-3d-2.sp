Code Model Test - 3d Table Model
* bsim4 transistor dc input and output characteristics
*
*** analysis type ***
.control
dc V1 -0.1 1.7 0.06 V2 0.2 1.7 0.25
plot i(Vs)
plot deriv(i(Vs))
reset
dc v2 0 1.7 0.04
plot i(Vs)
plot deriv(i(Vs))
reset
dc V1 -0.1 1.7 0.06
plot i(Vs2)
.endc
*
*** input sources ***
*
v1 d 0 DC 0.1
*
v2 g 0 DC 1.5
*
Vs s 0 0
Vb b 0 -0.5
Vs2 s2 0 0
*
*** table model of mos transistor ***
amos1 %vd(d s) %vd(g s) %vd(b s) %id(d s) mostable1
.model mostable1 table3d (offset=0.0 gain=1 order=4 file="bsim4n-3d-1.table")
* L=0.13u W=10.0u rgeoMod=1
* BSIM 4.7
* change width of transistor by modifying parameter "gain"

amos2 %vd(d s2) %vd(d s2) %vd(b s2) %id(d s2) mostable1

.end
