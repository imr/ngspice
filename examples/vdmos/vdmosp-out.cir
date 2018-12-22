VDMOS p channel output

m1 d g s p1
.model p1 vdmos pchan vto=-1.2 is=10n kp=2 bv=-12 rb=1k

vd d 0 -5
vg g 0 -5
vs s 0 0

.dc vd -15 1 0.1 vg 0 -5 -1

.control
run
plot vs#branch
.endc

.end
