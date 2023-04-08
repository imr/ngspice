VBIC Output Test Ic=f(Vc,Ib) vs self heating
vc c 0 0
ib 0 b 10u
ve e 0 0
vs s 0 0
vc1 c c1 0
vb1 b b1 0
ve1 e e1 0
vs1 s s1 0
.temp 27
Q1 c1 b1 e1 s1 dt M_BFP780 area=1

.include Infineon_VBIC.lib

.control
dc vc 0.0 5.0 0.05 ib 50u 500u 50u
settype temperature v(dt)
plot v(dt)
altermod @M_BFP780[RTH]=0
dc vc 0.0 5.0 0.05 ib 50u 500u 50u
plot dc1.vc1#branch dc2.vc1#branch
.endc
.end

