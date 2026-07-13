Plot inner small signal parameter

v1 1 0 dc 0
d1 1 0 myd
.model myd D(IS = 1.50E-07
+ N = 1.0
+ RS = 9
+ TT = 100n
+ CJ0 = 1.01p
+ VJ = 0.44
+ M = 0.5
+ EG = 1.11
+ XTI = 3
+ KF = 0
+ AF = 1
+ FC = 0.5
+ BV = 22
+ IBV = 10u)

.control
save @d1[gd] @d1[cd] @d1[qd]
dc v1 -5 .1 0.01
plot @d1[gd]
plot @d1[cd]
plot @d1[qd]
.endc
*
.END
