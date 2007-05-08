EPFL-EKV version 2.6 nch
.model nch nmos level=44
*** Electrical Parameter
+ vto   = 0.7      gamma = 0.7       phi   = 0.5
+ kp    = 150e-06  theta = 50e-03
+ tnom  = 25.0
*
vd  d 0 dc 0.1
vg  g 0 dc 0.0
vs  s 0 dc 0.0
vb  b 0 dc 0.0
m1  d g s b nch
*
* Transfer
.control
dc vd 0 5 0.01 vg 1 5 1
plot abs(i(vd))
.endc
*
.end
