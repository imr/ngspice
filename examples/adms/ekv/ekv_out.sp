EPFL-EKV version 2.6 nch

.include ekv26_0u5.par

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
