* Sample netlist: Gummel symmetry test *

.option post ingold numdgt=10
.temp 27

*.hdl "bsimbulk.va"
.include "model.l"

vd d 0 dc=0
vg g 0 dc=1.0
es s 0 d 0 -1
vb b 0 dc=0

m1 d g s b nch W=10u L=10u

.dc vd -0.1 0.1 0.001 vg 0.0 1.0 0.2
.probe dc ids=par'-i(vd)'
.probe dc gx=deriv(ids)
.probe dc gx2=deriv(gx)
.probe dc gx3=deriv(gx2)
.print dc par'ids' par'gx' par'gx2' par'gx3'

.end
