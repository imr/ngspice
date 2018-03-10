* Sample netlist: Id-Vg plot *

.option post ingold numdgt=10
.temp 27

*.hdl "bsimbulk.va"
.include "model.l"

vd d 0 dc=-0.05
vg g 0 dc=0
vs s 0 dc=0
vb b 0 dc=0

m1 d g s b pch W=10e-6 L=10e-6

.dc vg -1.3.0 1.3 0.01 vb 0 -0.3 -0.1
*.probe dc ids=par'i(vd)'
*.probe dc gm=deriv(ids)
*.probe dc gm2=deriv(gm)
*.print dc par'ids' par'gm' par'gm2'


.control
run
plot i(vs)
.endc

.end
