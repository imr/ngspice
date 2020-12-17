* Sample netlist: Id-Vd plot *

.option post ingold numdgt=10
.temp 27

*.hdl "bsimbulk.va"
.include "model.l"

vd d 0 dc=1.3
vg g 0 dc=0
vs s 0 dc=0
vb b 0 dc=0

m1 d g s b nch W=10e-6 L=10e-6

.dc vd 0.0 1.3 0.01 vg 0.4 1 0.3

.control
run
plot i(vs)
.endc

.end
