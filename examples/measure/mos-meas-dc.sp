***** Single NMOS Transistor .measure (Id-Vd) ***
m1 d g s b nch L=0.6u W=10.0u

vgs g 0 3.5 
vds d 0 3.5 
vs s 0 dc 0
vb b 0 dc 0

.dc vds 0 3.5 0.05 vgs 0.5 3.5 0.5 

.print dc v(1) i(vs)

* model binning
.model nch.1 nmos ( version=4.7 level=54 lmin=0.1u lmax=20u wmin=0.1u wmax=10u  )
.model nch.2 nmos ( version=4.7 level=54 lmin=0.1u lmax=20u wmin=10u  wmax=100u )
.model pch.1 pmos ( version=4.7 level=54 lmin=0.1u lmax=20u wmin=0.1u wmax=10u  )
.model pch.2 pmos ( version=4.7 level=54 lmin=0.1u lmax=20u wmin=10u  wmax=100u )

.meas dc is_at FIND i(vs) AT=1
.meas dc is_max max i(vs) from=0 to=3.5
.meas dc vds_at2 when i(vs)=10m
.meas dc vd_diff1 trig i(vs)  val=0.005   rise=1 targ i(vs) val=0.01  rise=1
.meas dc vd_diff2 trig i(vs)  val=0.005   rise=1 targ i(vs) val=0.01  rise=2

.control
run
*rusage all
plot i(vs)
.endc


.end





