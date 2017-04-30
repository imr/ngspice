***** Single NMOS Transistor .measure (Id-Vd) ***
* Do measurements
* Clip the output current vector.
* Do some measurements
* Unclip the vector
* Do all measurements again

m1 d g s b nch L=0.6u W=9.99u  ; W is slightly below binning limit

vgs g 0 3.5 
vds d 0 3.5 
vs s 0 dc 0
vb b 0 dc 0

* model binning
* uses default parameters, except toxe
.model nch.1 nmos ( version=4.7 level=54 lmin=0.1u lmax=20u wmin=0.1u wmax=10u toxe=3n )
.model nch.2 nmos ( version=4.7 level=54 lmin=0.1u lmax=20u wmin=10u  wmax=100u toxe=4n)

.control
dc vds 0 3.5 0.05 vgs 3.5 0.5 -0.5
meas dc is_at FIND i(vs) AT=1
meas dc is_max max i(vs)
meas dc vds_at2 when i(vs)=10m rise=1
meas dc vds_at2 when i(vs)=10m rise=2
* starting with branches in descending order of vgs
* trig ist the first branch which crosses 5mA
* Targ is the first branch crossing 10mA
meas dc vd_diff1 trig i(vs)  val=0.005   rise=1 targ i(vs) val=0.01  rise=1
* trig ist the first branch which crosses 5mA
* Targ is the second branch crossing 10mA
meas dc vd_diff2 trig i(vs)  val=0.005   rise=2 targ i(vs) val=0.01  rise=2

plot i(vs)

* clip the vector
clip i(vs) 1 1.5
plot i(vs)
echo
echo after clipping
meas dc vds_at2 when vs#branch=10m rise=1
meas dc vds_at2 when vs#branch=10m rise=2

* restore the unclipped vector
echo
echo after unclipping
clip i(vs) 0 0
meas dc is_at FIND i(vs) AT=1
meas dc is_max max i(vs)
meas dc vds_at2 when i(vs)=10m rise=1
meas dc vds_at2 when i(vs)=10m rise=2
* starting with branches in descending order of vgs
* trig ist the first branch which crosses 5mA
* Targ is the first branch crossing 10mA
meas dc vd_diff1 trig i(vs)  val=0.005   rise=1 targ i(vs) val=0.01  rise=1
* trig ist the first branch which crosses 5mA
* Targ is the second branch crossing 10mA
meas dc vd_diff2 trig i(vs)  val=0.005   rise=2 targ i(vs) val=0.01  rise=2
plot i(vs)

.endc


.end





