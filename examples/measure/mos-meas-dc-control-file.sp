***** Single NMOS Transistor .measure (Id-Vd) ***
* Altering device witdth leads to select new model due to binning limits.
* New model has artificially thick gate oxide (changed from default 3n to 4n)
* to demonstrate the effect.
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
set file = $inputdir/meas.out
echo measurements nch.1 model > $file
dc vds 0 3.5 0.05 vgs 3.5 0.5 -0.5
meas dc is_at FIND i(vs) AT=1 >> $file
meas dc is_max max i(vs) >> $file
meas dc vds_at2 when i(vs)=10m >> $file
* starting with branches in descending order of vgs
* trig ist the first branch which crosses 5mA
* Targ is the first branch crossing 10mA
meas dc vd_diff1 trig i(vs)  val=0.005   rise=1 targ i(vs) val=0.01  rise=1 >> $file
* trig ist the first branch which crosses 5mA
* Targ is the second branch crossing 10mA
meas dc vd_diff2 trig i(vs)  val=0.005   rise=2 targ i(vs) val=0.01  rise=2 >> $file
alter @m1[w]=10.01u   ; W is slightly above binning limit
dc vds 0 3.5 0.05 vgs 3.5 0.5 -0.5
echo measurements nch.2 model >> $file
meas dc is_at FIND i(vs) AT=1 >> $file
meas dc is_max max i(vs) >> $file
meas dc vds_at2 when i(vs)=10m >> $file
meas dc vd_diff1 trig i(vs)  val=0.005   rise=1 targ i(vs) val=0.01  rise=1 >> $file
* there is only one branch crossing 10mA, so this second meas fails with targ out of interval
echo
echo The next one will fail (no two branches crossing 10 mA):
meas dc vd_diff2 trig i(vs)  val=0.005   rise=2 targ i(vs) val=0.01  rise=2 >> $file
*rusage all
plot dc1.i(vs) i(vs)
.endc


.end





