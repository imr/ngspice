* 51 stage Ring-Osc.

vin in out 2 pulse 2 0 0.1n 5n 1 1 1
vdd dd 0 dc 0 pulse 0 1.5 0 1n 1 1 1
vss ss 0 dc 0
ve  sub  0 dc 0
vn n1 0 0

xinv1 dd ss sub in out25 inv25 
xinv2 dd ss sub out25 out50 inv25
xinv5 dd ss sub out50 out inv1
xinv11 dd ss sub out buf inv1
cout  buf ss 1pF

* this is needed
.option reltol=1e-4

.tran 0.2n 16n
.print tran v(out25) v(out50)
.include ./bsim4soi/nmos4p0.mod
.include ./bsim4soi/pmos4p0.mod


.subckt inv1 dd ss sub in out
mn1  out in  ss  sub  n1  w=4u  l=0.15u  AS=6p AD=6p PS=7u PD=7u pdbcp=0u
mp1  out in  dd  sub  p1  w=10u l=0.15u  AS=15p AD=15p PS=13u PD=13u pdbcp=0u
.ends inv1

.subckt inv5 dd ss sub in out
xinv1 dd ss sub in 1 inv1
xinv2 dd ss sub 1  2 inv1
xinv3 dd ss sub 2  3 inv1
xinv4 dd ss sub 3  4 inv1
xinv5 dd ss sub 4 out inv1
.ends inv5

.subckt inv25 dd ss sub in out
xinv1 dd ss sub in 1 inv5
xinv2 dd ss sub 1  2 inv5
xinv3 dd ss sub 2  3 inv5
xinv4 dd ss sub 3  4 inv5
xinv5 dd ss sub 4 out inv5
.ends inv25

.control
if $?batchmode
* do nothing
else
  save  out25 out50
  run
  plot out25 out50
  let lin-tstart = 4n $ skip the start-up phase
  let lin-tstop = 14n $ end earlier(just for demonstration)
  let lin-tstep = 5p
  linearize out25 out50
  plot out25 out50
endif
.endc

.end
