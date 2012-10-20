File: simple-meas-tran.sp
* Simple .measurement examples
* transient simulation of two sine signals with different frequencies
vac1 1 0 DC 0 sin(0 1 1k 0 0)
R1 1 0 100k
vac2 2 0 DC 0 sin(0 1.2 0.9k 0 0)
.tran 10u 5m
*
.measure tran tdiff TRIG v(1) VAL=0.5 RISE=1 TARG v(1) VAL=0.5 RISE=2
.measure tran tdiff TRIG v(1) VAL=0.5 RISE=1 TARG v(1) VAL=0.5 RISE=3
.measure tran tdiff TRIG v(1) VAL=0.5 RISE=1 TARG v(1) VAL=0.5 FALL=1
.measure tran tdiff TRIG v(1) VAL=0 FALL=3 TARG v(2) VAL=0 FALL=3
.measure tran tdiff TRIG v(1) VAL=-0.6 CROSS=1 TARG v(2) VAL=-0.8 CROSS=1
.measure tran tdiff TRIG AT=1m TARG v(2) VAL=-0.8 CROSS=3
.measure tran teval WHEN v(2)=0.7 CROSS=LAST
.measure tran teval WHEN v(2)=v(1) FALL=LAST
.measure tran teval WHEN v(1)=v(2) CROSS=LAST
.measure tran yeval FIND v(2) WHEN v(1)=0.2 FALL=2
.measure tran yeval FIND v(2) AT=2m
.measure tran ymax MAX v(2) from=2m to=3m
.measure tran tymax MAX_AT v(2) from=2m to=3m
.measure tran ypp PP v(1) from=2m to=4m
.measure tran yrms RMS v(1) from=2m to=3.5m
.measure tran yavg AVG v(1) from=2m to=4m
.measure tran yint INTEG v(2) from=2m to=3m
.param fval=5
.measure tran yadd param='fval + 7'
.param vout_diff=50k
.meas tran bw_chk param='(vout_diff < 100k) ? 1 : 0'
.measure tran vtest find par('v(2)*v(1)') AT=2.3m
*
.control
run
plot v(1) v(2)
gnuplot ttt i(vac1)
meas tran tdiff TRIG v(1) VAL=0.5 RISE=1 TARG v(1) VAL=0.5 RISE=2
meas tran tdiff TRIG v(1) VAL=0.5 RISE=1 TARG v(1) VAL=0.5 RISE=3
meas tran tdiff TRIG v(1) VAL=0.5 RISE=1 TARG v(1) VAL=0.5 FALL=1
meas tran tdiff TRIG v(1) VAL=0 FALL=3 TARG v(2) VAL=0 FALL=3
meas tran tdiff TRIG v(1) VAL=-0.6 CROSS=1 TARG v(2) VAL=-0.8 CROSS=1
meas tran tdiff TRIG AT=1m TARG v(2) VAL=-0.8 CROSS=3
meas tran teval WHEN v(2)=0.7 CROSS=LAST
meas tran teval WHEN v(2)=v(1) FALL=LAST
meas tran teval WHEN v(1)=v(2) CROSS=LAST
meas tran yeval FIND v(2) WHEN v(1)=0.2 FALL=2
meas tran yeval FIND v(2) AT=2m
meas tran ymax MAX v(2) from=2m to=3m
meas tran tymax MAX_AT v(2) from=2m to=3m
meas tran ypp PP v(1) from=2m to=4m
meas tran yrms RMS v(1) from=2m to=3.5m
meas tran yavg AVG v(1) from=2m to=4m
meas tran yint INTEG v(2) from=2m to=3m
meas tran ymax MAX v(2) from=2m to=3m
meas tran tmax WHEN v(2)=YMAX from=1m to=2m  $ from..to.. not recognized!

.endc
.end
