OSDI Diode Test
.options abstol=1e-15


* one voltage source for sweeping, one for sensing:
VD Dx 0 DC 0 AC 1 SIN (0.5 0.2 1M)
Vsense Dx D DC 0 
* Rt T 0 1e10 *not supported Pascal?

* model definitions:
.model dmod_built_in d( bv=5.0000000000e+01 is=1e-13 n=1.05 thermal=1 tnom=27 rth0=100 rs=5 cj0=0 vj=0.5 m=0.6 )
.model dmod_osdi osdi diode_va rs=5 is=1e-13 n=1.05 Rth=100 cj0=0 vj=0.5 m=0.6

*OSDI Diode:
A1 D 0 T dmod_osdi

*Built-in Diode:
*BUILT_IN_ACTIVATE*D1 D 0 T dmod_built_in


.control
set filetype=ascii
set wr_vecnames
set wr_singlescale

* a DC sweep from 0.3V to 1V
*dc Vd 0.3 1.0 0.01
*wrdata dc_sim.ngspice v(d) i(vsense) v(t)

* an AC sweep at Vd=0.5V
* alter VD=0.5
* ac dec 10 .01 10
* wrdata ac_sim.ngspice v(d) i(vsense)

* a transient analysis
tran 100ms 500000ms 
wrdata tr_sim.ngspice v(d) i(vsense)

* print number of iterations
rusage totiter

.endc

.end