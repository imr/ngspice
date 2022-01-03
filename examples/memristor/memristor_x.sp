Memristor with threshold as XSPICE code model
* Y. V. Pershin, M. Di Ventra: "SPICE model of memristive devices with threshold", 
* arXiv:1204.2600v1 [physics.comp-ph] 12 Apr 2012, 
* http://arxiv.org/pdf/1204.2600.pdf

* XSPICE code model, parameter selection and plotting by
* Holger Vogt 2012

* ac and op (dc) simulation just use start resistance rinit!

.param stime=10n
.param vmax = 4.2

* send parameters to the .control section
.csparam stime={stime}
.csparam vmax={vmax}

*Xmem 1 0 memristor
* triangular sweep (you have to adapt the parameters to 'alter' command in the .control section)
*V1 1 0 DC 0 PWL(0 0 '0.25*stime' 'vmax' '0.5*stime' 0 '0.75*stime' '-vmax' 'stime' 0)
* sinusoidal sweep for transient, dc for op, ac
V1 0 1 DC 0.1 ac 1 sin(0 'vmax' '1/stime')

Rl 1 11 1k

* memristor model with limits and threshold
* "artificial" parameters alpha, beta, and vt. beta and vt adapted to basic programming frequency 
* just to obtain nice results!
* You have to care for the physics and set real values! 
amen 11 2 memr
.model memr memristor (rmin=1k rmax=10k rinit=7k alpha=0 beta='20e3/stime' vt=1.6)

vgnd 2 0 dc 0

* This is the original subcircuit model
.subckt memristor plus minus PARAMS: Ron=1K Roff=10K Rinit=7.0K alpha=0 beta=20e3/stime Vt=1.6 
Bx 0 x I='((f1(V(plus)-V(minus))> 0) && (V(x) < Roff)) ? {f1(V(plus)-V(minus))}: ((((f1(V(plus)-V(minus)) < 0) && (V(x)>Ron)) ? {f1(V(plus)-V(minus))}: 0)) '
Vx x x1 dc 0
Cx x1 0 1 IC={Rinit} 
Rmem plus minus r={V(x)} 
.func f1(y)={beta*y+0.5*(alpha-beta)*(abs(y+Vt)-abs(y-Vt))} 
.ends

* transient simulation same programming voltage but rising frequencies
.control
*** first simulation ***
op 
print all
ac lin 101 1 100k
*plot v(11)
* approx. 100 simulation points
let deltime = stime/100
tran $&deltime $&stime uic
*plot i(v1) vs v(1) retrace
*** you may just stop here ***
* raise the frequency
let newfreq = 1.2/stime
let newstime = stime/1.2
let deltime = newstime/100
alter @V1[sin] [ 0 $&vmax $&newfreq ]
tran $&deltime $&newstime uic
* raise the frequency even more
let newfreq = 1.4/stime
let newstime = stime/1.4
let deltime = newstime/100
alter @V1[sin] [ 0 $&vmax $&newfreq ]
tran $&deltime $&newstime uic
* the resistor currents
plot tran1.alli tran2.alli alli title 'Memristor with threshold: currents'
* calculate resistance (avoid dividing by zero)
let res = v(1)/(I(v1) + 1e-16)
let res1 = tran1.v(1)/(tran1.I(v1) + 1e-16)
let res2 = tran2.v(1)/(tran2.I(v1) + 1e-16)
* resistance versus time plot
settype impedance res res1 res2
plot res vs time res1 vs tran1.time res2 vs tran2.time  title 'Memristor with threshold: resistance'
* resistance versus voltage (change occurs only above threshold!)
plot res vs v(1) res1 vs tran1.v(1) res2 vs tran2.v(1) retraceplot title 'Memristor with threshold: resistance'
* current through resistor for all plots versus voltage
plot i(v1) vs v(1) tran1.i(v1) vs tran1.v(1) tran2.i(v1) vs tran2.v(1) retraceplot title 'Memristor with threshold: external current loops'
rusage
.endc

.end
