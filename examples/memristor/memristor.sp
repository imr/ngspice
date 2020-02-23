Memristor with threshold
* Y. V. Pershin, M. Di Ventra: "SPICE model of memristive devices with threshold", 
* arXiv:1204.2600v1 [physics.comp-ph] 12 Apr 2012, 
* http://arxiv.org/pdf/1204.2600.pdf

* Parameter selection and plotting by
* Holger Vogt 2012

.param stime=10n
.param vmax = 3

* send parameters to the .control section
.csparam stime={stime}
.csparam vmax={vmax}

Xmem 1 0 memristor
* triangular sweep (you have to adapt the parameters to 'alter' command in the .control section)
*V1 1 0 DC 0 PWL(0 0 '0.25*stime' 'vmax' '0.5*stime' 0 '0.75*stime' '-vmax' 'stime' 0)
* sinusoidal sweep
V1 0 1 DC 0 sin(0 'vmax' '1/stime')

* memristor model with limits and threshold
* "artificial" parameters alpha, beta, and vt. beta and vt adapted to basic programming frequency 
* just to obtain nice results!
* You have to care for the physics and set real values! 
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
* approx. 100 simulation points
let deltime = stime/100
tran $&deltime $&stime uic
* plot i(v1) vs v(1)
*** you may just stop here ***
* raise the frequency
let newfreq = 1.1/stime
let newstime = stime/1.1
let deltime = newstime/100
alter @V1[sin] [ 0 $&vmax $&newfreq ]
tran $&deltime $&newstime uic
* raise the frequency even more
let newfreq = 1.4/stime
let newstime = stime/1.4
let deltime = newstime/100
alter @V1[sin] [ 0 $&vmax $&newfreq ]
tran $&deltime $&newstime uic
* the 'programming' currents
plot tran1.alli tran2.alli alli title 'Memristor with threshold: Internal Programming currents'
* resistance versus time plot
settype impedance xmem.x1 tran1.xmem.x1 tran2.xmem.x1
plot xmem.x1 tran1.xmem.x1 tran2.xmem.x1 title 'Memristor with threshold: resistance'
* resistance versus voltage (change occurs only above threshold!)
plot xmem.x1 vs v(1) tran1.xmem.x1 vs tran1.v(1) tran2.xmem.x1 vs tran2.v(1) retraceplot title 'Memristor with threshold: resistance'
* current through resistor for all plots versus voltage
plot i(v1) vs v(1) tran1.i(v1) vs tran1.v(1) tran2.i(v1) vs tran2.v(1) retraceplot title 'Memristor with threshold: external current loops'
.endc

.end
